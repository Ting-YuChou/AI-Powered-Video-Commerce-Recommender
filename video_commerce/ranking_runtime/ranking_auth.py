"""Connection-level authentication for the internal ranking TCP protocol.

Every coordinator/runner connection may open with an ``A`` frame whose JSON
body is ``{"v": 1, "role": <server role>, "ts": <unix seconds>,
"nonce": <hex>, "mac": <hex>}``.
``mac = HMAC-SHA256(K, "ranking-auth|v1|<role>|<ts>|<nonce>")`` where
``K = HMAC-SHA256(SECURITY_INTERNAL_SERVICE_KEY, "ranking-runtime-v1")`` and
``role`` is the role of the server being dialled (``coordinator`` or
``runner``), so a handshake captured for one role is useless against the other.
Servers accept timestamps within the clock-skew window, reject nonces they
have already seen, and treat the rest of the connection as authenticated.

This authenticates peers; it does not encrypt or integrity-protect later
frames, and the nonce cache is per process (a captured handshake replays
against another replica of the same role, or after a restart, within the
skew window). Network policy remains the boundary against on-path attackers;
mTLS is future work.

``RANKING_RUNTIME_AUTH_MODE`` controls rollout:

- ``disabled``: clients never send ``A``; servers treat ``A`` as unknown.
- ``permissive`` (default): clients send ``A`` when a key is configured and
  tolerate a legacy server answering 400; servers accept unauthenticated
  connections but count them.
- ``required``: clients fail when the handshake is not accepted; servers
  reject any other frame on an unauthenticated connection.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import os
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


AUTH_OPERATION = b"A"
AUTH_PROTOCOL_VERSION = 1
AUTH_MODE_DISABLED = "disabled"
AUTH_MODE_PERMISSIVE = "permissive"
AUTH_MODE_REQUIRED = "required"
AUTH_MODES = (AUTH_MODE_DISABLED, AUTH_MODE_PERMISSIVE, AUTH_MODE_REQUIRED)
ROLE_COORDINATOR = "coordinator"
ROLE_RUNNER = "runner"
SERVER_ROLES = (ROLE_COORDINATOR, ROLE_RUNNER)
DEFAULT_AUTH_MODE = AUTH_MODE_PERMISSIVE
DEFAULT_MAX_CLOCK_SKEW_SECONDS = 60
DEFAULT_NONCE_CACHE_SIZE = 65536
DEFAULT_HANDSHAKE_TIMEOUT_SECONDS = 10.0
MAX_AUTH_BODY_BYTES = 1024
# Operation byte plus the largest accepted handshake body.
MAX_PRE_AUTH_FRAME_BYTES = 1 + MAX_AUTH_BODY_BYTES
_KEY_DERIVATION_LABEL = b"ranking-runtime-v1"
_MAC_PREFIX = "ranking-auth|v1"


def normalize_auth_mode(raw_mode: Optional[str]) -> str:
    mode = (raw_mode or DEFAULT_AUTH_MODE).strip().lower()
    if mode not in AUTH_MODES:
        raise ValueError(
            f"RANKING_RUNTIME_AUTH_MODE must be one of {', '.join(AUTH_MODES)}"
        )
    return mode


def derive_runtime_key(internal_service_key: str) -> bytes:
    return hmac.new(
        internal_service_key.encode("utf-8"),
        _KEY_DERIVATION_LABEL,
        hashlib.sha256,
    ).digest()


def compute_auth_mac(runtime_key: bytes, role: str, ts: int, nonce: str) -> str:
    message = f"{_MAC_PREFIX}|{role}|{int(ts)}|{nonce}".encode("utf-8")
    return hmac.new(runtime_key, message, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class RankingRuntimeAuthSettings:
    """Resolved auth mode plus the derived runtime key (never the raw secret)."""

    mode: str = AUTH_MODE_DISABLED
    runtime_key: Optional[bytes] = None

    @classmethod
    def build(
        cls, mode: Optional[str], internal_service_key: Optional[str]
    ) -> "RankingRuntimeAuthSettings":
        normalized = normalize_auth_mode(mode)
        key = (internal_service_key or "").strip()
        return cls(
            mode=normalized,
            runtime_key=derive_runtime_key(key) if key else None,
        )

    @classmethod
    def from_config(cls, config: Any) -> "RankingRuntimeAuthSettings":
        return cls.build(
            getattr(config.ranking_config, "runtime_auth_mode", DEFAULT_AUTH_MODE),
            config.security_config.internal_service_key,
        )

    @classmethod
    def from_env(cls) -> "RankingRuntimeAuthSettings":
        """Lightweight resolution for probes that must not build a full Config."""
        key = os.getenv("SECURITY_INTERNAL_SERVICE_KEY") or ""
        key_file = os.getenv("SECURITY_INTERNAL_SERVICE_KEY_FILE") or ""
        if not key and key_file:
            with open(key_file, "r", encoding="utf-8") as handle:
                key = handle.read().strip()
        return cls.build(os.getenv("RANKING_RUNTIME_AUTH_MODE"), key)

    @property
    def client_sends_handshake(self) -> bool:
        return self.mode != AUTH_MODE_DISABLED and self.runtime_key is not None

    @property
    def client_tolerates_legacy_server(self) -> bool:
        return self.mode == AUTH_MODE_PERMISSIVE

    def build_handshake_body(
        self,
        server_role: str,
        *,
        now: Optional[float] = None,
        nonce: Optional[str] = None,
    ) -> bytes:
        if self.runtime_key is None:
            raise ValueError("ranking runtime auth key is not configured")
        if server_role not in SERVER_ROLES:
            raise ValueError(f"unknown ranking runtime server role: {server_role}")
        ts = int(time.time() if now is None else now)
        nonce = nonce or secrets.token_hex(16)
        return json.dumps(
            {
                "v": AUTH_PROTOCOL_VERSION,
                "role": server_role,
                "ts": ts,
                "nonce": nonce,
                "mac": compute_auth_mac(self.runtime_key, server_role, ts, nonce),
            },
            separators=(",", ":"),
        ).encode("utf-8")


@dataclass
class ConnectionAuthState:
    authenticated: bool = False
    unauthenticated_recorded: bool = False


@dataclass(frozen=True)
class AuthDecision:
    """Response the auth layer sends instead of dispatching the frame."""

    status_code: int
    payload: Dict[str, Any]
    close_connection: bool


class RankingRuntimeAuthenticator:
    """Server-side handshake verification with a bounded replay cache."""

    def __init__(
        self,
        settings: RankingRuntimeAuthSettings,
        *,
        role: str,
        observability: Any = None,
        max_clock_skew_seconds: float = DEFAULT_MAX_CLOCK_SKEW_SECONDS,
        nonce_cache_size: int = DEFAULT_NONCE_CACHE_SIZE,
        handshake_timeout_seconds: float = DEFAULT_HANDSHAKE_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self.role = role
        self.observability = observability
        self.max_clock_skew_seconds = float(max_clock_skew_seconds)
        self.nonce_cache_size = max(1, int(nonce_cache_size))
        self.handshake_timeout_seconds = max(0.1, float(handshake_timeout_seconds))
        self._clock = clock
        self._seen_nonces: "OrderedDict[str, float]" = OrderedDict()

    @property
    def verifies_handshakes(self) -> bool:
        return (
            self.settings.mode != AUTH_MODE_DISABLED
            and self.settings.runtime_key is not None
        )

    def pre_auth_limits(
        self, state: ConnectionAuthState
    ) -> Optional[Tuple[int, float]]:
        """``(max_frame_bytes, timeout_seconds)`` for the next read, if bounded.

        In required mode an unauthenticated peer may only send a small
        handshake, promptly; otherwise it could make the server buffer a
        maximum-size frame, or hold the connection, before auth runs.
        """
        if self.settings.mode != AUTH_MODE_REQUIRED or state.authenticated:
            return None
        return MAX_PRE_AUTH_FRAME_BYTES, self.handshake_timeout_seconds

    def record_pre_auth_violation(self) -> None:
        self._record("rejected")

    def screen_frame(
        self,
        state: ConnectionAuthState,
        operation: bytes,
        body: bytes,
    ) -> Optional[AuthDecision]:
        """Return a decision when the auth layer answers this frame itself.

        ``None`` means dispatch the frame normally.
        """
        if self.settings.mode == AUTH_MODE_DISABLED:
            return None
        if operation == AUTH_OPERATION and self.verifies_handshakes:
            failure = self.verify_handshake(body)
            if failure is None:
                state.authenticated = True
                self._record("authenticated")
                return AuthDecision(200, {"status": "authenticated"}, False)
            self._record(failure)
            return AuthDecision(
                401,
                {"detail": "ranking_runtime_auth_failed", "reason": failure},
                True,
            )
        if state.authenticated:
            return None
        if operation == AUTH_OPERATION:
            # No key on this server: answer like a server that predates ``A``.
            return None
        if self.settings.mode == AUTH_MODE_REQUIRED:
            self._record("rejected")
            return AuthDecision(
                401, {"detail": "ranking_runtime_auth_required"}, True
            )
        if not state.unauthenticated_recorded:
            state.unauthenticated_recorded = True
            self._record("unauthenticated")
        return None

    def verify_handshake(self, body: bytes) -> Optional[str]:
        """Return ``None`` on success or a short failure reason."""
        runtime_key = self.settings.runtime_key
        if runtime_key is None:
            return "unconfigured"
        if len(body) > MAX_AUTH_BODY_BYTES:
            return "malformed"
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError, RecursionError):
            return "malformed"
        if not isinstance(payload, dict):
            return "malformed"
        version = payload.get("v")
        role = payload.get("role")
        ts = payload.get("ts")
        nonce = payload.get("nonce")
        mac = payload.get("mac")
        if version != AUTH_PROTOCOL_VERSION or isinstance(version, bool):
            return "unsupported_version"
        if not isinstance(role, str):
            return "malformed"
        if role != self.role:
            return "wrong_role"
        if not isinstance(ts, int) or isinstance(ts, bool):
            return "malformed"
        if not isinstance(nonce, str) or not 16 <= len(nonce) <= 128:
            return "malformed"
        if not isinstance(mac, str) or len(mac) > 128:
            return "malformed"
        expected = compute_auth_mac(runtime_key, self.role, ts, nonce)
        if not hmac.compare_digest(
            expected.encode("ascii"), mac.encode("utf-8", errors="replace")
        ):
            return "invalid_mac"
        now = self._clock()
        if abs(now - ts) > self.max_clock_skew_seconds:
            return "expired"
        self._evict_expired_nonces(now)
        if nonce in self._seen_nonces:
            return "replayed"
        self._seen_nonces[nonce] = float(ts)
        while len(self._seen_nonces) > self.nonce_cache_size:
            self._seen_nonces.popitem(last=False)
        return None

    def _evict_expired_nonces(self, now: float) -> None:
        # Nonces older than the skew window would fail the timestamp check, so
        # they no longer need to occupy the replay cache. Insertion order is
        # close to timestamp order, so stop at the first fresh entry.
        cutoff = now - self.max_clock_skew_seconds
        while self._seen_nonces:
            nonce, ts = next(iter(self._seen_nonces.items()))
            if ts >= cutoff:
                break
            self._seen_nonces.pop(nonce, None)

    def _record(self, result: str) -> None:
        if self.observability is not None and hasattr(
            self.observability, "record_ranking_runtime_auth"
        ):
            self.observability.record_ranking_runtime_auth(self.role, result)


def warn_if_weak_production_auth(
    settings: RankingRuntimeAuthSettings, *, role: str, logger: logging.Logger
) -> bool:
    """Log a startup warning when production still accepts unauthenticated peers.

    Returns True when the warning was emitted. Config validation already
    rejects ``disabled`` in production; ``permissive`` is allowed for rolling
    upgrades but should be flipped to ``required`` once
    ``video_commerce_ranking_runtime_auth_total{result="unauthenticated"}``
    stays at zero (docs/operations/ranking-runtime-auth.md).
    """
    if os.getenv("ENVIRONMENT", "").lower() != "production":
        return False
    if settings.mode == AUTH_MODE_REQUIRED and settings.runtime_key is not None:
        return False
    logger.warning(
        "ranking_runtime_auth_not_enforced",
        extra={
            "role": role,
            "runtime_auth_mode": settings.mode,
            "runtime_auth_key_configured": settings.runtime_key is not None,
            "runbook": "docs/operations/ranking-runtime-auth.md",
        },
    )
    return True


def is_loopback_peer(peername: Any) -> bool:
    """True only for TCP peers on a loopback address."""
    if not isinstance(peername, (tuple, list)) or not peername:
        return False
    try:
        address = ipaddress.ip_address(str(peername[0]).split("%", 1)[0])
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped is not None and mapped.is_loopback)
