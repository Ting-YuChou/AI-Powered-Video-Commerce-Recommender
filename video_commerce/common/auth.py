"""
Authentication helpers for gateway bearer token validation.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, Optional

from video_commerce.common.config import SecurityConfig


class AuthValidationError(Exception):
    """Raised when client authentication fails."""


class BearerTokenValidator:
    """Validate bearer tokens using JWKS or a shared JWT secret."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.enabled = config.oidc_enabled
        self.required = config.oidc_required or config.auth_mode == "bearer"
        self._jwt = None
        self._jwk_client = None
        self._refresh_lock = threading.Lock()
        self._last_forced_refresh = float("-inf")

        if not self.enabled:
            return

        import jwt

        self._jwt = jwt
        if config.oidc_jwks_url:
            self._jwk_client = jwt.PyJWKClient(
                config.oidc_jwks_url,
                timeout=config.jwks_timeout_seconds,
            )
        elif not config.jwt_shared_secret:
            raise ValueError(
                "OIDC/JWT auth requires SECURITY_OIDC_JWKS_URL or SECURITY_JWT_SHARED_SECRET"
            )

    @property
    def uses_jwks(self) -> bool:
        return self._jwk_client is not None and not self.config.jwt_shared_secret

    async def authenticate_async(
        self, authorization_header: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Validate a bearer token without blocking the event loop.

        JWKS fetches use synchronous urllib, so they run in a worker thread.
        """
        if self.uses_jwks:
            return await asyncio.to_thread(self.authenticate, authorization_header)
        return self.authenticate(authorization_header)

    def authenticate(self, authorization_header: Optional[str]) -> Optional[Dict[str, Any]]:
        """Validate a bearer token when bearer auth is enabled."""
        if not self.enabled:
            return None

        if not authorization_header:
            if self.required:
                raise AuthValidationError("Missing bearer token")
            return None

        scheme, _, token = authorization_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise AuthValidationError("Invalid bearer token")

        return self._decode(token)

    def _decode(self, token: str) -> Dict[str, Any]:
        assert self._jwt is not None

        decode_kwargs: Dict[str, Any] = {
            "algorithms": self.config.jwt_algorithms,
        }
        options: Dict[str, Any] = {"require": ["exp"]}
        if self.config.oidc_audience:
            decode_kwargs["audience"] = self.config.oidc_audience
        else:
            options["verify_aud"] = False
        if self.config.oidc_issuer:
            decode_kwargs["issuer"] = self.config.oidc_issuer
        else:
            options["verify_iss"] = False
        decode_kwargs["options"] = options

        try:
            if self.config.jwt_shared_secret:
                signing_key = self.config.jwt_shared_secret
            else:
                signing_key = self._jwks_signing_key(token)
            return self._jwt.decode(token, signing_key, **decode_kwargs)
        except self._jwt.PyJWTError as exc:
            raise AuthValidationError("Invalid bearer token") from exc

    def _jwks_signing_key(self, token: str):
        """Look up the token's key, forcing at most one JWKS refresh per interval.

        PyJWKClient refetches the JWKS for every unknown ``kid``, so a stream of
        tokens with random key IDs would otherwise turn into one IdP request
        each.
        """
        assert self._jwt is not None and self._jwk_client is not None
        kid = self._jwt.get_unverified_header(token).get("kid")
        signing_key = self._jwk_client.match_kid(
            self._jwk_client.get_signing_keys(), kid
        )
        if signing_key is None:
            with self._refresh_lock:
                now = time.monotonic()
                if (
                    now - self._last_forced_refresh
                    >= self.config.jwks_min_refresh_interval_seconds
                ):
                    self._last_forced_refresh = now
                    signing_key = self._jwk_client.match_kid(
                        self._jwk_client.get_signing_keys(refresh=True), kid
                    )
        if signing_key is None:
            raise self._jwt.PyJWKClientError(
                "Unable to find a signing key for the token"
            )
        return signing_key.key
