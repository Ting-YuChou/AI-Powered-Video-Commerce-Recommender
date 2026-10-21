"""Length-prefixed client protocol for the internal ranking coordinator."""

from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import dataclass
from typing import Optional

from video_commerce.ranking_runtime.ranking_auth import (
    AUTH_OPERATION,
    RankingRuntimeAuthSettings,
)


RANK_OPERATION = b"R"
HEALTH_OPERATION = b"H"
METRICS_OPERATION = b"M"
DRAIN_OPERATION = b"D"
UNDRAIN_OPERATION = b"U"
MAX_FRAME_BYTES = 32 * 1024 * 1024
LEGACY_SERVER_AUTH_STATUS = 400


class RankingCoordinatorError(RuntimeError):
    """Base error for coordinator protocol failures."""


class RankingCoordinatorUnavailable(RankingCoordinatorError):
    """Raised when the coordinator cannot be reached."""


class RankingCoordinatorTimeout(RankingCoordinatorUnavailable):
    """Raised when the coordinator does not respond within the configured timeout."""


class RankingCoordinatorProtocolError(RankingCoordinatorError):
    """Raised when the coordinator sends an invalid frame."""


class RankingRuntimeAuthRejected(RankingCoordinatorUnavailable):
    """Raised when a ranking runtime server does not accept the auth handshake."""


@dataclass
class RankingCoordinatorResponse:
    status_code: int
    content_type: str
    body: bytes


def encode_request(operation: bytes, body: bytes = b"") -> bytes:
    if len(operation) != 1:
        raise ValueError("operation must be a single byte")
    payload = operation + body
    if len(payload) > MAX_FRAME_BYTES:
        raise RankingCoordinatorProtocolError("ranking coordinator frame too large")
    return struct.pack("!I", len(payload)) + payload


def encode_response(status_code: int, content_type: str, body: bytes) -> bytes:
    content_type_bytes = content_type.encode("ascii", errors="ignore")
    if len(content_type_bytes) > 1024:
        raise ValueError("content_type is too long")
    payload = (
        struct.pack("!HH", int(status_code), len(content_type_bytes))
        + content_type_bytes
        + body
    )
    if len(payload) > MAX_FRAME_BYTES:
        raise RankingCoordinatorProtocolError("ranking coordinator response too large")
    return struct.pack("!I", len(payload)) + payload


def decode_response(payload: bytes) -> RankingCoordinatorResponse:
    if len(payload) < 4:
        raise RankingCoordinatorProtocolError("ranking coordinator response too short")
    status_code, content_type_length = struct.unpack("!HH", payload[:4])
    content_type_end = 4 + content_type_length
    if content_type_end > len(payload):
        raise RankingCoordinatorProtocolError("invalid coordinator content type length")
    content_type = payload[4:content_type_end].decode("ascii", errors="replace")
    return RankingCoordinatorResponse(
        status_code=status_code,
        content_type=content_type or "application/octet-stream",
        body=payload[content_type_end:],
    )


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError as exc:
        raise RankingCoordinatorUnavailable(
            "ranking coordinator connection closed"
        ) from exc
    frame_length = struct.unpack("!I", header)[0]
    if frame_length <= 0 or frame_length > MAX_FRAME_BYTES:
        raise RankingCoordinatorProtocolError(
            "invalid ranking coordinator frame length"
        )
    try:
        return await reader.readexactly(frame_length)
    except asyncio.IncompleteReadError as exc:
        raise RankingCoordinatorUnavailable(
            "ranking coordinator frame truncated"
        ) from exc


async def perform_client_handshake(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    auth_settings: RankingRuntimeAuthSettings,
    *,
    timeout_seconds: float,
    peer_label: str,
) -> bool:
    """Send the ``A`` frame on a fresh connection.

    Returns True when the server authenticated the connection and False when a
    permissive client keeps using a legacy server that rejected ``A`` as an
    unknown operation. Raises ``RankingRuntimeAuthRejected`` otherwise.
    """
    writer.write(encode_request(AUTH_OPERATION, auth_settings.build_handshake_body()))
    await asyncio.wait_for(writer.drain(), timeout=timeout_seconds)
    response = decode_response(
        await asyncio.wait_for(read_frame(reader), timeout=timeout_seconds)
    )
    if response.status_code == 200:
        return True
    if (
        response.status_code == LEGACY_SERVER_AUTH_STATUS
        and auth_settings.client_tolerates_legacy_server
    ):
        return False
    detail = ""
    try:
        payload = json.loads(response.body)
        if isinstance(payload, dict):
            detail = str(payload.get("reason") or payload.get("detail") or "")
    except Exception:
        pass
    raise RankingRuntimeAuthRejected(
        f"ranking runtime auth rejected by {peer_label}: "
        f"status={response.status_code} {detail}".strip()
    )


def auth_rejection_detail(response: RankingCoordinatorResponse) -> Optional[str]:
    """Return the detail when a server refused a frame for missing/failed auth."""
    if response.status_code != 401:
        return None
    try:
        payload = json.loads(response.body)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    detail = str(payload.get("detail") or "")
    return detail if detail.startswith("ranking_runtime_auth") else None


def abort_writer(writer: Optional[asyncio.StreamWriter]) -> None:
    """Close a transport without awaiting, safe while a task is being cancelled."""
    if writer is None:
        return
    try:
        writer.close()
    except Exception:
        pass


class _CoordinatorConnection:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_seconds: float,
        request_timeout_seconds: float,
        auth_settings: Optional[RankingRuntimeAuthSettings] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.connect_timeout_seconds = connect_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.auth_settings = auth_settings or RankingRuntimeAuthSettings()
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.authenticated = False
        # True between writing a request frame and decoding its response. A
        # connection left in this state has an unread (or partially read)
        # response on the wire and must never serve another request.
        self.in_flight = False

    @property
    def reusable(self) -> bool:
        return not self.in_flight and (
            self.writer is None or not self.writer.is_closing()
        )

    async def request(
        self, operation: bytes, body: bytes = b""
    ) -> RankingCoordinatorResponse:
        await self._ensure_connected()
        assert self.writer is not None
        assert self.reader is not None
        frame = encode_request(operation, body)
        try:
            self.in_flight = True
            self.writer.write(frame)
            await asyncio.wait_for(
                self.writer.drain(),
                timeout=self.request_timeout_seconds,
            )
            response_frame = await asyncio.wait_for(
                read_frame(self.reader),
                timeout=self.request_timeout_seconds,
            )
            response = decode_response(response_frame)
            self.in_flight = False
        except asyncio.TimeoutError as exc:
            self.close_nowait()
            raise RankingCoordinatorTimeout(
                "ranking coordinator request timeout"
            ) from exc
        except BaseException:
            # Includes CancelledError: the response may still arrive later, so
            # the stream can no longer be trusted for the next request.
            self.close_nowait()
            raise
        rejection = auth_rejection_detail(response)
        if rejection is not None:
            # The server closes the connection after an auth rejection.
            self.close_nowait()
            raise RankingRuntimeAuthRejected(
                f"ranking runtime auth rejected by {self.host}:{self.port}: {rejection}"
            )
        return response

    async def _ensure_connected(self) -> None:
        if self.writer is not None and not self.writer.is_closing():
            return
        self.close_nowait()
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.connect_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise RankingCoordinatorTimeout(
                "ranking coordinator connect timeout"
            ) from exc
        except Exception as exc:
            raise RankingCoordinatorUnavailable(
                "ranking coordinator unavailable"
            ) from exc
        authenticated = False
        if self.auth_settings.client_sends_handshake:
            try:
                authenticated = await perform_client_handshake(
                    reader,
                    writer,
                    self.auth_settings,
                    timeout_seconds=self.connect_timeout_seconds,
                    peer_label=f"{self.host}:{self.port}",
                )
            except asyncio.TimeoutError as exc:
                abort_writer(writer)
                raise RankingCoordinatorTimeout(
                    "ranking coordinator auth handshake timeout"
                ) from exc
            except RankingCoordinatorError:
                abort_writer(writer)
                raise
            except BaseException as exc:
                abort_writer(writer)
                if isinstance(exc, Exception):
                    raise RankingCoordinatorUnavailable(
                        "ranking coordinator auth handshake failed"
                    ) from exc
                raise
        self.reader, self.writer = reader, writer
        self.authenticated = authenticated
        self.in_flight = False

    def close_nowait(self) -> None:
        writer = self.writer
        self.reader = None
        self.writer = None
        self.authenticated = False
        self.in_flight = False
        abort_writer(writer)

    async def close(self) -> None:
        writer = self.writer
        self.close_nowait()
        if writer is None:
            return
        try:
            await writer.wait_closed()
        except Exception:
            pass


class RankingCoordinatorClientPool:
    """Persistent connection pool to the single ranking coordinator process."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        pool_size: int = 64,
        connect_timeout_seconds: float = 1.0,
        request_timeout_seconds: float = 10.0,
        auth_settings: Optional[RankingRuntimeAuthSettings] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.pool_size = max(1, int(pool_size))
        self.connect_timeout_seconds = max(0.1, float(connect_timeout_seconds))
        self.request_timeout_seconds = max(0.1, float(request_timeout_seconds))
        self.auth_settings = auth_settings or RankingRuntimeAuthSettings()
        self._queue: asyncio.Queue[_CoordinatorConnection] = asyncio.Queue(
            maxsize=self.pool_size
        )
        for _ in range(self.pool_size):
            self._queue.put_nowait(
                _CoordinatorConnection(
                    self.host,
                    self.port,
                    connect_timeout_seconds=self.connect_timeout_seconds,
                    request_timeout_seconds=self.request_timeout_seconds,
                    auth_settings=self.auth_settings,
                )
            )
        self._closed = False

    async def rank(self, raw_body: bytes) -> RankingCoordinatorResponse:
        return await self.request(RANK_OPERATION, raw_body)

    async def health(self) -> RankingCoordinatorResponse:
        return await self.request(HEALTH_OPERATION)

    async def metrics(self) -> RankingCoordinatorResponse:
        return await self.request(METRICS_OPERATION)

    async def request(
        self, operation: bytes, body: bytes = b""
    ) -> RankingCoordinatorResponse:
        if self._closed:
            raise RankingCoordinatorUnavailable("ranking coordinator client is closed")
        try:
            connection = await asyncio.wait_for(
                self._queue.get(),
                timeout=self.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise RankingCoordinatorTimeout(
                "ranking coordinator client pool timeout"
            ) from exc
        try:
            try:
                return await connection.request(operation, body)
            except (RankingCoordinatorTimeout, RankingRuntimeAuthRejected):
                raise
            except RankingCoordinatorUnavailable:
                return await connection.request(operation, body)
        finally:
            # Never hand a connection with an interrupted request back to the
            # pool; reset it so the next borrower reconnects cleanly.
            if self._closed or not connection.reusable:
                connection.close_nowait()
            if not self._closed:
                self._queue.put_nowait(connection)

    async def aclose(self) -> None:
        self._closed = True
        connections = []
        while not self._queue.empty():
            connections.append(self._queue.get_nowait())
        await asyncio.gather(
            *(connection.close() for connection in connections),
            return_exceptions=True,
        )
