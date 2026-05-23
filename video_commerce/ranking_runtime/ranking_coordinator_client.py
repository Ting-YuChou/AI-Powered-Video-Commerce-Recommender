"""Length-prefixed client protocol for the internal ranking coordinator."""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from typing import Optional


RANK_OPERATION = b"R"
HEALTH_OPERATION = b"H"
METRICS_OPERATION = b"M"
MAX_FRAME_BYTES = 32 * 1024 * 1024


class RankingCoordinatorError(RuntimeError):
    """Base error for coordinator protocol failures."""


class RankingCoordinatorUnavailable(RankingCoordinatorError):
    """Raised when the coordinator cannot be reached."""


class RankingCoordinatorTimeout(RankingCoordinatorUnavailable):
    """Raised when the coordinator does not respond within the configured timeout."""


class RankingCoordinatorProtocolError(RankingCoordinatorError):
    """Raised when the coordinator sends an invalid frame."""


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


class _CoordinatorConnection:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_seconds: float,
        request_timeout_seconds: float,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.connect_timeout_seconds = connect_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None

    async def request(
        self, operation: bytes, body: bytes = b""
    ) -> RankingCoordinatorResponse:
        await self._ensure_connected()
        assert self.writer is not None
        assert self.reader is not None
        frame = encode_request(operation, body)
        try:
            self.writer.write(frame)
            await asyncio.wait_for(
                self.writer.drain(),
                timeout=self.request_timeout_seconds,
            )
            response_frame = await asyncio.wait_for(
                read_frame(self.reader),
                timeout=self.request_timeout_seconds,
            )
            return decode_response(response_frame)
        except asyncio.TimeoutError as exc:
            await self.close()
            raise RankingCoordinatorTimeout(
                "ranking coordinator request timeout"
            ) from exc
        except Exception:
            await self.close()
            raise

    async def _ensure_connected(self) -> None:
        if self.writer is not None and not self.writer.is_closing():
            return
        try:
            self.reader, self.writer = await asyncio.wait_for(
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

    async def close(self) -> None:
        writer = self.writer
        self.reader = None
        self.writer = None
        if writer is None:
            return
        writer.close()
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
    ) -> None:
        self.host = host
        self.port = int(port)
        self.pool_size = max(1, int(pool_size))
        self.connect_timeout_seconds = max(0.1, float(connect_timeout_seconds))
        self.request_timeout_seconds = max(0.1, float(request_timeout_seconds))
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
            except RankingCoordinatorTimeout:
                raise
            except RankingCoordinatorUnavailable:
                return await connection.request(operation, body)
        finally:
            if self._closed:
                await connection.close()
            else:
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
