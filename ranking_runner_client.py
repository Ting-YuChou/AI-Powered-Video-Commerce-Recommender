"""Client for internal ranking-runner batch execution."""

from __future__ import annotations

import asyncio
import json
import socket
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ranking_coordinator_client import (
    HEALTH_OPERATION,
    RankingCoordinatorProtocolError,
    RankingCoordinatorResponse,
    decode_response,
    encode_request,
    read_frame,
)


BATCH_RANK_OPERATION = b"B"
RUNNER_OVERLOADED_STATUS = 429
NON_UNHEALTHY_5XX_DETAILS = {
    "ranking_runner_deadline_exceeded",
    "ranking_runner_overloaded",
    "ranking_runner_unavailable",
}
ENDPOINT_STATES = ("active", "draining", "failed", "overloaded")


class RankingRunnerUnavailable(RuntimeError):
    """Raised when no ranking runner can accept a batch."""


class RankingRunnerTimeout(RankingRunnerUnavailable):
    """Raised when an accepted runner batch does not complete before timeout."""


@dataclass
class RankingRunnerEndpoint:
    host: str
    port: int
    label: str
    source_host: str = ""
    source_port: int = 0
    failed_until: float = 0.0
    overloaded_until: float = 0.0
    inflight_batches: int = 0
    max_inflight_batches: int = 1
    last_seen_at: float = 0.0
    missing_since: float = 0.0
    missing_refresh_count: int = 0
    draining: bool = False
    batch_payload_versions: Tuple[int, ...] = (1,)
    connections: Optional[asyncio.Queue["_RunnerConnection"]] = field(
        default=None, init=False, repr=False
    )

    @property
    def key(self) -> Tuple[str, int]:
        return (self.host, int(self.port))


class _RunnerConnection:
    def __init__(
        self,
        endpoint: RankingRunnerEndpoint,
        *,
        connect_timeout_seconds: float,
        request_timeout_seconds: float,
    ) -> None:
        self.endpoint = endpoint
        self.connect_timeout_seconds = connect_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None

    async def request(
        self,
        operation: bytes,
        body: bytes = b"",
        *,
        timeout_seconds: Optional[float] = None,
    ) -> RankingCoordinatorResponse:
        await self._ensure_connected()
        assert self.writer is not None
        assert self.reader is not None
        request_timeout_seconds = (
            self.request_timeout_seconds
            if timeout_seconds is None
            else max(0.001, float(timeout_seconds))
        )
        try:
            self.writer.write(encode_request(operation, body))
            await asyncio.wait_for(
                self.writer.drain(),
                timeout=request_timeout_seconds,
            )
            response_frame = await asyncio.wait_for(
                read_frame(self.reader),
                timeout=request_timeout_seconds,
            )
            return decode_response(response_frame)
        except asyncio.TimeoutError as exc:
            await self.close()
            raise RankingRunnerTimeout(
                f"ranking runner request timeout: {self.endpoint.label}"
            ) from exc
        except Exception:
            await self.close()
            raise

    async def _ensure_connected(self) -> None:
        if self.writer is not None and not self.writer.is_closing():
            return
        try:
            self.reader, self.writer = await asyncio.wait_for(
                asyncio.open_connection(self.endpoint.host, self.endpoint.port),
                timeout=self.connect_timeout_seconds,
            )
        except Exception as exc:
            raise RankingRunnerUnavailable(
                f"ranking runner unavailable: {self.endpoint.label}"
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


def parse_runner_urls(raw_urls: str, default_port: int) -> List[Tuple[str, int]]:
    endpoints: List[Tuple[str, int]] = []
    for raw_item in (raw_urls or "").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "://" in item:
            item = item.split("://", 1)[1]
        item = item.rstrip("/")
        if ":" in item:
            host, port_text = item.rsplit(":", 1)
            port = int(port_text)
        else:
            host = item
            port = int(default_port)
        endpoints.append((host, port))
    return endpoints


class RankingRunnerClientPool:
    """Capacity-aware persistent client pool for remote ranking runners."""

    def __init__(
        self,
        endpoints: Sequence[RankingRunnerEndpoint],
        *,
        configured_endpoints: Sequence[Tuple[str, int]] | None = None,
        dispatch_concurrency: int = 16,
        runner_max_inflight_batches: int = 1,
        connect_timeout_seconds: float = 1.0,
        request_timeout_seconds: float = 1.0,
        unhealthy_cooldown_seconds: float = 1.0,
        overload_backoff_seconds: float = 0.05,
        dns_refresh_seconds: float = 5.0,
        endpoint_missing_refreshes: int = 3,
        endpoint_missing_grace_seconds: float = 30.0,
        observability=None,
    ) -> None:
        if not endpoints:
            raise ValueError("at least one ranking runner endpoint is required")
        self.endpoints = list(endpoints)
        self.configured_endpoints = list(configured_endpoints or [])
        self.dispatch_concurrency = max(1, int(dispatch_concurrency))
        self.runner_max_inflight_batches = max(1, int(runner_max_inflight_batches))
        self.connect_timeout_seconds = max(0.1, float(connect_timeout_seconds))
        self.request_timeout_seconds = max(0.1, float(request_timeout_seconds))
        self.unhealthy_cooldown_seconds = max(0.01, float(unhealthy_cooldown_seconds))
        self.overload_backoff_seconds = max(0.001, float(overload_backoff_seconds))
        self.dns_refresh_seconds = max(0.0, float(dns_refresh_seconds))
        self.endpoint_missing_refreshes = max(1, int(endpoint_missing_refreshes))
        self.endpoint_missing_grace_seconds = max(
            0.0, float(endpoint_missing_grace_seconds)
        )
        self.observability = observability
        self._semaphore = asyncio.Semaphore(self.dispatch_concurrency)
        self._next_index = 0
        self._closed = False
        self._last_dns_refresh = time.monotonic()
        self._refresh_lock = asyncio.Lock()
        for endpoint in self.endpoints:
            self._prepare_endpoint(endpoint)

    @classmethod
    async def create(
        cls,
        raw_urls: str,
        *,
        default_port: int,
        dispatch_concurrency: int,
        runner_max_inflight_batches: int = 1,
        connect_timeout_seconds: float,
        request_timeout_seconds: float,
        unhealthy_cooldown_seconds: float,
        dns_refresh_seconds: float = 5.0,
        endpoint_missing_refreshes: int = 3,
        endpoint_missing_grace_seconds: float = 30.0,
        observability=None,
    ) -> "RankingRunnerClientPool":
        configured = parse_runner_urls(raw_urls, default_port)
        endpoints: List[RankingRunnerEndpoint] = []
        for host, port in configured:
            endpoints.extend(
                await cls._resolve_endpoint(
                    host,
                    port,
                    connect_timeout_seconds=connect_timeout_seconds,
                )
            )
        if not endpoints:
            endpoints = [
                RankingRunnerEndpoint(
                    host=host,
                    port=port,
                    label=f"{host}:{port}",
                    source_host=host,
                    source_port=port,
                )
                for host, port in configured
            ]
        return cls(
            endpoints,
            configured_endpoints=configured,
            dispatch_concurrency=dispatch_concurrency,
            runner_max_inflight_batches=runner_max_inflight_batches,
            connect_timeout_seconds=connect_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
            unhealthy_cooldown_seconds=unhealthy_cooldown_seconds,
            dns_refresh_seconds=dns_refresh_seconds,
            endpoint_missing_refreshes=endpoint_missing_refreshes,
            endpoint_missing_grace_seconds=endpoint_missing_grace_seconds,
            observability=observability,
        )

    @staticmethod
    async def _resolve_endpoint(
        host: str,
        port: int,
        *,
        connect_timeout_seconds: float,
    ) -> List[RankingRunnerEndpoint]:
        loop = asyncio.get_running_loop()
        try:
            infos = await asyncio.wait_for(
                loop.getaddrinfo(
                    host,
                    port,
                    type=socket.SOCK_STREAM,
                ),
                timeout=connect_timeout_seconds,
            )
        except Exception:
            return [
                RankingRunnerEndpoint(
                    host=host,
                    port=port,
                    label=f"{host}:{port}",
                    source_host=host,
                    source_port=port,
                )
            ]

        endpoints: List[RankingRunnerEndpoint] = []
        seen = set()
        for _family, _type, _proto, _canon, sockaddr in infos:
            address = sockaddr[0]
            resolved_port = int(sockaddr[1])
            key = (address, resolved_port)
            if key in seen:
                continue
            seen.add(key)
            endpoints.append(
                RankingRunnerEndpoint(
                    host=address,
                    port=resolved_port,
                    label=f"{host}->{address}:{resolved_port}",
                    source_host=host,
                    source_port=port,
                )
            )
        return endpoints or [
            RankingRunnerEndpoint(
                host=host,
                port=port,
                label=f"{host}:{port}",
                source_host=host,
                source_port=port,
            )
        ]

    @property
    def capacity(self) -> int:
        endpoint_capacity = sum(
            max(1, endpoint.max_inflight_batches)
            for endpoint in self.endpoints
            if not endpoint.draining
        )
        return max(1, min(self.dispatch_concurrency, endpoint_capacity))

    def has_available_endpoint(self) -> bool:
        now = time.monotonic()
        return any(
            not endpoint.draining and endpoint.failed_until <= now
            for endpoint in self.endpoints
        )

    def has_dispatch_capacity(self) -> bool:
        now = time.monotonic()
        return any(
            self._endpoint_has_capacity(endpoint, now) for endpoint in self.endpoints
        )

    def supports_batch_payload_version(self, version: int) -> bool:
        now = time.monotonic()
        active_endpoints = [
            endpoint
            for endpoint in self.endpoints
            if not endpoint.draining
            and endpoint.failed_until <= now
            and endpoint.overloaded_until <= now
        ]
        if not active_endpoints:
            return False
        return all(
            int(version) in endpoint.batch_payload_versions
            for endpoint in active_endpoints
        )

    async def rank_batch(
        self,
        body: bytes,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> RankingCoordinatorResponse:
        if self._closed:
            raise RankingRunnerUnavailable("ranking runner client is closed")
        await self._refresh_endpoints_if_needed()
        async with self._semaphore:
            wait_started = time.monotonic()
            timeout_budget = self.request_timeout_seconds
            if timeout_seconds is not None:
                timeout_budget = min(timeout_budget, max(0.001, float(timeout_seconds)))
            deadline = wait_started + timeout_budget
            last_error: Optional[BaseException] = None
            recorded_slot_wait = False
            while not self._closed and time.monotonic() <= deadline:
                acquired = self._acquire_endpoint_connection()
                if acquired is None:
                    await self._refresh_endpoints_if_needed()
                    await asyncio.sleep(self._next_capacity_sleep_seconds(deadline))
                    continue
                endpoint, connection = acquired
                slot_wait_seconds = time.monotonic() - wait_started
                self._record_wait_for_slot(slot_wait_seconds)
                recorded_slot_wait = True
                try:
                    response = await self._request(
                        endpoint,
                        connection,
                        BATCH_RANK_OPERATION,
                        body,
                        timeout_seconds=deadline - time.monotonic(),
                    )
                    setattr(response, "runner_slot_wait_seconds", slot_wait_seconds)
                    if response.status_code == RUNNER_OVERLOADED_STATUS:
                        self._mark_overloaded(endpoint)
                        last_error = RankingRunnerUnavailable(
                            f"ranking runner overloaded: {endpoint.label}"
                        )
                        continue
                    if response.status_code >= 500:
                        detail = self._response_detail(response)
                        if detail == "ranking_runner_deadline_exceeded":
                            raise RankingRunnerTimeout(
                                f"ranking runner deadline exceeded: {endpoint.label}"
                            )
                        if detail in NON_UNHEALTHY_5XX_DETAILS:
                            self._mark_overloaded(endpoint)
                            last_error = RankingRunnerUnavailable(
                                f"ranking runner capacity unavailable: {endpoint.label}: {detail}"
                            )
                            continue
                        last_error = RankingRunnerUnavailable(
                            f"ranking runner returned {response.status_code}"
                        )
                        self._mark_failed(endpoint)
                        await self._refresh_endpoints_if_needed(force=True)
                        continue
                    return response
                except RankingRunnerTimeout as exc:
                    last_error = exc
                    self._mark_slow_timeout(endpoint)
                    raise
                except RankingCoordinatorProtocolError as exc:
                    last_error = exc
                    self._mark_failed(endpoint)
                    await self._refresh_endpoints_if_needed(force=True)
                except Exception as exc:
                    last_error = exc
                    self._mark_failed(endpoint)
                    await self._refresh_endpoints_if_needed(force=True)
            if not recorded_slot_wait:
                self._record_wait_for_slot(time.monotonic() - wait_started)
            if last_error:
                if isinstance(last_error, RankingRunnerTimeout):
                    raise last_error
                raise RankingRunnerUnavailable(str(last_error)) from last_error
            raise RankingRunnerUnavailable("no ranking runner dispatch capacity")

    async def health_check(self) -> dict:
        await self._refresh_endpoints_if_needed()
        checks = await asyncio.gather(
            *(self._health_endpoint(endpoint) for endpoint in self.endpoints),
            return_exceptions=True,
        )
        healthy = 0
        details = []
        for endpoint, check in zip(self.endpoints, checks):
            if isinstance(check, Exception):
                details.append(
                    {
                        "endpoint": endpoint.label,
                        "status": "unhealthy",
                        "state": self._endpoint_state(endpoint),
                        "error": type(check).__name__,
                        "inflight_batches": endpoint.inflight_batches,
                        "missing_refresh_count": endpoint.missing_refresh_count,
                    }
                )
                continue
            if check.status_code == 200:
                self._update_endpoint_capabilities(endpoint, check)
                healthy += 1
                details.append(
                    {
                        "endpoint": endpoint.label,
                        "status": "healthy",
                        "state": self._endpoint_state(endpoint),
                        "inflight_batches": endpoint.inflight_batches,
                        "max_inflight_batches": endpoint.max_inflight_batches,
                        "missing_refresh_count": endpoint.missing_refresh_count,
                        "batch_payload_versions": list(
                            endpoint.batch_payload_versions
                        ),
                    }
                )
            else:
                details.append(
                    {
                        "endpoint": endpoint.label,
                        "status": "unhealthy",
                        "state": self._endpoint_state(endpoint),
                        "status_code": check.status_code,
                        "inflight_batches": endpoint.inflight_batches,
                        "missing_refresh_count": endpoint.missing_refresh_count,
                    }
                )
        return {
            "status": "healthy" if healthy > 0 else "unhealthy",
            "healthy_count": healthy,
            "total_count": len(self.endpoints),
            "dispatch_concurrency": self.dispatch_concurrency,
            "runner_max_inflight_batches": self.runner_max_inflight_batches,
            "endpoints": details,
        }

    async def aclose(self) -> None:
        self._closed = True
        await asyncio.gather(
            *(
                self._close_endpoint_connections(endpoint)
                for endpoint in self.endpoints
            ),
            return_exceptions=True,
        )

    def _prepare_endpoint(self, endpoint: RankingRunnerEndpoint) -> None:
        now = time.monotonic()
        endpoint.max_inflight_batches = max(1, self.runner_max_inflight_batches)
        if not endpoint.source_host:
            endpoint.source_host = endpoint.host
        if not endpoint.source_port:
            endpoint.source_port = endpoint.port
        if endpoint.last_seen_at <= 0.0:
            endpoint.last_seen_at = now
        endpoint.connections = asyncio.Queue(maxsize=endpoint.max_inflight_batches)
        for _ in range(endpoint.max_inflight_batches):
            endpoint.connections.put_nowait(
                _RunnerConnection(
                    endpoint,
                    connect_timeout_seconds=self.connect_timeout_seconds,
                    request_timeout_seconds=self.request_timeout_seconds,
                )
            )
        self._record_endpoint_capacity(endpoint)

    def _endpoint_has_capacity(
        self,
        endpoint: RankingRunnerEndpoint,
        now: float,
    ) -> bool:
        available_connections = (
            endpoint.connections.qsize() if endpoint.connections is not None else 0
        )
        return (
            not endpoint.draining
            and endpoint.failed_until <= now
            and endpoint.overloaded_until <= now
            and endpoint.inflight_batches < endpoint.max_inflight_batches
            and available_connections > 0
        )

    def _acquire_endpoint_connection(
        self,
    ) -> Optional[Tuple[RankingRunnerEndpoint, _RunnerConnection]]:
        now = time.monotonic()
        for _ in range(len(self.endpoints)):
            endpoint = self.endpoints[self._next_index % len(self.endpoints)]
            self._next_index = (self._next_index + 1) % len(self.endpoints)
            if self._endpoint_has_capacity(endpoint, now):
                if endpoint.connections is None:
                    self._prepare_endpoint(endpoint)
                assert endpoint.connections is not None
                try:
                    connection = endpoint.connections.get_nowait()
                except asyncio.QueueEmpty:
                    self._record_endpoint_capacity(endpoint)
                    continue
                endpoint.inflight_batches += 1
                self._record_endpoint_inflight(endpoint)
                self._record_endpoint_capacity(endpoint)
                return endpoint, connection
        return None

    def _release_endpoint(
        self,
        endpoint: RankingRunnerEndpoint,
        connection: _RunnerConnection,
    ) -> None:
        if endpoint.connections is not None:
            try:
                endpoint.connections.put_nowait(connection)
            except asyncio.QueueFull:
                asyncio.create_task(connection.close())
        endpoint.inflight_batches = max(0, endpoint.inflight_batches - 1)
        self._record_endpoint_inflight(endpoint)
        self._record_endpoint_capacity(endpoint)

    def _mark_failed(self, endpoint: RankingRunnerEndpoint) -> None:
        endpoint.failed_until = time.monotonic() + self.unhealthy_cooldown_seconds
        self._record_endpoint_event(endpoint, "unhealthy")
        self._record_endpoint_capacity(endpoint)

    def _mark_overloaded(self, endpoint: RankingRunnerEndpoint) -> None:
        endpoint.overloaded_until = time.monotonic() + self.overload_backoff_seconds
        self._record_endpoint_event(endpoint, "overloaded")
        self._record_endpoint_capacity(endpoint)

    def _mark_slow_timeout(self, endpoint: RankingRunnerEndpoint) -> None:
        endpoint.overloaded_until = time.monotonic() + self.overload_backoff_seconds
        self._record_endpoint_event(endpoint, "slow_timeout")
        self._record_endpoint_capacity(endpoint)

    @staticmethod
    def _response_detail(response: RankingCoordinatorResponse) -> str:
        try:
            payload = json.loads(response.body)
        except Exception:
            return ""
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("detail") or "")

    @staticmethod
    def _update_endpoint_capabilities(
        endpoint: RankingRunnerEndpoint,
        response: RankingCoordinatorResponse,
    ) -> None:
        try:
            payload = json.loads(response.body)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        versions = payload.get("batch_payload_versions")
        if versions is None:
            capabilities = payload.get("capabilities")
            if isinstance(capabilities, dict):
                versions = capabilities.get("batch_payload_versions")
        if not isinstance(versions, list):
            return

        normalized_versions = []
        for version in versions:
            try:
                normalized = int(version)
            except (TypeError, ValueError, OverflowError):
                continue
            if normalized > 0 and normalized not in normalized_versions:
                normalized_versions.append(normalized)
        if normalized_versions:
            endpoint.batch_payload_versions = tuple(sorted(normalized_versions))

    async def _health_endpoint(
        self,
        endpoint: RankingRunnerEndpoint,
    ) -> RankingCoordinatorResponse:
        connection = _RunnerConnection(
            endpoint,
            connect_timeout_seconds=self.connect_timeout_seconds,
            request_timeout_seconds=self.request_timeout_seconds,
        )
        try:
            return await connection.request(HEALTH_OPERATION)
        finally:
            await connection.close()

    async def _request(
        self,
        endpoint: RankingRunnerEndpoint,
        connection: _RunnerConnection,
        operation: bytes,
        body: bytes = b"",
        *,
        timeout_seconds: Optional[float] = None,
    ) -> RankingCoordinatorResponse:
        try:
            return await connection.request(
                operation,
                body,
                timeout_seconds=timeout_seconds,
            )
        finally:
            self._release_endpoint(endpoint, connection)

    async def _refresh_endpoints_if_needed(self, *, force: bool = False) -> None:
        if not self.configured_endpoints:
            return
        now = time.monotonic()
        if (
            not force
            and self.dns_refresh_seconds > 0
            and now - self._last_dns_refresh < self.dns_refresh_seconds
        ):
            return
        async with self._refresh_lock:
            now = time.monotonic()
            if (
                not force
                and self.dns_refresh_seconds > 0
                and now - self._last_dns_refresh < self.dns_refresh_seconds
            ):
                return
            resolved: List[RankingRunnerEndpoint] = []
            for host, port in self.configured_endpoints:
                resolved.extend(
                    await self._resolve_endpoint(
                        host,
                        port,
                        connect_timeout_seconds=self.connect_timeout_seconds,
                    )
                )
            if not resolved:
                self._last_dns_refresh = now
                return
            old_by_key = {endpoint.key: endpoint for endpoint in self.endpoints}
            next_endpoints: List[RankingRunnerEndpoint] = []
            next_keys = set()
            for endpoint in resolved:
                existing = old_by_key.get(endpoint.key)
                if existing is not None:
                    existing.last_seen_at = now
                    existing.missing_since = 0.0
                    existing.missing_refresh_count = 0
                    existing.draining = False
                    next_endpoints.append(existing)
                    next_keys.add(existing.key)
                    self._record_endpoint_missing_refreshes(existing)
                    self._record_endpoint_capacity(existing)
                    continue
                endpoint.last_seen_at = now
                self._prepare_endpoint(endpoint)
                next_endpoints.append(endpoint)
                next_keys.add(endpoint.key)
                self._record_endpoint_event(endpoint, "discovered")
            for endpoint in self.endpoints:
                if endpoint.key in next_keys:
                    continue
                self._mark_endpoint_missing(endpoint, now)
                if not endpoint.draining:
                    next_endpoints.append(endpoint)
                    continue
                if endpoint.inflight_batches > 0:
                    next_endpoints.append(endpoint)
                    continue
                await self._close_endpoint_connections(endpoint)
                self._record_endpoint_event(endpoint, "removed")
                self._record_endpoint_removed(endpoint, "dns_missing")
            self.endpoints = next_endpoints
            self._next_index %= max(1, len(self.endpoints))
            self._last_dns_refresh = now

    def _mark_endpoint_missing(
        self,
        endpoint: RankingRunnerEndpoint,
        now: float,
    ) -> None:
        if endpoint.missing_since <= 0.0:
            endpoint.missing_since = now
        endpoint.missing_refresh_count += 1
        self._record_endpoint_missing_refreshes(endpoint)
        missing_age_seconds = max(0.0, now - endpoint.missing_since)
        if (
            endpoint.missing_refresh_count >= self.endpoint_missing_refreshes
            and missing_age_seconds >= self.endpoint_missing_grace_seconds
        ):
            if not endpoint.draining:
                endpoint.draining = True
                self._record_endpoint_event(endpoint, "draining")
        self._record_endpoint_capacity(endpoint)

    async def _close_endpoint_connections(
        self, endpoint: RankingRunnerEndpoint
    ) -> None:
        if endpoint.connections is None:
            return
        connections = []
        while not endpoint.connections.empty():
            connections.append(endpoint.connections.get_nowait())
        await asyncio.gather(
            *(connection.close() for connection in connections),
            return_exceptions=True,
        )

    def _next_capacity_sleep_seconds(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 0.0
        next_ready = None
        now = time.monotonic()
        for endpoint in self.endpoints:
            for ready_at in (endpoint.failed_until, endpoint.overloaded_until):
                if ready_at > now:
                    next_ready = (
                        ready_at if next_ready is None else min(next_ready, ready_at)
                    )
        if next_ready is not None:
            return max(0.001, min(0.01, next_ready - now, remaining))
        return min(0.005, remaining)

    def _record_endpoint_inflight(self, endpoint: RankingRunnerEndpoint) -> None:
        if self.observability and hasattr(
            self.observability, "set_ranking_runner_endpoint_inflight"
        ):
            self.observability.set_ranking_runner_endpoint_inflight(
                endpoint.label,
                endpoint.inflight_batches,
            )
        self._record_endpoint_capacity(endpoint)

    def _record_endpoint_capacity(self, endpoint: RankingRunnerEndpoint) -> None:
        if self.observability and hasattr(
            self.observability,
            "set_ranking_runner_endpoint_available_connections",
        ):
            available = 0
            if endpoint.connections is not None:
                available = endpoint.connections.qsize()
            self.observability.set_ranking_runner_endpoint_available_connections(
                endpoint.label,
                available,
            )
        self._record_endpoint_state(endpoint)
        self._record_endpoint_missing_refreshes(endpoint)

    def _endpoint_state(self, endpoint: RankingRunnerEndpoint) -> str:
        now = time.monotonic()
        if endpoint.draining:
            return "draining"
        if endpoint.failed_until > now:
            return "failed"
        if endpoint.overloaded_until > now:
            return "overloaded"
        return "active"

    def _record_endpoint_state(self, endpoint: RankingRunnerEndpoint) -> None:
        if self.observability and hasattr(
            self.observability, "set_ranking_runner_endpoint_state"
        ):
            self.observability.set_ranking_runner_endpoint_state(
                endpoint.label,
                self._endpoint_state(endpoint),
            )

    def _record_endpoint_missing_refreshes(
        self, endpoint: RankingRunnerEndpoint
    ) -> None:
        if self.observability and hasattr(
            self.observability, "set_ranking_runner_endpoint_missing_refreshes"
        ):
            self.observability.set_ranking_runner_endpoint_missing_refreshes(
                endpoint.label,
                endpoint.missing_refresh_count,
            )

    def _record_endpoint_event(
        self,
        endpoint: RankingRunnerEndpoint,
        event: str,
    ) -> None:
        if self.observability and hasattr(
            self.observability, "record_ranking_runner_endpoint_event"
        ):
            self.observability.record_ranking_runner_endpoint_event(
                endpoint.label,
                event,
            )

    def _record_endpoint_removed(
        self,
        endpoint: RankingRunnerEndpoint,
        reason: str,
    ) -> None:
        if self.observability and hasattr(
            self.observability, "record_ranking_runner_endpoint_removed"
        ):
            self.observability.record_ranking_runner_endpoint_removed(
                endpoint.label,
                reason,
            )

    def _record_wait_for_slot(self, duration_seconds: float) -> None:
        if self.observability and hasattr(
            self.observability, "record_ranking_wait_for_runner_slot"
        ):
            self.observability.record_ranking_wait_for_runner_slot(duration_seconds)
