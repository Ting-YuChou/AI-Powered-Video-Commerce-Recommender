import asyncio
import json
import logging
import struct
import time

import pytest

from video_commerce.ranking_runtime.ranking_auth import (
    AUTH_OPERATION,
    RankingRuntimeAuthenticator,
    RankingRuntimeAuthSettings,
)
from video_commerce.ranking_runtime.ranking_coordinator_client import (
    HEALTH_OPERATION,
    RANK_OPERATION,
    RankingCoordinatorClientPool,
    RankingCoordinatorResponse,
    RankingCoordinatorTimeout,
    RankingCoordinatorUnavailable,
    RankingRuntimeAuthRejected,
    decode_response,
    encode_request,
    encode_response,
    read_frame,
)
from video_commerce.ranking_runtime.ranking_runner_client import (
    BATCH_RANK_OPERATION,
    RankingRunnerClientPool,
    RankingRunnerEndpoint,
    RankingRunnerOverloaded,
    RankingRunnerRequestUncertain,
    RankingRunnerTimeout,
    RankingRunnerUnavailable,
    parse_runner_urls,
)


def test_runner_effective_capacity_excludes_non_dispatchable_endpoints():
    endpoints = [
        RankingRunnerEndpoint(host="127.0.0.1", port=8100 + index, label=f"r{index}")
        for index in range(4)
    ]
    client = RankingRunnerClientPool(
        endpoints,
        dispatch_concurrency=16,
        runner_max_inflight_batches=1,
        connect_timeout_seconds=0.1,
        request_timeout_seconds=0.1,
        unhealthy_cooldown_seconds=1.0,
    )
    endpoints[1].draining = True
    endpoints[2].failed_until = time.monotonic() + 10
    endpoints[3].overloaded_until = time.monotonic() + 10

    assert client.effective_capacity == 1
    assert client.capacity == 1


def test_coordinator_response_round_trips_status_content_type_and_body():
    frame = encode_response(503, "application/json", b'{"detail":"busy"}')
    payload = frame[4:]

    response = decode_response(payload)

    assert response.status_code == 503
    assert response.content_type == "application/json"
    assert response.body == b'{"detail":"busy"}'


@pytest.mark.asyncio
async def test_coordinator_client_sends_rank_frame_to_tcp_server():
    seen = {}

    async def handle_client(reader, writer):
        frame = await read_frame(reader)
        seen["operation"] = frame[:1]
        seen["body"] = frame[1:]
        writer.write(encode_response(200, "application/json", b'{"ok":true}'))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    client = RankingCoordinatorClientPool(
        "127.0.0.1",
        port,
        pool_size=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
    )
    try:
        response = await client.rank(b'{"request_id":"r1"}')
    finally:
        await client.aclose()
        server.close()
        await server.wait_closed()

    assert seen["operation"] == RANK_OPERATION
    assert seen["body"] == b'{"request_id":"r1"}'
    assert response.status_code == 200
    assert response.body == b'{"ok":true}'


@pytest.mark.asyncio
async def test_coordinator_client_pool_acquisition_times_out():
    client = RankingCoordinatorClientPool(
        "127.0.0.1",
        1,
        pool_size=1,
        connect_timeout_seconds=0.1,
        request_timeout_seconds=0.1,
    )
    held_connection = await client._queue.get()
    try:
        with pytest.raises(RankingCoordinatorUnavailable):
            await client.rank(b'{"request_id":"r1"}')
    finally:
        client._queue.put_nowait(held_connection)
        await client.aclose()


@pytest.mark.asyncio
async def test_coordinator_client_request_timeout_is_wrapped():
    async def handle_client(reader, writer):
        await read_frame(reader)
        await asyncio.sleep(0.2)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    client = RankingCoordinatorClientPool(
        "127.0.0.1",
        port,
        pool_size=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=0.05,
    )
    try:
        with pytest.raises(RankingCoordinatorTimeout):
            await client.rank(b'{"request_id":"r1"}')
    finally:
        await client.aclose()
        server.close()
        await server.wait_closed()


def test_parse_runner_urls_accepts_http_and_tcp_style_hosts():
    assert parse_runner_urls(
        "ranking-runner:8014,http://other-runner:9000,bare-runner",
        8014,
    ) == [
        ("ranking-runner", 8014),
        ("other-runner", 9000),
        ("bare-runner", 8014),
    ]


@pytest.mark.asyncio
async def test_runner_client_sends_batch_frame_to_tcp_server():
    seen = {}

    async def handle_client(reader, writer):
        frame = await read_frame(reader)
        seen["operation"] = frame[:1]
        seen["body"] = frame[1:]
        writer.write(encode_response(200, "application/json", b'{"results":[]}'))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    client = RankingRunnerClientPool(
        [RankingRunnerEndpoint(host="127.0.0.1", port=port, label="local")],
        dispatch_concurrency=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        unhealthy_cooldown_seconds=1.0,
    )
    try:
        response = await client.rank_batch(b'{"requests":[]}')
    finally:
        await client.aclose()
        server.close()
        await server.wait_closed()

    assert seen["operation"] == BATCH_RANK_OPERATION
    assert seen["body"] == b'{"requests":[]}'
    assert response.status_code == 200
    assert response.body == b'{"results":[]}'


@pytest.mark.asyncio
async def test_runner_client_reuses_persistent_connection():
    connection_count = 0
    operations = []

    async def handle_client(reader, writer):
        nonlocal connection_count
        connection_count += 1
        for _ in range(2):
            frame = await read_frame(reader)
            operations.append(frame[:1])
            writer.write(encode_response(200, "application/json", b'{"results":[]}'))
            await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    client = RankingRunnerClientPool(
        [RankingRunnerEndpoint(host="127.0.0.1", port=port, label="local")],
        dispatch_concurrency=1,
        runner_max_inflight_batches=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        unhealthy_cooldown_seconds=0.1,
    )
    try:
        first = await client.rank_batch(b'{"requests":[]}')
        second = await client.rank_batch(b'{"requests":[]}')
    finally:
        await client.aclose()
        server.close()
        await server.wait_closed()

    assert connection_count == 1
    assert operations == [BATCH_RANK_OPERATION, BATCH_RANK_OPERATION]
    assert first.status_code == 200
    assert second.status_code == 200


@pytest.mark.asyncio
async def test_runner_health_check_does_not_consume_dispatch_connection():
    operations = []

    async def handle_client(reader, writer):
        frame = await read_frame(reader)
        operations.append(frame[:1])
        writer.write(encode_response(200, "application/json", b'{"status":"ready"}'))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    endpoint = RankingRunnerEndpoint(host="127.0.0.1", port=port, label="local")
    client = RankingRunnerClientPool(
        [endpoint],
        dispatch_concurrency=1,
        runner_max_inflight_batches=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        unhealthy_cooldown_seconds=0.1,
    )
    held_connection = endpoint.connections.get_nowait()
    try:
        health = await client.health_check()
    finally:
        endpoint.connections.put_nowait(held_connection)
        await client.aclose()
        server.close()
        await server.wait_closed()

    assert health["status"] == "healthy"
    assert operations == [HEALTH_OPERATION]


@pytest.mark.asyncio
async def test_runner_client_health_check_records_batch_payload_capability():
    async def handle_client(reader, writer):
        await read_frame(reader)
        writer.write(
            encode_response(
                200,
                "application/json",
                b'{"status":"ready","batch_payload_versions":[1,2],"is_trained":true,"artifact_verified":true,"model_version":"ranking-v1","feature_schema_version":"ranking_v3_00_temporal_multimodal"}',
            )
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    endpoint = RankingRunnerEndpoint(host="127.0.0.1", port=port, label="local")
    client = RankingRunnerClientPool(
        [endpoint],
        dispatch_concurrency=1,
        runner_max_inflight_batches=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        unhealthy_cooldown_seconds=0.1,
    )
    try:
        assert client.supports_batch_payload_version(2) is False
        health = await client.health_check()
    finally:
        await client.aclose()
        server.close()
        await server.wait_closed()

    assert client.supports_batch_payload_version(2) is True
    assert health["endpoints"][0]["batch_payload_versions"] == [1, 2]
    assert health["endpoints"][0]["is_trained"] is True
    assert health["endpoints"][0]["artifact_verified"] is True
    assert health["endpoints"][0]["model_version"] == "ranking-v1"


@pytest.mark.asyncio
async def test_runner_client_counts_degraded_separately_from_healthy():
    async def handle_client(reader, writer):
        await read_frame(reader)
        writer.write(
            encode_response(
                200,
                "application/json",
                b'{"status":"degraded","failure_reason":"model_untrained"}',
            )
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    client = RankingRunnerClientPool(
        [
            RankingRunnerEndpoint(
                host="127.0.0.1",
                port=server.sockets[0].getsockname()[1],
                label="local",
            )
        ],
        dispatch_concurrency=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        unhealthy_cooldown_seconds=0.1,
    )
    try:
        health = await client.health_check()
    finally:
        await client.aclose()
        server.close()
        await server.wait_closed()

    assert health["healthy_count"] == 0
    assert health["degraded_count"] == 1
    assert health["endpoints"][0]["status"] == "degraded"


@pytest.mark.asyncio
async def test_runner_client_retries_overload_without_marking_unhealthy():
    attempts = []

    async def overloaded_runner(reader, writer):
        try:
            await read_frame(reader)
            attempts.append("overloaded")
            writer.write(
                encode_response(
                    429,
                    "application/json",
                    b'{"detail":"ranking_runner_overloaded"}',
                )
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def healthy_runner(reader, writer):
        try:
            await read_frame(reader)
            attempts.append("healthy")
            writer.write(encode_response(200, "application/json", b'{"results":[]}'))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    overloaded_server = await asyncio.start_server(
        overloaded_runner, host="127.0.0.1", port=0
    )
    healthy_server = await asyncio.start_server(
        healthy_runner, host="127.0.0.1", port=0
    )
    overloaded_endpoint = RankingRunnerEndpoint(
        host="127.0.0.1",
        port=overloaded_server.sockets[0].getsockname()[1],
        label="overloaded",
    )
    healthy_endpoint = RankingRunnerEndpoint(
        host="127.0.0.1",
        port=healthy_server.sockets[0].getsockname()[1],
        label="healthy",
    )
    client = RankingRunnerClientPool(
        [overloaded_endpoint, healthy_endpoint],
        dispatch_concurrency=2,
        runner_max_inflight_batches=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        unhealthy_cooldown_seconds=10.0,
    )
    try:
        response = await client.rank_batch(b'{"requests":[]}')
    finally:
        await client.aclose()
        overloaded_server.close()
        healthy_server.close()
        await overloaded_server.wait_closed()
        await healthy_server.wait_closed()

    assert response.status_code == 200
    assert attempts == ["overloaded", "healthy"]
    assert overloaded_endpoint.failed_until == 0.0


@pytest.mark.asyncio
async def test_single_runner_overload_fails_fast_without_same_endpoint_retry():
    attempts = 0

    async def handle_client(reader, writer):
        nonlocal attempts
        try:
            while True:
                await read_frame(reader)
                attempts += 1
                writer.write(
                    encode_response(
                        429,
                        "application/json",
                        b'{"detail":"ranking_runner_overloaded"}',
                    )
                )
                await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    endpoint = RankingRunnerEndpoint(host="127.0.0.1", port=port, label="local")
    client = RankingRunnerClientPool(
        [endpoint],
        dispatch_concurrency=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        unhealthy_cooldown_seconds=1.0,
    )
    try:
        with pytest.raises(RankingRunnerOverloaded):
            await client.rank_batch(b'{"requests":[]}')
    finally:
        await client.aclose()
        server.close()
        await server.wait_closed()

    assert attempts == 1


@pytest.mark.asyncio
async def test_runner_client_marks_draining_response_immediately():
    async def handle_client(reader, writer):
        await read_frame(reader)
        writer.write(
            encode_response(
                429,
                "application/json",
                b'{"detail":"ranking_runner_draining"}',
            )
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    endpoint = RankingRunnerEndpoint(host="127.0.0.1", port=port, label="local")
    client = RankingRunnerClientPool(
        [endpoint],
        dispatch_concurrency=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        unhealthy_cooldown_seconds=1.0,
    )
    try:
        with pytest.raises(RankingRunnerOverloaded, match="draining"):
            await client.rank_batch(b'{"requests":[]}')
    finally:
        await client.aclose()
        server.close()
        await server.wait_closed()

    assert endpoint.draining is True
    assert client.effective_capacity == 0


@pytest.mark.asyncio
async def test_protocol_draining_state_survives_dns_presence(monkeypatch):
    endpoint = RankingRunnerEndpoint(
        host="10.0.0.1",
        port=8014,
        label="runner->10.0.0.1:8014",
        source_host="runner",
        source_port=8014,
    )

    async def resolve_same(host, port, *, connect_timeout_seconds):
        return [
            RankingRunnerEndpoint(
                host="10.0.0.1",
                port=8014,
                label="runner->10.0.0.1:8014",
                source_host="runner",
                source_port=8014,
            )
        ]

    monkeypatch.setattr(
        RankingRunnerClientPool,
        "_resolve_endpoint",
        staticmethod(resolve_same),
    )
    client = RankingRunnerClientPool(
        [endpoint],
        configured_endpoints=[("runner", 8014)],
        dispatch_concurrency=1,
        connect_timeout_seconds=0.1,
        request_timeout_seconds=0.1,
        unhealthy_cooldown_seconds=0.1,
        dns_refresh_seconds=0.0,
    )
    health_responses = []

    async def fake_health(target):
        return health_responses.pop(0)

    monkeypatch.setattr(client, "_health_endpoint", fake_health)
    client._mark_draining(endpoint)
    try:
        # DNS presence alone never proves the runner was undrained.
        await client._refresh_endpoints_if_needed(force=True)
        after_dns = (endpoint.draining, endpoint.drain_reason)

        # A runner that still reports draining stays out of rotation.
        health_responses.append(
            RankingCoordinatorResponse(503, "application/json", b'{"status":"draining"}')
        )
        await client.health_check()
        after_draining_health = endpoint.draining

        # Only an explicit healthy, non-draining health response clears it.
        health_responses.append(
            RankingCoordinatorResponse(200, "application/json", b'{"status":"ready"}')
        )
        await client.health_check()
        await client._refresh_endpoints_if_needed(force=True)
    finally:
        await client.aclose()

    assert after_dns == (True, "protocol")
    assert after_draining_health is True
    assert endpoint.draining is False
    assert endpoint.drain_reason == ""
    assert client.effective_capacity == 1


@pytest.mark.asyncio
async def test_healthy_response_does_not_clear_dns_missing_drain(monkeypatch):
    endpoint = RankingRunnerEndpoint(host="10.0.0.2", port=8014, label="runner-b")
    client = RankingRunnerClientPool(
        [endpoint],
        dispatch_concurrency=1,
        connect_timeout_seconds=0.1,
        request_timeout_seconds=0.1,
    )

    async def fake_health(target):
        return RankingCoordinatorResponse(
            200, "application/json", b'{"status":"ready"}'
        )

    monkeypatch.setattr(client, "_health_endpoint", fake_health)
    endpoint.draining = True
    endpoint.drain_reason = "dns_missing"
    try:
        await client.health_check()
    finally:
        await client.aclose()

    assert endpoint.draining is True
    assert endpoint.drain_reason == "dns_missing"


@pytest.mark.asyncio
async def test_runner_client_marks_5xx_unhealthy():
    async def handle_client(reader, writer):
        await read_frame(reader)
        writer.write(
            encode_response(503, "application/json", b'{"detail":"runner failed"}')
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    endpoint = RankingRunnerEndpoint(host="127.0.0.1", port=port, label="local")
    client = RankingRunnerClientPool(
        [endpoint],
        configured_endpoints=[("runner", port)],
        dispatch_concurrency=1,
        runner_max_inflight_batches=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=0.1,
        unhealthy_cooldown_seconds=10.0,
    )
    force_refresh_calls = []
    original_refresh = client._refresh_endpoints_if_needed

    async def capture_refresh(*, force=False):
        force_refresh_calls.append(force)
        return await original_refresh(force=force)

    client._refresh_endpoints_if_needed = capture_refresh
    try:
        with pytest.raises(Exception):
            await client.rank_batch(b'{"requests":[]}')
    finally:
        await client.aclose()
        server.close()
        await server.wait_closed()

    assert endpoint.failed_until > 0.0
    assert True in force_refresh_calls


@pytest.mark.asyncio
async def test_runner_client_timeout_backoffs_without_marking_unhealthy():
    async def handle_client(reader, writer):
        await read_frame(reader)
        await asyncio.sleep(0.2)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    endpoint = RankingRunnerEndpoint(host="127.0.0.1", port=port, label="local")
    client = RankingRunnerClientPool(
        [endpoint],
        configured_endpoints=[("runner", port)],
        dispatch_concurrency=1,
        runner_max_inflight_batches=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=0.05,
        unhealthy_cooldown_seconds=10.0,
    )
    force_refresh_calls = []
    original_refresh = client._refresh_endpoints_if_needed

    async def capture_refresh(*, force=False):
        force_refresh_calls.append(force)
        return await original_refresh(force=force)

    client._refresh_endpoints_if_needed = capture_refresh
    try:
        with pytest.raises(RankingRunnerTimeout):
            await client.rank_batch(b'{"requests":[]}')
    finally:
        await client.aclose()
        server.close()
        await server.wait_closed()

    assert endpoint.failed_until == 0.0
    assert endpoint.overloaded_until > 0.0
    assert True not in force_refresh_calls


@pytest.mark.asyncio
async def test_runner_client_deadline_503_backoffs_without_marking_unhealthy():
    async def handle_client(reader, writer):
        await read_frame(reader)
        writer.write(
            encode_response(
                503,
                "application/json",
                b'{"detail":"ranking_runner_deadline_exceeded"}',
            )
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    endpoint = RankingRunnerEndpoint(host="127.0.0.1", port=port, label="local")
    client = RankingRunnerClientPool(
        [endpoint],
        dispatch_concurrency=1,
        runner_max_inflight_batches=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        unhealthy_cooldown_seconds=10.0,
    )
    try:
        with pytest.raises(RankingRunnerTimeout):
            await client.rank_batch(b'{"requests":[]}')
    finally:
        await client.aclose()
        server.close()
        await server.wait_closed()

    assert endpoint.failed_until == 0.0
    assert endpoint.overloaded_until > 0.0


@pytest.mark.asyncio
async def test_runner_client_single_dns_miss_keeps_endpoint_sticky(monkeypatch):
    async def resolve_only_first(host, port, *, connect_timeout_seconds):
        return [
            RankingRunnerEndpoint(
                host="10.0.0.1",
                port=port,
                label=f"{host}->10.0.0.1:{port}",
                source_host=host,
                source_port=port,
            )
        ]

    monkeypatch.setattr(
        RankingRunnerClientPool,
        "_resolve_endpoint",
        staticmethod(resolve_only_first),
    )
    missing = RankingRunnerEndpoint(
        host="10.0.0.2",
        port=8014,
        label="runner->10.0.0.2:8014",
        source_host="runner",
        source_port=8014,
    )
    client = RankingRunnerClientPool(
        [
            RankingRunnerEndpoint(
                host="10.0.0.1",
                port=8014,
                label="runner->10.0.0.1:8014",
                source_host="runner",
                source_port=8014,
            ),
            missing,
        ],
        configured_endpoints=[("runner", 8014)],
        dispatch_concurrency=2,
        runner_max_inflight_batches=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        unhealthy_cooldown_seconds=1.0,
        dns_refresh_seconds=0.0,
        endpoint_missing_refreshes=3,
        endpoint_missing_grace_seconds=0.0,
    )
    try:
        await client._refresh_endpoints_if_needed()
    finally:
        await client.aclose()

    assert missing in client.endpoints
    assert missing.missing_refresh_count == 1
    assert missing.draining is False


@pytest.mark.asyncio
async def test_runner_client_missing_endpoint_drains_before_removal(monkeypatch):
    async def resolve_only_first(host, port, *, connect_timeout_seconds):
        return [
            RankingRunnerEndpoint(
                host="10.0.0.1",
                port=port,
                label=f"{host}->10.0.0.1:{port}",
                source_host=host,
                source_port=port,
            )
        ]

    monkeypatch.setattr(
        RankingRunnerClientPool,
        "_resolve_endpoint",
        staticmethod(resolve_only_first),
    )
    missing = RankingRunnerEndpoint(
        host="10.0.0.2",
        port=8014,
        label="runner->10.0.0.2:8014",
        source_host="runner",
        source_port=8014,
        inflight_batches=1,
    )
    client = RankingRunnerClientPool(
        [
            RankingRunnerEndpoint(
                host="10.0.0.1",
                port=8014,
                label="runner->10.0.0.1:8014",
                source_host="runner",
                source_port=8014,
            ),
            missing,
        ],
        configured_endpoints=[("runner", 8014)],
        dispatch_concurrency=2,
        runner_max_inflight_batches=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        unhealthy_cooldown_seconds=1.0,
        dns_refresh_seconds=0.0,
        endpoint_missing_refreshes=1,
        endpoint_missing_grace_seconds=0.0,
    )
    try:
        await client._refresh_endpoints_if_needed()
        assert missing in client.endpoints
        assert missing.draining is True
        assert client._endpoint_has_capacity(missing, time.monotonic()) is False

        missing.inflight_batches = 0
        await client._refresh_endpoints_if_needed()
        assert missing not in client.endpoints
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_runner_client_timeout_does_not_retry_sent_batch():
    attempts = 0
    server_done = asyncio.Event()

    async def handle_client(reader, writer):
        nonlocal attempts
        try:
            await read_frame(reader)
            attempts += 1
            await asyncio.sleep(0.2)
        finally:
            writer.close()
            await writer.wait_closed()
            server_done.set()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    endpoint = RankingRunnerEndpoint(host="127.0.0.1", port=port, label="local")
    client = RankingRunnerClientPool(
        [endpoint],
        dispatch_concurrency=1,
        runner_max_inflight_batches=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=0.05,
        unhealthy_cooldown_seconds=10.0,
    )
    try:
        with pytest.raises(RankingRunnerTimeout):
            await client.rank_batch(b'{"requests":[]}')
        await asyncio.wait_for(server_done.wait(), timeout=1.0)
    finally:
        await client.aclose()
        server.close()
        await server.wait_closed()

    assert attempts == 1
    assert endpoint.failed_until == 0.0


@pytest.mark.asyncio
async def test_runner_client_connection_close_after_send_does_not_retry():
    attempts = []

    async def closes_after_read(reader, writer):
        await read_frame(reader)
        attempts.append("accepted")
        writer.close()
        await writer.wait_closed()

    async def healthy_runner(reader, writer):
        await read_frame(reader)
        attempts.append("retried")
        writer.write(encode_response(200, "application/json", b'{"results":[]}'))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    first = await asyncio.start_server(closes_after_read, host="127.0.0.1", port=0)
    second = await asyncio.start_server(healthy_runner, host="127.0.0.1", port=0)
    client = RankingRunnerClientPool(
        [
            RankingRunnerEndpoint(
                host="127.0.0.1",
                port=first.sockets[0].getsockname()[1],
                label="first",
            ),
            RankingRunnerEndpoint(
                host="127.0.0.1",
                port=second.sockets[0].getsockname()[1],
                label="second",
            ),
        ],
        dispatch_concurrency=2,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        unhealthy_cooldown_seconds=1.0,
    )
    try:
        with pytest.raises(RankingRunnerRequestUncertain):
            await client.rank_batch(b'{"requests":[]}')
    finally:
        await client.aclose()
        first.close()
        second.close()
        await first.wait_closed()
        await second.wait_closed()

    assert attempts == ["accepted"]


async def _close_server(server):
    server.close()
    await server.wait_closed()
    # Let per-connection handlers observe EOF before the loop shuts down.
    await asyncio.sleep(0.05)


# --- Connection auth (ranking-auth v1) -------------------------------------

TEST_INTERNAL_KEY = "ranking-test-internal-key"


class _AuthRecorder:
    def __init__(self):
        self.events = []

    def record_ranking_runtime_auth(self, role, result):
        self.events.append((role, result))


def _auth_settings(mode, key=TEST_INTERNAL_KEY):
    return RankingRuntimeAuthSettings.build(mode, key)


async def _start_coordinator_server(mode, *, key=TEST_INTERNAL_KEY, stub=True):
    """Serve the real coordinator connection loop with a stub frame handler."""
    from video_commerce.services.ranking_coordinator.main import RankingCoordinator

    coordinator = RankingCoordinator()
    recorder = _AuthRecorder()
    coordinator.authenticator = RankingRuntimeAuthenticator(
        _auth_settings(mode, key),
        role="coordinator",
        observability=recorder,
    )
    handled = []

    real_handle_frame = coordinator._handle_frame

    async def handle_frame(frame):
        handled.append(frame[:1])
        if not stub:
            return await real_handle_frame(frame)
        return encode_response(200, "application/json", b'{"ok":true}')

    coordinator._handle_frame = handle_frame
    server = await asyncio.start_server(
        coordinator._handle_client, host="127.0.0.1", port=0
    )
    return server, server.sockets[0].getsockname()[1], handled, recorder


async def _start_legacy_server():
    """A server that predates the A frame: unknown operations get a 400."""
    handled = []

    async def handle_client(reader, writer):
        try:
            while True:
                try:
                    frame = await read_frame(reader)
                except Exception:
                    break
                handled.append(frame[:1])
                if frame[:1] == RANK_OPERATION:
                    writer.write(encode_response(200, "application/json", b'{"ok":true}'))
                else:
                    writer.write(
                        encode_response(
                            400,
                            "application/json",
                            b'{"detail":"invalid coordinator operation"}',
                        )
                    )
                await writer.drain()
        finally:
            writer.close()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    return server, server.sockets[0].getsockname()[1], handled


async def _raw_request(port, *frames):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    responses = []
    try:
        for operation, body in frames:
            writer.write(encode_request(operation, body))
            await writer.drain()
            try:
                responses.append(decode_response(await read_frame(reader)))
            except RankingCoordinatorUnavailable:
                responses.append(None)
                break
    finally:
        writer.close()
    return responses


def test_auth_handshake_round_trip_and_replay_rejected():
    settings = _auth_settings("required")
    now = 1_800_000_000
    authenticator = RankingRuntimeAuthenticator(
        settings, role="runner", clock=lambda: now
    )
    body = settings.build_handshake_body("runner", now=now, nonce="a" * 32)

    assert authenticator.verify_handshake(body) is None
    assert authenticator.verify_handshake(body) == "replayed"


def test_auth_handshake_rejects_expired_wrong_key_and_malformed():
    settings = _auth_settings("required")
    now = 1_800_000_000
    authenticator = RankingRuntimeAuthenticator(
        settings, role="runner", clock=lambda: now
    )

    stale = settings.build_handshake_body("runner", now=now - 61, nonce="b" * 32)
    future = settings.build_handshake_body("runner", now=now + 61, nonce="c" * 32)
    wrong_key = _auth_settings("required", "other-key").build_handshake_body(
        "runner", now=now, nonce="d" * 32
    )

    assert authenticator.verify_handshake(stale) == "expired"
    assert authenticator.verify_handshake(future) == "expired"
    assert authenticator.verify_handshake(wrong_key) == "invalid_mac"
    assert authenticator.verify_handshake(b"not-json") == "malformed"
    assert authenticator.verify_handshake(b'{"v":2}') == "unsupported_version"
    # Failed attempts must not consume nonces from the replay cache.
    assert (
        authenticator.verify_handshake(
            settings.build_handshake_body("runner", now=now, nonce="d" * 32)
        )
        is None
    )


def test_auth_nonce_cache_is_bounded():
    settings = _auth_settings("required")
    now = 1_800_000_000
    authenticator = RankingRuntimeAuthenticator(
        settings, role="runner", clock=lambda: now, nonce_cache_size=2
    )
    for index in range(3):
        body = settings.build_handshake_body("runner", now=now, nonce=f"{index:032d}")
        assert authenticator.verify_handshake(body) is None

    assert len(authenticator._seen_nonces) == 2


def test_auth_settings_never_send_handshake_without_key_or_when_disabled():
    assert not RankingRuntimeAuthSettings.build("permissive", "").client_sends_handshake
    assert not RankingRuntimeAuthSettings.build(
        "disabled", TEST_INTERNAL_KEY
    ).client_sends_handshake
    assert RankingRuntimeAuthSettings.build(
        "permissive", TEST_INTERNAL_KEY
    ).client_sends_handshake
    with pytest.raises(ValueError):
        RankingRuntimeAuthSettings.build("strict", TEST_INTERNAL_KEY)


@pytest.mark.asyncio
async def test_required_server_rejects_unauthenticated_frame_and_closes():
    server, port, handled, recorder = await _start_coordinator_server("required")
    try:
        responses = await _raw_request(
            port,
            (RANK_OPERATION, b'{"request_id":"r1"}'),
            (RANK_OPERATION, b'{"request_id":"r2"}'),
        )
    finally:
        await _close_server(server)

    assert responses[0].status_code == 401
    assert json.loads(responses[0].body)["detail"] == "ranking_runtime_auth_required"
    # The connection is closed after the rejection: no second response.
    assert responses[1:] == [None]
    assert handled == []
    assert recorder.events == [("coordinator", "rejected")]


def test_auth_handshake_rejects_deeply_nested_json_as_malformed():
    authenticator = RankingRuntimeAuthenticator(
        _auth_settings("required"), role="runner"
    )

    assert authenticator.verify_handshake(b"[" * 1000) == "malformed"


@pytest.mark.asyncio
async def test_required_server_closes_oversized_pre_auth_frame_without_buffering():
    server, port, handled, recorder = await _start_coordinator_server("required")
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        # Declare a 1 MiB frame but send only the header: the server must close
        # on the length alone instead of waiting to buffer the body.
        writer.write(struct.pack("!I", 1024 * 1024) + RANK_OPERATION)
        await writer.drain()
        closed = await asyncio.wait_for(reader.read(), timeout=1.0)
    finally:
        writer.close()
        await _close_server(server)

    assert closed == b""
    assert handled == []
    assert recorder.events == [("coordinator", "rejected")]


@pytest.mark.asyncio
async def test_required_server_closes_idle_unauthenticated_connection():
    from video_commerce.services.ranking_coordinator.main import RankingCoordinator

    coordinator = RankingCoordinator()
    recorder = _AuthRecorder()
    coordinator.authenticator = RankingRuntimeAuthenticator(
        _auth_settings("required"),
        role="coordinator",
        observability=recorder,
        handshake_timeout_seconds=0.1,
    )
    server = await asyncio.start_server(
        coordinator._handle_client, host="127.0.0.1", port=0
    )
    port = server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        closed = await asyncio.wait_for(reader.read(), timeout=1.0)
    finally:
        writer.close()
        await _close_server(server)

    assert closed == b""
    assert recorder.events == [("coordinator", "rejected")]


@pytest.mark.asyncio
async def test_permissive_server_does_not_bound_unauthenticated_frames():
    server, port, handled, recorder = await _start_coordinator_server("permissive")
    try:
        responses = await _raw_request(port, (RANK_OPERATION, b"x" * 4096))
    finally:
        await _close_server(server)

    assert responses[0].status_code == 200
    assert handled == [RANK_OPERATION]


@pytest.mark.asyncio
async def test_required_server_accepts_authenticated_pool_and_skips_per_frame_checks():
    server, port, handled, recorder = await _start_coordinator_server("required")
    client = RankingCoordinatorClientPool(
        "127.0.0.1",
        port,
        pool_size=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        auth_settings=_auth_settings("required"),
    )
    try:
        first = await client.rank(b'{"request_id":"r1"}')
        second = await client.health()
    finally:
        await client.aclose()
        await _close_server(server)

    assert first.status_code == 200
    assert second.status_code == 200
    assert handled == [RANK_OPERATION, HEALTH_OPERATION]
    assert recorder.events == [("coordinator", "authenticated")]


@pytest.mark.asyncio
async def test_required_server_rejects_replayed_handshake():
    server, port, handled, recorder = await _start_coordinator_server("required")
    body = _auth_settings("required").build_handshake_body("coordinator")
    try:
        first = await _raw_request(
            port, (AUTH_OPERATION, body), (HEALTH_OPERATION, b"")
        )
        replay = await _raw_request(
            port, (AUTH_OPERATION, body), (HEALTH_OPERATION, b"")
        )
    finally:
        await _close_server(server)

    assert [response.status_code for response in first] == [200, 200]
    assert replay[0].status_code == 401
    assert json.loads(replay[0].body)["reason"] == "replayed"
    assert replay[1:] == [None]
    assert handled == [HEALTH_OPERATION]
    assert recorder.events == [
        ("coordinator", "authenticated"),
        ("coordinator", "replayed"),
    ]


@pytest.mark.asyncio
async def test_keyless_pool_against_required_server_raises_auth_rejected():
    server, port, handled, recorder = await _start_coordinator_server("required")
    client = RankingCoordinatorClientPool(
        "127.0.0.1",
        port,
        pool_size=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        auth_settings=RankingRuntimeAuthSettings.build("permissive", ""),
    )
    try:
        with pytest.raises(RankingRuntimeAuthRejected, match="auth_required"):
            await client.rank(b'{"request_id":"r1"}')
    finally:
        await client.aclose()
        await _close_server(server)

    assert handled == []
    assert recorder.events == [("coordinator", "rejected")]


@pytest.mark.asyncio
async def test_keyless_runner_pool_marks_required_runner_failed():
    from video_commerce.services.ranking_runner.main import RankingRunner

    runner = RankingRunner()
    runner.authenticator = RankingRuntimeAuthenticator(
        _auth_settings("required"), role="runner"
    )
    server = await asyncio.start_server(
        runner._handle_client, host="127.0.0.1", port=0
    )
    port = server.sockets[0].getsockname()[1]
    endpoint = RankingRunnerEndpoint(host="127.0.0.1", port=port, label="local")
    client = RankingRunnerClientPool(
        [endpoint],
        dispatch_concurrency=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=0.2,
        unhealthy_cooldown_seconds=5.0,
    )
    try:
        with pytest.raises(RankingRunnerUnavailable, match="auth"):
            await client.rank_batch(b'{"requests":[]}')
    finally:
        await client.aclose()
        await _close_server(server)

    assert endpoint.failed_until > time.monotonic()


@pytest.mark.asyncio
async def test_client_with_wrong_key_is_rejected_without_retry():
    server, port, handled, recorder = await _start_coordinator_server("required")
    client = RankingCoordinatorClientPool(
        "127.0.0.1",
        port,
        pool_size=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        auth_settings=_auth_settings("permissive", "wrong-key"),
    )
    try:
        with pytest.raises(RankingRuntimeAuthRejected, match="invalid_mac"):
            await client.rank(b'{"request_id":"r1"}')
    finally:
        await client.aclose()
        await _close_server(server)

    assert handled == []
    assert recorder.events == [("coordinator", "invalid_mac")]


@pytest.mark.asyncio
async def test_permissive_server_accepts_and_counts_legacy_unauthenticated_client():
    server, port, handled, recorder = await _start_coordinator_server("permissive")
    client = RankingCoordinatorClientPool(
        "127.0.0.1",
        port,
        pool_size=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
    )
    try:
        first = await client.rank(b'{"request_id":"r1"}')
        second = await client.rank(b'{"request_id":"r2"}')
    finally:
        await client.aclose()
        await _close_server(server)

    assert first.status_code == 200
    assert second.status_code == 200
    assert handled == [RANK_OPERATION, RANK_OPERATION]
    # Counted once per connection, not per frame.
    assert recorder.events == [("coordinator", "unauthenticated")]


@pytest.mark.asyncio
async def test_permissive_client_tolerates_legacy_server_rejecting_auth_op():
    server, port, handled = await _start_legacy_server()
    client = RankingCoordinatorClientPool(
        "127.0.0.1",
        port,
        pool_size=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        auth_settings=_auth_settings("permissive"),
    )
    try:
        response = await client.rank(b'{"request_id":"r1"}')
        connection = client._queue.get_nowait()
        client._queue.put_nowait(connection)
    finally:
        await client.aclose()
        await _close_server(server)

    assert response.status_code == 200
    assert handled == [AUTH_OPERATION, RANK_OPERATION]
    assert connection.authenticated is False


@pytest.mark.asyncio
async def test_required_client_refuses_legacy_server():
    server, port, handled = await _start_legacy_server()
    client = RankingCoordinatorClientPool(
        "127.0.0.1",
        port,
        pool_size=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        auth_settings=_auth_settings("required"),
    )
    try:
        with pytest.raises(RankingRuntimeAuthRejected):
            await client.rank(b'{"request_id":"r1"}')
    finally:
        await client.aclose()
        await _close_server(server)

    assert handled == [AUTH_OPERATION]


@pytest.mark.asyncio
async def test_disabled_server_treats_auth_op_as_unknown():
    server, port, handled, recorder = await _start_coordinator_server(
        "disabled", stub=False
    )
    try:
        responses = await _raw_request(
            port,
            (AUTH_OPERATION, _auth_settings("required").build_handshake_body("coordinator")),
        )
    finally:
        await _close_server(server)

    # Disabled auth dispatches A like any frame, so it is an unknown op (400),
    # which permissive clients already tolerate.
    assert handled == [AUTH_OPERATION]
    assert responses[0].status_code == 400
    assert recorder.events == []


@pytest.mark.asyncio
async def test_runner_pool_authenticates_dispatch_and_health_connections():
    from video_commerce.services.ranking_runner.main import RankingRunner

    runner = RankingRunner()
    recorder = _AuthRecorder()
    runner.authenticator = RankingRuntimeAuthenticator(
        _auth_settings("required"), role="runner", observability=recorder
    )
    handled = []

    async def handle_frame(frame, *, peer=None):
        handled.append(frame[:1])
        return encode_response(200, "application/json", b'{"status":"ready"}')

    runner._handle_frame = handle_frame
    server = await asyncio.start_server(
        runner._handle_client, host="127.0.0.1", port=0
    )
    port = server.sockets[0].getsockname()[1]
    client = RankingRunnerClientPool(
        [RankingRunnerEndpoint(host="127.0.0.1", port=port, label="local")],
        dispatch_concurrency=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        unhealthy_cooldown_seconds=1.0,
        auth_settings=_auth_settings("required"),
    )
    try:
        batch = await client.rank_batch(b'{"requests":[]}')
        health = await client.health_check()
    finally:
        await client.aclose()
        await _close_server(server)

    assert batch.status_code == 200
    assert health["healthy_count"] == 1
    assert handled == [BATCH_RANK_OPERATION, HEALTH_OPERATION]
    assert recorder.events == [("runner", "authenticated"), ("runner", "authenticated")]


# --- Cancellation safety ---------------------------------------------------


async def _start_slow_echo_server():
    """Echo each frame body; a body of b"slow" is answered after a delay."""
    stats = {"connections": 0}

    async def handle_client(reader, writer):
        stats["connections"] += 1
        try:
            while True:
                try:
                    frame = await read_frame(reader)
                except Exception:
                    break
                body = frame[1:]
                if body == b"slow":
                    await asyncio.sleep(0.3)
                writer.write(encode_response(200, "application/octet-stream", body))
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    return server, server.sockets[0].getsockname()[1], stats


@pytest.mark.asyncio
async def test_cancelled_coordinator_request_connection_is_not_reused():
    server, port, stats = await _start_slow_echo_server()
    client = RankingCoordinatorClientPool(
        "127.0.0.1",
        port,
        pool_size=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=2.0,
    )
    try:
        slow = asyncio.create_task(client.rank(b"slow"))
        await asyncio.sleep(0.05)
        slow.cancel()
        with pytest.raises(asyncio.CancelledError):
            await slow
        # Without the fix the next borrower would read the late b"slow" reply.
        response = await client.rank(b"fast")
        await asyncio.sleep(0.35)
        follow_up = await client.rank(b"again")
    finally:
        await client.aclose()
        await _close_server(server)

    assert response.body == b"fast"
    assert follow_up.body == b"again"
    assert stats["connections"] == 2


@pytest.mark.asyncio
async def test_cancelled_runner_batch_connection_is_not_reused():
    server, port, stats = await _start_slow_echo_server()
    endpoint = RankingRunnerEndpoint(host="127.0.0.1", port=port, label="local")
    client = RankingRunnerClientPool(
        [endpoint],
        dispatch_concurrency=1,
        runner_max_inflight_batches=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=2.0,
        unhealthy_cooldown_seconds=0.1,
    )
    try:
        slow = asyncio.create_task(client.rank_batch(b"slow"))
        await asyncio.sleep(0.05)
        slow.cancel()
        with pytest.raises(asyncio.CancelledError):
            await slow
        assert endpoint.inflight_batches == 0
        assert endpoint.connections.qsize() == 1
        response = await client.rank_batch(b"fast")
        await asyncio.sleep(0.35)
        follow_up = await client.rank_batch(b"again")
    finally:
        await client.aclose()
        await _close_server(server)

    assert response.body == b"fast"
    assert follow_up.body == b"again"
    assert stats["connections"] == 2


def test_pool_release_resets_connection_left_in_flight():
    endpoint = RankingRunnerEndpoint(host="127.0.0.1", port=1, label="local")
    client = RankingRunnerClientPool(
        [endpoint],
        dispatch_concurrency=1,
        connect_timeout_seconds=0.1,
        request_timeout_seconds=0.1,
    )
    acquired = client._acquire_endpoint_connection()
    assert acquired is not None
    _, connection = acquired

    class _Writer:
        closed = False

        def is_closing(self):
            return self.closed

        def close(self):
            self.closed = True

    writer = _Writer()
    connection.writer = writer
    connection.in_flight = True
    client._release_endpoint(endpoint, connection)

    assert writer.closed is True
    assert connection.closed is True
    assert connection.reusable is False
    assert endpoint.connections.qsize() == 1
    replacement = endpoint.connections.get_nowait()
    assert replacement is not connection
    assert replacement.reusable is True


# --- Review follow-ups: role binding, skew boundary, per-replica replay ----


def test_auth_skew_boundary_is_inclusive_at_sixty_seconds():
    settings = _auth_settings("required")
    now = 1_800_000_000
    authenticator = RankingRuntimeAuthenticator(
        settings, role="runner", clock=lambda: now
    )

    def verify(offset, nonce_char):
        return authenticator.verify_handshake(
            settings.build_handshake_body("runner", now=now + offset, nonce=nonce_char * 32)
        )

    assert verify(-60, "a") is None
    assert verify(60, "b") is None
    assert verify(-61, "c") == "expired"
    assert verify(61, "d") == "expired"


def test_handshake_is_bound_to_server_role():
    settings = _auth_settings("required")
    now = 1_800_000_000
    runner = RankingRuntimeAuthenticator(settings, role="runner", clock=lambda: now)
    coordinator_body = settings.build_handshake_body(
        "coordinator", now=now, nonce="e" * 32
    )
    # Relabelling the role without the key cannot produce a valid MAC.
    relabelled = json.loads(coordinator_body)
    relabelled["role"] = "runner"

    assert runner.verify_handshake(coordinator_body) == "wrong_role"
    assert runner.verify_handshake(json.dumps(relabelled).encode()) == "invalid_mac"


def test_replay_cache_is_per_replica():
    """Documents a known limit: replicas of one role share no nonce state."""
    settings = _auth_settings("required")
    now = 1_800_000_000
    replica_a = RankingRuntimeAuthenticator(settings, role="runner", clock=lambda: now)
    replica_b = RankingRuntimeAuthenticator(settings, role="runner", clock=lambda: now)
    body = settings.build_handshake_body("runner", now=now, nonce="f" * 32)

    assert replica_a.verify_handshake(body) is None
    assert replica_a.verify_handshake(body) == "replayed"
    assert replica_b.verify_handshake(body) is None


def test_weak_production_auth_logs_warning(monkeypatch, caplog):
    from video_commerce.ranking_runtime.ranking_auth import warn_if_weak_production_auth

    logger = logging.getLogger("test.ranking_auth")
    monkeypatch.setenv("ENVIRONMENT", "production")
    with caplog.at_level(logging.WARNING, logger="test.ranking_auth"):
        assert warn_if_weak_production_auth(
            _auth_settings("permissive"), role="runner", logger=logger
        )
        assert not warn_if_weak_production_auth(
            _auth_settings("required"), role="runner", logger=logger
        )
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert not warn_if_weak_production_auth(
        _auth_settings("permissive"), role="runner", logger=logger
    )
    assert [record.message for record in caplog.records] == [
        "ranking_runtime_auth_not_enforced"
    ]


# --- Review follow-ups: unstubbed end-to-end -------------------------------


@pytest.mark.asyncio
async def test_unstubbed_required_coordinator_rejects_unauthenticated_rank():
    server, port, handled, recorder = await _start_coordinator_server(
        "required", stub=False
    )
    try:
        responses = await _raw_request(port, (RANK_OPERATION, b'{"request_id":"r1"}'))
        authenticated = await _raw_request(
            port,
            (AUTH_OPERATION, _auth_settings("required").build_handshake_body("coordinator")),
            (HEALTH_OPERATION, b""),
        )
    finally:
        await _close_server(server)

    assert responses[0].status_code == 401
    assert json.loads(responses[0].body)["detail"] == "ranking_runtime_auth_required"
    # The real handler only ran for the authenticated health frame (503: no
    # model loaded in this bare coordinator).
    assert handled == [HEALTH_OPERATION]
    assert authenticated[0].status_code == 200
    assert authenticated[1].status_code == 503
    assert json.loads(authenticated[1].body)["status"] == "not_ready"


@pytest.mark.asyncio
async def test_unstubbed_runner_pool_health_against_required_runner():
    from video_commerce.services.ranking_runner.main import RankingRunner

    class _HealthyModel:
        def health_check(self):
            return {"status": "healthy"}

    runner = RankingRunner()
    runner.ranking_model = _HealthyModel()
    runner.authenticator = RankingRuntimeAuthenticator(
        _auth_settings("required"), role="runner"
    )
    server = await asyncio.start_server(
        runner._handle_client, host="127.0.0.1", port=0
    )
    port = server.sockets[0].getsockname()[1]
    endpoint = RankingRunnerEndpoint(host="127.0.0.1", port=port, label="local")
    authed = RankingRunnerClientPool(
        [endpoint],
        dispatch_concurrency=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        auth_settings=_auth_settings("required"),
    )
    keyless = RankingRunnerClientPool(
        [RankingRunnerEndpoint(host="127.0.0.1", port=port, label="keyless")],
        dispatch_concurrency=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
    )
    try:
        healthy = await authed.health_check()
        rejected = await keyless.health_check()
    finally:
        await authed.aclose()
        await keyless.aclose()
        await _close_server(server)

    assert healthy["healthy_count"] == 1
    assert rejected["healthy_count"] == 0
    assert rejected["endpoints"][0]["error"] == "RankingRunnerAuthRejected"


# --- Review follow-ups: pool acquire/release under cancellation ------------


@pytest.mark.asyncio
async def test_cancel_racing_acquire_keeps_pool_size_and_honours_cancel():
    server, port, stats = await _start_slow_echo_server()
    client = RankingCoordinatorClientPool(
        "127.0.0.1",
        port,
        pool_size=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=2.0,
    )
    held = client._queue.get_nowait()
    try:
        task = asyncio.create_task(client.rank(b"fast"))
        await asyncio.sleep(0)  # task is now waiting for a connection
        client._queue.put_nowait(held)
        # Step until the getter has dequeued the connection, then cancel the
        # waiter before it resumes: the window where wait_for(queue.get())
        # either drops the item (3.12+) or swallows the cancel (3.10).
        for _ in range(20):
            if client._queue.qsize() == 0:
                break
            await asyncio.sleep(0)
        assert client._queue.qsize() == 0
        task.cancel()
        result = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        assert client._queue.qsize() == 1
        response = await client.rank(b"after")
    finally:
        await client.aclose()
        await _close_server(server)

    assert response.body == b"after"


@pytest.mark.asyncio
async def test_acquire_timeout_and_cancel_while_waiting_keep_pool_size():
    client = RankingCoordinatorClientPool(
        "127.0.0.1",
        1,
        pool_size=1,
        connect_timeout_seconds=0.1,
        request_timeout_seconds=0.1,
    )
    held = client._queue.get_nowait()
    try:
        with pytest.raises(RankingCoordinatorTimeout, match="pool timeout"):
            await client.rank(b"x")
        waiting = asyncio.create_task(client.rank(b"y"))
        await asyncio.sleep(0.01)
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        client._queue.put_nowait(held)
        await asyncio.sleep(0)
        assert client._queue.qsize() == 1
        assert client._queue._getters == type(client._queue._getters)() or all(
            getter.cancelled() for getter in client._queue._getters
        )
    finally:
        await client.aclose()


async def _start_unresponsive_auth_server():
    """Reads the A frame, never answers, and records when the peer hangs up."""
    events = {"auth_frames": 0, "peer_closed": asyncio.Event()}

    async def handle_client(reader, writer):
        try:
            frame = await read_frame(reader)
            if frame[:1] == AUTH_OPERATION:
                events["auth_frames"] += 1
            await reader.read()  # returns b"" (or raises) once the client drops
        except Exception:
            pass
        finally:
            events["peer_closed"].set()
            writer.close()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    return server, server.sockets[0].getsockname()[1], events


@pytest.mark.asyncio
async def test_cancel_during_handshake_closes_socket_and_keeps_pool_size():
    server, port, events = await _start_unresponsive_auth_server()
    client = RankingCoordinatorClientPool(
        "127.0.0.1",
        port,
        pool_size=1,
        connect_timeout_seconds=2.0,
        request_timeout_seconds=2.0,
        auth_settings=_auth_settings("required"),
    )
    try:
        task = asyncio.create_task(client.rank(b"x"))
        await asyncio.sleep(0.05)
        assert events["auth_frames"] == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(events["peer_closed"].wait(), timeout=1.0)
        assert client._queue.qsize() == 1
        connection = client._queue.get_nowait()
        assert connection.writer is None
        assert connection.reusable is True
        client._queue.put_nowait(connection)
    finally:
        await client.aclose()
        await _close_server(server)


@pytest.mark.asyncio
async def test_cancelled_request_connection_is_closed_and_replaced():
    peer_closed = asyncio.Event()

    async def handle_client(reader, writer):
        try:
            await read_frame(reader)
            await reader.read()
        except Exception:
            pass
        finally:
            peer_closed.set()
            writer.close()

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    client = RankingCoordinatorClientPool(
        "127.0.0.1",
        port,
        pool_size=1,
        connect_timeout_seconds=1.0,
        request_timeout_seconds=2.0,
    )
    original = client._queue._queue[0]
    try:
        task = asyncio.create_task(client.rank(b"slow"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(peer_closed.wait(), timeout=1.0)
        replacement = client._queue._queue[0]
        replacement_reusable = replacement.reusable
    finally:
        await client.aclose()
        await _close_server(server)

    assert original.closed is True
    assert original.reusable is False
    assert replacement is not original
    assert replacement_reusable is True


@pytest.mark.asyncio
async def test_probe_started_before_drain_does_not_clear_it(monkeypatch):
    endpoint = RankingRunnerEndpoint(host="10.0.0.3", port=8014, label="runner-c")
    client = RankingRunnerClientPool(
        [endpoint],
        dispatch_concurrency=1,
        connect_timeout_seconds=0.1,
        request_timeout_seconds=0.1,
    )
    release_probe = asyncio.Event()

    async def slow_ready_health(target):
        # The runner answered "ready" before it started draining.
        await release_probe.wait()
        return RankingCoordinatorResponse(200, "application/json", b'{"status":"ready"}')

    monkeypatch.setattr(client, "_health_endpoint", slow_ready_health)
    try:
        probe = asyncio.create_task(client.health_check())
        await asyncio.sleep(0)
        client._mark_draining(endpoint)  # a batch just got ranking_runner_draining
        release_probe.set()
        await probe
        stale_probe_result = endpoint.draining
        await client.health_check()  # a probe started after the drain
    finally:
        await client.aclose()

    assert stale_probe_result is True
    assert endpoint.draining is False
