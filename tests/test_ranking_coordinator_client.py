import asyncio
import time

import pytest

from video_commerce.ranking_runtime.ranking_coordinator_client import (
    HEALTH_OPERATION,
    RANK_OPERATION,
    RankingCoordinatorClientPool,
    RankingCoordinatorTimeout,
    RankingCoordinatorUnavailable,
    decode_response,
    encode_response,
    read_frame,
)
from video_commerce.ranking_runtime.ranking_runner_client import (
    BATCH_RANK_OPERATION,
    RankingRunnerClientPool,
    RankingRunnerEndpoint,
    RankingRunnerTimeout,
    parse_runner_urls,
)


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
                b'{"status":"ready","batch_payload_versions":[1,2]}',
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


@pytest.mark.asyncio
async def test_runner_client_retries_overload_without_marking_unhealthy():
    attempts = 0

    async def handle_client(reader, writer):
        nonlocal attempts
        while True:
            try:
                await read_frame(reader)
            except Exception:
                break
            attempts += 1
            if attempts == 1:
                writer.write(
                    encode_response(
                        429,
                        "application/json",
                        b'{"detail":"ranking_runner_overloaded"}',
                    )
                )
            else:
                writer.write(
                    encode_response(200, "application/json", b'{"results":[]}')
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
        response = await client.rank_batch(b'{"requests":[]}')
    finally:
        await client.aclose()
        server.close()
        await server.wait_closed()

    assert response.status_code == 200
    assert attempts == 2
    assert endpoint.failed_until == 0.0


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
