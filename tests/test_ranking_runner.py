import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from video_commerce.common.cache_codec import json_dumps, json_loads
from video_commerce.ranking_runtime.ranking_auth import (
    RankingRuntimeAuthenticator,
    RankingRuntimeAuthSettings,
    is_loopback_peer,
)
from video_commerce.ranking_runtime.ranking_coordinator_client import (
    DRAIN_OPERATION,
    HEALTH_OPERATION,
    UNDRAIN_OPERATION,
    RankingRuntimeAuthRejected,
    decode_response,
    encode_request,
    read_frame,
)
from video_commerce.services.ranking_runner.main import RankingRunner
from video_commerce.ranking_runtime.runner_healthcheck import probe_runner


class SlowRankingModel:
    def prepare_batch_matrix(self, requests):
        matrix = np.zeros((len(requests), 1), dtype=np.float32)
        prepared = []
        for index, request in enumerate(requests):
            candidate = request["candidates"][0]
            prepared.append(
                {
                    "index": request.get("index", index),
                    "k": request.get("k", 1),
                    "batch_wait_ms": request.get("batch_wait_ms", 0.0),
                    "valid_candidates": [
                        (
                            candidate,
                            {
                                "title": candidate.product_id,
                                "price": 1.0,
                                "category": "test",
                                "brand": "test",
                            },
                        )
                    ],
                    "row_start": index,
                    "row_end": index + 1,
                    "candidate_count": 1,
                    "feature_extraction_ms": 0.0,
                }
            )
        return matrix, prepared, 0.0

    def run_inference_batch(self, feature_matrix):
        time.sleep(0.1)
        scores = np.ones(feature_matrix.shape[0], dtype=np.float32)
        return (
            {
                "ctr": scores,
                "cvr": scores,
                "gmv": scores,
                "ranking_score": scores,
            },
            {"tensor_prep_ms": 0.0, "model_forward_ms": 0.0},
        )

    def build_recommendations_from_predictions(self, valid_candidates, predictions, k):
        recommendations = []
        for candidate, metadata in valid_candidates[:k]:
            recommendations.append(
                {
                    "product_id": candidate.product_id,
                    "title": metadata["title"],
                    "price": metadata["price"],
                    "category": metadata["category"],
                    "brand": metadata["brand"],
                    "confidence_score": 1.0,
                    "ranking_score": 1.0,
                }
            )
        return recommendations, 0.0


def _request_body(product_id):
    return json_dumps(
        {
            "requests": [
                {
                    "index": 0,
                    "candidates": [
                        {
                            "product_id": product_id,
                            "combined_score": 1.0,
                            "source": "test",
                        }
                    ],
                    "user_features": {"user_id": "u1"},
                    "context": {},
                    "product_metadata_map": {},
                    "k": 1,
                    "batch_wait_ms": 0.0,
                }
            ]
        }
    )


def _decode_runner_response(frame):
    response = decode_response(frame[4:])
    payload = json_loads(response.body)
    return response.status_code, payload


class HealthyRankingModel:
    def health_check(self):
        return {"status": "healthy"}


class UnreadyRankingModel:
    def health_check(self):
        return {"status": "unhealthy", "failure_reason": "artifact_unverified"}


class DegradedRankingModel:
    def health_check(self):
        return {"status": "degraded", "failure_reason": "model_untrained"}


@pytest.mark.asyncio
async def test_runner_health_exposes_batch_payload_capabilities():
    runner = RankingRunner()
    runner.ranking_model = HealthyRankingModel()

    status, payload = _decode_runner_response(await runner._handle_health())

    assert status == 200
    assert payload["batch_payload_versions"] == [1, 2, 3, 4]
    assert payload["capabilities"]["batch_payload_versions"] == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_runner_rejects_inference_when_model_is_not_ready():
    runner = RankingRunner()
    runner.ranking_model = UnreadyRankingModel()
    runner._executor = object()
    runner._batch_queue = asyncio.Queue(maxsize=1)

    status, payload = _decode_runner_response(
        await asyncio.wait_for(
            runner._handle_batch_rank(_request_body("p1")), timeout=0.1
        )
    )

    assert status == 503
    assert payload["detail"] == "ranking_model_not_ready"


@pytest.mark.asyncio
async def test_runner_local_fallback_health_is_degraded_but_reachable():
    runner = RankingRunner()
    runner.ranking_model = DegradedRankingModel()

    status, payload = _decode_runner_response(await runner._handle_health())

    assert status == 200
    assert payload["status"] == "degraded"


@pytest.mark.asyncio
async def test_runner_queue_accepts_busy_batch_without_immediate_503():
    runner = RankingRunner()
    runner.ranking_model = SlowRankingModel()
    runner._executor = ThreadPoolExecutor(max_workers=1)
    runner._batch_queue = asyncio.Queue(maxsize=1)
    runner._batch_workers = [
        asyncio.create_task(runner._run_batch_worker(0)),
    ]

    try:
        first = asyncio.create_task(runner._handle_batch_rank(_request_body("p1")))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(runner._handle_batch_rank(_request_body("p2")))
        first_status, _ = _decode_runner_response(await first)
        second_status, _ = _decode_runner_response(await second)
    finally:
        await runner._batch_queue.put(None)
        await asyncio.gather(*runner._batch_workers, return_exceptions=True)
        runner._executor.shutdown(wait=True, cancel_futures=False)

    assert first_status == 200
    assert second_status == 200


@pytest.mark.asyncio
async def test_runner_queue_full_returns_429_overloaded():
    runner = RankingRunner()
    runner.ranking_model = SlowRankingModel()
    runner._executor = ThreadPoolExecutor(max_workers=1)
    runner._batch_queue = asyncio.Queue(maxsize=1)
    runner._batch_workers = [
        asyncio.create_task(runner._run_batch_worker(0)),
    ]

    try:
        first = asyncio.create_task(runner._handle_batch_rank(_request_body("p1")))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(runner._handle_batch_rank(_request_body("p2")))
        await asyncio.sleep(0.01)
        third_status, third_payload = _decode_runner_response(
            await runner._handle_batch_rank(_request_body("p3"))
        )
        await asyncio.gather(first, second)
    finally:
        await runner._batch_queue.put(None)
        await asyncio.gather(*runner._batch_workers, return_exceptions=True)
        runner._executor.shutdown(wait=True, cancel_futures=False)

    assert third_status == 429
    assert third_payload["detail"] == "ranking_runner_overloaded"


@pytest.mark.asyncio
async def test_runner_queue_size_zero_rejects_when_worker_active():
    runner = RankingRunner()
    runner.ranking_model = SlowRankingModel()
    runner._executor = ThreadPoolExecutor(max_workers=1)
    runner._batch_queue = asyncio.Queue(maxsize=1)
    runner._runner_batch_concurrency = 1
    runner._runner_queue_size = 0
    runner._batch_workers = [
        asyncio.create_task(runner._run_batch_worker(0)),
    ]

    try:
        first = asyncio.create_task(runner._handle_batch_rank(_request_body("p1")))
        await asyncio.sleep(0.01)
        second_status, second_payload = _decode_runner_response(
            await runner._handle_batch_rank(_request_body("p2"))
        )
        first_status, _ = _decode_runner_response(await first)
    finally:
        await runner._batch_queue.put(None)
        await asyncio.gather(*runner._batch_workers, return_exceptions=True)
        runner._executor.shutdown(wait=True, cancel_futures=False)

    assert first_status == 200
    assert second_status == 429
    assert second_payload["detail"] == "ranking_runner_overloaded"


@pytest.mark.asyncio
async def test_runner_drain_rejects_new_batches_and_reports_not_ready():
    runner = RankingRunner()
    runner.ranking_model = HealthyRankingModel()
    runner._batch_queue = asyncio.Queue(maxsize=1)

    drain_status, drain_payload = _decode_runner_response(
        await runner._handle_frame(DRAIN_OPERATION, peer=("127.0.0.1", 51000))
    )
    batch_status, batch_payload = _decode_runner_response(
        await runner._handle_batch_rank(_request_body("p1"))
    )
    health_status, health_payload = _decode_runner_response(
        await runner._handle_health()
    )

    assert drain_status == 200
    assert drain_payload["status"] == "drained"
    assert batch_status == 429
    assert batch_payload["detail"] == "ranking_runner_draining"
    assert health_status == 503
    assert health_payload["status"] == "draining"


@pytest.mark.asyncio
async def test_runner_drain_waits_for_in_progress_admission_section():
    runner = RankingRunner()
    runner._batch_queue = asyncio.Queue(maxsize=1)
    await runner._admission_lock.acquire()

    drain = asyncio.create_task(runner._handle_drain())
    await asyncio.sleep(0)

    assert runner._draining is True
    assert drain.done() is False

    runner._admission_lock.release()
    status, payload = _decode_runner_response(await drain)

    assert status == 200
    assert payload["status"] == "drained"


@pytest.mark.asyncio
async def test_protocol_healthcheck_validates_ready_payload():
    runner = RankingRunner()
    runner.ranking_model = HealthyRankingModel()

    async def handle_client(reader, writer):
        frame = await read_frame(reader)
        writer.write(await runner._handle_frame(frame))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    from video_commerce.ranking_runtime.ranking_coordinator_client import read_frame

    server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    try:
        payload = await probe_runner(
            host="127.0.0.1",
            port=port,
            operation="health",
            timeout_seconds=1.0,
        )
    finally:
        server.close()
        await server.wait_closed()

    assert payload["status"] == "ready"


@pytest.mark.parametrize(
    "peer",
    [None, ("10.1.2.3", 51000), ("192.168.0.9", 51000, 0, 0), "/tmp/sock"],
)
@pytest.mark.parametrize("operation", [DRAIN_OPERATION, UNDRAIN_OPERATION])
@pytest.mark.asyncio
async def test_runner_lifecycle_operations_reject_non_loopback_peers(peer, operation):
    runner = RankingRunner()
    runner.ranking_model = HealthyRankingModel()
    runner._batch_queue = asyncio.Queue(maxsize=1)
    runner._draining = operation == UNDRAIN_OPERATION

    status, payload = _decode_runner_response(
        await runner._handle_frame(operation, peer=peer)
    )

    assert status == 403
    assert payload["detail"] == "ranking_runner_lifecycle_loopback_only"
    assert runner._draining is (operation == UNDRAIN_OPERATION)


def test_loopback_peer_detection_covers_ipv4_ipv6_and_mapped():
    assert is_loopback_peer(("127.0.0.1", 1))
    assert is_loopback_peer(("127.10.0.1", 1))
    assert is_loopback_peer(("::1", 1, 0, 0))
    assert is_loopback_peer(("::ffff:127.0.0.1", 1, 0, 0))
    assert not is_loopback_peer(("::ffff:10.0.0.1", 1, 0, 0))
    assert not is_loopback_peer(("0.0.0.0", 1))
    assert not is_loopback_peer(("not-an-ip", 1))


@pytest.mark.asyncio
async def test_runner_undrain_restores_admission_and_readiness():
    runner = RankingRunner()
    runner.ranking_model = HealthyRankingModel()
    runner._batch_queue = asyncio.Queue(maxsize=1)
    loopback = ("127.0.0.1", 51000)

    await runner._handle_frame(DRAIN_OPERATION, peer=loopback)
    undrain_status, undrain_payload = _decode_runner_response(
        await runner._handle_frame(UNDRAIN_OPERATION, peer=loopback)
    )
    health_status, health_payload = _decode_runner_response(
        await runner._handle_health()
    )

    assert undrain_status == 200
    assert undrain_payload["status"] == "undrained"
    assert runner._draining is False
    assert health_status == 200
    assert health_payload["status"] == "ready"


async def _close_server(server):
    server.close()
    await server.wait_closed()
    # Let per-connection handlers observe EOF before the loop shuts down.
    await asyncio.sleep(0.05)


async def _serve_runner(runner):
    server = await asyncio.start_server(
        runner._handle_client, host="127.0.0.1", port=0
    )
    return server, server.sockets[0].getsockname()[1]


@pytest.mark.asyncio
async def test_healthcheck_drain_and_undrain_over_loopback_with_required_auth():
    runner = RankingRunner()
    runner.ranking_model = HealthyRankingModel()
    runner._batch_queue = asyncio.Queue(maxsize=1)
    settings = RankingRuntimeAuthSettings.build("required", "runner-test-key")
    runner.authenticator = RankingRuntimeAuthenticator(settings, role="runner")
    server, port = await _serve_runner(runner)
    try:
        drained = await probe_runner(
            host="127.0.0.1",
            port=port,
            operation="drain",
            timeout_seconds=1.0,
            auth_settings=settings,
        )
        draining_flag = runner._draining
        undrained = await probe_runner(
            host="127.0.0.1",
            port=port,
            operation="undrain",
            timeout_seconds=1.0,
            auth_settings=settings,
        )
        healthy = await probe_runner(
            host="127.0.0.1",
            port=port,
            operation="health",
            timeout_seconds=1.0,
            auth_settings=settings,
        )
    finally:
        await _close_server(server)

    assert drained["status"] == "drained"
    assert draining_flag is True
    assert undrained["status"] == "undrained"
    assert healthy["status"] == "ready"
    assert runner._draining is False


@pytest.mark.asyncio
async def test_required_runner_rejects_unauthenticated_probe_and_frames():
    runner = RankingRunner()
    runner.ranking_model = HealthyRankingModel()
    runner._batch_queue = asyncio.Queue(maxsize=1)
    runner.authenticator = RankingRuntimeAuthenticator(
        RankingRuntimeAuthSettings.build("required", "runner-test-key"),
        role="runner",
    )
    server, port = await _serve_runner(runner)
    try:
        with pytest.raises(RuntimeError, match="status=401"):
            await probe_runner(
                host="127.0.0.1",
                port=port,
                operation="drain",
                timeout_seconds=1.0,
                auth_settings=RankingRuntimeAuthSettings(),
            )
        with pytest.raises(RankingRuntimeAuthRejected):
            await probe_runner(
                host="127.0.0.1",
                port=port,
                operation="health",
                timeout_seconds=1.0,
                auth_settings=RankingRuntimeAuthSettings.build(
                    "required", "wrong-key"
                ),
            )
    finally:
        await _close_server(server)

    assert runner._draining is False
