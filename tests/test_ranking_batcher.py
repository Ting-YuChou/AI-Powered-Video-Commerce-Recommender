import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from cache_codec import json_dumps, json_loads
from models import CandidateProduct, ProductRecommendation, UserFeatures
from ranking_batcher import (
    RankingBatcher,
    RankingQueueTimeoutError,
    normalize_ranking_batch_payloads,
    run_ranking_batch_payloads,
)
from ranking_coordinator_client import RankingCoordinatorResponse
from ranking_runner_client import RankingRunnerTimeout


class FakeRankingModel:
    def __init__(self):
        self.prepare_calls = 0
        self.batch_sizes = []

    def prepare_request_matrix(
        self,
        candidates,
        user_features,
        context,
        product_metadata_map=None,
    ):
        self.prepare_calls += 1
        valid_candidates = [
            (
                candidate,
                {
                    "title": candidate.product_id,
                    "price": 1.0,
                    "category": "test",
                    "brand": "test",
                },
            )
            for candidate in candidates
        ]
        return np.zeros((len(candidates), 1), dtype=np.float32), valid_candidates, 0.0

    def prepare_batch_matrix(self, requests):
        self.prepare_calls += 1
        total_candidates = sum(len(request["candidates"]) for request in requests)
        matrix = np.zeros((total_candidates, 1), dtype=np.float32)
        prepared = []
        row_offset = 0
        for fallback_index, request in enumerate(requests):
            candidates = request["candidates"]
            product_metadata_map = request.get("product_metadata_map") or {}
            valid_candidates = [
                (
                    candidate,
                    {
                        "title": product_metadata_map.get(candidate.product_id, {}).get(
                            "title", candidate.product_id
                        ),
                        "price": product_metadata_map.get(candidate.product_id, {}).get(
                            "price", 1.0
                        ),
                        "category": product_metadata_map.get(
                            candidate.product_id, {}
                        ).get("category", "test"),
                        "brand": product_metadata_map.get(candidate.product_id, {}).get(
                            "brand", "test"
                        ),
                    },
                )
                for candidate in candidates
            ]
            prepared.append(
                {
                    "index": request.get("index", fallback_index),
                    "k": request.get("k"),
                    "batch_wait_ms": request.get("batch_wait_ms", 0.0),
                    "valid_candidates": valid_candidates,
                    "row_start": row_offset,
                    "row_end": row_offset + len(candidates),
                    "candidate_count": len(candidates),
                    "feature_extraction_ms": 0.0,
                }
            )
            row_offset += len(candidates)
        return matrix, prepared, 0.0

    def run_inference_batch(self, feature_matrix):
        self.batch_sizes.append(feature_matrix.shape[0])
        count = feature_matrix.shape[0]
        values = np.linspace(1.0, 0.1, count, dtype=np.float32)
        return (
            {
                "ctr": values,
                "cvr": values,
                "gmv": values,
                "ranking_score": values,
            },
            {"tensor_prep_ms": 0.0, "model_forward_ms": 0.0},
        )

    def build_recommendations_from_predictions(self, valid_candidates, predictions, k):
        recommendations = []
        for candidate, metadata in valid_candidates[:k]:
            recommendations.append(
                ProductRecommendation(
                    product_id=candidate.product_id,
                    title=metadata["title"],
                    price=metadata["price"],
                    category=metadata["category"],
                    brand=metadata["brand"],
                    confidence_score=0.5,
                    ranking_score=0.5,
                )
            )
        return recommendations, 0.0


def _config(**overrides):
    defaults = {
        "enable_async_batching": True,
        "batch_max_requests": 16,
        "batch_target_requests": 16,
        "batch_wait_ms": 25.0,
        "batch_queue_size": 128,
        "batch_runner_count": 4,
        "coordinator_dispatch_concurrency": 4,
        "inference_executor_workers": 0,
        "inference_process_workers": 0,
        "offload_inference_to_thread": False,
        "max_queue_wait_ms": 10_000.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_global_dispatcher_drains_full_batches_before_dispatch():
    ranking = FakeRankingModel()
    batcher = RankingBatcher(ranking, _config())
    user_features = UserFeatures(user_id="u1")

    tasks = [
        asyncio.create_task(
            batcher.rank_candidates(
                [
                    CandidateProduct(
                        product_id=f"p{i}", combined_score=1.0, source="test"
                    )
                ],
                user_features,
                {},
                k=1,
                include_profile=True,
            )
        )
        for i in range(20)
    ]
    await asyncio.sleep(0)
    await batcher.start()
    try:
        results = await asyncio.gather(*tasks)
    finally:
        await batcher.close()

    batch_counts = [profile["batch_request_count"] for _, profile in results]
    assert max(batch_counts) == 16
    assert max(ranking.batch_sizes) == 16


@pytest.mark.asyncio
async def test_cancelled_requests_are_skipped_before_feature_prep():
    ranking = FakeRankingModel()
    batcher = RankingBatcher(ranking, _config())
    user_features = UserFeatures(user_id="u1")

    task = asyncio.create_task(
        batcher.rank_candidates(
            [CandidateProduct(product_id="p1", combined_score=1.0, source="test")],
            user_features,
            {},
            k=1,
            include_profile=True,
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await batcher.start()
    await asyncio.sleep(0.05)
    await batcher.close()

    assert ranking.prepare_calls == 0


@pytest.mark.asyncio
async def test_deadline_expired_requests_are_skipped_before_feature_prep():
    ranking = FakeRankingModel()
    batcher = RankingBatcher(
        ranking,
        _config(batch_wait_ms=50.0, max_queue_wait_ms=1000.0),
    )
    user_features = UserFeatures(user_id="u1")

    task = asyncio.create_task(
        batcher.rank_candidates(
            [CandidateProduct(product_id="p1", combined_score=1.0, source="test")],
            user_features,
            {},
            k=1,
            include_profile=True,
            deadline_unix_seconds=0.0,
        )
    )

    await batcher.start()
    try:
        with pytest.raises(RankingQueueTimeoutError):
            await task
    finally:
        await batcher.close()

    assert ranking.prepare_calls == 0


@pytest.mark.asyncio
async def test_synthetic_1000_enqueue_forms_large_batches():
    ranking = FakeRankingModel()
    batcher = RankingBatcher(
        ranking,
        _config(
            batch_max_requests=64,
            batch_target_requests=32,
            batch_wait_ms=64.0,
            batch_queue_size=2048,
            batch_runner_count=8,
            coordinator_dispatch_concurrency=8,
        ),
    )
    user_features = UserFeatures(user_id="u1")

    await batcher.start()
    try:
        tasks = [
            asyncio.create_task(
                batcher.rank_candidates(
                    [
                        CandidateProduct(
                            product_id=f"p{i}", combined_score=1.0, source="test"
                        )
                    ],
                    user_features,
                    {},
                    k=1,
                    include_profile=True,
                )
            )
            for i in range(1000)
        ]
        results = await asyncio.gather(*tasks)
    finally:
        await batcher.close()

    batch_counts = [profile["batch_request_count"] for _, profile in results]
    assert sum(batch_counts) / len(batch_counts) >= 24
    assert max(batch_counts) >= 32


@pytest.mark.asyncio
async def test_estimated_queue_wait_rejects_before_unbounded_backlog():
    ranking = FakeRankingModel()
    batcher = RankingBatcher(
        ranking,
        _config(
            batch_max_requests=10,
            batch_wait_ms=100.0,
            batch_queue_size=100,
            batch_runner_count=1,
            max_queue_wait_ms=50.0,
        ),
    )
    for index in range(10):
        batcher.queue.put_nowait(
            SimpleNamespace(
                candidates=[],
                user_features=UserFeatures(user_id=f"u{index}"),
                context={},
                product_metadata_map={},
                k=1,
                include_profile=True,
                future=asyncio.get_running_loop().create_future(),
                enqueued_at=0.0,
            )
        )

    with pytest.raises(RankingQueueTimeoutError):
        await batcher.rank_candidates(
            [CandidateProduct(product_id="p1", combined_score=1.0, source="test")],
            UserFeatures(user_id="u-new"),
            {},
            k=1,
            include_profile=True,
        )


def test_admission_uses_runner_service_time_not_dispatch_wait_feedback():
    runner_pool = SimpleNamespace(
        capacity=1,
        has_available_endpoint=lambda: True,
    )
    batcher = RankingBatcher(
        None,
        _config(
            batch_max_requests=64,
            batch_target_requests=32,
            batch_wait_ms=1.0,
            batch_queue_size=2048,
            batch_runner_count=1,
            coordinator_dispatch_concurrency=1,
            max_queue_wait_ms=100.0,
        ),
        runner_pool=runner_pool,
    )
    batcher._active_batch_count = 1
    batcher._batch_execution_ewma_seconds = 10.0
    batcher._runner_service_ewma_seconds = 0.001
    batcher._dispatch_wait_ewma_seconds = 10.0

    assert batcher._estimated_queue_wait_exceeded() is False


class FakeRunnerPool:
    capacity = 2

    def __init__(self, supported_payload_versions=(1, 2)):
        self.bodies = []
        self.supported_payload_versions = set(supported_payload_versions)

    def has_available_endpoint(self):
        return True

    def supports_batch_payload_version(self, version):
        return int(version) in self.supported_payload_versions

    async def rank_batch(self, body, *, timeout_seconds=None):
        payload = json_loads(body)
        self.bodies.append(payload)
        results = []
        for item in payload["requests"]:
            candidates = item["candidates"]
            product_id = candidates[0]["product_id"]
            results.append(
                (
                    item["index"],
                    [
                        {
                            "product_id": product_id,
                            "title": product_id,
                            "price": 1.0,
                            "category": "test",
                            "brand": "test",
                            "confidence_score": 0.5,
                            "ranking_score": 0.5,
                        }
                    ],
                    {
                        "path": "torch_microbatch_runner",
                        "batch_request_count": len(payload["requests"]),
                        "batch_candidate_count": len(payload["requests"]),
                        "batch_wait_ms": item["batch_wait_ms"],
                        "feature_extraction_ms": 0.0,
                        "tensor_prep_ms": 0.0,
                        "model_forward_ms": 0.0,
                        "response_build_ms": 0.0,
                        "total_ms": 0.0,
                        "candidate_count": len(candidates),
                        "ranked_count": 1,
                    },
                )
            )
        return RankingCoordinatorResponse(
            200,
            "application/json",
            json_dumps(
                {
                    "results": results,
                    "stages": {"total_execution": 0.001},
                    "runner_process_id": 1234,
                }
            ),
        )


@pytest.mark.asyncio
async def test_remote_runner_batch_executes_without_local_model():
    runner_pool = FakeRunnerPool()
    batcher = RankingBatcher(
        None,
        _config(batch_max_requests=4, batch_wait_ms=5.0),
        runner_pool=runner_pool,
    )
    user_features = UserFeatures(user_id="u1")

    await batcher.start()
    try:
        tasks = [
            asyncio.create_task(
                batcher.rank_candidates(
                    [
                        CandidateProduct(
                            product_id=f"p{i}", combined_score=1.0, source="test"
                        )
                    ],
                    user_features,
                    {},
                    k=1,
                    include_profile=True,
                )
            )
            for i in range(4)
        ]
        results = await asyncio.gather(*tasks)
    finally:
        await batcher.close()

    assert runner_pool.bodies[0]["requests"]
    assert all(
        recommendations[0].product_id == f"p{i}"
        for i, (recommendations, _) in enumerate(results)
    )
    assert all(profile["ranking_runner_process_id"] == 1234 for _, profile in results)


@pytest.mark.asyncio
async def test_remote_runner_payload_v2_deduplicates_and_trims_metadata():
    runner_pool = FakeRunnerPool()
    batcher = RankingBatcher(
        None,
        _config(batch_max_requests=2, batch_wait_ms=0.0),
        runner_pool=runner_pool,
    )
    user_features = UserFeatures(user_id="u1")
    metadata = {
        "p1": {
            "title": "Product p1",
            "price": 1.0,
            "category": "test",
            "brand": "brand",
            "rating": 4.5,
            "num_reviews": 10,
            "in_stock": True,
            "created_at": 123.0,
            "tags": ["tag"],
            "description": "large field that should not be shipped",
            "image_url": "https://example.invalid/image.jpg",
        }
    }

    await batcher.start()
    try:
        await batcher.rank_candidates(
            [CandidateProduct(product_id="p1", combined_score=1.0, source="test")],
            user_features,
            {},
            product_metadata_map=metadata,
            k=1,
            include_profile=True,
        )
    finally:
        await batcher.close()

    body = runner_pool.bodies[0]
    assert body["payload_version"] == 2
    assert body["requests"][0]["product_ids"] == ["p1"]
    assert "product_metadata_map" not in body["requests"][0]
    assert body["product_metadata_map"]["p1"] == {
        "title": "Product p1",
        "price": 1.0,
        "category": "test",
        "brand": "brand",
        "rating": 4.5,
        "num_reviews": 10,
        "in_stock": True,
        "created_at": 123.0,
        "tags": ["tag"],
    }


@pytest.mark.asyncio
async def test_remote_runner_payload_v2_falls_back_to_v1_without_capability():
    runner_pool = FakeRunnerPool(supported_payload_versions=(1,))
    batcher = RankingBatcher(
        None,
        _config(batch_max_requests=1, batch_wait_ms=0.0),
        runner_pool=runner_pool,
    )

    await batcher.start()
    try:
        await batcher.rank_candidates(
            [CandidateProduct(product_id="p1", combined_score=1.0, source="test")],
            UserFeatures(user_id="u1"),
            {},
            product_metadata_map={
                "p1": {
                    "title": "Product p1",
                    "price": 1.0,
                    "category": "test",
                    "brand": "brand",
                }
            },
            k=1,
            include_profile=True,
        )
    finally:
        await batcher.close()

    body = runner_pool.bodies[0]
    assert body["payload_version"] == 1
    assert body["requests"][0]["product_metadata_map"]["p1"]["title"] == "Product p1"
    assert "product_ids" not in body["requests"][0]


def test_runner_batch_result_validation_rejects_missing_duplicate_and_bad_indexes():
    batcher = RankingBatcher(
        None,
        _config(batch_max_requests=2, batch_wait_ms=0.0),
        runner_pool=FakeRunnerPool(),
    )
    batch = [SimpleNamespace(), SimpleNamespace()]

    with pytest.raises(RankingQueueTimeoutError, match="incomplete_response"):
        batcher._decode_batch_payload_result(batch, {"results": [(0, [], {})]})

    with pytest.raises(RankingQueueTimeoutError, match="incomplete_response"):
        batcher._decode_batch_payload_result(
            batch,
            {"results": [(0, [], {}), (0, [], {})]},
        )

    with pytest.raises(RankingQueueTimeoutError, match="invalid_response"):
        batcher._decode_batch_payload_result(
            batch,
            {"results": [(0, [], {}), (2, [], {})]},
        )

    with pytest.raises(RankingQueueTimeoutError, match="invalid_response"):
        batcher._decode_batch_payload_result(batch, {"results": ["malformed"]})


class PartialRunnerPool(FakeRunnerPool):
    async def rank_batch(self, body, *, timeout_seconds=None):
        payload = json_loads(body)
        self.bodies.append(payload)
        return RankingCoordinatorResponse(
            200,
            "application/json",
            json_dumps(
                {
                    "results": [
                        (
                            0,
                            [
                                {
                                    "product_id": "p0",
                                    "title": "p0",
                                    "price": 1.0,
                                    "category": "test",
                                    "brand": "test",
                                    "confidence_score": 0.5,
                                    "ranking_score": 0.5,
                                }
                            ],
                            {"path": "partial"},
                        )
                    ]
                }
            ),
        )


@pytest.mark.asyncio
async def test_remote_runner_partial_response_fails_whole_batch_without_pending():
    runner_pool = PartialRunnerPool()
    batcher = RankingBatcher(
        None,
        _config(batch_max_requests=2, batch_target_requests=2, batch_wait_ms=5.0),
        runner_pool=runner_pool,
    )
    user_features = UserFeatures(user_id="u1")

    await batcher.start()
    try:
        tasks = [
            asyncio.create_task(
                batcher.rank_candidates(
                    [
                        CandidateProduct(
                            product_id=f"p{i}", combined_score=1.0, source="test"
                        )
                    ],
                    user_features,
                    {},
                    k=1,
                    include_profile=True,
                )
            )
            for i in range(2)
        ]
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=1.0,
        )
    finally:
        await batcher.close()

    assert all(isinstance(result, RankingQueueTimeoutError) for result in results)
    assert all("incomplete_response" in str(result) for result in results)


def test_ranking_runner_payload_v2_round_trip_and_v1_fallback():
    v2_payload = {
        "payload_version": 2,
        "product_metadata_map": {
            "p1": {
                "title": "V2 title",
                "price": 2.0,
                "category": "test",
                "brand": "brand",
            }
        },
        "requests": [
            {
                "index": 0,
                "candidates": [
                    {
                        "product_id": "p1",
                        "combined_score": 1.0,
                        "source": "test",
                    }
                ],
                "product_ids": ["p1"],
                "user_features": {"user_id": "u1"},
                "context": {},
                "k": 1,
                "batch_wait_ms": 0.0,
            }
        ],
    }
    normalized = normalize_ranking_batch_payloads(v2_payload)
    assert normalized[0]["product_metadata_map"]["p1"]["title"] == "V2 title"

    result = run_ranking_batch_payloads(FakeRankingModel(), v2_payload)
    assert result["results"][0][1][0]["title"] == "V2 title"

    v1_payload = {"requests": normalized}
    assert normalize_ranking_batch_payloads(v1_payload)[0]["product_metadata_map"][
        "p1"
    ]["title"] == "V2 title"


@pytest.mark.asyncio
async def test_remote_runner_payload_size_guard_fails_before_send():
    runner_pool = FakeRunnerPool()
    batcher = RankingBatcher(
        None,
        _config(
            batch_max_requests=1,
            batch_wait_ms=0.0,
            runner_payload_max_bytes=1024,
        ),
        runner_pool=runner_pool,
    )

    await batcher.start()
    try:
        with pytest.raises(RankingQueueTimeoutError, match="payload_too_large"):
            await batcher.rank_candidates(
                [
                    CandidateProduct(
                        product_id="p1",
                        combined_score=1.0,
                        source="test",
                    )
                ],
                UserFeatures(user_id="u1"),
                {},
                product_metadata_map={
                    "p1": {
                        "title": "x" * 2048,
                        "price": 1.0,
                        "category": "test",
                        "brand": "brand",
                    }
                },
                k=1,
                include_profile=True,
            )
    finally:
        await batcher.close()

    assert runner_pool.bodies == []


class FailingRunnerPool:
    capacity = 1

    def has_available_endpoint(self):
        return True

    async def rank_batch(self, body, *, timeout_seconds=None):
        raise RuntimeError("runner down")


class TimeoutRunnerPool:
    capacity = 1

    def has_available_endpoint(self):
        return True

    async def rank_batch(self, body, *, timeout_seconds=None):
        raise RankingRunnerTimeout("ranking runner request timeout: runner-1")


class StageObservability:
    def __init__(self):
        self.stages = []

    def record_ranking_batch_stage(self, path, stage, duration_seconds):
        self.stages.append((path, stage, duration_seconds))


@pytest.mark.asyncio
async def test_remote_runner_failure_does_not_fallback_to_local_ranker():
    batcher = RankingBatcher(
        FakeRankingModel(),
        _config(batch_max_requests=1, batch_wait_ms=0.0),
        runner_pool=FailingRunnerPool(),
    )

    await batcher.start()
    try:
        with pytest.raises(RankingQueueTimeoutError):
            await batcher.rank_candidates(
                [CandidateProduct(product_id="p1", combined_score=1.0, source="test")],
                UserFeatures(user_id="u1"),
                {},
                k=1,
                include_profile=True,
            )
    finally:
        await batcher.close()


@pytest.mark.asyncio
async def test_remote_runner_timeout_records_specific_failure_stage():
    observability = StageObservability()
    batcher = RankingBatcher(
        FakeRankingModel(),
        _config(batch_max_requests=1, batch_wait_ms=0.0),
        runner_pool=TimeoutRunnerPool(),
        observability=observability,
    )

    await batcher.start()
    try:
        with pytest.raises(RankingQueueTimeoutError):
            await batcher.rank_candidates(
                [CandidateProduct(product_id="p1", combined_score=1.0, source="test")],
                UserFeatures(user_id="u1"),
                {},
                k=1,
                include_profile=True,
            )
    finally:
        await batcher.close()

    assert any(
        path == "microbatch_runner" and stage == "remote_timeout"
        for path, stage, _duration in observability.stages
    )
