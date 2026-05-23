import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from video_commerce.common.cache_codec import json_dumps, json_loads
from video_commerce.ranking_runtime.ranking_coordinator_client import decode_response
from video_commerce.services.ranking_runner.main import RankingRunner


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


@pytest.mark.asyncio
async def test_runner_health_exposes_batch_payload_capabilities():
    runner = RankingRunner()
    runner.ranking_model = HealthyRankingModel()

    status, payload = _decode_runner_response(await runner._handle_health())

    assert status == 200
    assert payload["batch_payload_versions"] == [1, 2]
    assert payload["capabilities"]["batch_payload_versions"] == [1, 2]


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
