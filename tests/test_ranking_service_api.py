import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

import ranking_service_api
from models import CandidateProduct, ProductRecommendation, UserFeatures
from ranking_batcher import RankingBatcher
from ranking_coordinator_client import RankingCoordinatorResponse
from ranking_payloads import RankRequest, coerce_rank_payload


class FakeRankingModel:
    def prepare_batch_matrix(self, requests):
        total_candidates = sum(len(request["candidates"]) for request in requests)
        matrix = np.zeros((total_candidates, 1), dtype=np.float32)
        prepared = []
        row_offset = 0
        for fallback_index, request in enumerate(requests):
            feature_matrix, valid_candidates, feature_ms = self.prepare_request_matrix(
                request["candidates"],
                request["user_features"],
                request.get("context") or {},
                request.get("product_metadata_map") or {},
            )
            candidate_count = feature_matrix.shape[0]
            prepared.append(
                {
                    "index": request.get("index", fallback_index),
                    "k": request.get("k"),
                    "batch_wait_ms": request.get("batch_wait_ms", 0.0),
                    "valid_candidates": valid_candidates,
                    "row_start": row_offset,
                    "row_end": row_offset + candidate_count,
                    "candidate_count": candidate_count,
                    "feature_extraction_ms": feature_ms,
                }
            )
            row_offset += candidate_count
        return matrix, prepared, 0.0

    def prepare_request_matrix(
        self,
        candidates,
        user_features,
        context,
        product_metadata_map=None,
    ):
        valid_candidates = []
        for candidate in candidates:
            metadata = (product_metadata_map or {}).get(candidate.product_id, {})
            valid_candidates.append(
                (
                    candidate,
                    {
                        "title": metadata.get("title", candidate.product_id),
                        "price": metadata.get("price", 1.0),
                        "category": metadata.get("category", "test"),
                        "brand": metadata.get("brand", "test"),
                    },
                )
            )
        return np.zeros((len(candidates), 1), dtype=np.float32), valid_candidates, 0.0

    def run_inference_batch(self, feature_matrix):
        scores = np.linspace(0.9, 0.1, feature_matrix.shape[0], dtype=np.float32)
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
        for index, (candidate, metadata) in enumerate(valid_candidates[:k]):
            score = float(predictions["ranking_score"][index])
            recommendations.append(
                ProductRecommendation(
                    product_id=candidate.product_id,
                    title=metadata["title"],
                    price=metadata["price"],
                    category=metadata["category"],
                    brand=metadata["brand"],
                    confidence_score=score,
                    ranking_score=score,
                )
            )
        return recommendations, 0.0

    def _rank_candidates_sync(
        self,
        candidates,
        user_features,
        context,
        k,
        include_profile,
        product_metadata_map=None,
    ):
        feature_matrix, valid_candidates, feature_ms = self.prepare_request_matrix(
            candidates,
            user_features,
            context,
            product_metadata_map,
        )
        predictions, inference_profile = self.run_inference_batch(feature_matrix)
        (
            recommendations,
            response_build_ms,
        ) = self.build_recommendations_from_predictions(
            valid_candidates,
            predictions,
            k,
        )
        profile = {
            "path": "fake_direct",
            "feature_extraction_ms": feature_ms,
            "tensor_prep_ms": inference_profile.get("tensor_prep_ms", 0.0),
            "model_forward_ms": inference_profile.get("model_forward_ms", 0.0),
            "response_build_ms": response_build_ms,
            "candidate_count": len(candidates),
            "ranked_count": len(recommendations),
        }
        if include_profile:
            return recommendations, profile
        return recommendations


def _config():
    return SimpleNamespace(
        enable_async_batching=True,
        batch_max_requests=1,
        batch_wait_ms=1.0,
        batch_queue_size=16,
        batch_runner_count=1,
        inference_executor_workers=0,
        offload_inference_to_thread=False,
        max_queue_wait_ms=1000.0,
    )


def test_body_with_deadline_preserves_earlier_deadline_and_fills_missing_or_invalid(
    monkeypatch,
):
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            service_topology_config=SimpleNamespace(
                ranking_coordinator_request_timeout_seconds=1.0
            )
        )
    )
    monkeypatch.setattr(ranking_service_api.time, "time", lambda: 100.0)

    earlier = ranking_service_api._body_with_deadline(
        b'{"deadline_unix_seconds":100.2}',
        runtime,
    )
    later = ranking_service_api._body_with_deadline(
        b'{"deadline_unix_seconds":200.0}',
        runtime,
    )
    missing = ranking_service_api._body_with_deadline(b"{}", runtime)
    invalid = ranking_service_api._body_with_deadline(
        b'{"deadline_unix_seconds":"bad"}',
        runtime,
    )

    assert json.loads(earlier)["deadline_unix_seconds"] == pytest.approx(100.2)
    assert json.loads(later)["deadline_unix_seconds"] == pytest.approx(100.9)
    assert json.loads(missing)["deadline_unix_seconds"] == pytest.approx(100.9)
    assert json.loads(invalid)["deadline_unix_seconds"] == pytest.approx(100.9)


@pytest.mark.asyncio
async def test_ranking_service_internal_rank_matches_in_process_batcher(monkeypatch):
    candidates = [
        CandidateProduct(product_id="p1", combined_score=1.0, source="test"),
        CandidateProduct(product_id="p2", combined_score=0.8, source="test"),
        CandidateProduct(product_id="p3", combined_score=0.5, source="test"),
    ]
    user_features = UserFeatures(user_id="u1")
    context = {"device": "mobile"}
    product_metadata_map = {
        "p1": {
            "title": "Product 1",
            "price": 10.0,
            "category": "cat",
            "brand": "brand",
        },
        "p2": {
            "title": "Product 2",
            "price": 11.0,
            "category": "cat",
            "brand": "brand",
        },
        "p3": {
            "title": "Product 3",
            "price": 12.0,
            "category": "cat",
            "brand": "brand",
        },
    }

    expected_batcher = RankingBatcher(FakeRankingModel(), _config())
    service_batcher = RankingBatcher(FakeRankingModel(), _config())
    await expected_batcher.start()
    await service_batcher.start()
    monkeypatch.setattr(ranking_service_api, "ranking_batcher", service_batcher)

    try:
        expected_recommendations, _ = await expected_batcher.rank_candidates(
            candidates=candidates,
            user_features=user_features,
            context=context,
            product_metadata_map=product_metadata_map,
            k=2,
            include_profile=True,
        )
        request_payload = RankRequest(
            request_id="req-1",
            candidates=candidates,
            user_features=user_features,
            context=context,
            product_metadata_map=product_metadata_map,
            k=2,
        )

        async def parse_rank_request(_request):
            return coerce_rank_payload(request_payload)

        monkeypatch.setattr(
            ranking_service_api,
            "_parse_rank_request",
            parse_rank_request,
        )
        response = await ranking_service_api.rank(
            SimpleNamespace(state=SimpleNamespace(request_id="req-1"))
        )
    finally:
        await asyncio.gather(
            expected_batcher.close(),
            service_batcher.close(),
            return_exceptions=True,
        )

    body = json.loads(response.body)
    actual_recommendations = [
        ProductRecommendation(**item) for item in body["recommendations"]
    ]

    assert [item.product_id for item in actual_recommendations] == [
        item.product_id for item in expected_recommendations
    ]
    assert [item.ranking_score for item in actual_recommendations] == pytest.approx(
        [item.ranking_score for item in expected_recommendations]
    )


@pytest.mark.asyncio
async def test_ranking_service_direct_local_path_passes_request_deadline(monkeypatch):
    class CapturingBatcher:
        def __init__(self):
            self.deadline_unix_seconds = None

        async def rank_candidates(
            self,
            *,
            candidates,
            user_features,
            context,
            product_metadata_map,
            k,
            include_profile,
            deadline_unix_seconds=None,
        ):
            self.deadline_unix_seconds = deadline_unix_seconds
            return [], {"path": "fake"}

    batcher = CapturingBatcher()
    request_payload = RankRequest(
        request_id="req-deadline",
        candidates=[
            CandidateProduct(product_id="p1", combined_score=1.0, source="test")
        ],
        user_features=UserFeatures(user_id="u1"),
        context={},
        product_metadata_map={},
        k=1,
        deadline_unix_seconds=123.4,
    )

    async def parse_rank_request(_request):
        return coerce_rank_payload(request_payload)

    monkeypatch.setattr(ranking_service_api, "ranking_coordinator_client", None)
    monkeypatch.setattr(ranking_service_api, "ranking_batcher", batcher)
    monkeypatch.setattr(ranking_service_api, "_parse_rank_request", parse_rank_request)

    response = await ranking_service_api.rank(
        SimpleNamespace(state=SimpleNamespace(request_id="req-deadline"))
    )

    assert response.status_code == 200
    assert batcher.deadline_unix_seconds == pytest.approx(123.4)


@pytest.mark.asyncio
async def test_ranking_service_proxy_forwards_raw_body_to_single_coordinator(
    monkeypatch,
):
    class FakeCoordinatorClient:
        def __init__(self):
            self.body = None

        async def rank(self, body):
            self.body = body
            return RankingCoordinatorResponse(
                status_code=200,
                content_type="application/json",
                body=b'{"recommendations":[],"profile":{"path":"coordinator"}}',
            )

    fake_client = FakeCoordinatorClient()
    monkeypatch.setattr(ranking_service_api, "ranking_coordinator_client", fake_client)

    class FakeRequest:
        state = SimpleNamespace(request_id="req-proxy")

        async def body(self):
            return b'{"request_id":"req-proxy","candidates":[]}'

    response = await ranking_service_api.rank(FakeRequest())

    assert fake_client.body == b'{"request_id":"req-proxy","candidates":[]}'
    assert response.status_code == 200
    assert json.loads(response.body)["profile"]["path"] == "coordinator"
