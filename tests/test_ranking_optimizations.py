import asyncio
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from config import RankingConfig, RecommendationConfig
from models import CandidateProduct, RecommendationRequest, UserFeatures
import recommendation_api as recommendation_api_module
from ranking import RankingModel
import ranking as ranking_module
from recommendation_api import (
    _bounded_hot_path_read,
    _build_cache_freshness_context,
    _build_candidate_cache_context,
    _build_recommendation_cache_context,
    _catalog_serving_version_context,
    _default_user_sequence_token,
    _filter_recommendable_candidates,
    _join_or_create_recommendation_singleflight,
    _resolve_recommendation_singleflight,
    _user_feature_cache_token,
)
from recommender import TwoTowerRetrievalEngine


def _hash_context(context):
    return hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()[:16]


def test_build_recommendations_constructs_only_top_k_in_score_order():
    ranking = RankingModel(RankingConfig())
    candidates = [
        (
            CandidateProduct(product_id=f"p{i}", combined_score=0.1, source="test"),
            {
                "title": f"Product {i}",
                "price": float(i),
                "category": "cat",
                "brand": "brand",
            },
        )
        for i in range(5)
    ]
    predictions = {
        "ctr": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "cvr": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "gmv": np.array([10, 20, 30, 40, 50]),
        "ranking_score": np.array([0.2, 0.9, 0.1, 0.8, 0.3]),
    }

    recommendations, _ = ranking.build_recommendations_from_predictions(
        candidates,
        predictions,
        k=2,
    )

    assert [item.product_id for item in recommendations] == ["p1", "p3"]


def test_prepare_request_matrix_reuses_static_product_features_without_changing_values(monkeypatch):
    fixed_now = 1_700_000_000.0
    monkeypatch.setattr(ranking_module.time, "time", lambda: fixed_now)

    ranking = RankingModel(RankingConfig(product_feature_cache_size=10))
    user_features = UserFeatures(
        user_id="u1",
        total_interactions=12,
        avg_session_length=120.0,
        preferred_categories=["cat"],
        last_active=fixed_now - 60,
    )
    context = {"device": "mobile", "session_position": 3, "time_on_page": 42}
    metadata = {
        "price": 25.0,
        "rating": 4.0,
        "num_reviews": 8,
        "in_stock": True,
        "created_at": fixed_now - 86400,
        "tags": ["a", "b"],
        "brand": "brand",
    }
    candidate = CandidateProduct(
        product_id="p1",
        collaborative_score=0.2,
        content_similarity_score=0.3,
        popularity_score=0.4,
        combined_score=0.5,
        source="test",
    )

    matrix, valid_candidates, _ = ranking.prepare_request_matrix(
        [candidate, candidate],
        user_features,
        context,
        product_metadata_map={"p1": metadata},
    )
    expected = ranking.feature_extractor.create_ranking_features(
        user_features,
        metadata,
        context,
        candidate,
    )

    assert len(valid_candidates) == 2
    assert len(ranking._product_feature_cache) == 1
    np.testing.assert_allclose(matrix[0], expected)
    np.testing.assert_allclose(matrix[1], expected)


def test_user_feature_cache_token_changes_when_personalization_changes():
    base = UserFeatures(user_id="u1", total_interactions=1, last_active=100.0)
    updated = base.copy(update={"total_interactions": 2})

    assert _user_feature_cache_token(base) != _user_feature_cache_token(updated)


def test_user_feature_cache_token_is_stable_for_cold_user_defaults():
    first = UserFeatures(user_id="u1", last_active=100.0)
    second = UserFeatures(user_id="u1", last_active=200.0)

    assert _user_feature_cache_token(first) == _user_feature_cache_token(second)


def test_recommendation_cache_context_ignores_tracing_fields_but_keeps_model_inputs():
    base = RecommendationRequest(
        user_id="u1",
        k=10,
        context={
            "source": "loadtest",
            "request_index": 1,
            "device": "mobile",
            "session_position": 2,
            "time_on_page": 3,
        },
    )
    same_model_inputs = base.copy(
        update={"context": {**base.context, "request_index": 2}}
    )
    changed_model_input = base.copy(
        update={"context": {**base.context, "device": "desktop"}}
    )

    assert _build_recommendation_cache_context(base, current_time=3600) == (
        _build_recommendation_cache_context(same_model_inputs, current_time=3600)
    )
    assert _build_recommendation_cache_context(base, current_time=3600) != (
        _build_recommendation_cache_context(changed_model_input, current_time=3600)
    )


def test_cache_keys_change_when_user_sequence_token_changes():
    payload = RecommendationRequest(
        user_id="u1",
        k=10,
        context={"device": "mobile", "page": "home"},
    )
    serving_versions = {"ranking_model": "ranking-1", "catalog": {"catalog_version": 1}}
    user_feature_token = {"total_interactions": 2}
    first_sequence_token = {
        "length": 1,
        "latest_event_id": "e1",
        "latest_occurred_at": 1.0,
        "latest_product_id": "p1",
        "latest_action": "view",
    }
    second_sequence_token = {
        **first_sequence_token,
        "length": 2,
        "latest_event_id": "e2",
        "latest_occurred_at": 2.0,
        "latest_product_id": "p2",
        "latest_action": "click",
    }

    recommendation_context = _build_recommendation_cache_context(payload, current_time=3600)
    first_recommendation_key = _hash_context(
        {
            **recommendation_context,
            **_build_cache_freshness_context(
                serving_versions,
                user_feature_token,
                first_sequence_token,
            ),
        }
    )
    second_recommendation_key = _hash_context(
        {
            **recommendation_context,
            **_build_cache_freshness_context(
                serving_versions,
                user_feature_token,
                second_sequence_token,
            ),
        }
    )
    assert first_recommendation_key != second_recommendation_key

    candidate_context = _build_candidate_cache_context(payload, k_per_source=100)
    first_candidate_key = _hash_context(
        {
            **candidate_context,
            **_build_cache_freshness_context(
                serving_versions,
                user_feature_token,
                first_sequence_token,
            ),
        }
    )
    second_candidate_key = _hash_context(
        {
            **candidate_context,
            **_build_cache_freshness_context(
                serving_versions,
                user_feature_token,
                second_sequence_token,
            ),
        }
    )
    assert first_candidate_key != second_candidate_key


def test_cache_freshness_context_uses_stable_cold_sequence_token():
    context = _build_cache_freshness_context(
        {"ranking_model": "ranking-1"},
        {"total_interactions": 0},
        None,
    )

    assert context["user_sequence_token"] == _default_user_sequence_token()


@pytest.mark.asyncio
async def test_user_sequence_token_timeout_falls_back_without_failing_request():
    async def slow_read():
        await asyncio.sleep(0.05)
        return {"length": 1}

    runtime = SimpleNamespace(
        config=SimpleNamespace(
            cache_config=SimpleNamespace(hot_path_read_timeout_ms=1)
        )
    )

    result = await _bounded_hot_path_read(
        runtime,
        "user_sequence_token",
        slow_read(),
        _default_user_sequence_token(),
    )

    assert result == _default_user_sequence_token()


def test_filter_recommendable_candidates_drops_unavailable_inventory():
    candidates = [
        CandidateProduct(product_id="active", source="test"),
        CandidateProduct(product_id="oos", source="test"),
        CandidateProduct(product_id="inactive", source="test"),
        CandidateProduct(product_id="deleted", source="test"),
        CandidateProduct(product_id="unknown", source="test"),
    ]
    metadata = {
        "active": {"in_stock": True, "active": True},
        "oos": {"in_stock": False, "active": True},
        "inactive": {"in_stock": True, "active": False},
        "deleted": {"deleted": True},
    }

    filtered = _filter_recommendable_candidates(candidates, metadata)

    assert [candidate.product_id for candidate in filtered] == ["active", "unknown"]


def test_catalog_version_context_uses_vector_search_catalog_token(monkeypatch):
    class FakeVectorSearch:
        def get_catalog_version_context(self):
            return {
                "catalog_version": 123,
                "last_updated": 100,
                "product_count": 2,
            }

    monkeypatch.setattr(recommendation_api_module, "vector_search", FakeVectorSearch())

    assert _catalog_serving_version_context() == {
        "catalog_version": 123,
        "last_updated": 100,
        "product_count": 2,
    }


@pytest.mark.asyncio
async def test_recommendation_singleflight_joiner_receives_owner_result():
    key = "test-singleflight-key"
    owner_future, owns = await _join_or_create_recommendation_singleflight(key)
    joined_future, joined_owns = await _join_or_create_recommendation_singleflight(key)

    assert owns is True
    assert joined_owns is False
    assert joined_future is owner_future

    await _resolve_recommendation_singleflight(
        key,
        owner_future,
        result={"recommendations": ["p1"], "total_candidates": 1},
    )

    assert await joined_future == {"recommendations": ["p1"], "total_candidates": 1}


def test_two_tower_user_embedding_cache_key_keeps_time_and_feature_freshness():
    class FakeVectorSearch:
        embedding_dim = 128

    engine = TwoTowerRetrievalEngine(
        RecommendationConfig(user_embedding_cache_time_bucket_seconds=1.0),
        FakeVectorSearch(),
    )
    engine.model_version = "two-tower-1"
    features = {
        "total_interactions": 1,
        "last_active": 100.0,
        "preferred_categories": ["cat"],
    }

    same_bucket = engine._user_embedding_cache_key("u1", features, current_time=1000.4)
    same_bucket_again = engine._user_embedding_cache_key("u1", features, current_time=1000.8)
    next_bucket = engine._user_embedding_cache_key("u1", features, current_time=1001.1)
    changed_features = engine._user_embedding_cache_key(
        "u1",
        {**features, "total_interactions": 2},
        current_time=1000.4,
    )

    assert same_bucket == same_bucket_again
    assert same_bucket != next_bucket
    assert same_bucket != changed_features


def test_two_tower_user_embedding_cache_reuses_same_fresh_key():
    class FakeVectorSearch:
        embedding_dim = 128

    class FakeTrainer:
        def __init__(self):
            self.calls = 0

        def encode_user(self, user_id, user_features, current_time=None):
            self.calls += 1
            return np.array([self.calls, current_time], dtype=np.float32)

    engine = TwoTowerRetrievalEngine(
        RecommendationConfig(user_embedding_cache_size=10),
        FakeVectorSearch(),
    )
    trainer = FakeTrainer()
    engine.trainer = trainer
    engine.model_version = "two-tower-1"

    first = engine._get_user_embedding("u1", {"last_active": 100.0}, 1000.0)
    second = engine._get_user_embedding("u1", {"last_active": 100.0}, 1000.0)

    np.testing.assert_allclose(first, second)
    assert trainer.calls == 1
