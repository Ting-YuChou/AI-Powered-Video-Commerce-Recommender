import asyncio
import hashlib
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi import HTTPException
from starlette.responses import Response

from config import RankingConfig, RecommendationConfig
from feature_store import FeatureStore
from models import (
    CandidateProduct,
    ContentFeatures,
    ProductRecommendation,
    RecommendationRequest,
    UserFeatures,
)
from model_trainer import ModelTrainerService
import recommendation_api as recommendation_api_module
from ranking import (
    RANKING_FEATURE_SCHEMA_VERSION,
    RANKING_TRAINING_DATA_SOURCE,
    RankingModel,
)
import ranking as ranking_module
from recommendation_api import (
    _BestEffortTaskQueue,
    _await_candidate_cache_race,
    _await_recommendation_cache_race,
    _bounded_hot_path_read,
    _build_cache_freshness_context,
    _build_candidate_cache_context,
    _build_recommendation_cache_context,
    _catalog_serving_version_context,
    _count_candidate_source_tokens,
    _count_ranked_source_tokens,
    _content_feature_cache_token,
    _default_user_sequence_token,
    _filter_recommendable_candidates,
    _join_or_create_recommendation_singleflight,
    _resolve_recommendation_singleflight,
    _serving_version_context,
    _refresh_serving_version_context,
    _user_feature_cache_token,
)
from recommender import TwoTowerRetrievalEngine
from slate_diversity import select_mmr_recommendations


def _hash_context(context):
    return hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()[:16]


def _recommendation(product_id: str, ranking_score: float) -> ProductRecommendation:
    return ProductRecommendation(
        product_id=product_id,
        title=f"Product {product_id}",
        price=10.0,
        confidence_score=0.8,
        ranking_score=ranking_score,
    )


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


def test_mmr_diversifies_high_relevance_duplicate_cluster():
    recommendations = [
        _recommendation("p1", 1.0),
        _recommendation("p2", 0.95),
        _recommendation("p3", 0.9),
    ]
    embeddings = {
        "p1": np.array([1.0, 0.0], dtype=np.float32),
        "p2": np.array([1.0, 0.0], dtype=np.float32),
        "p3": np.array([0.0, 1.0], dtype=np.float32),
    }

    selected = select_mmr_recommendations(
        recommendations,
        k=2,
        embedding_lookup=embeddings.get,
        lambda_weight=0.5,
    )

    assert [item.product_id for item in selected] == ["p1", "p3"]


def test_mmr_lambda_one_preserves_relevance_order():
    recommendations = [
        _recommendation("p1", 1.0),
        _recommendation("p2", 0.8),
        _recommendation("p3", 0.7),
    ]
    embeddings = {
        "p1": np.array([1.0, 0.0], dtype=np.float32),
        "p2": np.array([1.0, 0.0], dtype=np.float32),
        "p3": np.array([0.0, 1.0], dtype=np.float32),
    }

    selected = select_mmr_recommendations(
        recommendations,
        k=3,
        embedding_lookup=embeddings.get,
        lambda_weight=1.0,
    )

    assert [item.product_id for item in selected] == ["p1", "p2", "p3"]


def test_mmr_missing_embeddings_do_not_remove_products():
    recommendations = [
        _recommendation("p1", 1.0),
        _recommendation("p2", 0.9),
        _recommendation("p3", 0.8),
    ]
    embeddings = {"p1": np.array([1.0, 0.0], dtype=np.float32)}

    selected = select_mmr_recommendations(
        recommendations,
        k=3,
        embedding_lookup=embeddings.get,
        lambda_weight=0.5,
    )

    assert [item.product_id for item in selected] == ["p1", "p2", "p3"]


def test_prepare_request_matrix_reuses_static_product_features_without_changing_values(
    monkeypatch,
):
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


def test_prepare_batch_matrix_matches_request_matrix_rows(monkeypatch):
    fixed_now = 1_700_000_000.0
    monkeypatch.setattr(ranking_module.time, "time", lambda: fixed_now)

    ranking = RankingModel(RankingConfig(product_feature_cache_size=10))
    user_features_one = UserFeatures(
        user_id="u1",
        total_interactions=12,
        avg_session_length=120.0,
        preferred_categories=["cat"],
        last_active=fixed_now - 60,
    )
    user_features_two = UserFeatures(
        user_id="u2",
        total_interactions=3,
        avg_session_length=40.0,
        preferred_categories=["other"],
        last_active=fixed_now - 3600,
    )
    context_one = {"device": "mobile", "session_position": 3, "time_on_page": 42}
    context_two = {"device": "desktop", "session_position": 1, "time_on_page": 7}
    candidates_one = [
        CandidateProduct(product_id="p1", combined_score=0.5, source="test"),
        CandidateProduct(product_id="p2", combined_score=0.4, source="test"),
    ]
    candidates_two = [
        CandidateProduct(product_id="p3", combined_score=0.3, source="test")
    ]
    metadata = {
        "p1": {"price": 25.0, "category": "cat", "brand": "brand"},
        "p2": {"price": 35.0, "category": "cat", "brand": "brand"},
        "p3": {"price": 45.0, "category": "other", "brand": "other"},
    }

    matrix_one, _, _ = ranking.prepare_request_matrix(
        candidates_one,
        user_features_one,
        context_one,
        product_metadata_map=metadata,
    )
    matrix_two, _, _ = ranking.prepare_request_matrix(
        candidates_two,
        user_features_two,
        context_two,
        product_metadata_map=metadata,
    )
    batch_matrix, prepared, _ = ranking.prepare_batch_matrix(
        [
            {
                "index": 0,
                "candidates": candidates_one,
                "user_features": user_features_one,
                "context": context_one,
                "product_metadata_map": metadata,
                "k": 2,
            },
            {
                "index": 1,
                "candidates": candidates_two,
                "user_features": user_features_two,
                "context": context_two,
                "product_metadata_map": metadata,
                "k": 1,
            },
        ]
    )

    assert [item["row_start"] for item in prepared] == [0, 2]
    assert [item["row_end"] for item in prepared] == [2, 3]
    np.testing.assert_allclose(batch_matrix[:2], matrix_one)
    np.testing.assert_allclose(batch_matrix[2:], matrix_two)


def test_missing_metadata_fallback_reuses_product_feature_cache(monkeypatch):
    current_time = 1_700_000_000.0
    monkeypatch.setattr(ranking_module.time, "time", lambda: current_time)

    ranking = RankingModel(RankingConfig(product_feature_cache_size=10))
    user_features = UserFeatures(user_id="u1", last_active=current_time)
    context = {"device": "mobile", "session_position": 1, "time_on_page": 0}
    candidate = CandidateProduct(
        product_id="missing-metadata",
        collaborative_score=0.2,
        content_similarity_score=0.3,
        popularity_score=0.4,
        combined_score=0.5,
        source="test",
    )

    matrix_one, _, _ = ranking.prepare_request_matrix(
        [candidate],
        user_features,
        context,
        product_metadata_map={},
    )
    current_time += 60
    matrix_two, _, _ = ranking.prepare_request_matrix(
        [candidate],
        user_features,
        context,
        product_metadata_map={},
    )

    product_age_index = ranking.feature_extractor.user_feature_dim + 4
    assert len(ranking._product_feature_cache) == 1
    assert matrix_one[0, product_age_index] == pytest.approx(0.0)
    assert matrix_two[0, product_age_index] == pytest.approx(0.0)


def test_sasrec_candidate_keeps_existing_ranking_feature_dimension():
    ranking = RankingModel(RankingConfig())
    candidate = CandidateProduct(
        product_id="seq",
        collaborative_score=0.8,
        combined_score=0.32,
        source="sasrec",
    )

    candidate_features = ranking.feature_extractor.extract_candidate_features(candidate)

    assert ranking.feature_extractor.candidate_feature_dim == 4
    assert ranking.feature_extractor.total_feature_dim == 28
    assert candidate_features.shape == (4,)
    assert candidate_features.tolist() == [0.8, 0.0, 0.0, 0.32]


def test_ranking_brand_feature_uses_stable_hash_bucket():
    ranking = RankingModel(RankingConfig())
    metadata = {
        "price": 25.0,
        "rating": 4.0,
        "num_reviews": 8,
        "in_stock": True,
        "created_at": time.time(),
        "tags": ["a"],
        "brand": "Acme",
    }
    expected_bucket = (
        int.from_bytes(
            hashlib.sha256(b"Acme").digest()[:8], byteorder="big", signed=False
        )
        % 100
    ) / 100

    dynamic_features = ranking.feature_extractor.extract_product_features(metadata)
    static_features, _ = ranking.feature_extractor.extract_static_product_features(
        metadata
    )

    np.testing.assert_allclose(dynamic_features[7], expected_bucket, rtol=1e-6)
    np.testing.assert_allclose(static_features[7], expected_bucket, rtol=1e-6)


def test_candidate_cache_schema_round_trips_merged_sasrec_source():
    candidate = CandidateProduct(
        product_id="seq",
        collaborative_score=0.9,
        combined_score=0.36,
        source="cf+sasrec",
    )

    restored = CandidateProduct(**candidate.dict())

    assert restored.source == "cf+sasrec"
    assert restored.collaborative_score == 0.9


def test_source_count_helpers_track_ranked_sasrec_coverage():
    candidates = [
        CandidateProduct(product_id="seq", source="cf+sasrec", collaborative_score=0.9),
        CandidateProduct(
            product_id="trend", source="trending_pool", popularity_score=0.5
        ),
    ]
    ranked = [
        ProductRecommendation(
            product_id="seq",
            title="Sequential product",
            price=10.0,
            confidence_score=0.8,
            ranking_score=0.7,
        )
    ]

    assert _count_candidate_source_tokens(candidates) == {
        "cf": 1,
        "sasrec": 1,
        "trending_pool": 1,
    }
    assert _count_ranked_source_tokens(
        ranked,
        {candidate.product_id: candidate.source for candidate in candidates},
    ) == {"cf": 1, "sasrec": 1}


def test_prepare_training_data_uses_online_equivalent_feature_extractor(monkeypatch):
    fixed_now = 1_700_000_000.0
    monkeypatch.setattr(ranking_module.time, "time", lambda: fixed_now)
    ranking = RankingModel(RankingConfig())
    user_features = UserFeatures(
        user_id="u1",
        total_interactions=12,
        avg_session_length=180.0,
        preferred_categories=["cat"],
        last_active=fixed_now - 30,
    )
    product_metadata = {
        "price": 42.0,
        "rating": 4.5,
        "num_reviews": 15,
        "in_stock": True,
        "created_at": fixed_now - 86400,
        "tags": ["video"],
        "brand": "brand",
        "category": "cat",
    }
    sample = {
        "user_id": "u1",
        "product_id": "p1",
        "action": "click",
        "context": {"device": "mobile", "session_position": 2, "time_on_page": 30},
        "collaborative_score": 0.2,
        "content_similarity_score": 0.3,
        "popularity_score": 0.4,
        "combined_score": 0.5,
        "source": "cf",
    }

    features, labels = ranking._prepare_training_data(
        [sample],
        user_features_map={"u1": user_features.dict()},
        product_metadata_map={"p1": product_metadata},
    )
    expected = ranking.feature_extractor.create_ranking_features(
        user_features,
        product_metadata,
        sample["context"],
        CandidateProduct(
            product_id="p1",
            collaborative_score=0.2,
            content_similarity_score=0.3,
            popularity_score=0.4,
            combined_score=0.5,
            source="cf",
        ),
    )

    np.testing.assert_allclose(features.cpu().numpy()[0], expected)
    assert labels["ctr"].item() == 1.0
    assert labels["cvr"].item() == 0.0


def test_prepare_training_data_does_not_use_random_vectors(monkeypatch):
    def fail_random(*args, **kwargs):
        raise AssertionError("random feature generation should not be used")

    monkeypatch.setattr(ranking_module.np.random, "normal", fail_random)
    ranking = RankingModel(RankingConfig())

    features, labels = ranking._prepare_training_data(
        [
            {
                "user_id": "u1",
                "product_id": "p1",
                "action": "view",
                "context": {},
            }
        ],
        user_features_map={},
        product_metadata_map={"p1": {"price": 10.0}},
    )

    assert features.shape == (1, ranking.feature_extractor.total_feature_dim)
    assert labels["ctr"].item() == 0.0


def test_prepare_training_data_labels_actions_and_gmv_sources():
    ranking = RankingModel(RankingConfig())
    samples = [
        {"user_id": "u1", "product_id": "p1", "action": "view", "context": {}},
        {"user_id": "u1", "product_id": "p2", "action": "click", "context": {}},
        {"user_id": "u1", "product_id": "p3", "action": "add_to_cart", "context": {}},
        {
            "user_id": "u1",
            "product_id": "p4",
            "action": "purchase",
            "context": {"purchase_value": 99.0},
        },
        {"user_id": "u1", "product_id": "p5", "action": "purchase", "context": {}},
    ]

    _, labels = ranking._prepare_training_data(
        samples,
        product_metadata_map={"p5": {"price": 25.0}},
    )

    assert labels["ctr"].squeeze(1).tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert labels["cvr"].squeeze(1).tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert labels["gmv"].squeeze(1).tolist() == [0.0, 0.0, 0.0, 99.0, 25.0]


class FakeTrainerFeatureStore:
    async def get_all_user_features_map(self):
        return {"u1": {"user_id": "u1", "total_interactions": 7}}


class FakeTrainerSystemStore:
    async def get_training_interactions(self, limit=50000):
        return [{"user_id": "u1", "product_id": "p1", "action": "click", "context": {}}]


class FakeTrainerRankingModel:
    def __init__(self):
        self.is_trained = True
        self.loaded_model_path = "/tmp/ranking.pt"
        self.model_version = "ranking-test"
        self.last_training_time = 123.0
        self.feature_schema_version = RANKING_FEATURE_SCHEMA_VERSION
        self.training_data_source = RANKING_TRAINING_DATA_SOURCE
        self.received = None

    async def train_model(
        self, training_data, *, user_features_map=None, product_metadata_map=None
    ):
        self.received = {
            "training_data": training_data,
            "user_features_map": user_features_map,
            "product_metadata_map": product_metadata_map,
        }


class FakeTrainerArtifactManager:
    def __init__(self):
        self.payload = None

    async def persist_ranking_checkpoint(
        self, *, local_path, model_version, payload=None
    ):
        self.payload = payload
        return SimpleNamespace(model_version=model_version)


class FakeTrainerObservability:
    def __init__(self):
        self.runs = []

    def record_training_run(self, trigger, status, duration):
        self.runs.append((trigger, status, duration))


@pytest.mark.asyncio
async def test_model_trainer_passes_real_feature_context_and_checkpoint_metadata():
    service = object.__new__(ModelTrainerService)
    service.config = SimpleNamespace(
        ranking_config=SimpleNamespace(
            enable_periodic_training=True, training_min_samples=1
        ),
        model_config=SimpleNamespace(ranking_model_path="/tmp/ranking.pt"),
    )
    service.feature_store = FakeTrainerFeatureStore()
    service.system_store = FakeTrainerSystemStore()
    service.ranking_model = FakeTrainerRankingModel()
    service.vector_search = SimpleNamespace(product_metadata={"p1": {"price": 12.0}})
    service.artifact_manager = FakeTrainerArtifactManager()
    service.observability = FakeTrainerObservability()

    await service._train_ranking_model(trigger="test")

    assert (
        service.ranking_model.received["user_features_map"]["u1"]["total_interactions"]
        == 7
    )
    assert service.ranking_model.received["product_metadata_map"]["p1"]["price"] == 12.0
    assert (
        service.artifact_manager.payload["feature_schema_version"]
        == RANKING_FEATURE_SCHEMA_VERSION
    )
    assert (
        service.artifact_manager.payload["training_data_source"]
        == RANKING_TRAINING_DATA_SOURCE
    )
    assert service.observability.runs[-1][1] == "success"


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

    recommendation_context = _build_recommendation_cache_context(
        payload, current_time=3600
    )
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


def test_cache_freshness_context_changes_when_content_features_become_ready():
    missing = _build_cache_freshness_context(
        {"ranking_model": "ranking-1"},
        {"total_interactions": 0},
        None,
        content_feature_token=_content_feature_cache_token("content-1", None, None),
    )
    ready = _build_cache_freshness_context(
        {"ranking_model": "ranking-1"},
        {"total_interactions": 0},
        None,
        content_feature_token=_content_feature_cache_token(
            "content-1",
            ContentFeatures(
                content_id="content-1",
                visual_embedding=[0.1, 0.2],
                created_at=123.0,
            ),
            456.0,
        ),
    )

    assert missing != ready
    assert missing["content_feature_token"]["content_features_present"] is False
    assert ready["content_feature_token"]["content_features_present"] is True


@pytest.mark.asyncio
async def test_serving_pools_return_candidate_clones():
    store = FeatureStore.__new__(FeatureStore)
    store._trending_pool_memory_cache = {
        "global": [
            CandidateProduct(
                product_id="trend",
                combined_score=0.5,
                source="trending_pool",
            )
        ]
    }
    store._category_pool_memory_cache = {
        "cat": [
            CandidateProduct(
                product_id="cat",
                combined_score=0.4,
                source="category_pool",
            )
        ]
    }

    first_trending = await FeatureStore.get_trending_pool(store, 1)
    first_trending[0].source = "mutated"
    second_trending = await FeatureStore.get_trending_pool(store, 1)

    first_category = await FeatureStore.get_category_pool(store, "cat", 1)
    first_category[0].combined_score = 99.0
    second_category = await FeatureStore.get_category_pool(store, "cat", 1)

    assert second_trending[0].source == "trending_pool"
    assert second_category[0].combined_score == 0.4


def test_recommendation_cache_context_changes_when_slate_diversity_is_enabled():
    payload = RecommendationRequest(user_id="u1", k=10, context={"device": "mobile"})

    disabled = _build_recommendation_cache_context(
        payload,
        current_time=3600,
        recommendation_config=RecommendationConfig(enable_slate_diversity=False),
    )
    enabled = _build_recommendation_cache_context(
        payload,
        current_time=3600,
        recommendation_config=RecommendationConfig(
            enable_slate_diversity=True,
            mmr_lambda=0.7,
        ),
    )

    assert disabled != enabled
    assert enabled["slate_diversity"]["enabled"] is True


@pytest.mark.asyncio
async def test_recommendation_path_expands_ranker_pool_and_caches_post_mmr(monkeypatch):
    recommendation_api_module._recommendation_singleflight.clear()
    scheduled_tasks = []

    class FakeFeatureStore:
        def __init__(self):
            self.cached_recommendations = None

        async def get_user_features(self, user_id, cache_default=False):
            return UserFeatures(user_id=user_id, total_interactions=3)

        async def get_user_sequence_token(self, user_id):
            return _default_user_sequence_token()

        async def get_user_serving_context(
            self, user_id, sequence_limit=200, cache_default=False
        ):
            return (
                await self.get_user_features(user_id, cache_default=cache_default),
                await self.get_user_sequence_token(user_id),
            )

        async def get_cached_recommendations(self, user_id, cache_key):
            return None

        async def get_cached_candidate_products(self, user_id, cache_key):
            return None

        def generate_context_hash(self, context):
            return hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()

        async def cache_candidate_products(
            self, user_id, cache_key, candidates, user_features=None
        ):
            return None

        async def get_product_metadata_batch(self, product_ids):
            return {
                product_id: {
                    "title": f"Product {product_id}",
                    "price": 10.0,
                    "active": True,
                    "in_stock": True,
                }
                for product_id in product_ids
            }

        def prime_product_metadata_memory_cache(self, metadata):
            return None

        async def store_product_metadata_batch(self, metadata):
            return None

        async def cache_recommendations(
            self, user_id, cache_key, recommendations, user_features=None
        ):
            self.cached_recommendations = recommendations

        async def log_recommendation_request(self, user_id, count, response_time):
            return None

    class FakeRecommendationEngine:
        loaded_two_tower_version = "two-tower-test"
        loaded_sasrec_version = None
        cf_engine = SimpleNamespace(model_version="two-tower-test")

        async def generate_candidates(
            self,
            user_id,
            content_features=None,
            context=None,
            k_per_source=100,
            include_profile=False,
            user_features=None,
            user_interactions=None,
        ):
            candidates = [
                CandidateProduct(
                    product_id=f"p{i}", combined_score=1.0 - i * 0.1, source="test"
                )
                for i in range(1, 6)
            ]
            return candidates, {"path": "fake", "candidate_count": len(candidates)}

        async def get_trending_recommendations(self, k):
            return []

    class FakeVectorSearch:
        def __init__(self):
            self.embeddings = {
                "p1": np.array([1.0, 0.0], dtype=np.float32),
                "p2": np.array([1.0, 0.0], dtype=np.float32),
                "p3": np.array([0.0, 1.0], dtype=np.float32),
                "p4": np.array([0.0, 1.0], dtype=np.float32),
            }

        def get_catalog_version_context(self):
            return {"catalog_version": 1, "last_updated": 1, "product_count": 5}

        async def get_product_metadata_batch(self, product_ids):
            return {}

        def get_product_embedding(self, product_id):
            return self.embeddings.get(product_id)

    class FakeRankingBatcher:
        def __init__(self):
            self.received_k = None

        async def rank_candidates(
            self,
            candidates,
            user_features,
            context,
            k,
            product_metadata_map=None,
            include_profile=False,
            deadline_unix_seconds=None,
        ):
            self.received_k = k
            recommendations = [
                _recommendation(candidate.product_id, 1.0 - index * 0.05)
                for index, candidate in enumerate(candidates[:k])
            ]
            return recommendations, {
                "path": "fake",
                "ranked_count": len(recommendations),
            }

    class FakeObservability:
        def record_recommendation(self, **kwargs):
            return None

    def schedule_task(task_name, awaitable, timeout_seconds=0.25):
        scheduled_tasks.append(asyncio.create_task(awaitable))

    feature_store = FakeFeatureStore()
    ranking_batcher = FakeRankingBatcher()
    runtime = SimpleNamespace(
        service_name="recommendation-service",
        active_requests=0,
        handled_requests=0,
        max_active_requests=0,
        observability=FakeObservability(),
        config=SimpleNamespace(
            recommendation_config=RecommendationConfig(
                enable_slate_diversity=True,
                mmr_lambda=0.5,
                mmr_rerank_pool_multiplier=2,
                mmr_min_rerank_pool_size=1,
                mmr_max_rerank_pool_size=10,
            ),
            cache_config=SimpleNamespace(
                hot_path_read_timeout_ms=1000,
                background_write_timeout_ms=1000,
            ),
            monitoring_config=SimpleNamespace(
                enable_profiling_logs=True,
                profiling_log_min_duration_ms=999999,
            ),
            model_config=SimpleNamespace(ranking_model_path=None),
            vector_config=SimpleNamespace(index_path=None),
        ),
    )
    request = SimpleNamespace(
        state=SimpleNamespace(
            worker_process_id=123,
            worker_active_requests_at_entry=1,
            worker_handled_requests=7,
            request_started_at=time.perf_counter(),
        )
    )

    monkeypatch.setattr(
        recommendation_api_module.app.state, "runtime", runtime, raising=False
    )
    monkeypatch.setattr(recommendation_api_module, "feature_store", feature_store)
    monkeypatch.setattr(
        recommendation_api_module, "recommendation_engine", FakeRecommendationEngine()
    )
    monkeypatch.setattr(recommendation_api_module, "vector_search", FakeVectorSearch())
    monkeypatch.setattr(recommendation_api_module, "ranking_batcher", ranking_batcher)
    monkeypatch.setattr(
        recommendation_api_module,
        "ranking_model",
        SimpleNamespace(model_version="ranking-test"),
    )
    monkeypatch.setattr(recommendation_api_module, "kafka_manager", None)
    monkeypatch.setattr(
        recommendation_api_module, "_schedule_best_effort_task", schedule_task
    )

    result = await recommendation_api_module.get_recommendations(
        request,
        Response(),
        RecommendationRequest(user_id="u1", k=2, context={"device": "mobile"}),
    )
    if scheduled_tasks:
        await asyncio.gather(*scheduled_tasks)
    payload = json.loads(result.body)

    assert ranking_batcher.received_k == 4
    assert [item["product_id"] for item in payload["recommendations"]] == ["p1", "p3"]
    assert len(payload["recommendations"]) == 2
    assert [item["product_id"] for item in feature_store.cached_recommendations] == [
        "p1",
        "p3",
    ]
    profile = payload["metadata"]["profile"]
    assert profile["slate_diversity_enabled"] is True
    assert profile["slate_diversity_method"] == "mmr"
    assert profile["slate_diversity_pool_size"] == 4
    assert profile["slate_diversity_selected_count"] == 2
    assert profile["slate_diversity_ms"] >= 0.0


@pytest.mark.asyncio
async def test_recommendation_admission_rejects_at_inflight_limit(monkeypatch):
    runtime = SimpleNamespace(
        service_name="recommendation-service",
        active_requests=1,
        handled_requests=0,
        max_active_requests=1,
        config=SimpleNamespace(
            service_topology_config=SimpleNamespace(max_inflight_recommendations=1),
        ),
    )
    request = SimpleNamespace(state=SimpleNamespace(request_started_at=time.perf_counter()))
    monkeypatch.setattr(
        recommendation_api_module.app.state,
        "runtime",
        runtime,
        raising=False,
    )

    with pytest.raises(HTTPException) as exc:
        await recommendation_api_module.get_recommendations(
            request,
            Response(),
            RecommendationRequest(user_id="u1", k=1),
        )

    assert exc.value.status_code == 503
    assert exc.value.detail == "Recommendation service overloaded"


@pytest.mark.asyncio
async def test_recommendation_local_ranking_path_passes_deadline(monkeypatch):
    class CapturingRankingBatcher:
        def __init__(self):
            self.deadline_unix_seconds = None

        async def rank_candidates(
            self,
            candidates,
            user_features,
            context,
            k,
            product_metadata_map=None,
            include_profile=False,
            deadline_unix_seconds=None,
        ):
            self.deadline_unix_seconds = deadline_unix_seconds
            return [_recommendation(candidates[0].product_id, 1.0)], {
                "path": "fake"
            }

    ranking_batcher = CapturingRankingBatcher()
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            service_topology_config=SimpleNamespace(
                ranking_coordinator_request_timeout_seconds=1.0
            )
        )
    )
    monkeypatch.setattr(
        recommendation_api_module.app.state,
        "runtime",
        runtime,
        raising=False,
    )
    monkeypatch.setattr(recommendation_api_module, "ranking_batcher", ranking_batcher)
    monkeypatch.setattr(
        recommendation_api_module,
        "ranking_coordinator_client_pool",
        None,
    )
    monkeypatch.setattr(recommendation_api_module, "ranking_client_pool", None)
    monkeypatch.setattr(recommendation_api_module.time, "time", lambda: 100.0)

    await recommendation_api_module._rank_candidates_for_request(
        candidates=[
            CandidateProduct(product_id="p1", combined_score=1.0, source="test")
        ],
        user_features=UserFeatures(user_id="u1"),
        context={},
        product_metadata_map={},
        k=1,
        http_request=SimpleNamespace(state=SimpleNamespace(request_id="req-1")),
    )

    assert ranking_batcher.deadline_unix_seconds == pytest.approx(100.9)


@pytest.mark.asyncio
async def test_missing_content_features_skips_candidate_and_recommendation_cache_writes(
    monkeypatch,
):
    recommendation_api_module._recommendation_singleflight.clear()
    scheduled_tasks = []
    scheduled_names = []

    class FakeFeatureStore:
        def __init__(self):
            self.candidate_cache_writes = 0
            self.recommendation_cache_writes = 0

        async def get_user_serving_context(
            self, user_id, sequence_limit=200, cache_default=False
        ):
            return UserFeatures(user_id=user_id, total_interactions=1), {
                **_default_user_sequence_token(),
                "length": 1,
            }

        async def get_content_features(self, content_id):
            return None

        async def get_content_processed_time(self, content_id):
            return 123.0

        async def get_cached_recommendations(self, user_id, cache_key):
            return None

        async def get_cached_candidate_products(self, user_id, cache_key):
            return None

        def generate_context_hash(self, context):
            return hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()

        async def cache_candidate_products(
            self, user_id, cache_key, candidates, user_features=None
        ):
            self.candidate_cache_writes += 1

        async def get_product_metadata_batch(self, product_ids):
            return {
                product_id: {
                    "title": f"Product {product_id}",
                    "price": 10.0,
                    "category": "cat",
                    "brand": "brand",
                    "active": True,
                    "in_stock": True,
                }
                for product_id in product_ids
            }

        def prime_product_metadata_memory_cache(self, metadata):
            return None

        async def store_product_metadata_batch(self, metadata):
            return None

        async def cache_recommendations(
            self, user_id, cache_key, recommendations, user_features=None
        ):
            self.recommendation_cache_writes += 1

        async def log_recommendation_request(self, user_id, count, response_time):
            return None

    class FakeRecommendationEngine:
        loaded_two_tower_version = "two-tower-test"
        loaded_sasrec_version = None
        cf_engine = SimpleNamespace(model_version="two-tower-test")

        async def generate_candidates(
            self,
            user_id,
            content_features=None,
            context=None,
            k_per_source=100,
            include_profile=False,
            user_features=None,
            user_interactions=None,
        ):
            return [
                CandidateProduct(
                    product_id="p1",
                    combined_score=1.0,
                    source="test",
                )
            ], {"path": "fake", "candidate_count": 1}

        async def get_trending_recommendations(self, k):
            return []

    class FakeVectorSearch:
        def get_catalog_version_context(self):
            return {"catalog_version": 1, "last_updated": 1, "product_count": 1}

        async def get_product_metadata_batch(self, product_ids):
            return {}

    class FakeRankingBatcher:
        async def rank_candidates(
            self,
            candidates,
            user_features,
            context,
            k,
            product_metadata_map=None,
            include_profile=False,
            deadline_unix_seconds=None,
        ):
            return [_recommendation("p1", 1.0)], {"path": "fake", "ranked_count": 1}

    class FakeObservability:
        def record_recommendation(self, **kwargs):
            return None

    def schedule_task(task_name, awaitable, timeout_seconds=0.25):
        scheduled_names.append(task_name)
        task = asyncio.create_task(awaitable)
        scheduled_tasks.append(task)
        return task

    feature_store = FakeFeatureStore()
    runtime = SimpleNamespace(
        service_name="recommendation-service",
        active_requests=0,
        handled_requests=0,
        max_active_requests=0,
        observability=FakeObservability(),
        config=SimpleNamespace(
            service_topology_config=SimpleNamespace(max_inflight_recommendations=0),
            recommendation_config=RecommendationConfig(),
            cache_config=SimpleNamespace(
                hot_path_read_timeout_ms=1000,
                candidate_cache_race_timeout_ms=5,
                recommendation_cache_race_timeout_ms=5,
                background_write_timeout_ms=1000,
            ),
            monitoring_config=SimpleNamespace(
                enable_profiling_logs=True,
                profiling_log_min_duration_ms=999999,
            ),
            model_config=SimpleNamespace(ranking_model_path=None),
            vector_config=SimpleNamespace(index_path=None),
        ),
    )
    request = SimpleNamespace(
        state=SimpleNamespace(
            worker_process_id=123,
            worker_active_requests_at_entry=1,
            worker_handled_requests=7,
            request_started_at=time.perf_counter(),
        )
    )

    monkeypatch.setattr(
        recommendation_api_module.app.state, "runtime", runtime, raising=False
    )
    monkeypatch.setattr(recommendation_api_module, "feature_store", feature_store)
    monkeypatch.setattr(
        recommendation_api_module, "recommendation_engine", FakeRecommendationEngine()
    )
    monkeypatch.setattr(recommendation_api_module, "vector_search", FakeVectorSearch())
    monkeypatch.setattr(recommendation_api_module, "ranking_batcher", FakeRankingBatcher())
    monkeypatch.setattr(recommendation_api_module, "ranking_coordinator_client_pool", None)
    monkeypatch.setattr(recommendation_api_module, "ranking_client_pool", None)
    monkeypatch.setattr(
        recommendation_api_module,
        "ranking_model",
        SimpleNamespace(model_version="ranking-test"),
    )
    monkeypatch.setattr(recommendation_api_module, "kafka_manager", None)
    monkeypatch.setattr(
        recommendation_api_module, "_schedule_best_effort_task", schedule_task
    )

    result = await recommendation_api_module.get_recommendations(
        request,
        Response(),
        RecommendationRequest(user_id="u1", content_id="content-1", k=1),
    )
    if scheduled_tasks:
        await asyncio.gather(*scheduled_tasks)
    payload = json.loads(result.body)

    assert payload["recommendations"][0]["product_id"] == "p1"
    assert feature_store.candidate_cache_writes == 0
    assert feature_store.recommendation_cache_writes == 0
    assert "cache_candidate_products" not in scheduled_names
    assert "cache_recommendations" not in scheduled_names


@pytest.mark.asyncio
async def test_user_sequence_token_timeout_falls_back_without_failing_request():
    async def slow_read():
        await asyncio.sleep(0.05)
        return {"length": 1}

    runtime = SimpleNamespace(
        config=SimpleNamespace(cache_config=SimpleNamespace(hot_path_read_timeout_ms=1))
    )

    result = await _bounded_hot_path_read(
        runtime,
        "user_sequence_token",
        slow_read(),
        _default_user_sequence_token(),
    )

    assert result == _default_user_sequence_token()


@pytest.mark.asyncio
async def test_candidate_cache_race_timeout_does_not_block_live_generation():
    async def slow_candidate_cache():
        await asyncio.sleep(0.05)
        return [CandidateProduct(product_id="cached", source="cache")]

    runtime = SimpleNamespace(
        config=SimpleNamespace(
            cache_config=SimpleNamespace(candidate_cache_race_timeout_ms=1)
        )
    )
    task = asyncio.create_task(slow_candidate_cache())

    candidates, elapsed_ms = await _await_candidate_cache_race(runtime, task)

    assert candidates is None
    assert elapsed_ms < 25
    assert task.cancelled()


@pytest.mark.asyncio
async def test_recommendation_cache_race_timeout_does_not_block_live_generation():
    async def slow_recommendation_cache():
        await asyncio.sleep(0.05)
        return [{"product_id": "cached"}]

    runtime = SimpleNamespace(
        config=SimpleNamespace(
            cache_config=SimpleNamespace(recommendation_cache_race_timeout_ms=1)
        )
    )
    task = asyncio.create_task(slow_recommendation_cache())

    cached, elapsed_ms = await _await_recommendation_cache_race(runtime, task)

    assert cached is None
    assert elapsed_ms < 25
    assert task.cancelled()


@pytest.mark.asyncio
async def test_best_effort_queue_full_drops_without_blocking_response_path():
    async def noop():
        return None

    class FakeObservability:
        def __init__(self):
            self.records = []
            self.depths = []

        def record_best_effort_task(self, task, status):
            self.records.append((task, status))

        def set_best_effort_queue_depth(self, depth):
            self.depths.append(depth)

    observability = FakeObservability()
    runtime = SimpleNamespace(
        observability=observability,
        config=SimpleNamespace(
            cache_config=SimpleNamespace(
                background_task_queue_size=1,
                background_task_worker_count=1,
            )
        ),
    )
    queue = _BestEffortTaskQueue(runtime)

    assert queue.enqueue("cache_recommendations", noop(), 0.1) is True
    assert queue.enqueue("cache_recommendations", noop(), 0.1) is False
    await queue.close()

    assert ("cache_recommendations", "enqueued") in observability.records
    assert ("cache_recommendations", "dropped_queue_full") in observability.records


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


def test_serving_version_context_includes_sasrec_versions(monkeypatch, tmp_path):
    checkpoint = tmp_path / "sasrec.pt"
    vocab = tmp_path / "sasrec_vocab.json"
    checkpoint.write_bytes(b"checkpoint")
    vocab.write_text("{}", encoding="utf-8")

    runtime = SimpleNamespace(
        config=SimpleNamespace(
            model_config=SimpleNamespace(
                ranking_model_path=str(tmp_path / "ranking.pt")
            ),
            vector_config=SimpleNamespace(index_path=str(tmp_path / "vector.faiss")),
            recommendation_config=SimpleNamespace(
                sasrec_checkpoint_path=str(checkpoint),
                sasrec_vocab_path=str(vocab),
            ),
        )
    )

    monkeypatch.setattr(
        recommendation_api_module,
        "recommendation_engine",
        SimpleNamespace(
            loaded_two_tower_version="two-tower-1",
            loaded_sasrec_version="sasrec-1",
            cf_engine=SimpleNamespace(model_version="two-tower-1"),
        ),
    )
    monkeypatch.setattr(
        recommendation_api_module,
        "ranking_model",
        SimpleNamespace(model_version="ranking-1"),
    )
    monkeypatch.setattr(
        recommendation_api_module,
        "vector_search",
        SimpleNamespace(
            get_catalog_version_context=lambda: {
                "catalog_version": 1,
                "last_updated": 1,
                "product_count": 1,
            }
        ),
    )

    context = _serving_version_context(runtime)

    assert context["sasrec_model"] == "sasrec-1"
    assert context["sasrec_checkpoint_mtime"] is not None
    assert context["sasrec_vocab_mtime"] is not None


def test_serving_version_context_reuses_cached_file_mtimes(monkeypatch, tmp_path):
    ranking_path = tmp_path / "ranking.pt"
    vector_path = tmp_path / "vector.faiss"
    sasrec_path = tmp_path / "sasrec.pt"
    vocab_path = tmp_path / "sasrec_vocab.json"
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            model_config=SimpleNamespace(ranking_model_path=str(ranking_path)),
            vector_config=SimpleNamespace(index_path=str(vector_path)),
            recommendation_config=SimpleNamespace(
                sasrec_checkpoint_path=str(sasrec_path),
                sasrec_vocab_path=str(vocab_path),
            ),
        )
    )
    calls = {"count": 0}

    def fake_mtime(path):
        calls["count"] += 1
        return calls["count"]

    monkeypatch.setattr(recommendation_api_module, "_safe_file_mtime", fake_mtime)
    monkeypatch.setattr(recommendation_api_module, "recommendation_engine", None)
    monkeypatch.setattr(recommendation_api_module, "ranking_model", None)
    monkeypatch.setattr(recommendation_api_module, "vector_search", None)

    _refresh_serving_version_context(runtime)
    first = _serving_version_context(runtime)
    second = _serving_version_context(runtime)

    assert first == second
    assert calls["count"] == 4


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
    same_bucket_again = engine._user_embedding_cache_key(
        "u1", features, current_time=1000.8
    )
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


@pytest.mark.asyncio
async def test_ranking_load_model_keeps_existing_model_when_checkpoint_load_fails(
    tmp_path,
    monkeypatch,
):
    ranking = RankingModel(RankingConfig(hidden_dims=[8]))
    await ranking.load_model()
    existing_model = ranking.model
    existing_optimizer = ranking.optimizer
    checkpoint_path = tmp_path / "ranking.pt"
    torch.save(
        {
            "model_state_dict": {},
            "config": {"architecture": "mlp", "hidden_dims": [8]},
        },
        checkpoint_path,
    )

    def fail_load(model, state_dict):
        raise RuntimeError("partial load")

    monkeypatch.setattr(
        ranking,
        "_load_shape_compatible_state_dict_into_model",
        fail_load,
    )

    with pytest.raises(RuntimeError, match="partial load"):
        await ranking.load_model(str(checkpoint_path))

    assert ranking.model is existing_model
    assert ranking.optimizer is existing_optimizer
