import numpy as np

from config import RankingConfig
from models import CandidateProduct, RecommendationRequest, UserFeatures
from ranking import RankingModel
import ranking as ranking_module
from recommendation_api import _build_recommendation_cache_context, _user_feature_cache_token


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
