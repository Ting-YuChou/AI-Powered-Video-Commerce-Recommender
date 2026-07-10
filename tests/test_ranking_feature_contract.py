import numpy as np
import pytest

from video_commerce.common.config import RankingConfig
from video_commerce.common.models import CandidateProduct, UserFeatures
from video_commerce.ml.ranking import FeatureExtractor, RankingModel
from video_commerce.ml.ranking_features import FeatureBundle, RankingFeatureAssembler


def _bundle() -> FeatureBundle:
    return FeatureBundle(
        user_features=UserFeatures(
            user_id="user-1",
            total_interactions=12,
            avg_session_length=120.0,
            preferred_categories=["beauty"],
            price_sensitivity=0.4,
            click_through_rate=0.25,
            conversion_rate=0.1,
            last_active=100.0,
            demographics={},
        ),
        product_metadata={
            "price": 42.0,
            "rating": 4.5,
            "num_reviews": 20,
            "in_stock": True,
            "created_at": 100.0,
            "tags": ["sale"],
            "brand": "brand-1",
        },
        context={"device": "mobile", "session_position": 2},
        candidate=CandidateProduct(
            product_id="product-1",
            combined_score=0.8,
            source="cf",
        ),
    )


def test_feature_assembler_uses_supplied_as_of_time_for_all_time_features():
    vector = RankingFeatureAssembler(FeatureExtractor()).build(
        _bundle(),
        as_of_ts=100.0 + 3 * 86400,
    )

    assert vector[6] == 3.0
    assert vector[14] == 3.0


def test_feature_assembler_matches_feature_extractor_for_a_fixed_observation_time():
    bundle = _bundle()
    as_of_ts = 100.0 + 3 * 86400
    extractor = FeatureExtractor()

    expected = extractor.create_ranking_features(
        bundle.user_features,
        bundle.product_metadata,
        bundle.context,
        bundle.candidate,
        as_of_ts=as_of_ts,
    )
    actual = RankingFeatureAssembler(extractor).build(bundle, as_of_ts=as_of_ts)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_ranking_training_uses_pit_bundle_and_observation_timestamp(monkeypatch):
    ranking = RankingModel(RankingConfig())
    captured = {}

    def capture_features(user_features, product_metadata, context, candidate, *, as_of_ts=None):
        captured["user_features"] = user_features
        captured["product_metadata"] = product_metadata
        captured["as_of_ts"] = as_of_ts
        return np.zeros(ranking.feature_extractor.total_feature_dim, dtype=np.float32)

    monkeypatch.setattr(
        ranking.feature_extractor, "create_ranking_features", capture_features
    )

    features, _ = ranking._prepare_training_data(
        [
            {
                "user_id": "user-1",
                "product_id": "product-1",
                "action": "click",
                "event_time": 1_700_000_000.0,
                "as_of_ts": 1_700_000_123.0,
                "user_features": {"total_interactions": 3},
                "product_metadata": {"price": 9.0, "category": "shoes"},
                "context": {},
            }
        ],
        user_features_map={"user-1": {"total_interactions": 999}},
        product_metadata_map={"product-1": {"price": 999.0}},
    )

    assert features.shape[0] == 1
    assert captured["user_features"].total_interactions == 3
    assert captured["product_metadata"]["price"] == 9.0
    assert captured["as_of_ts"] == pytest.approx(1_700_000_123.0)
