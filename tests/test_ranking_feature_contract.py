import asyncio
import numpy as np
import pytest
import threading
import time

from video_commerce.common.config import RankingConfig
from video_commerce.common.models import CandidateProduct, UserFeatures
from video_commerce.ml.ranking import FeatureExtractor, RankingModel
from video_commerce.ml.ranking_features import FeatureBundle, RankingFeatureAssembler
from video_commerce.ml.ranking_training import (
    AttributionFacts,
    RankingTrainingExample,
)


def _bundle() -> FeatureBundle:
    return FeatureBundle(
        as_of_ts=100.0 + 3 * 86400,
        feature_definition_version="ranking_ltr_v1",
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
    vector = RankingFeatureAssembler(FeatureExtractor()).build(_bundle())

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
    actual = RankingFeatureAssembler(extractor).build(bundle)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_feature_assembler_build_many_matches_single_bundle_contract():
    assembler = RankingFeatureAssembler(FeatureExtractor())
    bundle = _bundle()

    matrix = assembler.build_many([bundle, bundle])

    assert matrix.shape == (2, assembler.extractor.total_feature_dim)
    np.testing.assert_allclose(matrix[0], assembler.build(bundle), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(matrix[1], assembler.build(bundle), rtol=1e-6, atol=1e-6)


def test_feature_assembler_fallback_metadata_never_reads_wall_clock(monkeypatch):
    bundle_payload = dict(_bundle().__dict__)
    bundle_payload["product_metadata"] = {"_ranking_fallback_metadata": True}

    monkeypatch.setattr(
        "video_commerce.ml.ranking.time.time",
        lambda: (_ for _ in ()).throw(AssertionError("wall clock read")),
    )

    vector = RankingFeatureAssembler(FeatureExtractor()).build(
        FeatureBundle(**bundle_payload)
    )

    assert np.isfinite(vector).all()


def test_feature_bundle_rejects_wrong_definition_version():
    payload = dict(_bundle().__dict__)
    payload["feature_definition_version"] = "ranking_ltr_v0"

    with pytest.raises(ValueError, match="feature definition"):
        FeatureBundle(**payload)


def test_online_request_matrix_routes_through_shared_assembler(monkeypatch):
    ranking = RankingModel(RankingConfig())
    calls = []
    original = ranking.feature_assembler.build_many

    def capture(bundles):
        calls.extend(bundles)
        return original(bundles)

    monkeypatch.setattr(ranking.feature_assembler, "build_many", capture)
    bundle = _bundle()
    matrix, candidates, _ = ranking.prepare_request_matrix(
        [bundle.candidate],
        bundle.user_features,
        {**bundle.context, "_feature_as_of_ts": bundle.as_of_ts},
        product_metadata_map={
            bundle.candidate.product_id: dict(bundle.product_metadata)
        },
    )

    assert matrix.shape[0] == 1
    assert len(candidates) == 1
    assert len(calls) == 1
    assert calls[0].as_of_ts == bundle.as_of_ts


@pytest.mark.asyncio
async def test_untrained_ranker_uses_candidate_score_fallback(monkeypatch):
    class ObservabilitySpy:
        calls = 0

        def record_ranking_untrained_fallback(self):
            self.calls += 1

    observability = ObservabilitySpy()
    ranking = RankingModel(
        RankingConfig(offload_inference_to_thread=False),
        observability=observability,
    )
    await ranking.load_model()
    assert ranking.is_trained is False
    ranking.model = None

    monkeypatch.setattr(
        ranking,
        "prepare_request_matrix",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("untrained neural inference executed")
        ),
    )
    recommendations, profile = await ranking.rank_candidates(
        [
            CandidateProduct(product_id="low", combined_score=0.2, source="test"),
            CandidateProduct(product_id="high", combined_score=0.9, source="test"),
            CandidateProduct(
                product_id="zero",
                combined_score=0.0,
                collaborative_score=1.0,
                source="test",
            ),
        ],
        UserFeatures(user_id="u1"),
        {},
        k=3,
        include_profile=True,
    )

    assert [item.product_id for item in recommendations] == ["high", "low", "zero"]
    assert profile["path"] == "fallback_untrained"
    assert ranking.untrained_fallback_count == 1
    assert observability.calls == 1


def test_ranking_training_uses_pit_bundle_and_observation_timestamp(monkeypatch):
    ranking = RankingModel(RankingConfig())
    captured = {}

    def capture_features(bundle):
        captured["user_features"] = bundle.user_features
        captured["product_metadata"] = bundle.product_metadata
        captured["as_of_ts"] = bundle.as_of_ts
        return np.zeros(ranking.feature_extractor.total_feature_dim, dtype=np.float32)

    monkeypatch.setattr(ranking.feature_assembler, "build", capture_features)

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


@pytest.mark.asyncio
async def test_async_training_cancellation_waits_for_sync_worker_exit(monkeypatch):
    ranking = RankingModel(RankingConfig(training_min_samples=1))
    worker_started = threading.Event()
    worker_stopped = threading.Event()
    cancellation_event = threading.Event()

    def cancellable_sync_worker(
        _training_data,
        _training_sample_source,
        *,
        cancellation_event,
        **_kwargs,
    ):
        worker_started.set()
        while not cancellation_event.is_set():
            time.sleep(0.001)
        worker_stopped.set()
        return None

    monkeypatch.setattr(ranking, "_train_model_sync", cancellable_sync_worker)
    example = RankingTrainingExample(
        observation_id="imp-1:p1",
        impression_id="imp-1",
        bundle=_bundle(),
        attribution=AttributionFacts("click", True, False),
    )
    task = asyncio.create_task(
        ranking.train_model([example], cancellation_event=cancellation_event)
    )
    assert await asyncio.to_thread(worker_started.wait, 1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert worker_stopped.is_set()
