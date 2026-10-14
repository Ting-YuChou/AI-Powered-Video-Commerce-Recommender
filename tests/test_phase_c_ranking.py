import math

import numpy as np
import pytest

from video_commerce.common.config import RankingConfig, RecommendationConfig
from video_commerce.common.models import CandidateProduct
from video_commerce.ml.candidate_fusion import (
    FUSION_POLICY_VERSION,
    SOURCE_REGISTRY_VERSION,
    merge_candidates_rrf,
    retrieval_lineage_snapshot,
)
from video_commerce.ml.ranking_training import AttributionFacts, RankingLabelBuilder
from video_commerce.ml.probability_calibration import (
    TemperatureCalibration,
    expected_calibration_error,
    fit_temperature,
)
from video_commerce.ml.ranking import RankingModel
from video_commerce.ml.ranking_categorical import (
    MISSING_ID,
    OOV_ID,
    RankingCategoricalVocabulary,
)
from video_commerce.ml.ranking_features import FeatureBundle
from video_commerce.common.models import UserFeatures


def test_v3_model_roundtrip_uses_frozen_categorical_features(tmp_path):
    import asyncio
    import torch

    config = RankingConfig(
        feature_contract_version="ranking_ltr_v3_categorical_sources",
        categorical_vocab_min_frequency=1,
        hidden_dims=[16],
    )
    ranker = RankingModel(config)
    bundle = FeatureBundle(
        as_of_ts=1000.0,
        feature_definition_version=config.feature_contract_version,
        user_features=UserFeatures(user_id="u1", last_active=900.0),
        product_metadata={
            "category": "cat",
            "brand": "brand",
            "price": 12.0,
            "created_at": 100.0,
        },
        context={"device": "mobile", "surface": "home"},
        candidate=CandidateProduct(product_id="p1", source="content"),
    )
    ranker.fit_categorical_features([bundle])
    ranker.fit_dense_normalization([bundle])
    ranker._initialize_model()
    ranker.is_trained = True
    matrix = ranker.feature_assembler.build_many([bundle])
    ranker.model.eval()
    with torch.no_grad():
        before = ranker.model(torch.tensor(matrix))["ranking_score"]
    checkpoint = tmp_path / "v3.pt"
    asyncio.run(ranker.save_model(str(checkpoint)))
    restored = RankingModel(config)
    asyncio.run(restored.load_model(str(checkpoint)))
    with torch.no_grad():
        after = restored.model(
            torch.tensor(restored.feature_assembler.build_many([bundle]))
        )["ranking_score"]
    assert torch.allclose(before, after)
    assert (
        restored.categorical_vocabulary.to_manifest()
        == ranker.categorical_vocabulary.to_manifest()
    )
    from types import SimpleNamespace

    verified_payload = {
        "artifact_sha256": "a" * 64,
        "feature_schema_version": restored.feature_schema_version,
        "categorical_vocabulary": restored.categorical_vocabulary.to_manifest(),
        "dense_normalization": restored.feature_assembler.dense_normalization.to_manifest(),
    }
    assert restored.mark_artifact_record_verified(
        SimpleNamespace(model_version="v3", payload=verified_payload)
    )
    mismatched = {**verified_payload, "categorical_vocabulary": {}}
    assert not restored.mark_artifact_record_verified(
        SimpleNamespace(model_version="v3", payload=mismatched)
    )
    old_model = restored.model
    broken = torch.load(checkpoint, map_location="cpu")
    broken["config"]["categorical_vocabulary"] = RankingCategoricalVocabulary.build(
        [
            {
                "category": "different-category",
                "brand": "different-brand",
                "device": "mobile",
                "surface": "home",
                "time_of_day": "night",
            }
        ],
        min_frequency=1,
    ).to_manifest()
    broken["config"]["probability_calibration"] = {
        "policy_version": "temperature_v1",
        "ctr_temperature": float("nan"),
        "ctcvr_temperature": 1.0,
    }
    corrupt_path = tmp_path / "broken.pt"
    torch.save(broken, corrupt_path)
    with pytest.raises(ValueError, match="finite"):
        asyncio.run(restored.load_model(str(corrupt_path)))
    assert restored.model is old_model
    assert (
        restored.categorical_vocabulary.to_manifest()
        == ranker.categorical_vocabulary.to_manifest()
    )


@pytest.mark.parametrize("architecture", ["dcn", "dcn_v2_low_rank"])
def test_v3_onnx_export_matches_torch(tmp_path, architecture):
    from video_commerce.ml.ranking_onnx import export_ranking_onnx

    ranker = RankingModel(
        RankingConfig(
            feature_contract_version="ranking_ltr_v3_categorical_sources",
            architecture=architecture,
            hidden_dims=[16],
        )
    )
    bundle = FeatureBundle(
        as_of_ts=1000.0,
        feature_definition_version=ranker.config.feature_contract_version,
        user_features=UserFeatures(user_id="u1", last_active=900.0),
        product_metadata={"price": 10.0, "created_at": 100.0},
        context={},
        candidate=CandidateProduct(product_id="p1", source="content"),
    )
    ranker.fit_categorical_features([bundle])
    ranker.fit_dense_normalization([bundle])
    ranker._initialize_model()
    ranker.model.eval()
    ranker.is_trained = ranker.artifact_verified = True
    contract = export_ranking_onnx(ranker, tmp_path / "v3.onnx")
    assert contract["input_dim"] == ranker.feature_assembler.output_dim


def test_dense_normalization_matches_clipped_numpy_and_preserves_missing():
    from video_commerce.ml.ranking_categorical import DenseFeatureNormalization

    values = np.asarray([[1.0, 3.0], [2.0, 4.0], [100.0, np.nan]])
    normalizer = DenseFeatureNormalization.fit(iter(values), dimension=2)
    assert normalizer.lower == pytest.approx(np.nanquantile(values, 0.01, axis=0))
    assert normalizer.upper == pytest.approx(np.nanquantile(values, 0.99, axis=0))
    clipped = np.clip(values, normalizer.lower, normalizer.upper)
    assert normalizer.mean == pytest.approx(np.nanmean(clipped, axis=0))
    result = normalizer.apply(values)
    assert np.isfinite(result).all()
    assert result[-1, -1] == 1.0
    assert result[-1, 1] == pytest.approx(0.0)


def test_streaming_v3_training_does_not_fit_holdout_or_calibration_vocab(tmp_path):
    from video_commerce.ml.ranking_training import RankingTrainingExample

    def example(identifier, category, clicked, purchased=False):
        bundle = FeatureBundle(
            as_of_ts=1000.0,
            feature_definition_version="ranking_ltr_v1",
            user_features=UserFeatures(user_id="u1", last_active=900.0),
            product_metadata={
                "category": category,
                "brand": category,
                "price": 10.0,
                "created_at": 100.0,
            },
            context={},
            candidate=CandidateProduct(product_id=identifier, source="content"),
        )
        return RankingTrainingExample(
            observation_id=identifier,
            impression_id=identifier,
            bundle=bundle,
            attribution=AttributionFacts(
                attributed_action="purchase"
                if purchased
                else "click"
                if clicked
                else "view",
                attributed_click=clicked,
                attributed_purchase=purchased,
                attributed_value=12.0 if purchased else None,
                attributed_value_source="purchase_value" if purchased else None,
                attributed_value_currency="USD" if purchased else None,
            ),
        )

    class Source:
        calls = []

        def row_count_for_split(self, split):
            return 2

        def reindex_value_buckets(self, resolver):
            pass

        def compute_value_transform_stats(self, **kwargs):
            assert kwargs["split"] == "training"
            return {
                "global": {"count": 1, "clip": 12.0, "mean": 0.0, "std": 1.0},
                "buckets": {},
            }

        def iter_example_batches(self, *, split, **kwargs):
            self.calls.append(split)
            assert split != "holdout"
            category = (
                "training-category" if split == "training" else f"{split}-category"
            )
            yield [
                example(f"{split}-1", category, True, True),
                example(f"{split}-2", category, False),
            ]

    source = Source()
    ranker = RankingModel(
        RankingConfig(
            feature_contract_version="ranking_ltr_v3_categorical_sources",
            probability_calibration="temperature_v1",
            categorical_vocab_min_frequency=1,
            epochs=1,
            training_min_samples=1,
            hidden_dims=[16],
            batch_size=2,
        )
    )
    ranker.loaded_model_path = str(tmp_path / "trained.pt")
    ranker._train_model_source_sync(source, batch_rows=2)
    assert ranker.is_trained
    assert set(ranker.categorical_vocabulary.fields["category"]) == {
        "training-category"
    }
    assert "calibration" in source.calls
    assert (
        ranker.artifact_metadata["probability_calibration_evidence"]["row_count"] == 2
    )


def test_phase_c_promotion_requires_bound_linux_performance_evidence():
    from video_commerce.ml.phase_c_evidence import (
        bundle_digest,
        performance_evidence_valid,
    )

    manifest = {"artifact_manifest": {"checkpoint": {"sha256": "a" * 64}}}
    mode = {
        "error_rate": 0.0,
        "five_xx_rate": 0.0,
        "successful_p95_ms": 500.0,
        "successful_p99_ms": 800.0,
        "p95_relative_regression": 0.05,
        "successful_qps_relative_regression": 0.05,
    }
    evidence = {
        "release_id": "release-1",
        "bundle_digest": bundle_digest(manifest),
        "platform": "linux",
        "resources": {"cpus": 4, "memory_bytes": 8_589_934_592},
        "results_sha256": "b" * 64,
        "hot": mode,
        "unique": mode,
    }
    assert performance_evidence_valid(
        evidence, release_id="release-1", manifest=manifest
    )
    assert not performance_evidence_valid(
        None, release_id="release-1", manifest=manifest
    )
    assert not performance_evidence_valid(
        evidence, release_id="other-release", manifest=manifest
    )
    assert not performance_evidence_valid(
        {**evidence, "platform": "darwin"}, release_id="release-1", manifest=manifest
    )
    assert not performance_evidence_valid(
        {**evidence, "unique": {**mode, "successful_p99_ms": 2000.0}},
        release_id="release-1",
        manifest=manifest,
    )


def test_v2_promotion_rejects_weakened_or_single_experiment_gate_snapshot():
    from dataclasses import asdict
    from video_commerce.ml.model_release_quality import RankingQualityGateConfig
    from video_commerce.ml.phase_c_evidence import quality_gate_snapshot_valid

    snapshot = asdict(RankingQualityGateConfig())
    assert quality_gate_snapshot_valid(snapshot)
    assert not quality_gate_snapshot_valid({**snapshot, "min_users": 1})
    assert not quality_gate_snapshot_valid({**snapshot, "experiment_kind": "features"})


def test_rrf_keeps_each_source_signal_and_ignores_raw_score_scale():
    merged = merge_candidates_rrf(
        {
            "cf": [
                CandidateProduct(
                    product_id="shared",
                    collaborative_score=0.01,
                    combined_score=0.01,
                    source="cf",
                ),
                CandidateProduct(
                    product_id="cf-only",
                    collaborative_score=0.009,
                    combined_score=0.009,
                    source="cf",
                ),
            ],
            "swing_itemcf": [
                CandidateProduct(
                    product_id="shared",
                    collaborative_score=9000.0,
                    combined_score=9000.0,
                    source="swing_itemcf",
                ),
                CandidateProduct(
                    product_id="swing-only",
                    collaborative_score=8000.0,
                    combined_score=8000.0,
                    source="swing_itemcf",
                ),
            ],
        },
        limit=3,
        rrf_k=60,
    )

    assert [candidate.product_id for candidate in merged] == [
        "shared",
        "cf-only",
        "swing-only",
    ]
    shared = merged[0]
    assert set(shared.retrieval_signals) == {"two_tower", "swing_itemcf"}
    assert shared.retrieval_signals["two_tower"].source_rank == 1
    assert shared.retrieval_signals["swing_itemcf"].raw_score == 9000.0
    assert shared.collaborative_score == 0.01
    assert shared.source == "two_tower+swing_itemcf"
    assert shared.fusion_policy_version == FUSION_POLICY_VERSION
    assert shared.source_registry_version == SOURCE_REGISTRY_VERSION
    assert shared.combined_score == 2.0 / 61.0


def test_rrf_uses_product_id_for_deterministic_ties():
    merged = merge_candidates_rrf(
        {
            "trending_pool": [
                CandidateProduct(
                    product_id="product-b",
                    popularity_score=0.9,
                    combined_score=0.9,
                    source="trending_pool",
                )
            ],
            "category_pool": [
                CandidateProduct(
                    product_id="product-a",
                    popularity_score=0.1,
                    combined_score=0.1,
                    source="category_pool",
                )
            ],
        },
        limit=2,
        rrf_k=60,
    )

    assert [candidate.product_id for candidate in merged] == [
        "product-a",
        "product-b",
    ]
    assert math.isclose(merged[0].combined_score, 1.0 / 61.0)


def test_candidate_fusion_config_rejects_unknown_mode_and_invalid_rrf_k():
    assert (
        RecommendationConfig(candidate_fusion_mode="RRF").candidate_fusion_mode == "rrf"
    )
    with pytest.raises(ValueError, match="candidate_fusion_mode"):
        RecommendationConfig(candidate_fusion_mode="learned")
    with pytest.raises(ValueError, match="candidate_rrf_k"):
        RecommendationConfig(candidate_rrf_k=0)


def test_retrieval_lineage_snapshot_is_json_ready_and_complete():
    candidate = merge_candidates_rrf(
        {
            "cf": [
                CandidateProduct(
                    product_id="p1",
                    collaborative_score=0.7,
                    combined_score=0.7,
                    source="cf",
                )
            ]
        },
        limit=1,
    )[0]

    assert retrieval_lineage_snapshot(candidate) == {
        "candidate_pool_position": 1,
        "fusion_policy_version": "candidate_rrf_v1",
        "source_registry_version": "retrieval_sources_v1",
        "retrieval_signals": {
            "two_tower": {
                "raw_source": "cf",
                "raw_score": 0.7,
                "source_rank": 1,
                "source_count": 1,
                "normalized_rank": 1.0,
                "reciprocal_rank": 1.0 / 61.0,
            }
        },
    }


@pytest.mark.parametrize(
    ("source", "currency", "conflict", "expected_mask", "expected_reason"),
    [
        ("purchase_value", "USD", False, 1.0, None),
        ("gmv", "usd", False, 1.0, None),
        ("margin", "USD", False, 0.0, "unsupported_value_source"),
        ("purchase_value", "EUR", False, 0.0, "currency_mismatch"),
        ("gmv", "USD", True, 0.0, "conflicting_gross_revenue_values"),
    ],
)
def test_gross_revenue_policy_masks_incompatible_value_labels(
    source, currency, conflict, expected_mask, expected_reason
):
    labels = RankingLabelBuilder(
        target_policy="gross_revenue_v1", value_currency="USD"
    ).build(
        AttributionFacts(
            attributed_action="purchase",
            attributed_click=True,
            attributed_purchase=True,
            attributed_value=25.0,
            attributed_value_source=source,
            attributed_value_currency=currency,
            attributed_value_conflict=conflict,
        )
    )

    assert labels.value_mask == expected_mask
    assert labels.business_value == (25.0 if expected_mask else 0.0)
    assert labels.value_rejection_reason == expected_reason


def test_temperature_fit_reduces_overconfident_probability_error():
    probabilities = [0.01, 0.02, 0.98, 0.99]
    labels = [0.0, 1.0, 0.0, 1.0]

    temperature = fit_temperature(probabilities, labels)
    calibrated = TemperatureCalibration(
        ctr_temperature=temperature,
        ctcvr_temperature=temperature,
    ).apply_numpy(probabilities, head="ctr")

    assert temperature > 1.0
    assert expected_calibration_error(calibrated, labels, bins=2) < (
        expected_calibration_error(probabilities, labels, bins=2)
    )


def test_temperature_calibration_numpy_and_torch_are_equivalent():
    import torch

    policy = TemperatureCalibration(ctr_temperature=1.7, ctcvr_temperature=2.3)
    probabilities = np.asarray([0.1, 0.5, 0.9], dtype=np.float32)

    numpy_values = policy.apply_numpy(probabilities, head="ctcvr")
    torch_values = policy.apply_torch(
        torch.from_numpy(probabilities), head="ctcvr"
    ).numpy()

    assert np.allclose(numpy_values, torch_values, atol=1e-6)
    assert policy.to_manifest()["policy_version"] == "temperature_v1"


def test_business_score_uses_calibrated_ctcvr_and_preserves_raw_probability():
    model = RankingModel(
        RankingConfig(
            business_score_enabled=True,
            probability_calibration="temperature_v1",
        )
    )
    model.probability_calibration = TemperatureCalibration(
        ctr_temperature=2.0,
        ctcvr_temperature=2.0,
    )
    predictions = {
        "ctr": np.asarray([0.9], dtype=np.float32),
        "cvr": np.asarray([1.0], dtype=np.float32),
        "ctcvr": np.asarray([0.9], dtype=np.float32),
        "gmv": np.asarray([1.0], dtype=np.float32),
        "ranking_score": np.asarray([0.1], dtype=np.float32),
    }

    model._add_business_predictions(predictions)

    assert predictions["raw_ctr"].tolist() == pytest.approx([0.9])
    assert predictions["raw_ctcvr"].tolist() == pytest.approx([0.9])
    assert predictions["ctcvr"][0] < predictions["raw_ctcvr"][0]
    assert predictions["business_score"][0] == pytest.approx(
        predictions["ctcvr"][0] * predictions["predicted_value"][0]
    )


def test_categorical_vocab_is_training_only_deterministic_and_reserves_special_ids():
    records = [
        {
            "category": "zeta" if index % 2 else "alpha",
            "brand": "brand-a",
            "tags": ["tag-b", "tag-a"],
            "device": "mobile",
            "surface": "home",
            "time_of_day": "3",
        }
        for index in range(6)
    ]

    first = RankingCategoricalVocabulary.build(records, min_frequency=2)
    second = RankingCategoricalVocabulary.build(
        list(reversed(records)), min_frequency=2
    )

    assert first.to_manifest() == second.to_manifest()
    assert first.fields["category"]["alpha"] == 3
    assert first.fields["category"]["zeta"] == 4
    assert first.lookup("category", None) == MISSING_ID
    assert first.lookup("category", "unseen") == OOV_ID


def test_categorical_vocab_truncates_multihot_fields_and_tracks_source_overlap():
    vocabulary = RankingCategoricalVocabulary.build(
        [
            {
                "category": "cat",
                "brand": "brand",
                "tags": [f"tag-{index}" for index in range(30)],
                "preferred_categories": [f"pref-{index}" for index in range(15)],
                "device": "mobile",
                "surface": "home",
                "time_of_day": "2",
            }
        ],
        min_frequency=1,
    )
    encoded = vocabulary.encode_record(
        {
            "category": "cat",
            "brand": "brand",
            "tags": [f"tag-{index}" for index in range(30)],
            "preferred_categories": [f"pref-{index}" for index in range(15)],
            "request_category": "cat",
            "device": "mobile",
            "surface": "home",
            "time_of_day": "2",
            "price_sensitivity": 0.5,
            "normalized_log_price": 2.0,
            "retrieval_source_overlap": 3,
        }
    )

    assert len(encoded["preferred_categories"]) == 10
    assert len(encoded["tags"]) == 20
    assert encoded["price_cross"] == pytest.approx(1.0)
    assert encoded["retrieval_source_overlap"] == pytest.approx(3.0)
