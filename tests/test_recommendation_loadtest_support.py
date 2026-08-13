import asyncio
import hashlib
import json

import torch

from video_commerce.common.config import RankingConfig
from video_commerce.loadtest_support import (
    build_synthetic_catalog,
    create_synthetic_ranking_checkpoint,
    summarize_k6_result,
)
from video_commerce.ml.ranking import RankingModel


def test_build_synthetic_catalog_is_deterministic_and_recommendable():
    first_candidates, first_metadata = build_synthetic_catalog(count=20, seed=17)
    second_candidates, second_metadata = build_synthetic_catalog(count=20, seed=17)

    assert [item.dict() for item in first_candidates] == [
        item.dict() for item in second_candidates
    ]
    assert first_metadata == second_metadata
    assert len(first_candidates) == 20
    assert set(first_metadata) == {item.product_id for item in first_candidates}
    assert all(metadata["is_active"] for metadata in first_metadata.values())
    assert all(metadata["in_stock"] for metadata in first_metadata.values())


def test_create_synthetic_checkpoint_loads_as_trained_and_runs_forward(tmp_path):
    checkpoint_path = tmp_path / "ranking_model.pt"
    config = RankingConfig(
        architecture="mlp",
        hidden_dims=[16],
        dropout_rate=0.0,
        require_verified_artifact=True,
        allow_untrained_fallback=False,
    )

    manifest = asyncio.run(
        create_synthetic_ranking_checkpoint(
            checkpoint_path=checkpoint_path,
            ranking_config=config,
            model_version="synthetic-loadtest-v1",
            seed=23,
        )
    )

    assert checkpoint_path.exists()
    assert manifest["purpose"] == "serving_capacity_only"
    assert manifest["synthetic_weights"] is True
    assert (
        manifest["artifact_sha256"]
        == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    )

    loaded = RankingModel(config)
    asyncio.run(loaded.load_model(str(checkpoint_path)))
    loaded.mark_artifact_verified(
        model_version=manifest["model_version"],
        artifact_sha256=manifest["artifact_sha256"],
        feature_schema_version=manifest["feature_schema_version"],
        metadata=manifest,
    )
    health = loaded.health_check(run_inference_test=True)

    assert loaded.is_trained is True
    assert health["status"] == "healthy"
    assert health["artifact_verified"] is True
    assert health["inference_test_passed"] is True
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert (
        checkpoint["config"]["feature_schema_version"]
        == manifest["feature_schema_version"]
    )


def test_summarize_k6_result_preserves_capacity_and_path_evidence(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "root_group": {"checks": {"passes": 1990, "fails": 10}},
                "metrics": {
                    "iterations": {"count": 2000, "rate": 997.4},
                    "dropped_iterations": {"count": 0, "rate": 0},
                    "http_req_duration": {
                        "avg": 72.0,
                        "med": 52.0,
                        "p(95)": 122.0,
                        "p(99)": 174.0,
                    },
                    "recommendation_success_latency": {
                        "avg": 70.0,
                        "med": 50.0,
                        "p(95)": 120.0,
                        "p(99)": 170.0,
                    },
                    "recommendation_errors": {"passes": 0, "fails": 2000},
                    "recommendation_5xx": {"passes": 0, "fails": 2000},
                    "recommendation_429": {"passes": 0, "fails": 2000},
                    "candidate_cache_rank_responses": {
                        "passes": 1988,
                        "fails": 12,
                    },
                    "recommendation_cache_responses": {
                        "passes": 0,
                        "fails": 2000,
                    },
                    "model_forward_responses": {
                        "passes": 1988,
                        "fails": 12,
                    },
                    "ranking_untrained_fallback_responses": {
                        "passes": 0,
                        "fails": 2000,
                    },
                    "recommendation_batch_request_count": {
                        "avg": 8.5,
                        "med": 8,
                        "p(95)": 12,
                    },
                    "recommendation_model_forward_latency": {
                        "avg": 40.0,
                        "med": 38.0,
                        "p(95)": 60.0,
                        "p(99)": 70.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_k6_result(
        result_path,
        target_qps=1000,
        duration="2s",
        scenario="candidate_cache_rank",
    )

    assert summary["actual_qps"] == 997.4
    assert summary["successful_qps"] == 997.4
    assert summary["error_rate"] == 0.0
    assert summary["p95_ms"] == 120.0
    assert summary["p99_ms"] == 170.0
    assert summary["dropped_iterations"] == 0
    assert summary["candidate_cache_rank_rate"] == 0.994
    assert summary["recommendation_cache_rate"] == 0.0
    assert summary["model_forward_evidence_rate"] == 0.994
    assert summary["untrained_fallback_rate"] == 0.0
    assert summary["batch_requests_avg"] == 8.5
    assert summary["model_forward_p95_ms"] == 60.0


def test_summarize_k6_result_marks_unexported_percentiles_as_missing(tmp_path):
    result_path = tmp_path / "result-without-p99.json"
    result_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "iterations": {"count": 10, "rate": 5.0},
                    "recommendation_success_latency": {
                        "avg": 20.0,
                        "med": 18.0,
                        "p(95)": 30.0,
                    },
                    "recommendation_model_forward_latency": {
                        "avg": 10.0,
                        "p(95)": 15.0,
                    },
                    "recommendation_errors": {"passes": 0, "fails": 10},
                }
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_k6_result(
        result_path,
        target_qps=5,
        duration="2s",
        scenario="candidate_cache_rank",
    )

    assert summary["p99_ms"] is None
    assert summary["model_forward_p99_ms"] is None
