from __future__ import annotations

import asyncio
import os
from pathlib import Path
import uuid

import pytest
from sqlalchemy import inspect

from video_commerce.common.config import DatabaseConfig, ModelReleaseConfig
from video_commerce.data_plane.system_store import (
    ModelRelease,
    ModelReleaseEvaluation,
    ModelReleasePointer,
    ModelReleaseTransition,
    SystemStore,
)
from video_commerce.ml.model_release import (
    ModelReleaseController,
    ReleaseLifecycleState,
    ReleasePromotionError,
    ReleaseValidationStatus,
    validate_release_transition,
)


def test_registered_release_cannot_skip_quality_gate():
    with pytest.raises(ReleasePromotionError, match="registered.*active"):
        validate_release_transition(
            ReleaseLifecycleState.REGISTERED,
            ReleaseLifecycleState.ACTIVE,
            validation_status=ReleaseValidationStatus.PASSED,
        )


def test_model_release_config_exposes_production_gate_defaults():
    config = ModelReleaseConfig()

    assert config.gate_mode == "observe"
    assert config.environment == "production"
    assert config.holdout_days == 7
    assert config.bootstrap_samples == 2000
    assert config.min_impressions == 1000
    assert config.evaluation_max_age_hours == 168


def test_model_release_config_rejects_unknown_gate_mode():
    with pytest.raises(ValueError, match="legacy, observe, or enforced"):
        ModelReleaseConfig(gate_mode="auto")


def test_validated_release_must_be_staged_before_activation():
    assert validate_release_transition(
        ReleaseLifecycleState.REGISTERED,
        ReleaseLifecycleState.VALIDATED,
        validation_status=ReleaseValidationStatus.PASSED,
    )
    assert validate_release_transition(
        ReleaseLifecycleState.VALIDATED,
        ReleaseLifecycleState.STAGING,
        validation_status=ReleaseValidationStatus.PASSED,
    )
    assert validate_release_transition(
        ReleaseLifecycleState.STAGING,
        ReleaseLifecycleState.ACTIVE,
        validation_status=ReleaseValidationStatus.PASSED,
    )


def test_failed_or_insufficient_release_cannot_leave_registered():
    for status in (
        ReleaseValidationStatus.FAILED,
        ReleaseValidationStatus.INSUFFICIENT_EVIDENCE,
    ):
        with pytest.raises(ReleasePromotionError, match="quality gate"):
            validate_release_transition(
                ReleaseLifecycleState.REGISTERED,
                ReleaseLifecycleState.VALIDATED,
                validation_status=status,
            )


class _PromotionStore:
    def __init__(self):
        self.generation = 3
        self.active_version = "champion"
        self.releases = {
            "challenger": {
                "model_version": "challenger",
                "lifecycle_state": "staging",
                "validation_status": "passed",
            },
            "champion": {
                "model_version": "champion",
                "lifecycle_state": "active",
                "validation_status": "passed",
            },
        }

    async def activate_model_release(self, **kwargs):
        if kwargs["expected_generation"] != self.generation:
            return None
        release = self.releases[kwargs["model_version"]]
        if release["lifecycle_state"] not in {"staging", "retired"}:
            return None
        self.releases[self.active_version]["lifecycle_state"] = "retired"
        release["lifecycle_state"] = "active"
        self.active_version = kwargs["model_version"]
        self.generation += 1
        return {
            "model_version": self.active_version,
            "generation": self.generation,
        }


def test_promotion_uses_expected_generation_cas():
    store = _PromotionStore()
    controller = ModelReleaseController(store)

    promoted = asyncio.run(
        controller.promote(
            model_name="ranking_model",
            model_version="challenger",
            environment="production",
            expected_generation=3,
            actor="release-bot",
            reason="offline gate passed",
        )
    )

    assert promoted == {"model_version": "challenger", "generation": 4}
    with pytest.raises(ReleasePromotionError, match="generation"):
        asyncio.run(
            controller.promote(
                model_name="ranking_model",
                model_version="champion",
                environment="production",
                expected_generation=3,
                actor="release-bot",
                reason="stale retry",
            )
        )


def test_promotion_requires_auditable_actor_and_reason():
    controller = ModelReleaseController(_PromotionStore())

    with pytest.raises(ValueError, match="actor"):
        asyncio.run(
            controller.promote(
                model_name="ranking_model",
                model_version="challenger",
                environment="production",
                expected_generation=3,
                actor="",
                reason="offline gate passed",
            )
        )


def test_release_migration_and_models_define_durable_state_machine():
    migration = (
        Path(__file__).resolve().parents[1]
        / "migrations/postgres/010_model_release_quality_gate.sql"
    ).read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS model_releases" in migration
    assert "CREATE TABLE IF NOT EXISTS model_release_evaluations" in migration
    assert "CREATE TABLE IF NOT EXISTS model_release_pointers" in migration
    assert "CREATE TABLE IF NOT EXISTS model_release_transitions" in migration
    assert "UNIQUE (model_name, environment, slot)" in migration
    assert {column.key for column in inspect(ModelRelease).columns} >= {
        "release_id",
        "checkpoint_id",
        "lifecycle_state",
        "validation_status",
        "bundle_manifest",
    }
    assert inspect(ModelReleaseEvaluation).primary_key[0].key == "evaluation_id"
    assert [column.key for column in inspect(ModelReleasePointer).primary_key] == [
        "model_name",
        "environment",
        "slot",
    ]
    assert inspect(ModelReleaseTransition).primary_key[0].key == "id"


def test_disabled_store_has_no_active_checkpoint_fallback():
    store = SystemStore(DatabaseConfig(enable=False))

    assert (
        asyncio.run(
            store.get_active_model_checkpoint("ranking_model", environment="production")
        )
        is None
    )


def test_postgres_release_activation_is_cas_guarded_and_audited(tmp_path):
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is required for model release integration test")

    async def exercise():
        suffix = uuid.uuid4().hex[:12]
        store = SystemStore(
            DatabaseConfig(
                enable=True,
                url=database_url,
                auto_create_schema=True,
                enable_retention_cleanup=False,
            )
        )
        await store.initialize()
        try:
            for version in (
                f"champion-{suffix}",
                f"challenger-{suffix}",
                f"regression-{suffix}",
            ):
                assert await store.record_model_checkpoint(
                    "ranking_model",
                    version,
                    str(tmp_path / f"{version}.pt"),
                    payload={"artifact_sha256": "a" * 64},
                )
                release = await store.register_model_release(
                    model_name="ranking_model",
                    model_version=version,
                    bundle_manifest={
                        "artifact_sha256": "a" * 64,
                        "artifact_manifest": {
                            "checkpoint": {
                                "path": str(tmp_path / f"{version}.pt"),
                                "sha256": "a" * 64,
                            }
                        },
                        "feature_schema_version": "ranking_v3_00_temporal_multimodal",
                        "score_policy_version": "business-value-v1",
                        "value_transform_stats": {},
                        "quality_gate_policy_version": "ranking_quality_gate_v1",
                    },
                )
                assert release["lifecycle_state"] == "registered"

            champion = await store.get_model_release(
                "ranking_model", f"champion-{suffix}"
            )
            bootstrapped = await store.activate_model_release(
                model_name="ranking_model",
                model_version=f"champion-{suffix}",
                environment=f"test-{suffix}",
                expected_generation=0,
                actor="test",
                reason="initial verified artifact",
                bootstrap=True,
                rollback=False,
            )
            assert bootstrapped["generation"] == 1
            assert bootstrapped["release_id"] == champion["release_id"]

            challenger = await store.get_model_release(
                "ranking_model", f"challenger-{suffix}"
            )
            evaluation = await store.record_model_release_evaluation(
                release_id=challenger["release_id"],
                champion_release_id=champion["release_id"],
                dataset_manifest_uri="s3://pit/manifest.json",
                dataset_manifest_sha256="b" * 64,
                holdout_start=100.0,
                holdout_end=200.0,
                policy_version="ranking_quality_gate_v1",
                decision="passed",
                metrics={"ndcg_at_10_delta": 0.01},
                slice_metrics={},
                bootstrap={"ndcg_at_10_delta_ci_lower": 0.0},
                gate_config={"bootstrap_samples": 2000},
            )
            assert evaluation["decision"] == "passed"
            assert await store.stage_model_release(
                release_id=challenger["release_id"], actor="evaluator"
            )

            regression = await store.get_model_release(
                "ranking_model", f"regression-{suffix}"
            )
            failed_evaluation = await store.record_model_release_evaluation(
                release_id=regression["release_id"],
                champion_release_id=champion["release_id"],
                dataset_manifest_uri="s3://pit/manifest.json",
                dataset_manifest_sha256="b" * 64,
                holdout_start=100.0,
                holdout_end=200.0,
                policy_version="ranking_quality_gate_v1",
                decision="failed",
                metrics={"ndcg_at_10_delta": -0.1},
                slice_metrics={},
                bootstrap={"ndcg_at_10_delta_ci_lower": -0.2},
                gate_config={"bootstrap_samples": 2000},
            )
            assert failed_evaluation["decision"] == "failed"
            assert not await store.stage_model_release(
                release_id=regression["release_id"], actor="evaluator"
            )

            promoted = await store.activate_model_release(
                model_name="ranking_model",
                model_version=f"challenger-{suffix}",
                environment=f"test-{suffix}",
                expected_generation=1,
                actor="release-manager",
                reason="quality gate passed",
                bootstrap=False,
                rollback=False,
            )
            assert promoted["generation"] == 2
            stale = await store.activate_model_release(
                model_name="ranking_model",
                model_version=f"champion-{suffix}",
                environment=f"test-{suffix}",
                expected_generation=1,
                actor="release-manager",
                reason="stale retry",
                bootstrap=False,
                rollback=True,
            )
            assert stale is None
            pointer = await store.get_model_release_pointer(
                "ranking_model", environment=f"test-{suffix}", slot="active"
            )
            assert pointer["model_version"] == f"challenger-{suffix}"
            assert pointer["generation"] == 2
            active_checkpoint = await store.get_active_model_checkpoint(
                "ranking_model", environment=f"test-{suffix}"
            )
            assert active_checkpoint["model_version"] == f"challenger-{suffix}"
            assert (
                active_checkpoint["payload"]["model_release_id"]
                == challenger["release_id"]
            )
            assert active_checkpoint["payload"]["active_generation"] == 2
        finally:
            await store.close()

    asyncio.run(exercise())
