import asyncio
import hashlib

from video_commerce.common.config import (
    ModelConfig,
    ObjectStorageConfig,
    RecommendationConfig,
)
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.ml.model_artifacts import ModelArtifactManager


class _Store:
    def __init__(self, active):
        self.active = active
        self.latest = None

    async def get_active_model_checkpoint(self, model_name, *, environment):
        assert model_name == "ranking_model"
        assert environment == "production"
        return self.active

    async def get_latest_model_checkpoint(self, _model_name):
        return self.latest


class _RegisteringStore:
    def __init__(self, *, insert_checkpoint=True):
        self.release = None
        self.insert_checkpoint = insert_checkpoint

    async def get_latest_model_checkpoint(self, _model_name):
        return None

    async def record_model_checkpoint(
        self, model_name, model_version, checkpoint_path, payload=None
    ):
        self.checkpoint = {
            "model_name": model_name,
            "model_version": model_version,
            "checkpoint_path": checkpoint_path,
            "payload": payload,
        }
        return self.insert_checkpoint

    async def get_model_checkpoint_for_materialization_run(self, model_name, run_id):
        return self.checkpoint

    async def register_model_release(
        self, *, model_name, model_version, bundle_manifest
    ):
        self.release = {
            "release_id": "release-1",
            "model_name": model_name,
            "model_version": model_version,
            "bundle_manifest": bundle_manifest,
        }
        return self.release


def test_sync_active_ranking_materializes_only_pointer_selected_checkpoint(tmp_path):
    source = tmp_path / "active.pt"
    source.write_bytes(b"active")
    destination = tmp_path / "cache" / "ranking.pt"
    store = _Store(
        {
            "model_name": "ranking_model",
            "model_version": "active-v1",
            "checkpoint_path": str(source),
            "payload": {
                "artifact_sha256": hashlib.sha256(b"active").hexdigest(),
                "feature_schema_version": "ranking_v3_00_temporal_multimodal",
                "model_release_id": "release-active",
                "active_generation": 7,
            },
            "created_at": 1.0,
        }
    )
    manager = ModelArtifactManager(
        system_store=store,
        object_storage=ObjectStorage(ObjectStorageConfig(backend="local")),
        model_config=ModelConfig(ranking_model_path=str(destination)),
        recommendation_config=RecommendationConfig(),
    )

    record = asyncio.run(
        manager.sync_active_ranking_checkpoint(
            environment="production",
            expected_feature_schema_version="ranking_v3_00_temporal_multimodal",
        )
    )

    assert record.model_version == "active-v1"
    assert record.payload["model_release_id"] == "release-active"
    assert record.payload["active_generation"] == 7
    assert destination.read_bytes() == b"active"


def test_persisted_ranking_checkpoint_is_registered_without_becoming_active(tmp_path):
    checkpoint = tmp_path / "ranking.pt"
    checkpoint.write_bytes(b"ranking")
    store = _RegisteringStore()
    manager = ModelArtifactManager(
        system_store=store,
        object_storage=ObjectStorage(ObjectStorageConfig(backend="local")),
        model_config=ModelConfig(ranking_model_path=str(checkpoint)),
        recommendation_config=RecommendationConfig(),
    )

    record = asyncio.run(
        manager.persist_ranking_checkpoint(
            local_path=str(checkpoint),
            model_version="ranking-v2",
            payload={
                "feature_schema_version": "ranking_v3_00_temporal_multimodal",
                "feature_lake_manifest_sha256": "b" * 64,
                "score_policy_version": "business-value-v1",
                "value_transform_stats": {},
            },
        )
    )

    assert store.release["model_version"] == "ranking-v2"
    assert store.release["bundle_manifest"]["quality_gate_policy_version"] == (
        "ranking_quality_gate_v1"
    )
    assert (
        store.release["bundle_manifest"]["artifact_manifest"]
        == record.payload["artifact_manifest"]
    )
    assert record.payload["model_release_id"] == "release-1"


def test_checkpoint_conflict_repairs_missing_release_registration(tmp_path):
    checkpoint = tmp_path / "ranking.pt"
    checkpoint.write_bytes(b"ranking")
    store = _RegisteringStore(insert_checkpoint=False)
    manager = ModelArtifactManager(
        system_store=store,
        object_storage=ObjectStorage(ObjectStorageConfig(backend="local")),
        model_config=ModelConfig(ranking_model_path=str(checkpoint)),
        recommendation_config=RecommendationConfig(),
    )

    record = asyncio.run(
        manager.persist_ranking_checkpoint(
            local_path=str(checkpoint),
            model_version="ranking-v2",
            payload={
                "feature_lake_materialization_run_id": "pit-run-1",
                "feature_schema_version": "ranking_v3_00_temporal_multimodal",
                "feature_lake_manifest_sha256": "b" * 64,
                "score_policy_version": "business-value-v1",
                "value_transform_stats": {},
            },
        )
    )

    assert store.release["model_version"] == "ranking-v2"
    assert record.payload["model_release_id"] == "release-1"


def test_enforced_selection_fails_closed_without_active_pointer(tmp_path):
    manager = ModelArtifactManager(
        system_store=_Store(None),
        object_storage=ObjectStorage(ObjectStorageConfig(backend="local")),
        model_config=ModelConfig(ranking_model_path=str(tmp_path / "ranking.pt")),
        recommendation_config=RecommendationConfig(),
    )

    try:
        asyncio.run(
            manager.sync_selected_ranking_checkpoint(
                gate_mode="enforced",
                environment="production",
                expected_feature_schema_version="ranking_v3_00_temporal_multimodal",
            )
        )
    except RuntimeError as exc:
        assert "active ranking release" in str(exc)
    else:
        raise AssertionError("enforced release selection must fail closed")


def test_observe_selection_falls_back_to_latest_checkpoint(tmp_path):
    source = tmp_path / "latest.pt"
    source.write_bytes(b"latest")
    store = _Store(None)
    store.latest = {
        "model_name": "ranking_model",
        "model_version": "latest-v1",
        "checkpoint_path": str(source),
        "payload": {
            "artifact_sha256": hashlib.sha256(b"latest").hexdigest(),
            "feature_schema_version": "ranking_v3_00_temporal_multimodal",
        },
        "created_at": 1.0,
    }
    destination = tmp_path / "cache" / "ranking.pt"
    manager = ModelArtifactManager(
        system_store=store,
        object_storage=ObjectStorage(ObjectStorageConfig(backend="local")),
        model_config=ModelConfig(ranking_model_path=str(destination)),
        recommendation_config=RecommendationConfig(),
    )

    record = asyncio.run(
        manager.sync_selected_ranking_checkpoint(
            gate_mode="observe",
            environment="production",
            expected_feature_schema_version="ranking_v3_00_temporal_multimodal",
        )
    )

    assert record.model_version == "latest-v1"
    assert destination.read_bytes() == b"latest"
