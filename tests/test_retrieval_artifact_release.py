import asyncio

from video_commerce.common.config import (
    ModelConfig,
    ObjectStorageConfig,
    RecommendationConfig,
)
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.ml.model_artifacts import ModelArtifactManager


class ReleaseStore:
    def __init__(self):
        self.checkpoints = []
        self.releases = []

    async def activate_product_catalog(self, source_version, metadata_map, **kwargs):
        return "catalog-1"

    async def record_model_checkpoint(
        self, model_name, model_version, checkpoint_path, payload=None
    ):
        self.checkpoints.append(
            {
                "model_name": model_name,
                "model_version": model_version,
                "checkpoint_path": checkpoint_path,
                "payload": dict(payload or {}),
            }
        )
        return True

    async def register_model_release(
        self, *, model_name, model_version, bundle_manifest
    ):
        result = {
            "release_id": "release-1",
            "model_name": model_name,
            "model_version": model_version,
            "bundle_manifest": dict(bundle_manifest),
        }
        self.releases.append(result)
        return result


def test_pit_two_tower_persistence_registers_but_does_not_activate_release(tmp_path):
    store = ReleaseStore()
    checkpoint = tmp_path / "model.pt"
    index = tmp_path / "model.faiss"
    metadata = tmp_path / "model.cf_meta.json"
    checkpoint.write_bytes(b"checkpoint")
    index.write_bytes(b"index")
    metadata.write_text("{}", encoding="utf-8")
    manager = ModelArtifactManager(
        system_store=store,
        object_storage=ObjectStorage(
            ObjectStorageConfig(
                backend="local", download_dir=str(tmp_path / "downloads")
            )
        ),
        model_config=ModelConfig(cache_dir=str(tmp_path / "cache")),
        recommendation_config=RecommendationConfig(),
    )

    record = asyncio.run(
        manager.persist_two_tower_artifacts(
            checkpoint_path=str(checkpoint),
            index_path=str(index),
            metadata_path=str(metadata),
            model_version="tt-pit-1",
            catalog_metadata={"p1": {"active": True, "in_stock": True}},
            payload={
                "retrieval_pit_manifest_sha256": "a" * 64,
                "catalog_manifest_sha256": "b" * 64,
                "eligibility_policy_version": "retrieval_eligibility_v1",
                "label_policy_version": "retrieval_label_v1",
                "quality_gate_policy_version": "retrieval_quality_gate_v1",
                "embedding_dimension": 128,
            },
        )
    )

    assert len(store.releases) == 1
    assert store.releases[0]["bundle_manifest"]["cf_index_path"]
    assert record.payload["model_release_id"] == "release-1"
