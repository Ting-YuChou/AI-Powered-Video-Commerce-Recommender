from __future__ import annotations

import asyncio
import hashlib

from video_commerce.common.config import (
    ModelConfig,
    ObjectStorageConfig,
    RecommendationConfig,
)
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.ml.model_artifacts import ModelArtifactManager


class ObservationStore:
    def __init__(self, active):
        self.active = active
        self.observations = []

    async def get_active_model_checkpoint(self, model_name, *, environment):
        return self.active

    async def record_model_release_instance_observation(self, **kwargs):
        self.observations.append(kwargs)


def test_active_generation_stays_loading_until_serving_smoke_marks_ready(
    tmp_path, monkeypatch
):
    remote = tmp_path / "remote.pt"
    remote.write_bytes(b"ranking")
    store = ObservationStore(
        {
            "model_name": "ranking_model",
            "model_version": "v2",
            "checkpoint_path": str(remote),
            "payload": {
                "artifact_sha256": hashlib.sha256(b"ranking").hexdigest(),
                "feature_schema_version": "ranking_v3_00_temporal_multimodal",
                "model_release_id": "release-v2",
                "active_generation": 2,
            },
        }
    )
    monkeypatch.setenv("MODEL_ARTIFACT_SERVICE_NAME", "ranking-runner")
    manager = ModelArtifactManager(
        system_store=store,
        object_storage=ObjectStorage(ObjectStorageConfig(backend="local")),
        model_config=ModelConfig(
            artifact_generation_root=str(tmp_path / "generations"),
            ranking_model_path=str(tmp_path / "legacy.pt"),
        ),
        recommendation_config=RecommendationConfig(),
    )

    record = asyncio.run(
        manager.sync_active_ranking_checkpoint(
            environment="production",
            expected_feature_schema_version="ranking_v3_00_temporal_multimodal",
        )
    )

    assert [row["state"] for row in store.observations] == ["loading"]
    asyncio.run(
        manager.record_generation_activation(
            record, environment="production", state="ready"
        )
    )
    assert [row["state"] for row in store.observations] == ["loading", "ready"]
    assert store.observations[-1]["bundle_digest"] == record.payload["bundle_digest"]
    assert store.observations[-1]["observed_generation"] == 2
