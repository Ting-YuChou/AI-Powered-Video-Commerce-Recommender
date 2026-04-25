import asyncio
import hashlib

from config import ModelConfig, ObjectStorageConfig, RecommendationConfig
from model_artifacts import ModelArtifactManager
from object_storage import ObjectStorage


class FakeSystemStore:
    def __init__(self):
        self.latest = {}
        self.recorded = []

    async def record_model_checkpoint(self, model_name, model_version, checkpoint_path, payload=None):
        record = {
            "model_name": model_name,
            "model_version": model_version,
            "checkpoint_path": checkpoint_path,
            "payload": payload or {},
            "created_at": 1.0,
        }
        self.recorded.append(record)
        self.latest[model_name] = record

    async def get_latest_model_checkpoint(self, model_name):
        return self.latest.get(model_name)


def test_persist_ranking_checkpoint_records_model_metadata(tmp_path):
    fake_store = FakeSystemStore()
    ranking_path = tmp_path / "models" / "ranking.pt"
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_path.write_bytes(b"ranking")

    manager = ModelArtifactManager(
        system_store=fake_store,
        object_storage=ObjectStorage(
            ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "downloads"))
        ),
        model_config=ModelConfig(ranking_model_path=str(ranking_path), cache_dir=str(tmp_path / "models")),
        recommendation_config=RecommendationConfig(cf_index_path=str(tmp_path / "cf.faiss")),
    )

    record = asyncio.run(
        manager.persist_ranking_checkpoint(
            local_path=str(ranking_path),
            model_version="ranking-123",
            payload={"trigger": "test"},
        )
    )

    assert record is not None
    assert fake_store.recorded[-1]["checkpoint_path"] == str(ranking_path)
    assert fake_store.recorded[-1]["payload"]["local_cache_path"] == str(ranking_path)
    assert fake_store.recorded[-1]["payload"]["artifact_sha256"] == hashlib.sha256(b"ranking").hexdigest()
    assert (
        fake_store.recorded[-1]["payload"]["artifact_manifest"]["checkpoint"]["sha256"]
        == hashlib.sha256(b"ranking").hexdigest()
    )


def test_sync_latest_two_tower_artifacts_copies_to_local_cache(tmp_path):
    fake_store = FakeSystemStore()
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    checkpoint = remote_dir / "two_tower.pt"
    index = remote_dir / "two_tower.faiss"
    metadata = remote_dir / "two_tower.cf_meta.json"
    checkpoint.write_bytes(b"pt")
    index.write_bytes(b"faiss")
    metadata.write_text("{}", encoding="utf-8")

    fake_store.latest[ModelArtifactManager.TWO_TOWER_MODEL_NAME] = {
        "model_name": ModelArtifactManager.TWO_TOWER_MODEL_NAME,
        "model_version": "two-tower-123",
        "checkpoint_path": str(checkpoint),
        "payload": {
            "cf_index_path": str(index),
            "cf_index_metadata_path": str(metadata),
        },
        "created_at": 1.0,
    }

    local_index = tmp_path / "cache" / "cf_index.faiss"
    manager = ModelArtifactManager(
        system_store=fake_store,
        object_storage=ObjectStorage(
            ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "downloads"))
        ),
        model_config=ModelConfig(
            ranking_model_path=str(tmp_path / "cache" / "ranking.pt"),
            cache_dir=str(tmp_path / "cache"),
        ),
        recommendation_config=RecommendationConfig(cf_index_path=str(local_index)),
    )

    record = asyncio.run(manager.sync_latest_two_tower_artifacts())

    assert record is not None
    assert (tmp_path / "cache" / "cf_index.pt").read_bytes() == b"pt"
    assert local_index.read_bytes() == b"faiss"
    assert local_index.with_suffix(".cf_meta.json").read_text(encoding="utf-8") == "{}"


def test_sync_latest_two_tower_artifacts_rejects_checksum_mismatch(tmp_path):
    fake_store = FakeSystemStore()
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    checkpoint = remote_dir / "two_tower.pt"
    index = remote_dir / "two_tower.faiss"
    metadata = remote_dir / "two_tower.cf_meta.json"
    checkpoint.write_bytes(b"pt")
    index.write_bytes(b"faiss")
    metadata.write_text("{}", encoding="utf-8")

    fake_store.latest[ModelArtifactManager.TWO_TOWER_MODEL_NAME] = {
        "model_name": ModelArtifactManager.TWO_TOWER_MODEL_NAME,
        "model_version": "two-tower-123",
        "checkpoint_path": str(checkpoint),
        "payload": {
            "cf_index_path": str(index),
            "cf_index_metadata_path": str(metadata),
            "artifact_manifest": {
                "checkpoint": {"sha256": "bad"},
                "cf_index": {"sha256": hashlib.sha256(b"faiss").hexdigest()},
                "cf_index_metadata": {"sha256": hashlib.sha256(b"{}").hexdigest()},
            },
        },
        "created_at": 1.0,
    }

    local_index = tmp_path / "cache" / "cf_index.faiss"
    manager = ModelArtifactManager(
        system_store=fake_store,
        object_storage=ObjectStorage(
            ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "downloads"))
        ),
        model_config=ModelConfig(
            ranking_model_path=str(tmp_path / "cache" / "ranking.pt"),
            cache_dir=str(tmp_path / "cache"),
        ),
        recommendation_config=RecommendationConfig(cf_index_path=str(local_index)),
    )

    try:
        asyncio.run(manager.sync_latest_two_tower_artifacts())
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("expected checksum mismatch")

    assert not (tmp_path / "cache" / "cf_index.pt").exists()
    assert not local_index.exists()
