import asyncio
import hashlib

from video_commerce.common.config import (
    ModelConfig,
    ObjectStorageConfig,
    RecommendationConfig,
)
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.data_plane.object_storage import ObjectStorage


class FakeSystemStore:
    def __init__(self):
        self.latest = {}
        self.recorded = []
        self.catalog_activations = []

    async def record_model_checkpoint(
        self, model_name, model_version, checkpoint_path, payload=None
    ):
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

    async def activate_product_catalog(self, source_version, metadata_map, **kwargs):
        self.catalog_activations.append(
            {
                "source_version": source_version,
                "metadata_map": metadata_map,
                **kwargs,
            }
        )
        return "activation-1"


def test_persist_two_tower_artifacts_activates_catalog_after_checkpoint(tmp_path):
    fake_store = FakeSystemStore()
    checkpoint = tmp_path / "two_tower.pt"
    index = tmp_path / "two_tower.faiss"
    metadata = tmp_path / "two_tower.cf_meta.json"
    checkpoint.write_bytes(b"checkpoint")
    index.write_bytes(b"index")
    metadata.write_text("{}", encoding="utf-8")
    manager = ModelArtifactManager(
        system_store=fake_store,
        object_storage=ObjectStorage(
            ObjectStorageConfig(
                backend="local", download_dir=str(tmp_path / "downloads")
            )
        ),
        model_config=ModelConfig(cache_dir=str(tmp_path / "cache")),
        recommendation_config=RecommendationConfig(cf_index_path=str(index)),
    )

    asyncio.run(
        manager.persist_two_tower_artifacts(
            checkpoint_path=str(checkpoint),
            index_path=str(index),
            metadata_path=str(metadata),
            model_version="catalog-v1",
            catalog_metadata={"product-1": {"price": 10.0}},
        )
    )

    assert fake_store.recorded[-1]["model_version"] == "catalog-v1"
    assert fake_store.catalog_activations == [
        {
            "source_version": "catalog-v1",
            "metadata_map": {"product-1": {"price": 10.0}},
        }
    ]


def test_persist_ranking_checkpoint_records_model_metadata(tmp_path):
    fake_store = FakeSystemStore()
    ranking_path = tmp_path / "models" / "ranking.pt"
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_path.write_bytes(b"ranking")

    manager = ModelArtifactManager(
        system_store=fake_store,
        object_storage=ObjectStorage(
            ObjectStorageConfig(
                backend="local", download_dir=str(tmp_path / "downloads")
            )
        ),
        model_config=ModelConfig(
            ranking_model_path=str(ranking_path), cache_dir=str(tmp_path / "models")
        ),
        recommendation_config=RecommendationConfig(
            cf_index_path=str(tmp_path / "cf.faiss")
        ),
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
    assert (
        fake_store.recorded[-1]["payload"]["artifact_sha256"]
        == hashlib.sha256(b"ranking").hexdigest()
    )
    assert (
        fake_store.recorded[-1]["payload"]["artifact_manifest"]["checkpoint"]["sha256"]
        == hashlib.sha256(b"ranking").hexdigest()
    )


def test_persist_ranking_shadow_checkpoint_uses_non_activating_model_namespace(tmp_path):
    fake_store = FakeSystemStore()
    shadow_path = tmp_path / "ranking.pit-shadow.pt"
    shadow_path.write_bytes(b"shadow")
    manager = ModelArtifactManager(
        system_store=fake_store,
        object_storage=ObjectStorage(
            ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "downloads"))
        ),
        model_config=ModelConfig(ranking_model_path=str(tmp_path / "ranking.pt")),
        recommendation_config=RecommendationConfig(
            cf_index_path=str(tmp_path / "cf.faiss")
        ),
    )

    record = asyncio.run(
        manager.persist_ranking_shadow_checkpoint(
            local_path=str(shadow_path),
            model_version="ranking-shadow-1",
            payload={"shadow": True},
        )
    )

    assert record.model_name == ModelArtifactManager.RANKING_SHADOW_MODEL_NAME
    assert fake_store.recorded[-1]["model_name"] == "ranking_model_pit_shadow"
    assert fake_store.latest.get(ModelArtifactManager.RANKING_MODEL_NAME) is None


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
            ObjectStorageConfig(
                backend="local", download_dir=str(tmp_path / "downloads")
            )
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


def test_sync_latest_two_tower_artifacts_removes_undeclared_optional_sidecars(tmp_path):
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
            ObjectStorageConfig(
                backend="local", download_dir=str(tmp_path / "downloads")
            )
        ),
        model_config=ModelConfig(
            ranking_model_path=str(tmp_path / "cache" / "ranking.pt"),
            cache_dir=str(tmp_path / "cache"),
        ),
        recommendation_config=RecommendationConfig(cf_index_path=str(local_index)),
    )
    sidecar_path = local_index.with_suffix(".cf_embeddings.npz")
    adapter_path = local_index.with_suffix(".cf_adapter.npz")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_bytes(b"stale-sidecar")
    adapter_path.write_bytes(b"stale-adapter")

    record = asyncio.run(manager.sync_latest_two_tower_artifacts())

    assert record is not None
    assert not sidecar_path.exists()
    assert not adapter_path.exists()


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
            ObjectStorageConfig(
                backend="local", download_dir=str(tmp_path / "downloads")
            )
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


def test_persist_sasrec_artifacts_records_manifest(tmp_path):
    fake_store = FakeSystemStore()
    checkpoint = tmp_path / "models" / "sasrec.pt"
    vocab = tmp_path / "models" / "sasrec_vocab.json"
    metadata = tmp_path / "models" / "sasrec_metadata.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    vocab.write_text('{"product_to_id": {}}', encoding="utf-8")
    metadata.write_text('{"model_version": "sasrec-123"}', encoding="utf-8")

    manager = ModelArtifactManager(
        system_store=fake_store,
        object_storage=ObjectStorage(
            ObjectStorageConfig(
                backend="local", download_dir=str(tmp_path / "downloads")
            )
        ),
        model_config=ModelConfig(cache_dir=str(tmp_path / "cache")),
        recommendation_config=RecommendationConfig(),
    )

    record = asyncio.run(
        manager.persist_sasrec_artifacts(
            checkpoint_path=str(checkpoint),
            vocab_path=str(vocab),
            metadata_path=str(metadata),
            model_version="sasrec-123",
            payload={"trigger": "test"},
        )
    )

    assert record is not None
    assert (
        fake_store.recorded[-1]["model_name"] == ModelArtifactManager.SASREC_MODEL_NAME
    )
    manifest = fake_store.recorded[-1]["payload"]["artifact_manifest"]
    assert manifest["checkpoint"]["sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert manifest["vocab"]["local_cache_path"].endswith("sasrec_vocab.json")
    assert manifest["metadata"]["local_cache_path"].endswith("sasrec_metadata.json")


def test_sync_latest_sasrec_artifacts_copies_to_local_cache(tmp_path):
    fake_store = FakeSystemStore()
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    checkpoint = remote_dir / "sasrec.pt"
    vocab = remote_dir / "sasrec_vocab.json"
    metadata = remote_dir / "sasrec_metadata.json"
    checkpoint.write_bytes(b"checkpoint")
    vocab.write_text("{}", encoding="utf-8")
    metadata.write_text("{}", encoding="utf-8")

    fake_store.latest[ModelArtifactManager.SASREC_MODEL_NAME] = {
        "model_name": ModelArtifactManager.SASREC_MODEL_NAME,
        "model_version": "sasrec-123",
        "checkpoint_path": str(checkpoint),
        "payload": {
            "vocab_path": str(vocab),
            "metadata_path": str(metadata),
        },
        "created_at": 1.0,
    }

    manager = ModelArtifactManager(
        system_store=fake_store,
        object_storage=ObjectStorage(
            ObjectStorageConfig(
                backend="local", download_dir=str(tmp_path / "downloads")
            )
        ),
        model_config=ModelConfig(cache_dir=str(tmp_path / "cache")),
        recommendation_config=RecommendationConfig(),
    )

    record = asyncio.run(manager.sync_latest_sasrec_artifacts())

    assert record is not None
    assert (tmp_path / "cache" / "sasrec_model.pt").read_bytes() == b"checkpoint"
    assert (tmp_path / "cache" / "sasrec_vocab.json").read_text(
        encoding="utf-8"
    ) == "{}"
    assert (tmp_path / "cache" / "sasrec_metadata.json").read_text(
        encoding="utf-8"
    ) == "{}"


def test_sync_latest_sasrec_artifacts_rejects_checksum_mismatch(tmp_path):
    fake_store = FakeSystemStore()
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    checkpoint = remote_dir / "sasrec.pt"
    vocab = remote_dir / "sasrec_vocab.json"
    metadata = remote_dir / "sasrec_metadata.json"
    checkpoint.write_bytes(b"checkpoint")
    vocab.write_text("{}", encoding="utf-8")
    metadata.write_text("{}", encoding="utf-8")

    fake_store.latest[ModelArtifactManager.SASREC_MODEL_NAME] = {
        "model_name": ModelArtifactManager.SASREC_MODEL_NAME,
        "model_version": "sasrec-123",
        "checkpoint_path": str(checkpoint),
        "payload": {
            "vocab_path": str(vocab),
            "metadata_path": str(metadata),
            "artifact_manifest": {
                "checkpoint": {"sha256": "bad"},
                "vocab": {"sha256": hashlib.sha256(b"{}").hexdigest()},
                "metadata": {"sha256": hashlib.sha256(b"{}").hexdigest()},
            },
        },
        "created_at": 1.0,
    }

    manager = ModelArtifactManager(
        system_store=fake_store,
        object_storage=ObjectStorage(
            ObjectStorageConfig(
                backend="local", download_dir=str(tmp_path / "downloads")
            )
        ),
        model_config=ModelConfig(cache_dir=str(tmp_path / "cache")),
        recommendation_config=RecommendationConfig(),
    )

    try:
        asyncio.run(manager.sync_latest_sasrec_artifacts())
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("expected checksum mismatch")

    assert not (tmp_path / "cache" / "sasrec_model.pt").exists()


def test_persist_swing_itemcf_artifact_records_manifest(tmp_path):
    fake_store = FakeSystemStore()
    index = tmp_path / "models" / "swing_itemcf.json.gz"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_bytes(b"swing-index")

    manager = ModelArtifactManager(
        system_store=fake_store,
        object_storage=ObjectStorage(
            ObjectStorageConfig(
                backend="local", download_dir=str(tmp_path / "downloads")
            )
        ),
        model_config=ModelConfig(cache_dir=str(tmp_path / "cache")),
        recommendation_config=RecommendationConfig(
            swing_itemcf_index_path=str(tmp_path / "cache" / "swing_itemcf.json.gz")
        ),
    )

    record = asyncio.run(
        manager.persist_swing_itemcf_artifact(
            index_path=str(index),
            model_version="swing-123",
            payload={"training_user_count": 2},
        )
    )

    assert record is not None
    assert (
        fake_store.recorded[-1]["model_name"]
        == ModelArtifactManager.SWING_ITEMCF_MODEL_NAME
    )
    manifest = fake_store.recorded[-1]["payload"]["artifact_manifest"]
    assert manifest["index"]["sha256"] == hashlib.sha256(b"swing-index").hexdigest()
    assert manifest["index"]["local_cache_path"].endswith("swing_itemcf.json.gz")
    assert fake_store.recorded[-1]["payload"]["training_user_count"] == 2


def test_sync_latest_swing_itemcf_artifact_copies_to_local_cache(tmp_path):
    fake_store = FakeSystemStore()
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    index = remote_dir / "swing_itemcf.json.gz"
    index.write_bytes(b"swing-index")

    fake_store.latest[ModelArtifactManager.SWING_ITEMCF_MODEL_NAME] = {
        "model_name": ModelArtifactManager.SWING_ITEMCF_MODEL_NAME,
        "model_version": "swing-123",
        "checkpoint_path": str(index),
        "payload": {"index_path": str(index)},
        "created_at": 1.0,
    }

    local_index = tmp_path / "cache" / "swing_itemcf.json.gz"
    manager = ModelArtifactManager(
        system_store=fake_store,
        object_storage=ObjectStorage(
            ObjectStorageConfig(
                backend="local", download_dir=str(tmp_path / "downloads")
            )
        ),
        model_config=ModelConfig(cache_dir=str(tmp_path / "cache")),
        recommendation_config=RecommendationConfig(
            swing_itemcf_index_path=str(local_index)
        ),
    )

    record = asyncio.run(manager.sync_latest_swing_itemcf_artifact())

    assert record is not None
    assert local_index.read_bytes() == b"swing-index"


def test_sync_latest_swing_itemcf_artifact_rejects_checksum_mismatch(tmp_path):
    fake_store = FakeSystemStore()
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    index = remote_dir / "swing_itemcf.json.gz"
    index.write_bytes(b"swing-index")

    fake_store.latest[ModelArtifactManager.SWING_ITEMCF_MODEL_NAME] = {
        "model_name": ModelArtifactManager.SWING_ITEMCF_MODEL_NAME,
        "model_version": "swing-123",
        "checkpoint_path": str(index),
        "payload": {
            "index_path": str(index),
            "artifact_manifest": {"index": {"sha256": "bad"}},
        },
        "created_at": 1.0,
    }

    local_index = tmp_path / "cache" / "swing_itemcf.json.gz"
    manager = ModelArtifactManager(
        system_store=fake_store,
        object_storage=ObjectStorage(
            ObjectStorageConfig(
                backend="local", download_dir=str(tmp_path / "downloads")
            )
        ),
        model_config=ModelConfig(cache_dir=str(tmp_path / "cache")),
        recommendation_config=RecommendationConfig(
            swing_itemcf_index_path=str(local_index)
        ),
    )

    try:
        asyncio.run(manager.sync_latest_swing_itemcf_artifact())
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("expected checksum mismatch")

    assert not local_index.exists()
