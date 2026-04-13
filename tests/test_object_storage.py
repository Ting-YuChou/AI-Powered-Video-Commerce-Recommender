import asyncio
import os

from config import ObjectStorageConfig
from object_storage import ObjectStorage


async def _persist_and_materialize(storage: ObjectStorage, path: str):
    persisted = await storage.persist_staged_file(path, object_name="uploads/test.mp4")
    materialized, cleanup = await storage.materialize_for_processing(persisted)
    return persisted, materialized, cleanup


def test_local_object_storage_keeps_local_path(tmp_path):
    staged_file = tmp_path / "video.mp4"
    staged_file.write_bytes(b"video")

    storage = ObjectStorage(
        ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "downloads"))
    )
    persisted, materialized, cleanup = asyncio.run(
        _persist_and_materialize(storage, str(staged_file))
    )

    assert persisted == str(staged_file)
    assert materialized == str(staged_file)
    assert cleanup is False


def test_local_object_storage_deletes_local_file(tmp_path):
    staged_file = tmp_path / "video.mp4"
    staged_file.write_bytes(b"video")

    storage = ObjectStorage(
        ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "downloads"))
    )
    asyncio.run(storage.delete_uploaded_object(str(staged_file)))

    assert not os.path.exists(staged_file)


def test_local_object_storage_builds_artifact_object_name(tmp_path):
    storage = ObjectStorage(
        ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "downloads"))
    )

    object_name = storage.build_artifact_object_name(
        model_name="ranking/model",
        model_version="v1/2",
        filename="checkpoint.pt",
    )

    assert object_name == "uploads/artifacts/ranking-model/v1-2/checkpoint.pt"


def test_local_object_storage_syncs_file_to_fixed_cache_path(tmp_path):
    source_file = tmp_path / "source.pt"
    target_file = tmp_path / "cache" / "ranking.pt"
    source_file.write_bytes(b"checkpoint")

    storage = ObjectStorage(
        ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "downloads"))
    )
    synced = asyncio.run(storage.sync_to_local_path(str(source_file), str(target_file)))

    assert synced == str(target_file)
    assert target_file.read_bytes() == b"checkpoint"
