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
