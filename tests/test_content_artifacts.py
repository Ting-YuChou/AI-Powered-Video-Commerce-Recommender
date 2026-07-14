import asyncio
import hashlib
import json
import pytest

from video_commerce.common.config import ObjectStorageConfig
from video_commerce.common.models import ContentFeatures
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.ml.content_artifacts import (
    canonical_content_feature_bytes,
    content_artifact_reference,
    load_content_feature_artifact,
    persist_content_features,
    publish_content_feature_artifact,
)


def test_publish_content_feature_artifact_is_canonical_and_attaches_reference(tmp_path):
    storage = ObjectStorage(
        ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "objects"))
    )
    features = ContentFeatures(
        content_id="content-1",
        visual_embedding=[0.1, 0.2],
        frame_embeddings=[[0.1, 0.2]],
        frame_timestamps_seconds=[1.25],
        text_embedding_model="intfloat/multilingual-e5-small",
        text_embedding_revision="pinned",
    )

    published = asyncio.run(publish_content_feature_artifact(storage, features))

    assert published.artifact_schema_version == "content_feature_artifact_v2"
    assert published.artifact_sha256
    assert published.artifact_uri
    payload_bytes = open(published.artifact_uri, "rb").read()
    assert hashlib.sha256(payload_bytes).hexdigest() == published.artifact_sha256
    payload = json.loads(payload_bytes)
    assert payload["schema_version"] == "content_feature_artifact_v2"
    assert payload["features"]["content_id"] == "content-1"
    assert payload["features"]["artifact_uri"] is None
    assert payload["features"]["artifact_sha256"] is None


def test_persist_content_features_publishes_durable_artifact_before_current_pointer(
    tmp_path,
):
    events = []

    class SystemStore:
        async def upsert_content_feature_artifact(
            self, content_id, features, *, schema_version
        ):
            events.append(
                ("postgres", content_id, schema_version, features["artifact_sha256"])
            )

    class FeatureStore:
        async def store_content_features(self, content_id, features):
            events.append(("redis", content_id, features.artifact_sha256))

    storage = ObjectStorage(
        ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "objects"))
    )
    features = ContentFeatures(content_id="content-1", visual_embedding=[1.0])

    published = asyncio.run(
        persist_content_features(
            features,
            object_storage=storage,
            system_store=SystemStore(),
            feature_store=FeatureStore(),
        )
    )

    assert published.artifact_sha256
    assert [event[0] for event in events] == ["postgres", "redis"]
    assert events[0][3] == events[1][2] == published.artifact_sha256


def test_content_artifact_reference_contains_only_immutable_lineage():
    features = ContentFeatures(
        content_id="content-1",
        visual_embedding=[1.0],
        artifact_uri="s3://bucket/content/hash.json",
        artifact_sha256="abc123",
        artifact_schema_version="content_feature_artifact_v2",
        artifact_created_at=123.0,
    )

    assert content_artifact_reference(features) == {
        "content_id": "content-1",
        "uri": "s3://bucket/content/hash.json",
        "sha256": "abc123",
        "schema_version": "content_feature_artifact_v2",
        "created_at": 123.0,
    }
    assert (
        content_artifact_reference(
            ContentFeatures(content_id="missing", visual_embedding=[1.0])
        )
        is None
    )


def test_load_content_feature_artifact_verifies_checksum_and_schema(tmp_path):
    features = ContentFeatures(
        content_id="content-1", visual_embedding=[1.0], created_at=10.0
    )
    payload = canonical_content_feature_bytes(features)
    path = tmp_path / "artifact.json"
    path.write_bytes(payload)
    reference = {
        "content_id": "content-1",
        "uri": str(path),
        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
        "schema_version": "content_feature_artifact_v2",
        "created_at": 10.0,
    }

    storage = ObjectStorage(
        ObjectStorageConfig(backend="local", download_dir=str(tmp_path))
    )
    loaded = asyncio.run(load_content_feature_artifact(storage, reference))
    assert loaded.content_id == "content-1"
    reference["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        asyncio.run(load_content_feature_artifact(storage, reference))
