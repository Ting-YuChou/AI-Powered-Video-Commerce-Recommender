import asyncio
import hashlib
import os

import pytest

from video_commerce.common.config import ObjectStorageConfig
from video_commerce.data_plane.object_storage import ObjectStorage


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


def test_local_object_listing_includes_size_and_last_modified(tmp_path):
    staged_file = tmp_path / "video.mp4"
    staged_file.write_bytes(b"video")
    storage = ObjectStorage(
        ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "downloads"))
    )

    objects = asyncio.run(storage.list_storage_objects(str(tmp_path)))

    assert len(objects) == 1
    assert objects[0].uri == str(staged_file)
    assert objects[0].size == 5
    assert objects[0].last_modified > 0


def test_local_object_storage_health_requires_writable_directory(tmp_path):
    storage = ObjectStorage(ObjectStorageConfig(backend="local"))

    healthy = asyncio.run(storage.health_check(local_path=str(tmp_path)))
    missing = asyncio.run(storage.health_check(local_path=str(tmp_path / "missing")))

    assert healthy["status"] == "healthy"
    assert missing["status"] == "unhealthy"


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


def test_local_object_storage_rejects_checksum_mismatch_without_replacing_target(
    tmp_path,
):
    source_file = tmp_path / "source.pt"
    target_file = tmp_path / "cache" / "ranking.pt"
    source_file.write_bytes(b"new checkpoint")
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_bytes(b"old checkpoint")

    storage = ObjectStorage(
        ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "downloads"))
    )

    with pytest.raises(ValueError):
        asyncio.run(
            storage.sync_to_local_path(
                str(source_file),
                str(target_file),
                expected_sha256=hashlib.sha256(b"different").hexdigest(),
            )
        )

    assert target_file.read_bytes() == b"old checkpoint"


def test_local_object_storage_persists_content_addressed_bytes_idempotently(tmp_path):
    storage = ObjectStorage(
        ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "objects"))
    )

    uri = asyncio.run(
        storage.persist_immutable_bytes(
            b'{"schema":"v2"}',
            object_name="content-features/content-1/abc.json",
            content_type="application/json",
        )
    )
    second_uri = asyncio.run(
        storage.persist_immutable_bytes(
            b'{"schema":"v2"}',
            object_name="content-features/content-1/abc.json",
            content_type="application/json",
        )
    )

    assert uri == second_uri
    assert uri == str(tmp_path / "objects/content-features/content-1/abc.json")
    assert (
        tmp_path / "objects/content-features/content-1/abc.json"
    ).read_bytes() == b'{"schema":"v2"}'


def test_local_object_storage_rejects_conflicting_immutable_bytes(tmp_path):
    storage = ObjectStorage(
        ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "objects"))
    )
    asyncio.run(
        storage.persist_immutable_bytes(
            b"first",
            object_name="content-features/content-1/hash.json",
        )
    )

    with pytest.raises(FileExistsError, match="different bytes"):
        asyncio.run(
            storage.persist_immutable_bytes(
                b"second",
                object_name="content-features/content-1/hash.json",
            )
        )


def test_remote_object_storage_cleans_temp_file_on_download_failure(tmp_path):
    class FailingClient:
        def download_file(self, bucket, key, filename):
            assert bucket == "bucket"
            assert key == "uploads/video.mp4"
            assert os.path.exists(filename)
            raise RuntimeError("download failed")

    download_dir = tmp_path / "downloads"
    storage = ObjectStorage(
        ObjectStorageConfig(
            backend="s3",
            bucket="bucket",
            download_dir=str(download_dir),
        )
    )
    storage._client = FailingClient()

    with pytest.raises(RuntimeError):
        asyncio.run(storage.materialize_for_processing("s3://bucket/uploads/video.mp4"))

    assert list(download_dir.iterdir()) == []


def test_remote_create_only_upload_injects_atomic_if_none_match_header(tmp_path):
    class Events:
        def __init__(self):
            self.handlers = {}

        def register_first(self, name, handler, *, unique_id):
            self.handlers[(name, unique_id)] = handler

        def unregister(self, name, *, unique_id):
            self.handlers.pop((name, unique_id))

    class Meta:
        def __init__(self):
            self.events = Events()

    class Client:
        def __init__(self):
            self.meta = Meta()
            self.headers = None

        def put_object(self, **kwargs):
            params = {"headers": {}}
            for (name, _), handler in list(self.meta.events.handlers.items()):
                if name == "before-call.s3.PutObject":
                    handler(params=params)
            self.headers = params["headers"]
            assert kwargs["Body"].read() == b"manifest"

    staged = tmp_path / "manifest.json"
    staged.write_bytes(b"manifest")
    storage = ObjectStorage(ObjectStorageConfig(backend="s3", bucket="bucket"))
    client = Client()
    storage._client = client

    uri = asyncio.run(
        storage.persist_staged_file_create_only(
            str(staged), object_name="training/run/manifest.json"
        )
    )

    assert uri == "s3://bucket/training/run/manifest.json"
    assert client.headers == {"If-None-Match": "*"}
    assert client.meta.events.handlers == {}


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("uploads", "uploads/raw/content-1.mp4"),
        ("/team/uploads/", "team/uploads/raw/content-1.mp4"),
        ("", "raw/content-1.mp4"),
    ],
)
def test_upload_object_names_use_raw_subprefix(tmp_path, prefix, expected):
    storage = ObjectStorage(
        ObjectStorageConfig(
            backend="s3", bucket="bucket", prefix=prefix, download_dir=str(tmp_path)
        )
    )

    assert storage.build_object_name(content_id="content-1", suffix=".mp4") == expected


def test_raw_subprefix_leaves_artifact_and_feature_keys_unchanged(tmp_path):
    storage = ObjectStorage(
        ObjectStorageConfig(
            backend="s3", bucket="bucket", prefix="uploads", download_dir=str(tmp_path)
        )
    )

    assert (
        storage.build_artifact_object_name(
            model_name="ranking", model_version="v1", filename="model.pt"
        )
        == "uploads/artifacts/ranking/v1/model.pt"
    )
    assert (
        storage.build_content_feature_object_name(content_id="c1", sha256="abc")
        == "uploads/content-features/c1/abc.json"
    )


@pytest.mark.parametrize(
    "backend,prefix,expected",
    [
        ("s3", "uploads", ["s3://bucket/uploads/raw", "s3://bucket/uploads"]),
        ("s3", "/uploads/", ["s3://bucket/uploads/raw", "s3://bucket/uploads"]),
        # The bucket root and a root-level raw/ may hold other systems' objects.
        ("s3", "", []),
        ("s3", "/", []),
        ("local", "uploads", ["/data/uploads"]),
    ],
)
def test_upload_reconcile_prefixes(tmp_path, backend, prefix, expected):
    storage = ObjectStorage(
        ObjectStorageConfig(
            backend=backend, bucket="bucket", prefix=prefix, download_dir=str(tmp_path)
        )
    )

    assert storage.upload_reconcile_prefixes(local_upload_dir="/data/uploads") == expected


def test_remote_reconcile_listing_covers_raw_and_legacy_uploads(tmp_path):
    calls = []

    class Client:
        def list_objects_v2(self, **kwargs):
            calls.append(kwargs)
            if kwargs["Prefix"] == "uploads/raw/":
                return {
                    "Contents": [{"Key": "uploads/raw/new.mp4", "Size": 1}],
                    "IsTruncated": False,
                }
            # The delimiter folds raw/ and artifacts/ into CommonPrefixes.
            return {
                "Contents": [{"Key": "uploads/legacy.mp4", "Size": 1}],
                "CommonPrefixes": [
                    {"Prefix": "uploads/raw/"},
                    {"Prefix": "uploads/artifacts/"},
                ],
                "IsTruncated": False,
            }

    storage = ObjectStorage(
        ObjectStorageConfig(
            backend="s3", bucket="bucket", prefix="uploads", download_dir=str(tmp_path)
        )
    )
    storage._client = Client()

    async def collect():
        return [
            item.uri
            for prefix in storage.upload_reconcile_prefixes(local_upload_dir="")
            async for page in storage.iter_upload_object_pages(prefix)
            for item in page
        ]

    assert asyncio.run(collect()) == [
        "s3://bucket/uploads/raw/new.mp4",
        "s3://bucket/uploads/legacy.mp4",
    ]
    assert [(call["Prefix"], call["Delimiter"]) for call in calls] == [
        ("uploads/raw/", "/"),
        ("uploads/", "/"),
    ]


def test_lifecycle_rules_match_upload_layout_and_s3_schema(tmp_path):
    import json
    from pathlib import Path

    import botocore.session
    from botocore.validate import ParamValidator

    lifecycle = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "deploy"
            / "object-storage"
            / "lifecycle.json"
        ).read_text()
    )
    operation = (
        botocore.session.get_session()
        .get_service_model("s3")
        .operation_model("PutBucketLifecycleConfiguration")
    )
    report = ParamValidator().validate(
        {"Bucket": "bucket", "LifecycleConfiguration": lifecycle},
        operation.input_shape,
    )
    assert not report.has_errors(), report.generate_report()

    default_prefix = ObjectStorageConfig.__fields__["prefix"].default
    storage = ObjectStorage(
        ObjectStorageConfig(
            backend="s3",
            bucket="bucket",
            prefix=default_prefix,
            download_dir=str(tmp_path),
        )
    )
    raw_key_prefix = storage.build_object_name(content_id="c1", suffix=".mp4")[
        : -len("c1.mp4")
    ]
    rules = {rule["ID"]: rule for rule in lifecycle["Rules"]}

    raw_rule = rules["raw-uploads-standard-ia-after-30-days"]
    assert raw_rule["Filter"] == {"Prefix": raw_key_prefix}
    assert raw_rule["Transitions"] == [{"Days": 30, "StorageClass": "STANDARD_IA"}]
    # Raw videos are kept forever: no rule may expire anything.
    assert all(
        "Expiration" not in rule and "NoncurrentVersionExpiration" not in rule
        for rule in lifecycle["Rules"]
    )
    multipart_rule = rules["abort-incomplete-multipart-uploads-after-7-days"]
    assert multipart_rule["Filter"] == {"Prefix": ""}
    assert multipart_rule["AbortIncompleteMultipartUpload"] == {
        "DaysAfterInitiation": 7
    }


def test_empty_s3_prefix_disables_orphan_cleanup_with_warning(tmp_path, caplog):
    storage = ObjectStorage(
        ObjectStorageConfig(
            backend="s3", bucket="bucket", prefix="", download_dir=str(tmp_path)
        )
    )

    with caplog.at_level("WARNING"):
        assert storage.upload_reconcile_prefixes(local_upload_dir="/data") == []

    assert "object_storage_orphan_reconcile_disabled" in caplog.text
