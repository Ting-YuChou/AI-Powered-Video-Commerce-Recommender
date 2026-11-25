import asyncio
import hashlib
import os

import pytest

from video_commerce.common.config import (
    DIRECT_UPLOAD_INCOMING_EXPIRY_SECONDS,
    ObjectStorageConfig,
)
from video_commerce.data_plane.object_storage import (
    ObjectStorage,
    UploadObjectChanged,
)


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
    incoming_key_prefix = storage.build_incoming_object_name(
        content_id="c1", suffix=".mp4"
    )[: -len("c1.mp4")]
    rules = {rule["ID"]: rule for rule in lifecycle["Rules"]}

    raw_rule = rules["raw-uploads-standard-ia-after-30-days"]
    assert raw_rule["Filter"] == {"Prefix": raw_key_prefix}
    assert raw_rule["Transitions"] == [{"Days": 30, "StorageClass": "STANDARD_IA"}]
    # Raw videos are kept forever: only incoming/ may expire, and its prefix
    # does not cover raw/.
    incoming_rule = rules["incoming-direct-uploads-expire-after-1-day"]
    assert incoming_rule["Filter"] == {"Prefix": incoming_key_prefix}
    assert incoming_rule["Expiration"] == {"Days": 1}
    assert incoming_key_prefix.startswith(f"{default_prefix}/")
    assert not raw_key_prefix.startswith(incoming_key_prefix)
    assert all(
        "Expiration" not in rule and "NoncurrentVersionExpiration" not in rule
        for rule in lifecycle["Rules"]
        if rule is not incoming_rule
    )
    # DATA_DIRECT_UPLOAD_* validation keeps URL TTL + grace under this expiry.
    assert incoming_rule["Expiration"]["Days"] * 86400 == (
        DIRECT_UPLOAD_INCOMING_EXPIRY_SECONDS
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


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("uploads", "uploads/incoming/content-1.mp4"),
        ("/team/uploads/", "team/uploads/incoming/content-1.mp4"),
        ("", "incoming/content-1.mp4"),
    ],
)
def test_incoming_object_names_use_incoming_subprefix(tmp_path, prefix, expected):
    storage = ObjectStorage(
        ObjectStorageConfig(
            backend="s3", bucket="bucket", prefix=prefix, download_dir=str(tmp_path)
        )
    )

    assert (
        storage.build_incoming_object_name(content_id="content-1", suffix=".mp4")
        == expected
    )


def _decode_policy(fields):
    import base64
    import json

    return json.loads(base64.b64decode(fields["policy"]))


def test_presigned_direct_upload_pins_key_size_type_and_sse(tmp_path):
    storage = ObjectStorage(
        ObjectStorageConfig(
            backend="s3",
            bucket="bucket",
            endpoint_url="http://minio:9000",
            public_endpoint_url="http://127.0.0.1:9000",
            access_key_id="test-access-key",
            secret_access_key="test-secret-key",
            server_side_encryption="AES256",
            download_dir=str(tmp_path),
        )
    )
    # initialize() builds both clients without any network call.
    asyncio.run(storage.initialize())

    post = storage.presign_direct_upload(
        "s3://bucket/uploads/incoming/content-1.mp4",
        content_type="video/mp4",
        size_bytes=1234,
        expires_in=900,
    )

    # Signed for the browser-reachable endpoint, not the in-cluster one.
    assert post["url"] == "http://127.0.0.1:9000/bucket"
    fields = post["fields"]
    assert fields["key"] == "uploads/incoming/content-1.mp4"
    assert fields["Content-Type"] == "video/mp4"
    assert fields["x-amz-server-side-encryption"] == "AES256"
    assert fields["x-amz-algorithm"] == "AWS4-HMAC-SHA256"
    conditions = _decode_policy(fields)["conditions"]
    assert {"bucket": "bucket"} in conditions
    assert {"key": "uploads/incoming/content-1.mp4"} in conditions
    assert ["content-length-range", 1234, 1234] in conditions
    assert {"Content-Type": "video/mp4"} in conditions
    assert {"x-amz-server-side-encryption": "AES256"} in conditions
    # The in-cluster client keeps the internal endpoint.
    assert storage._client.meta.endpoint_url == "http://minio:9000"


def test_presigned_direct_upload_defaults_to_internal_endpoint(tmp_path):
    storage = ObjectStorage(
        ObjectStorageConfig(
            backend="s3",
            bucket="bucket",
            endpoint_url="http://minio:9000",
            access_key_id="test-access-key",
            secret_access_key="test-secret-key",
            download_dir=str(tmp_path),
        )
    )
    asyncio.run(storage.initialize())

    post = storage.presign_direct_upload(
        "s3://bucket/uploads/incoming/content-1.mp4",
        content_type="video/mp4",
        size_bytes=10,
        expires_in=60,
    )

    assert post["url"] == "http://minio:9000/bucket"
    assert "x-amz-server-side-encryption" not in post["fields"]


def test_bad_public_endpoint_only_fails_when_presigning(tmp_path):
    storage = ObjectStorage(
        ObjectStorageConfig(
            backend="s3",
            bucket="bucket",
            endpoint_url="http://minio:9000",
            public_endpoint_url="127.0.0.1:9000",
            access_key_id="test-access-key",
            secret_access_key="test-secret-key",
            download_dir=str(tmp_path),
        )
    )

    # Services that never presign start normally.
    asyncio.run(storage.initialize())

    with pytest.raises(ValueError):
        storage.presign_direct_upload(
            "s3://bucket/uploads/incoming/c.mp4",
            content_type="video/mp4",
            size_bytes=1,
            expires_in=60,
        )


def test_presigned_direct_upload_requires_s3(tmp_path):
    storage = ObjectStorage(
        ObjectStorageConfig(backend="local", download_dir=str(tmp_path))
    )

    with pytest.raises(RuntimeError):
        storage.presign_direct_upload(
            "s3://bucket/uploads/incoming/c.mp4",
            content_type="video/mp4",
            size_bytes=1,
            expires_in=60,
        )


def _client_error(status, code):
    from botocore.exceptions import ClientError

    return ClientError(
        {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "Operation",
    )


def test_head_upload_object_reports_size_etag_and_type_or_none(tmp_path):
    class Client:
        def head_object(self, *, Bucket, Key):
            assert Bucket == "bucket"
            if Key == "uploads/incoming/missing.mp4":
                raise _client_error(404, "404")
            assert Key == "uploads/incoming/c.mp4"
            return {"ContentLength": 42, "ETag": '"abc"', "ContentType": "video/mp4"}

    storage = ObjectStorage(ObjectStorageConfig(backend="s3", bucket="bucket"))
    storage._client = Client()

    head = asyncio.run(storage.head_upload_object("s3://bucket/uploads/incoming/c.mp4"))
    assert (head.size, head.etag, head.content_type) == (42, '"abc"', "video/mp4")
    assert (
        asyncio.run(
            storage.head_upload_object("s3://bucket/uploads/incoming/missing.mp4")
        )
        is None
    )


def test_head_upload_object_raises_other_errors(tmp_path):
    class Client:
        def head_object(self, **_kwargs):
            raise _client_error(403, "AccessDenied")

    storage = ObjectStorage(ObjectStorageConfig(backend="s3", bucket="bucket"))
    storage._client = Client()

    with pytest.raises(Exception, match="AccessDenied"):
        asyncio.run(storage.head_upload_object("s3://bucket/uploads/incoming/c.mp4"))


def test_copy_upload_object_is_conditional_on_the_checked_etag(tmp_path):
    class Client:
        def __init__(self):
            self.calls = []

        def copy_object(self, **kwargs):
            self.calls.append(kwargs)

    storage = ObjectStorage(
        ObjectStorageConfig(
            backend="s3",
            bucket="bucket",
            server_side_encryption="AES256",
            checksum_algorithm="SHA256",
        )
    )
    client = Client()
    storage._client = client

    uri = asyncio.run(
        storage.copy_upload_object(
            "s3://bucket/uploads/incoming/c.mp4",
            object_name="uploads/raw/c-1.mp4",
            if_match='"abc"',
        )
    )

    assert uri == "s3://bucket/uploads/raw/c-1.mp4"
    assert client.calls == [
        {
            "Bucket": "bucket",
            "Key": "uploads/raw/c-1.mp4",
            "CopySource": {"Bucket": "bucket", "Key": "uploads/incoming/c.mp4"},
            "CopySourceIfMatch": '"abc"',
            "MetadataDirective": "COPY",
            "ServerSideEncryption": "AES256",
            "ChecksumAlgorithm": "SHA256",
        }
    ]


@pytest.mark.parametrize(
    "error",
    [_client_error(412, "PreconditionFailed"), _client_error(404, "NoSuchKey")],
)
def test_copy_upload_object_reports_a_changed_or_removed_source(tmp_path, error):
    class Client:
        def copy_object(self, **_kwargs):
            raise error

    storage = ObjectStorage(ObjectStorageConfig(backend="s3", bucket="bucket"))
    storage._client = Client()

    with pytest.raises(UploadObjectChanged):
        asyncio.run(
            storage.copy_upload_object(
                "s3://bucket/uploads/incoming/c.mp4",
                object_name="uploads/raw/c-1.mp4",
                if_match='"abc"',
            )
        )


def test_copy_upload_object_requires_an_etag(tmp_path):
    storage = ObjectStorage(ObjectStorageConfig(backend="s3", bucket="bucket"))

    with pytest.raises(ValueError):
        asyncio.run(
            storage.copy_upload_object(
                "s3://bucket/uploads/incoming/c.mp4",
                object_name="uploads/raw/c-1.mp4",
                if_match="",
            )
        )


def test_cors_rule_allows_presigned_posts_and_matches_the_s3_schema():
    import json
    from pathlib import Path

    import botocore.session
    from botocore.validate import ParamValidator

    cors = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "deploy"
            / "object-storage"
            / "cors.json"
        ).read_text()
    )
    operation = (
        botocore.session.get_session()
        .get_service_model("s3")
        .operation_model("PutBucketCors")
    )
    report = ParamValidator().validate(
        {"Bucket": "bucket", "CORSConfiguration": cors},
        operation.input_shape,
    )
    assert not report.has_errors(), report.generate_report()

    [rule] = cors["CORSRules"]
    # Presigned POSTs only; the upload-progress listener forces a preflight,
    # which must accept the browser's request headers.
    assert rule["AllowedMethods"] == ["POST"]
    assert rule["AllowedHeaders"] == ["*"]
    assert "*" not in rule["AllowedOrigins"]


def test_empty_endpoints_use_the_aws_endpoint(tmp_path):
    # Helm and .env send unset endpoints as "", which boto3 would reject.
    storage = ObjectStorage(
        ObjectStorageConfig(
            backend="s3",
            bucket="bucket",
            region="eu-west-1",
            endpoint_url="",
            public_endpoint_url="",
            access_key_id="test-access-key",
            secret_access_key="test-secret-key",
            force_path_style=False,
            download_dir=str(tmp_path),
        )
    )
    asyncio.run(storage.initialize())

    assert storage._client.meta.endpoint_url == "https://s3.eu-west-1.amazonaws.com"
    post = storage.presign_direct_upload(
        "s3://bucket/uploads/incoming/c.mp4",
        content_type="video/mp4",
        size_bytes=1,
        expires_in=60,
    )
    assert post["url"] == "https://bucket.s3.eu-west-1.amazonaws.com/"


def test_compose_public_endpoint_keeps_an_explicit_empty_value():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    compose = (root / "docker-compose.yml").read_text()
    # "-", not ":-": an empty value means "presign for the endpoint URL".
    assert (
        "OBJECT_STORAGE_PUBLIC_ENDPOINT_URL: "
        "${OBJECT_STORAGE_PUBLIC_ENDPOINT_URL-http://127.0.0.1:9000}"
    ) in compose
    env_example = (root / ".env.example").read_text()
    assert "\nOBJECT_STORAGE_PUBLIC_ENDPOINT_URL=http://127.0.0.1:9000\n" in env_example
