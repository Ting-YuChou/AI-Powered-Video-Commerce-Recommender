import io
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from video_commerce.common.config import Config
from video_commerce.services.gateway import api as gateway_api
from video_commerce.services.gateway.api import (
    _content_upload_dependency_error,
    _proxy_json_request,
    _stage_and_publish_content_task,
    content_status,
    stream_upload_to_temp_file,
    validate_upload_file,
)


@pytest.mark.asyncio
async def test_content_upload_dependencies_require_healthy_object_storage(monkeypatch):
    class Store:
        async def health_check(self):
            return SimpleNamespace(status="healthy")

    class Kafka:
        async def health_check(self):
            return {"producer": {"connected": True}}

    class Storage:
        async def health_check(self, *, local_path):
            assert local_path == "/uploads"
            return {"status": "unhealthy"}

    monkeypatch.setattr(gateway_api, "system_store", Store())
    monkeypatch.setattr(gateway_api, "kafka_manager", Kafka())
    monkeypatch.setattr(gateway_api, "object_storage", Storage())
    runtime = SimpleNamespace(
        config=SimpleNamespace(data_config=SimpleNamespace(upload_dir="/uploads"))
    )

    assert (
        await _content_upload_dependency_error(runtime)
        == "Content upload object storage is unavailable"
    )


def test_validate_upload_file_rejects_bad_extension():
    config = Config()
    file = UploadFile(
        file=io.BytesIO(b"abc"),
        filename="video.exe",
        headers=Headers({"content-type": "video/mp4"}),
    )

    with pytest.raises(HTTPException) as exc:
        validate_upload_file(file, config)

    assert exc.value.status_code == 400


def test_validate_upload_file_rejects_bad_mime():
    config = Config()
    file = UploadFile(
        file=io.BytesIO(b"abc"),
        filename="video.mp4",
        headers=Headers({"content-type": "application/octet-stream"}),
    )

    with pytest.raises(HTTPException) as exc:
        validate_upload_file(file, config)

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_stream_upload_to_temp_file_enforces_size_limit(tmp_path):
    file = UploadFile(
        file=io.BytesIO(b"a" * 20),
        filename="video.mp4",
        headers=Headers({"content-type": "video/mp4"}),
    )

    with pytest.raises(HTTPException) as exc:
        await stream_upload_to_temp_file(
            file=file,
            upload_dir=str(tmp_path),
            suffix=".mp4",
            max_size=10,
            chunk_size=4,
        )

    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_stream_upload_to_temp_file_streams_successfully(tmp_path):
    content = b"video-bytes"
    file = UploadFile(
        file=io.BytesIO(content),
        filename="video.mp4",
        headers=Headers({"content-type": "video/mp4"}),
    )

    path, size_bytes = await stream_upload_to_temp_file(
        file=file,
        upload_dir=str(tmp_path),
        suffix=".mp4",
        max_size=1024,
        chunk_size=4,
    )

    assert size_bytes == len(content)
    with open(path, "rb") as handle:
        assert handle.read() == content


@pytest.mark.asyncio
async def test_gateway_proxy_preserves_retry_after(monkeypatch):
    class FakePool:
        async def post(self, path, content, headers):
            return httpx.Response(
                429,
                content=b'{"detail":"ranking_overloaded","retry_after_seconds":1}',
                headers={"content-type": "application/json", "retry-after": "1"},
            )

    class FakeRequest:
        state = SimpleNamespace(request_id="req-1")

        async def body(self):
            return b"{}"

    monkeypatch.setattr(
        gateway_api.app.state.runtime,
        "config",
        SimpleNamespace(
            monitoring_config=SimpleNamespace(request_id_header="X-Request-ID"),
            security_config=SimpleNamespace(
                internal_service_header="X-Internal-Service-Key",
                internal_service_key="",
            ),
        ),
    )

    response = await _proxy_json_request(
        proxy_pool=FakePool(),
        path="/api/recommendations",
        target="recommendation-service",
        payload={},
        request=FakeRequest(),
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"


class DurableContentStore:
    def __init__(self):
        self.task = None
        self.published = []
        self.failed = []

    async def stage_content_task(self, task, **_job):
        self.task = task
        return "inserted"

    async def claim_content_task_outbox_event(
        self, event_id, *, worker_id, lease_seconds
    ):
        assert event_id == self.task.event_id
        assert worker_id.startswith("gateway-")
        assert lease_seconds == 30
        return dict(self.task.payload)

    async def mark_content_task_outbox_published(self, event_id, *, worker_id):
        self.published.append((event_id, worker_id))
        return True

    async def mark_content_task_outbox_failed(self, event_id, error, *, worker_id):
        self.failed.append((event_id, error, worker_id))
        return True


class DurableContentKafka:
    def __init__(self, acknowledged):
        self.acknowledged = acknowledged
        self.events = []

    async def publish_video_processing_task_payload(self, event):
        self.events.append(event)
        return self.acknowledged


@pytest.mark.asyncio
async def test_gateway_content_upload_marks_queued_only_after_kafka_ack():
    store = DurableContentStore()
    kafka = DurableContentKafka(acknowledged=True)

    task, acknowledged = await _stage_and_publish_content_task(
        system_store=store,
        kafka_manager=kafka,
        content_id="content-1",
        pipeline_version="pipeline-v1",
        storage_path="s3://uploads/content-1.mp4",
        filename="clip.mp4",
        user_id="user-1",
        priority="normal",
        request_id="request-1",
        occurred_at=100.0,
        lease_seconds=30,
    )

    assert acknowledged is True
    assert kafka.events == [task.payload]
    assert store.published[0][0] == task.event_id
    assert store.failed == []


@pytest.mark.asyncio
async def test_gateway_content_upload_keeps_outbox_pending_without_kafka_ack():
    store = DurableContentStore()
    kafka = DurableContentKafka(acknowledged=False)

    task, acknowledged = await _stage_and_publish_content_task(
        system_store=store,
        kafka_manager=kafka,
        content_id="content-1",
        pipeline_version="pipeline-v1",
        storage_path="s3://uploads/content-1.mp4",
        filename="clip.mp4",
        user_id="user-1",
        priority="normal",
        request_id="request-1",
        occurred_at=100.0,
        lease_seconds=30,
    )

    assert acknowledged is False
    assert store.published == []
    assert store.failed[0][0] == task.event_id


@pytest.mark.asyncio
async def test_gateway_keeps_durable_task_when_inline_claim_fails():
    class ClaimFailureStore(DurableContentStore):
        async def claim_content_task_outbox_event(self, *_args, **_kwargs):
            raise RuntimeError("database unavailable after commit")

        async def mark_content_task_outbox_failed(self, *_args, **_kwargs):
            raise RuntimeError("database still unavailable")

    store = ClaimFailureStore()
    task, acknowledged = await _stage_and_publish_content_task(
        system_store=store,
        kafka_manager=DurableContentKafka(acknowledged=True),
        content_id="content-1",
        pipeline_version="pipeline-v1",
        storage_path="s3://uploads/content-1.mp4",
        filename="clip.mp4",
        user_id="user-1",
        priority="normal",
        request_id="request-1",
        occurred_at=100.0,
        lease_seconds=30,
    )

    assert acknowledged is False
    assert store.task == task


@pytest.mark.asyncio
async def test_content_status_prefers_authoritative_postgres_job(monkeypatch):
    class Store:
        async def get_content_job(self, content_id):
            return {
                "content_id": content_id,
                "status": "pending_publish",
                "updated_at": 123.0,
            }

    class FeatureStore:
        async def get_content_status(self, _content_id):
            return "queued"

        async def get_content_processed_time(self, _content_id):
            return 456.0

    monkeypatch.setattr(gateway_api, "system_store", Store())
    monkeypatch.setattr(gateway_api, "feature_store", FeatureStore())

    response = await content_status("content-1")

    assert response["status"] == "pending_publish"
    assert response["processed_at"] == 123.0


class UploadObservability:
    def __init__(self):
        self.uploads = []

    def record_content_upload(self, priority, status):
        self.uploads.append((priority, status))


class UploadStore:
    """In-memory content_jobs/outbox with the gateway-facing store contract."""

    def __init__(self):
        self.jobs = {}
        self.staged = []
        self.stage_error = None
        self.lookup_error = None
        self.published = []

    async def get_content_job_by_idempotency_key(self, *, user_id, idempotency_key):
        for job in self.jobs.values():
            if (
                job["user_id"] == user_id
                and job["idempotency_key"] == idempotency_key
            ):
                return dict(job)
        return None

    async def get_content_job(self, content_id):
        if self.lookup_error is not None:
            raise self.lookup_error
        job = self.jobs.get(content_id)
        return dict(job) if job else None

    async def stage_content_task(self, task, **job):
        self.staged.append((task, job))
        if self.stage_error is not None:
            error, commit = self.stage_error
            if commit:
                self._commit(task, job)
            raise error
        self._commit(task, job)
        return "inserted"

    def _commit(self, task, job):
        self.jobs[task.content_id] = {
            "content_id": task.content_id,
            "filename": job["filename"],
            "storage_path": job["storage_path"],
            "user_id": job["user_id"],
            "status": "pending_publish",
            "task_event_id": task.event_id,
            "idempotency_key": job.get("idempotency_key"),
            "request_fingerprint": job.get("request_fingerprint"),
        }

    async def claim_content_task_outbox_event(self, event_id, **_kwargs):
        return {"event_id": event_id}

    async def mark_content_task_outbox_published(self, event_id, *, worker_id):
        self.published.append(event_id)
        for job in self.jobs.values():
            if job["task_event_id"] == event_id:
                job["status"] = "queued"
        return True

    async def mark_content_task_outbox_failed(self, *_args, **_kwargs):
        return True


class UploadObjectStorage:
    def __init__(self):
        self.persisted = []
        self.deleted = []

    def build_object_name(self, *, content_id, suffix):
        return f"uploads/{content_id}{suffix}"

    async def persist_staged_file(self, local_path, *, object_name, content_type):
        self.persisted.append(object_name)
        return f"s3://bucket/{object_name}"

    async def delete_uploaded_object(self, storage_path):
        self.deleted.append(storage_path)


class UploadFeatureStore:
    def __init__(self):
        self.statuses = []

    async def update_content_status(self, content_id, status):
        self.statuses.append((content_id, status))


@pytest.fixture
def upload_env(monkeypatch, tmp_path):
    config = Config()
    config.data_config.upload_dir = str(tmp_path)
    runtime = SimpleNamespace(config=config, observability=UploadObservability())
    store = UploadStore()
    storage = UploadObjectStorage()
    features = UploadFeatureStore()
    monkeypatch.setattr(gateway_api.app.state, "runtime", runtime, raising=False)
    monkeypatch.setattr(gateway_api, "system_store", store)
    monkeypatch.setattr(gateway_api, "object_storage", storage)
    monkeypatch.setattr(gateway_api, "feature_store", features)
    monkeypatch.setattr(
        gateway_api, "kafka_manager", DurableContentKafka(acknowledged=True)
    )

    async def healthy(_runtime):
        return None

    monkeypatch.setattr(gateway_api, "_content_upload_dependency_error", healthy)
    return SimpleNamespace(
        runtime=runtime, store=store, storage=storage, tmp_path=tmp_path
    )


async def _upload(body, *, key=None, user_id="user-1"):
    headers = {gateway_api.IDEMPOTENCY_KEY_HEADER: key} if key is not None else {}
    request = SimpleNamespace(
        headers=headers, state=SimpleNamespace(request_id="request-1")
    )
    file = UploadFile(
        file=io.BytesIO(body),
        filename="clip.mp4",
        headers=Headers({"content-type": "video/mp4"}),
    )
    return await gateway_api.upload_content(
        request, file=file, user_id=user_id, priority="normal"
    )


def _json(response):
    import json

    return json.loads(response.body)


@pytest.mark.asyncio
async def test_upload_retry_with_same_idempotency_key_replays_original(upload_env):
    first = await _upload(b"video-bytes", key="retry-1")
    second = await _upload(b"video-bytes", key="retry-1")

    assert first["status"] == "queued"
    assert second["content_id"] == first["content_id"]
    assert second["idempotent_replay"] is True
    assert second["status"] == "queued"
    # The retry neither re-uploads nor re-stages, and its temp file is gone.
    assert len(upload_env.storage.persisted) == 1
    assert len(upload_env.store.staged) == 1
    assert list(upload_env.tmp_path.iterdir()) == []
    _, job = upload_env.store.staged[0]
    assert job["idempotency_key"] == "retry-1"
    import hashlib

    assert job["request_fingerprint"] == (
        f"sha256:{hashlib.sha256(b'video-bytes').hexdigest()}:11"
    )
    assert ("normal", "idempotent_replay") in upload_env.runtime.observability.uploads


@pytest.mark.asyncio
async def test_upload_reusing_key_with_different_bytes_is_409(upload_env):
    first = await _upload(b"video-bytes", key="retry-1")
    conflict = await _upload(b"other-bytes", key="retry-1")

    assert conflict.status_code == 409
    body = _json(conflict)
    assert body["error"]["code"] == "IDEMPOTENCY_KEY_REUSED"
    assert body["content_id"] == first["content_id"]
    assert len(upload_env.store.staged) == 1
    assert list(upload_env.tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_upload_key_is_scoped_to_user(upload_env):
    first = await _upload(b"video-bytes", key="retry-1", user_id="user-1")
    second = await _upload(b"video-bytes", key="retry-1", user_id="user-2")

    assert second["content_id"] != first["content_id"]
    assert "idempotent_replay" not in second


@pytest.mark.asyncio
async def test_upload_replay_republishes_pending_job(upload_env):
    gateway_api.kafka_manager.acknowledged = False
    first = await _upload(b"video-bytes", key="retry-1")
    assert first.status_code == 503
    assert _json(first)["status"] == "pending_publish"

    gateway_api.kafka_manager.acknowledged = True
    second = await _upload(b"video-bytes", key="retry-1")

    assert second["content_id"] == _json(first)["content_id"]
    assert second["status"] == "queued"
    assert len(upload_env.store.published) == 1


@pytest.mark.asyncio
async def test_concurrent_same_key_loser_deletes_only_its_own_object(upload_env):
    from video_commerce.data_plane.system_store import ContentIdempotencyKeyInUse

    winner = await _upload(b"video-bytes", key="race-1")
    original_lookup = upload_env.store.get_content_job_by_idempotency_key
    calls = {"count": 0}

    async def miss_then_hit(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return None
        return await original_lookup(**kwargs)

    upload_env.store.get_content_job_by_idempotency_key = miss_then_hit
    upload_env.store.stage_error = (ContentIdempotencyKeyInUse("race-1"), False)

    loser = await _upload(b"video-bytes", key="race-1")

    assert loser["content_id"] == winner["content_id"]
    assert loser["idempotent_replay"] is True
    loser_object = f"s3://bucket/{upload_env.storage.persisted[-1]}"
    assert upload_env.storage.deleted == [loser_object]


@pytest.mark.asyncio
async def test_unknown_commit_outcome_keeps_object_when_lookup_fails(upload_env):
    upload_env.store.stage_error = (RuntimeError("connection lost in COMMIT"), False)
    upload_env.store.lookup_error = RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="COMMIT"):
        await _upload(b"video-bytes")

    assert upload_env.storage.deleted == []


@pytest.mark.asyncio
async def test_unknown_commit_outcome_deletes_object_only_when_job_absent(
    upload_env,
):
    upload_env.store.stage_error = (RuntimeError("connection lost in COMMIT"), False)

    with pytest.raises(RuntimeError, match="COMMIT"):
        await _upload(b"video-bytes")

    assert upload_env.storage.deleted == [
        f"s3://bucket/{upload_env.storage.persisted[-1]}"
    ]


@pytest.mark.asyncio
async def test_commit_that_succeeded_despite_error_keeps_object_and_is_pending(
    upload_env,
):
    upload_env.store.stage_error = (RuntimeError("connection lost in COMMIT"), True)

    response = await _upload(b"video-bytes")

    assert response.status_code == 503
    body = _json(response)
    assert body["error"]["code"] == "CONTENT_ENQUEUE_PENDING"
    assert body["content_id"] in upload_env.store.jobs
    assert upload_env.storage.deleted == []


@pytest.mark.parametrize("key", ["", " ", "a" * 129, "bad key", "key/1", "ключ"])
def test_invalid_idempotency_key_is_rejected(key):
    with pytest.raises(HTTPException) as exc:
        gateway_api.parse_idempotency_key(key)

    assert exc.value.status_code == 400


def test_valid_idempotency_key_is_accepted():
    assert gateway_api.parse_idempotency_key(None) is None
    assert gateway_api.parse_idempotency_key("a" * 128) == "a" * 128
    assert (
        gateway_api.parse_idempotency_key("2026-10-20T00:00:00Z.retry_1~x")
        == "2026-10-20T00:00:00Z.retry_1~x"
    )
