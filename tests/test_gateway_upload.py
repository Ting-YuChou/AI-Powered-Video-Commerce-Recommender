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
