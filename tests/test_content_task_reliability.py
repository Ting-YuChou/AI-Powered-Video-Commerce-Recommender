import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

from sqlalchemy import inspect

from video_commerce.common.config import DatabaseConfig
from video_commerce.data_plane.system_store import (
    ContentProcessingRun,
    ContentTaskOutbox,
    SystemStore,
    prepare_content_task,
)
from video_commerce.services.content_task_publisher.main import ContentTaskPublisher


def test_prepare_content_task_is_stable_for_retries():
    first = prepare_content_task(
        content_id="content-1",
        pipeline_version="temporal-multimodal-v3",
        storage_path="s3://uploads/content-1.mp4",
        filename="clip.mp4",
        user_id="user-1",
        priority="high",
        request_id="request-1",
        timestamp=100.0,
    )
    second = prepare_content_task(
        content_id="content-1",
        pipeline_version="temporal-multimodal-v3",
        storage_path="s3://uploads/content-1.mp4",
        filename="clip.mp4",
        user_id="user-1",
        priority="high",
        request_id="request-1",
        timestamp=100.0,
    )

    assert first.event_id == second.event_id
    assert first.payload == second.payload
    assert first.payload["event_id"] == first.event_id
    assert first.payload["pipeline_version"] == "temporal-multimodal-v3"
    assert first.payload_hash == second.payload_hash


def test_content_task_migration_declares_outbox_and_processing_lease():
    migration = (
        Path(__file__).resolve().parents[1]
        / "migrations/postgres/009_content_task_reliability.sql"
    ).read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS content_task_outbox" in migration
    assert "CREATE TABLE IF NOT EXISTS content_processing_runs" in migration
    assert "PRIMARY KEY (content_id, pipeline_version)" in migration
    assert "claim_expires_at" in migration
    assert "next_attempt_at" in migration
    assert "ADD COLUMN IF NOT EXISTS pipeline_version" in migration
    assert "ADD COLUMN IF NOT EXISTS task_event_id" in migration


def test_content_task_models_keep_stable_identity_and_composite_run_key():
    outbox = inspect(ContentTaskOutbox)
    processing_run = inspect(ContentProcessingRun)

    assert [column.key for column in outbox.primary_key] == ["event_id"]
    assert [column.key for column in processing_run.primary_key] == [
        "content_id",
        "pipeline_version",
    ]
    assert {column.key for column in outbox.columns} >= {
        "payload_hash",
        "claimed_by",
        "claim_expires_at",
        "next_attempt_at",
        "published_at",
    }
    assert {column.key for column in processing_run.columns} >= {
        "lease_owner",
        "lease_expires_at",
        "attempts",
        "status",
    }


class FakeStore:
    def __init__(self, rows):
        self.rows = rows
        self.published = []
        self.failed = []

    async def claim_content_task_outbox(self, *, worker_id, batch_size, lease_seconds):
        assert worker_id == "publisher-1"
        assert batch_size == 10
        assert lease_seconds == 60
        return list(self.rows)

    async def mark_content_task_outbox_published(self, event_id, *, worker_id):
        self.published.append((event_id, worker_id))
        return True

    async def mark_content_task_outbox_failed(self, event_id, error, *, worker_id):
        self.failed.append((event_id, error, worker_id))
        return True


class FakeKafka:
    def __init__(self, succeeds=True):
        self.succeeds = succeeds
        self.events = []

    async def publish_video_processing_task_payload(self, event):
        self.events.append(event)
        return self.succeeds


def test_content_task_publisher_marks_published_only_after_ack():
    event = {"event_id": "event-1", "content_id": "content-1"}
    store = FakeStore([event])
    publisher = ContentTaskPublisher(
        system_store=store,
        kafka_manager=FakeKafka(succeeds=True),
        config=SimpleNamespace(
            content_outbox_batch_size=10,
            content_outbox_lease_seconds=60,
        ),
        worker_id="publisher-1",
    )

    assert asyncio.run(publisher.publish_once()) == 1
    assert store.published == [("event-1", "publisher-1")]
    assert store.failed == []


def test_content_task_publisher_releases_failed_row_for_retry():
    event = {"event_id": "event-1", "content_id": "content-1"}
    store = FakeStore([event])
    publisher = ContentTaskPublisher(
        system_store=store,
        kafka_manager=FakeKafka(succeeds=False),
        config=SimpleNamespace(
            content_outbox_batch_size=10,
            content_outbox_lease_seconds=60,
        ),
        worker_id="publisher-1",
    )

    assert asyncio.run(publisher.publish_once()) == 0
    assert store.published == []
    assert store.failed[0][0] == "event-1"
    assert "Kafka" in store.failed[0][1]


def test_content_task_outbox_and_processing_lease_are_owner_safe():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        import pytest

        pytest.skip("DATABASE_URL is required for Postgres reliability test")

    async def exercise() -> None:
        content_id = f"content-test-{uuid.uuid4().hex}"
        task = prepare_content_task(
            content_id=content_id,
            pipeline_version="pipeline-v1",
            storage_path=f"s3://uploads/{content_id}.mp4",
            filename="clip.mp4",
            user_id="user-1",
            priority="normal",
            request_id="request-1",
            timestamp=100.0,
        )
        store = SystemStore(
            DatabaseConfig(
                enable=True,
                url=database_url,
                auto_create_schema=True,
                enable_retention_cleanup=False,
            )
        )
        await store.initialize()
        try:
            assert (
                await store.stage_content_task(
                    task,
                    filename="clip.mp4",
                    storage_path=task.payload["file_path"],
                    user_id="user-1",
                    priority="normal",
                    request_id="request-1",
                )
                == "inserted"
            )
            claimed = await store.claim_content_task_outbox(
                worker_id="publisher-1", batch_size=10, lease_seconds=60
            )
            assert [row["event_id"] for row in claimed] == [task.event_id]
            assert (
                await store.claim_content_task_outbox(
                    worker_id="publisher-2", batch_size=10, lease_seconds=60
                )
                == []
            )
            assert not await store.mark_content_task_outbox_published(
                task.event_id, worker_id="publisher-2"
            )
            assert await store.mark_content_task_outbox_published(
                task.event_id, worker_id="publisher-1"
            )

            first = await store.claim_content_processing_run(
                content_id=content_id,
                pipeline_version="pipeline-v1",
                task_event_id=task.event_id,
                worker_id="worker-1",
                lease_seconds=60,
            )
            assert first["outcome"] == "acquired"
            assert (
                await store.claim_content_processing_run(
                    content_id=content_id,
                    pipeline_version="pipeline-v1",
                    task_event_id=task.event_id,
                    worker_id="worker-2",
                    lease_seconds=60,
                )
            )["outcome"] == "lease_held"
            assert not await store.fail_content_processing_run(
                content_id=content_id,
                pipeline_version="pipeline-v1",
                worker_id="worker-2",
                error="stale",
            )
            assert await store.fail_content_processing_run(
                content_id=content_id,
                pipeline_version="pipeline-v1",
                worker_id="worker-1",
                error="retry",
            )
            takeover = await store.claim_content_processing_run(
                content_id=content_id,
                pipeline_version="pipeline-v1",
                task_event_id=task.event_id,
                worker_id="worker-2",
                lease_seconds=60,
            )
            assert takeover["outcome"] == "acquired"
            features = {
                "content_id": content_id,
                "visual_embedding": [0.1, 0.2],
                "multimodal_schema_version": "temporal_multimodal_v2",
            }
            assert not await store.complete_content_processing_run(
                content_id=content_id,
                pipeline_version="pipeline-v1",
                worker_id="worker-1",
                features=features,
                schema_version="temporal_multimodal_v2",
                artifact_uri="s3://artifacts/stale.json",
                artifact_sha256="stale",
            )
            assert await store.complete_content_processing_run(
                content_id=content_id,
                pipeline_version="pipeline-v1",
                worker_id="worker-2",
                features=features,
                schema_version="temporal_multimodal_v2",
                artifact_uri="s3://artifacts/current.json",
                artifact_sha256="current",
            )
            completed = await store.claim_content_processing_run(
                content_id=content_id,
                pipeline_version="pipeline-v1",
                task_event_id=task.event_id,
                worker_id="worker-3",
                lease_seconds=60,
            )
            assert completed["outcome"] == "completed"
            assert (await store.get_content_job(content_id))["status"] == "completed"

            replacement = prepare_content_task(
                content_id=content_id,
                pipeline_version="pipeline-v2",
                storage_path=task.payload["file_path"],
                filename="clip.mp4",
                user_id="user-1",
                priority="normal",
                request_id="request-2",
                timestamp=200.0,
            )
            await store.stage_content_task(
                replacement,
                filename="clip.mp4",
                storage_path=replacement.payload["file_path"],
                user_id="user-1",
                priority="normal",
                request_id="request-2",
            )
            superseded = await store.claim_content_processing_run(
                content_id=content_id,
                pipeline_version="pipeline-v1",
                task_event_id=task.event_id,
                worker_id="worker-4",
                lease_seconds=60,
            )
            assert superseded["outcome"] == "superseded"
            assert superseded["current_pipeline_version"] == "pipeline-v2"
        finally:
            async with store.session_factory.begin() as session:
                from sqlalchemy import text

                await session.execute(
                    text("DELETE FROM content_feature_artifacts WHERE content_id=:id"),
                    {"id": content_id},
                )
                await session.execute(
                    text("DELETE FROM content_processing_runs WHERE content_id=:id"),
                    {"id": content_id},
                )
                await session.execute(
                    text("DELETE FROM content_task_outbox WHERE content_id=:id"),
                    {"id": content_id},
                )
                await session.execute(
                    text("DELETE FROM content_jobs WHERE content_id=:id"),
                    {"id": content_id},
                )
            await store.close()

    asyncio.run(exercise())


def test_content_task_publisher_terminalizes_missing_object_without_kafka_publish():
    class MissingStore(FakeStore):
        def __init__(self):
            super().__init__([{"event_id": "event-1", "file_path": "/missing.mp4"}])
            self.missing = []

        async def mark_content_task_missing_object(self, event_id, *, worker_id):
            self.missing.append((event_id, worker_id))
            return True

    class MissingStorage:
        async def storage_path_exists(self, _path):
            return False

    store = MissingStore()
    kafka = FakeKafka()
    publisher = ContentTaskPublisher(
        system_store=store,
        kafka_manager=kafka,
        object_storage=MissingStorage(),
        config=SimpleNamespace(
            content_outbox_batch_size=10,
            content_outbox_lease_seconds=60,
        ),
        worker_id="publisher-1",
    )

    assert asyncio.run(publisher.publish_once()) == 0
    assert store.missing == [("event-1", "publisher-1")]
    assert kafka.events == []
    assert store.failed == []


def test_content_reconciler_deletes_only_old_unreferenced_objects():
    class ReconcileStore:
        async def list_content_job_storage_paths(self):
            return {"/uploads/referenced.mp4"}

    class ReconcileStorage:
        def __init__(self):
            self.deleted = []

        async def list_storage_objects(self, _prefix):
            return [
                SimpleNamespace(
                    uri="/uploads/referenced.mp4", size=1, last_modified=1.0
                ),
                SimpleNamespace(uri="/uploads/orphan.mp4", size=1, last_modified=1.0),
                SimpleNamespace(uri="/uploads/young.mp4", size=1, last_modified=95.0),
            ]

        async def delete_uploaded_object(self, uri):
            self.deleted.append(uri)

    storage = ReconcileStorage()
    publisher = ContentTaskPublisher(
        system_store=ReconcileStore(),
        kafka_manager=FakeKafka(),
        object_storage=storage,
        upload_prefix="/uploads",
        config=SimpleNamespace(
            content_outbox_batch_size=10,
            content_outbox_lease_seconds=60,
            content_orphan_grace_seconds=10,
        ),
        worker_id="publisher-1",
    )

    stats = asyncio.run(publisher.reconcile_objects(now=100.0))

    assert storage.deleted == ["/uploads/orphan.mp4"]
    assert stats == {"orphan_objects": 2, "deleted_objects": 1}
