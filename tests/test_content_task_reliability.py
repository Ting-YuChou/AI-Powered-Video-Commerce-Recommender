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
        def __init__(self):
            self.batches = []

        async def filter_referenced_storage_paths(self, paths):
            self.batches.append(list(paths))
            return {path for path in paths if path == "/uploads/referenced.mp4"}

    class ReconcileStorage:
        def __init__(self):
            self.deleted = []
            self.page_size = None

        async def iter_upload_object_pages(self, _prefix, *, page_size):
            self.page_size = page_size
            yield [
                SimpleNamespace(
                    uri="/uploads/referenced.mp4", size=1, last_modified=1.0
                ),
                SimpleNamespace(uri="/uploads/orphan.mp4", size=1, last_modified=1.0),
            ]
            yield [
                SimpleNamespace(uri="/uploads/young.mp4", size=1, last_modified=95.0),
            ]

        async def delete_uploaded_object(self, uri):
            self.deleted.append(uri)

    storage = ReconcileStorage()
    store = ReconcileStore()
    publisher = ContentTaskPublisher(
        system_store=store,
        kafka_manager=FakeKafka(),
        object_storage=storage,
        upload_prefix="/uploads",
        config=SimpleNamespace(
            content_outbox_batch_size=10,
            content_outbox_lease_seconds=60,
            content_orphan_grace_seconds=10,
            content_reconcile_page_size=2,
        ),
        worker_id="publisher-1",
    )

    stats = asyncio.run(publisher.reconcile_objects(now=100.0))

    assert storage.deleted == ["/uploads/orphan.mp4"]
    assert stats == {"orphan_objects": 2, "deleted_objects": 1}
    # One batched DB lookup per listed page; the whole listing is never held.
    assert storage.page_size == 2
    assert store.batches == [
        ["/uploads/referenced.mp4", "/uploads/orphan.mp4"],
        ["/uploads/young.mp4"],
    ]


def test_local_upload_pages_skip_nested_artifacts_and_respect_page_size(tmp_path):
    from video_commerce.common.config import ObjectStorageConfig
    from video_commerce.data_plane.object_storage import ObjectStorage

    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (tmp_path / name).write_bytes(b"video")
    nested = tmp_path / "content-features" / "content-1"
    nested.mkdir(parents=True)
    (nested / "features.json").write_bytes(b"{}")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "model.pt").write_bytes(b"model")
    storage = ObjectStorage(
        ObjectStorageConfig(backend="local", download_dir=str(tmp_path / "dl"))
    )

    async def collect():
        return [
            page
            async for page in storage.iter_upload_object_pages(
                str(tmp_path), page_size=2
            )
        ]

    pages = asyncio.run(collect())

    assert [len(page) for page in pages] == [2, 1]
    assert sorted(item.uri for page in pages for item in page) == [
        str(tmp_path / name) for name in ("a.mp4", "b.mp4", "c.mp4")
    ]


def test_remote_upload_pages_use_delimiter_and_continuation(tmp_path):
    from video_commerce.common.config import ObjectStorageConfig
    from video_commerce.data_plane.object_storage import ObjectStorage

    calls = []

    class Client:
        def list_objects_v2(self, **kwargs):
            calls.append(kwargs)
            if "ContinuationToken" not in kwargs:
                return {
                    "Contents": [{"Key": "uploads/a.mp4", "Size": 1}],
                    "CommonPrefixes": [{"Prefix": "uploads/artifacts/"}],
                    "IsTruncated": True,
                    "NextContinuationToken": "next",
                }
            return {
                "Contents": [{"Key": "uploads/b.mp4", "Size": 2}],
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
            [item.uri for item in page]
            async for page in storage.iter_upload_object_pages(
                "s3://bucket/uploads", page_size=1
            )
        ]

    assert asyncio.run(collect()) == [
        ["s3://bucket/uploads/a.mp4"],
        ["s3://bucket/uploads/b.mp4"],
    ]
    assert all(call["Delimiter"] == "/" for call in calls)
    assert all(call["Prefix"] == "uploads/" for call in calls)
    assert calls[1]["ContinuationToken"] == "next"


class RecordingObservability:
    def __init__(self):
        self.redrives = []
        self.errors = []
        self.outbox = []
        self.orphans = []

    def record_content_processing_redrives(self, outcome, count=1):
        self.redrives.append((outcome, count))

    def record_content_task_publisher_error(self, stage):
        self.errors.append(stage)

    def update_content_task_outbox(self, **kwargs):
        self.outbox.append(kwargs)

    def update_content_orphans(self, **kwargs):
        self.orphans.append(kwargs)


def _reaper_config(**overrides):
    values = {
        "content_outbox_batch_size": 10,
        "content_outbox_lease_seconds": 60,
        "content_outbox_retention_days": 7,
        "content_orphan_grace_seconds": 10,
        "content_processing_redrive_grace_seconds": 30,
        "content_processing_max_redrives": 3,
        "content_processing_reaper_interval_seconds": 15.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_reaper_passes_policy_to_store_and_counts_outcomes():
    class ReaperStore:
        def __init__(self):
            self.calls = []

        async def redrive_expired_content_processing_runs(self, **kwargs):
            self.calls.append(kwargs)
            return {"redriven": 2, "exhausted": 1, "superseded": 0}

    store = ReaperStore()
    observability = RecordingObservability()
    publisher = ContentTaskPublisher(
        system_store=store,
        kafka_manager=FakeKafka(),
        config=_reaper_config(),
        worker_id="publisher-1",
        observability=observability,
    )

    counts = asyncio.run(publisher.reap_expired_processing_runs())

    assert counts == {"redriven": 2, "exhausted": 1, "superseded": 0}
    assert store.calls == [
        {"grace_seconds": 30, "max_redrives": 3, "batch_size": 10}
    ]
    assert observability.redrives == [("redriven", 2), ("exhausted", 1)]


def test_publisher_loop_survives_failing_iterations_with_backoff(monkeypatch):
    from video_commerce.services.content_task_publisher import main as publisher_main

    class FlakyStore(FakeStore):
        def __init__(self, stop_event):
            super().__init__([])
            self.stop_event = stop_event
            self.claims = 0
            self.reaps = 0

        async def claim_content_task_outbox(self, **_kwargs):
            self.claims += 1
            if self.claims <= 2:
                raise RuntimeError("database unavailable")
            if self.claims >= 4:
                self.stop_event.set()
            return []

        async def get_content_task_outbox_stats(self):
            return {"pending": 0, "oldest_age_seconds": 0.0}

        async def redrive_expired_content_processing_runs(self, **_kwargs):
            self.reaps += 1
            return {"redriven": 0, "exhausted": 0, "superseded": 0}

        async def prune_content_task_outbox(self, **_kwargs):
            return 0

    delays = []

    async def exercise():
        stop_event = asyncio.Event()
        store = FlakyStore(stop_event)
        observability = RecordingObservability()
        publisher = ContentTaskPublisher(
            system_store=store,
            kafka_manager=FakeKafka(),
            config=_reaper_config(),
            worker_id="publisher-1",
            observability=observability,
        )
        original_wait_for = asyncio.wait_for

        async def recording_wait_for(awaitable, timeout):
            delays.append(timeout)
            return await original_wait_for(awaitable, timeout=0)

        monkeypatch.setattr(publisher_main.asyncio, "wait_for", recording_wait_for)
        await original_wait_for(
            publisher_main.run_publisher_loop(
                publisher, stop_event, poll_seconds=0.25, observability=observability
            ),
            timeout=5,
        )
        return store, observability

    store, observability = asyncio.run(exercise())

    assert store.claims == 4
    assert observability.errors == ["publish", "publish"]
    # Failures back off exponentially from the poll interval, then reset.
    assert delays[:3] == [0.25, 0.5, 0.25]
    # Maintenance ran once the loop recovered.
    assert store.reaps == 1


def test_failure_backoff_is_exponential_and_capped():
    from video_commerce.services.content_task_publisher.main import (
        failure_backoff_seconds,
    )

    assert failure_backoff_seconds(1, base_seconds=0.25) == 0.25
    assert failure_backoff_seconds(3, base_seconds=0.25) == 1.0
    assert failure_backoff_seconds(50, base_seconds=0.25) == 30.0


def test_upload_idempotency_migration_is_additive():
    migration = (
        Path(__file__).resolve().parents[1]
        / "migrations/postgres/013_content_upload_idempotency.sql"
    ).read_text(encoding="utf-8")

    assert "ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(128)" in migration
    assert "ADD COLUMN IF NOT EXISTS request_fingerprint" in migration
    assert "CREATE UNIQUE INDEX IF NOT EXISTS ux_content_jobs_user_idempotency_key" in (
        migration
    )
    assert "(COALESCE(user_id, ''))" in migration
    assert "ix_content_jobs_storage_path" in migration
    assert "ADD COLUMN IF NOT EXISTS redrive_count INTEGER NOT NULL DEFAULT 0" in (
        migration
    )
    assert "DROP " not in migration.upper()


async def _cleanup_content(store, content_ids):
    from sqlalchemy import text

    async with store.session_factory.begin() as session:
        for content_id in content_ids:
            for table in (
                "content_feature_artifacts",
                "content_processing_runs",
                "content_task_outbox",
                "content_jobs",
            ):
                await session.execute(
                    text(f"DELETE FROM {table} WHERE content_id=:id"),
                    {"id": content_id},
                )


def _postgres_store():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        import pytest

        pytest.skip("DATABASE_URL is required for Postgres reliability test")
    return SystemStore(
        DatabaseConfig(
            enable=True,
            url=database_url,
            auto_create_schema=True,
            enable_retention_cleanup=False,
        )
    )


async def _stage_published_task(store, content_id, **stage_kwargs):
    task = prepare_content_task(
        content_id=content_id,
        pipeline_version="pipeline-v1",
        storage_path=f"s3://uploads/{content_id}.mp4",
        filename="clip.mp4",
        user_id=stage_kwargs.get("user_id", "user-1"),
        priority="normal",
        request_id="request-1",
        timestamp=100.0,
    )
    await store.stage_content_task(
        task,
        filename="clip.mp4",
        storage_path=task.payload["file_path"],
        user_id=stage_kwargs.pop("user_id", "user-1"),
        priority="normal",
        request_id="request-1",
        **stage_kwargs,
    )
    claimed = await store.claim_content_task_outbox_event(
        task.event_id, worker_id="publisher-1", lease_seconds=60
    )
    assert claimed is not None
    assert await store.mark_content_task_outbox_published(
        task.event_id, worker_id="publisher-1"
    )
    return task


async def _expire_lease(store, content_id):
    """Simulate a dead worker: lease expired, task published before the lease."""
    from sqlalchemy import text

    async with store.session_factory.begin() as session:
        await session.execute(
            text(
                "UPDATE content_processing_runs "
                "SET lease_expires_at = now() - interval '5 minutes' "
                "WHERE content_id=:id"
            ),
            {"id": content_id},
        )
        await session.execute(
            text(
                "UPDATE content_task_outbox "
                "SET published_at = now() - interval '10 minutes' "
                "WHERE content_id=:id"
            ),
            {"id": content_id},
        )


def test_reaper_redrives_expired_lease_then_terminalizes_after_max():
    store = _postgres_store()

    async def exercise() -> None:
        content_id = f"content-reaper-{uuid.uuid4().hex}"
        await store.initialize()
        try:
            task = await _stage_published_task(store, content_id)
            claim = await store.claim_content_processing_run(
                content_id=content_id,
                pipeline_version="pipeline-v1",
                task_event_id=task.event_id,
                worker_id="worker-dead",
                lease_seconds=60,
            )
            assert claim["outcome"] == "acquired"
            # A live lease is never redriven.
            assert (
                await store.redrive_expired_content_processing_runs(
                    grace_seconds=0, max_redrives=1
                )
            )["redriven"] == 0
            # Duplicate delivery while the holder lives: lease_held (offset acked).
            assert (
                await store.claim_content_processing_run(
                    content_id=content_id,
                    pipeline_version="pipeline-v1",
                    task_event_id=task.event_id,
                    worker_id="worker-2",
                    lease_seconds=60,
                )
            )["outcome"] == "lease_held"

            await _expire_lease(store, content_id)
            assert (
                await store.redrive_expired_content_processing_runs(
                    grace_seconds=3600, max_redrives=1
                )
            )["redriven"] == 0
            assert await store.redrive_expired_content_processing_runs(
                grace_seconds=0, max_redrives=1
            ) == {"redriven": 1, "exhausted": 0, "superseded": 0}
            # Until republished and reclaimed the run is not redriven twice.
            assert (
                await store.redrive_expired_content_processing_runs(
                    grace_seconds=0, max_redrives=1
                )
            )["redriven"] == 0
            assert (await store.get_content_job(content_id))[
                "status"
            ] == "pending_publish"
            republished = await store.claim_content_task_outbox(
                worker_id="publisher-2", batch_size=10, lease_seconds=60
            )
            assert task.event_id in [row["event_id"] for row in republished]
            assert await store.mark_content_task_outbox_published(
                task.event_id, worker_id="publisher-2"
            )
            takeover = await store.claim_content_processing_run(
                content_id=content_id,
                pipeline_version="pipeline-v1",
                task_event_id=task.event_id,
                worker_id="worker-3",
                lease_seconds=60,
            )
            assert takeover["outcome"] == "acquired"

            # The second dead worker exhausts the redrive budget.
            await _expire_lease(store, content_id)
            assert await store.redrive_expired_content_processing_runs(
                grace_seconds=0, max_redrives=1
            ) == {"redriven": 0, "exhausted": 1, "superseded": 0}
            job = await store.get_content_job(content_id)
            assert job["status"] == "failed"
            assert "processing_attempts_exhausted" in job["error_message"]
            async with store.session_factory() as session:
                outbox = await session.get(ContentTaskOutbox, task.event_id)
                run = await session.get(
                    ContentProcessingRun, (content_id, "pipeline-v1")
                )
            assert outbox.terminal_reason == "processing_attempts_exhausted"
            assert outbox.terminal_at is not None
            assert run.status == "failed"
            assert run.redrive_count == 1
            terminal_claims = await store.claim_content_task_outbox(
                worker_id="publisher-3", batch_size=100, lease_seconds=60
            )
            assert task.event_id not in [row["event_id"] for row in terminal_claims]
        finally:
            await _cleanup_content(store, [content_id])
            await store.close()

    asyncio.run(exercise())


def test_upload_idempotency_key_is_unique_per_user_in_postgres():
    from video_commerce.data_plane.system_store import ContentIdempotencyKeyInUse

    store = _postgres_store()

    async def exercise() -> None:
        first_id = f"content-idem-{uuid.uuid4().hex}"
        second_id = f"content-idem-{uuid.uuid4().hex}"
        other_user_id = f"content-idem-{uuid.uuid4().hex}"
        anon_ids = [f"content-idem-{uuid.uuid4().hex}" for _ in range(2)]
        key = f"key-{uuid.uuid4().hex}"
        await store.initialize()
        try:
            await _stage_published_task(
                store,
                first_id,
                idempotency_key=key,
                request_fingerprint="sha256:abc:5",
            )
            import pytest

            with pytest.raises(ContentIdempotencyKeyInUse):
                await _stage_published_task(
                    store,
                    second_id,
                    idempotency_key=key,
                    request_fingerprint="sha256:def:5",
                )
            assert await store.get_content_job(second_id) is None
            existing = await store.get_content_job_by_idempotency_key(
                user_id="user-1", idempotency_key=key
            )
            assert existing["content_id"] == first_id
            assert existing["request_fingerprint"] == "sha256:abc:5"
            # Keys are scoped per user, and anonymous uploads are scoped too.
            await _stage_published_task(
                store,
                other_user_id,
                user_id="user-2",
                idempotency_key=key,
                request_fingerprint="sha256:abc:5",
            )
            await _stage_published_task(
                store,
                anon_ids[0],
                user_id=None,
                idempotency_key=key,
                request_fingerprint="sha256:abc:5",
            )
            with pytest.raises(ContentIdempotencyKeyInUse):
                await _stage_published_task(
                    store,
                    anon_ids[1],
                    user_id=None,
                    idempotency_key=key,
                    request_fingerprint="sha256:abc:5",
                )
            assert (
                await store.filter_referenced_storage_paths(
                    [f"s3://uploads/{first_id}.mp4", "s3://uploads/unknown.mp4"]
                )
            ) == {f"s3://uploads/{first_id}.mp4"}
        finally:
            await _cleanup_content(
                store, [first_id, second_id, other_user_id, *anon_ids]
            )
            await store.close()

    asyncio.run(exercise())
