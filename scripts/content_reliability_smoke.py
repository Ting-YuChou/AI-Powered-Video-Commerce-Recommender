"""Exercise content outbox and processing recovery against real Kafka/Postgres."""

from __future__ import annotations

import asyncio
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import time
import uuid

from aiokafka import AIOKafkaConsumer
from sqlalchemy import text

from video_commerce.common.config import Config
from video_commerce.common.models import ContentFeatures
from video_commerce.data_plane.kafka_client import KafkaManager
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.data_plane.system_store import SystemStore, prepare_content_task
from video_commerce.services.content_task_publisher.main import ContentTaskPublisher
from video_commerce.services.content_worker.video_processor import VideoProcessorWorker


class _SmokeFeatureStore:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.features = None

    async def update_content_status(self, _content_id, status):
        self.statuses.append(status)

    async def store_content_features(self, _content_id, features):
        self.features = features


class _SmokeVectorSearch:
    def __init__(self) -> None:
        self.updates = 0

    async def add_content_embedding(self, _content_id, _embedding):
        self.updates += 1


class _SmokeProcessor:
    def __init__(self) -> None:
        self.calls = 0

    async def process_video(self, _path, content_id):
        self.calls += 1
        return ContentFeatures(
            content_id=content_id,
            visual_embedding=[0.3, 0.4],
            artifact_uri="smoke://worker-artifact",
            artifact_sha256="worker-sha256",
        )


class _SmokeObservability:
    def record_content_processing(self, _outcome):
        return None

    def record_asr_transcription(self, _status, _duration):
        return None

    def record_asr_alignment(self, _status, _duration, _segments):
        return None

    def record_worker_message(self, _worker, _topic, _status, _duration):
        return None

    def record_kafka_produce(self, _topic, _status):
        return None


async def _collect_event_ids(config, expected: dict[str, int]) -> Counter:
    consumer = AIOKafkaConsumer(
        config.kafka_config.video_processing_topic,
        bootstrap_servers=config.kafka_config.bootstrap_servers,
        group_id=f"content-reliability-smoke-{uuid.uuid4()}",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )
    counts: Counter = Counter()
    await consumer.start()
    try:
        for _ in range(4):
            batches = await consumer.getmany(timeout_ms=2500)
            for records in batches.values():
                for record in records:
                    event_id = str(record.value.get("event_id") or "")
                    if event_id in expected:
                        counts[event_id] += 1
            if all(counts[event_id] >= count for event_id, count in expected.items()):
                break
    finally:
        await consumer.stop()
    return counts


async def main() -> None:
    config = Config()
    store = SystemStore(config.database_config)
    storage = ObjectStorage(config.object_storage_config)
    kafka = KafkaManager(config.kafka_config)
    content_ids: list[str] = []
    paths: list[Path] = []
    await store.initialize()
    await storage.initialize()
    await kafka.start()
    try:
        pipeline_version = config.model_config.content_pipeline_version
        normal_content_id = f"smoke-normal-{uuid.uuid4().hex}"
        crash_content_id = f"smoke-crash-{uuid.uuid4().hex}"
        content_ids.extend([normal_content_id, crash_content_id])

        tasks = []
        for content_id in content_ids:
            path = Path(config.data_config.upload_dir) / f"{content_id}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"phase-a-content-smoke")
            paths.append(path)
            task = prepare_content_task(
                content_id=content_id,
                pipeline_version=pipeline_version,
                storage_path=str(path),
                filename=path.name,
                user_id="smoke-user",
                priority="normal",
                request_id=f"request-{content_id}",
                timestamp=time.time(),
            )
            await store.stage_content_task(
                task,
                filename=path.name,
                storage_path=str(path),
                user_id="smoke-user",
                priority="normal",
                request_id=f"request-{content_id}",
            )
            assert (await store.get_content_job(content_id))[
                "status"
            ] == "pending_publish"
            tasks.append(task)

        publisher = ContentTaskPublisher(
            system_store=store,
            kafka_manager=kafka,
            object_storage=storage,
            config=SimpleNamespace(
                content_outbox_batch_size=1,
                content_outbox_lease_seconds=1,
            ),
            worker_id="smoke-publisher",
        )
        assert await publisher.publish_once() == 1
        assert (await store.get_content_job(normal_content_id))["status"] == "queued"

        crash_task = tasks[1]
        claimed = await store.claim_content_task_outbox_event(
            crash_task.event_id,
            worker_id="crashed-after-ack",
            lease_seconds=1,
        )
        assert claimed is not None
        assert await kafka.publish_video_processing_task_payload(claimed)
        assert (await store.get_content_job(crash_content_id))[
            "status"
        ] == "pending_publish"
        await asyncio.sleep(1.2)

        retry_publisher = ContentTaskPublisher(
            system_store=store,
            kafka_manager=kafka,
            object_storage=storage,
            config=SimpleNamespace(
                content_outbox_batch_size=10,
                content_outbox_lease_seconds=30,
            ),
            worker_id="smoke-retry-publisher",
        )
        assert await retry_publisher.publish_once() == 1
        assert (await store.get_content_job(crash_content_id))["status"] == "queued"

        counts = await _collect_event_ids(
            config,
            {tasks[0].event_id: 1, crash_task.event_id: 2},
        )
        assert counts[tasks[0].event_id] >= 1
        assert counts[crash_task.event_id] >= 2

        worker = VideoProcessorWorker(config)
        processor = _SmokeProcessor()
        feature_store = _SmokeFeatureStore()
        vector_search = _SmokeVectorSearch()
        worker.content_processor = processor
        worker.feature_store = feature_store
        worker.vector_search = vector_search
        worker.system_store = store
        worker.object_storage = None
        worker.observability = _SmokeObservability()
        task_payload = dict(crash_task.payload)
        await worker._handle_video_task(
            config.kafka_config.video_processing_topic,
            crash_content_id,
            task_payload,
            None,
        )
        await worker._handle_video_task(
            config.kafka_config.video_processing_topic,
            crash_content_id,
            task_payload,
            None,
        )
        assert processor.calls == 1
        assert vector_search.updates == 2
        assert feature_store.features.content_id == crash_content_id

        claim = await store.claim_content_processing_run(
            content_id=normal_content_id,
            pipeline_version=pipeline_version,
            task_event_id=tasks[0].event_id,
            worker_id="smoke-worker",
            lease_seconds=60,
        )
        assert claim["outcome"] == "acquired"
        features = {
            "content_id": normal_content_id,
            "visual_embedding": [0.1, 0.2],
            "multimodal_schema_version": "temporal_multimodal_v2",
        }
        assert await store.complete_content_processing_run(
            content_id=normal_content_id,
            pipeline_version=pipeline_version,
            worker_id="smoke-worker",
            features=features,
            schema_version="temporal_multimodal_v2",
            artifact_uri="smoke://immutable-artifact",
            artifact_sha256="smoke-sha256",
        )
        repair = await store.claim_content_processing_run(
            content_id=normal_content_id,
            pipeline_version=pipeline_version,
            task_event_id=tasks[0].event_id,
            worker_id="smoke-repair-worker",
            lease_seconds=60,
        )
        assert repair["outcome"] == "completed"
        assert (await store.get_content_feature_artifact(normal_content_id)) == features

        print(
            json.dumps(
                {
                    "status": "passed",
                    "normal_event_count": counts[tasks[0].event_id],
                    "redelivered_event_count": counts[crash_task.event_id],
                    "worker_model_calls": processor.calls,
                    "worker_projection_updates": vector_search.updates,
                    "durable_repair_outcome": repair["outcome"],
                },
                sort_keys=True,
            )
        )
    finally:
        async with store.session_factory.begin() as session:
            for content_id in content_ids:
                for table in (
                    "content_feature_artifacts",
                    "content_processing_runs",
                    "content_task_outbox",
                    "content_jobs",
                ):
                    await session.execute(
                        text(f"DELETE FROM {table} WHERE content_id = :content_id"),
                        {"content_id": content_id},
                    )
        for path in paths:
            path.unlink(missing_ok=True)
        await kafka.stop()
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
