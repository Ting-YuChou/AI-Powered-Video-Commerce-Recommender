"""Publish durable content processing tasks from Postgres to Kafka."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import time
from typing import Any


class ContentTaskPublisher:
    def __init__(
        self,
        *,
        system_store: Any,
        kafka_manager: Any,
        config: Any,
        worker_id: str,
        object_storage: Any = None,
        upload_prefix: str = "",
        observability: Any = None,
    ) -> None:
        self.system_store = system_store
        self.kafka_manager = kafka_manager
        self.config = config
        self.worker_id = worker_id
        self.object_storage = object_storage
        self.upload_prefix = upload_prefix
        self.observability = observability

    async def publish_once(self) -> int:
        rows = await self.system_store.claim_content_task_outbox(
            worker_id=self.worker_id,
            batch_size=int(self.config.content_outbox_batch_size),
            lease_seconds=int(self.config.content_outbox_lease_seconds),
        )
        published = 0
        for event in rows:
            event_id = str(event.get("event_id") or "")
            try:
                if (
                    self.object_storage is not None
                    and not await self.object_storage.storage_path_exists(
                        str(event.get("file_path") or "")
                    )
                ):
                    await self.system_store.mark_content_task_missing_object(
                        event_id, worker_id=self.worker_id
                    )
                    if self.observability is not None:
                        self.observability.record_content_task_publish_retry(
                            "missing_object"
                        )
                    continue
                acknowledged = (
                    await self.kafka_manager.publish_video_processing_task_payload(
                        event
                    )
                )
                if not acknowledged:
                    raise RuntimeError("Kafka broker did not acknowledge content task")
                marked = await self.system_store.mark_content_task_outbox_published(
                    event_id, worker_id=self.worker_id
                )
                if not marked:
                    raise RuntimeError("content task outbox publish lease was lost")
                published += 1
            except Exception as exc:
                await self.system_store.mark_content_task_outbox_failed(
                    event_id, str(exc), worker_id=self.worker_id
                )
                if self.observability is not None:
                    self.observability.record_content_task_publish_retry(
                        "publish_error"
                    )
        return published

    async def reconcile_objects(self, *, now: float | None = None) -> dict[str, int]:
        if self.object_storage is None or not self.upload_prefix:
            return {"orphan_objects": 0, "deleted_objects": 0}
        current_time = time.time() if now is None else float(now)
        grace = int(self.config.content_orphan_grace_seconds)
        referenced = await self.system_store.list_content_job_storage_paths()
        objects = await self.object_storage.list_storage_objects(self.upload_prefix)
        orphan_objects = 0
        deleted_objects = 0
        for item in objects:
            if item.uri in referenced:
                continue
            orphan_objects += 1
            if current_time - float(item.last_modified) >= grace:
                await self.object_storage.delete_uploaded_object(item.uri)
                deleted_objects += 1
        return {
            "orphan_objects": orphan_objects,
            "deleted_objects": deleted_objects,
        }


async def main() -> None:
    from video_commerce.common.config import Config
    from video_commerce.common.observability import (
        ObservabilityManager,
        configure_logging,
        start_worker_metrics_server,
    )
    from video_commerce.common.telemetry import configure_tracing
    from video_commerce.data_plane.kafka_client import close_kafka, init_kafka
    from video_commerce.data_plane.object_storage import ObjectStorage
    from video_commerce.data_plane.system_store import SystemStore

    config = Config()
    configure_logging(config.monitoring_config)
    configure_tracing("content-task-publisher", config.monitoring_config)
    observability = ObservabilityManager()
    start_worker_metrics_server(
        observability, config.monitoring_config, default_port=9105
    )
    store = SystemStore(config.database_config, observability=observability)
    await store.initialize()
    storage = ObjectStorage(config.object_storage_config)
    await storage.initialize()
    kafka = await init_kafka(config.kafka_config, observability=observability)
    worker_id = f"content-publisher-{socket.gethostname()}-{os.getpid()}"
    upload_prefix = (
        f"s3://{config.object_storage_config.bucket}/{config.object_storage_config.prefix}"
        if storage.is_remote
        else config.data_config.upload_dir
    )
    publisher = ContentTaskPublisher(
        system_store=store,
        kafka_manager=kafka,
        object_storage=storage,
        upload_prefix=upload_prefix,
        config=config.service_topology_config,
        worker_id=worker_id,
        observability=observability,
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)
    last_reconciled_at = 0.0
    last_pruned_at = 0.0
    try:
        while not stop_event.is_set():
            published = await publisher.publish_once()
            observability.update_content_task_outbox(
                **(await store.get_content_task_outbox_stats())
            )
            now = time.monotonic()
            if now - last_reconciled_at >= 300.0:
                observability.update_content_orphans(
                    **(await publisher.reconcile_objects())
                )
                last_reconciled_at = now
            if now - last_pruned_at >= 3600.0:
                await store.prune_content_task_outbox(
                    retention_days=(
                        config.service_topology_config.content_outbox_retention_days
                    )
                )
                last_pruned_at = now
            if published == 0:
                try:
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=(
                            config.service_topology_config.content_outbox_poll_seconds
                        ),
                    )
                except asyncio.TimeoutError:
                    pass
    finally:
        await close_kafka()
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
