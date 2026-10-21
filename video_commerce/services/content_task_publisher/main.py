"""Publish durable content processing tasks from Postgres to Kafka."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import time
from typing import Any

logger = logging.getLogger(__name__)


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

    async def reap_expired_processing_runs(self) -> dict[str, int]:
        """Redrive runs whose worker died while holding the processing lease."""
        counts = await self.system_store.redrive_expired_content_processing_runs(
            grace_seconds=int(self.config.content_processing_redrive_grace_seconds),
            max_redrives=int(self.config.content_processing_max_redrives),
            batch_size=int(self.config.content_outbox_batch_size),
        )
        if self.observability is not None:
            for outcome, count in counts.items():
                if count:
                    self.observability.record_content_processing_redrives(
                        outcome, count
                    )
        return counts

    async def reconcile_objects(self, *, now: float | None = None) -> dict[str, int]:
        """Delete old upload objects no content job references, one page at a time."""
        if self.object_storage is None or not self.upload_prefix:
            return {"orphan_objects": 0, "deleted_objects": 0}
        current_time = time.time() if now is None else float(now)
        grace = int(self.config.content_orphan_grace_seconds)
        page_size = int(getattr(self.config, "content_reconcile_page_size", 1000))
        orphan_objects = 0
        deleted_objects = 0
        async for page in self.object_storage.iter_upload_object_pages(
            self.upload_prefix, page_size=page_size
        ):
            referenced = await self.system_store.filter_referenced_storage_paths(
                [item.uri for item in page]
            )
            for item in page:
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

    async def run_iteration(self, state: "_LoopState") -> int:
        """One publisher pass; maintenance runs on its own cadence."""
        state.stage = "publish"
        published = await self.publish_once()
        if self.observability is not None:
            state.stage = "outbox_stats"
            self.observability.update_content_task_outbox(
                **(await self.system_store.get_content_task_outbox_stats())
            )
        now = time.monotonic()
        if now - state.last_reaped_at >= float(
            self.config.content_processing_reaper_interval_seconds
        ):
            state.stage = "reaper"
            await self.reap_expired_processing_runs()
            state.last_reaped_at = now
        if now - state.last_reconciled_at >= 300.0:
            state.stage = "reconcile"
            counts = await self.reconcile_objects()
            if self.observability is not None:
                self.observability.update_content_orphans(**counts)
            state.last_reconciled_at = now
        if now - state.last_pruned_at >= 3600.0:
            state.stage = "prune"
            await self.system_store.prune_content_task_outbox(
                retention_days=self.config.content_outbox_retention_days
            )
            state.last_pruned_at = now
        return published


class _LoopState:
    def __init__(self) -> None:
        self.stage = "publish"
        self.last_reaped_at = float("-inf")
        self.last_reconciled_at = float("-inf")
        self.last_pruned_at = float("-inf")
        self.consecutive_failures = 0


def failure_backoff_seconds(
    consecutive_failures: int, *, base_seconds: float, max_seconds: float = 30.0
) -> float:
    """Exponential backoff for a failing publisher loop, capped at max_seconds."""
    exponent = min(max(0, int(consecutive_failures) - 1), 16)
    return min(float(max_seconds), max(0.05, float(base_seconds)) * (2**exponent))


async def run_publisher_loop(
    publisher: ContentTaskPublisher,
    stop_event: asyncio.Event,
    *,
    poll_seconds: float,
    observability: Any = None,
) -> None:
    """Keep publishing until stopped; one failing iteration never ends the loop."""
    state = _LoopState()
    while not stop_event.is_set():
        try:
            published = await publisher.run_iteration(state)
            state.consecutive_failures = 0
            delay = 0.0 if published else float(poll_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            state.consecutive_failures += 1
            delay = failure_backoff_seconds(
                state.consecutive_failures, base_seconds=poll_seconds
            )
            logger.exception(
                "content_task_publisher_iteration_failed",
                extra={
                    "stage": state.stage,
                    "consecutive_failures": state.consecutive_failures,
                    "backoff_seconds": delay,
                },
            )
            if observability is not None:
                observability.record_content_task_publisher_error(state.stage)
        if delay > 0:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass


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
    try:
        await run_publisher_loop(
            publisher,
            stop_event,
            poll_seconds=config.service_topology_config.content_outbox_poll_seconds,
            observability=observability,
        )
    finally:
        await close_kafka()
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
