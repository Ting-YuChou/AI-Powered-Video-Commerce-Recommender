"""Reliable Postgres-outbox publisher for versioned catalog feature events."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import time
from typing import Any

from video_commerce.common.config import Config
from video_commerce.data_plane.kafka_client import close_kafka, init_kafka
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.common.observability import (
    ObservabilityManager,
    configure_logging,
    start_worker_metrics_server,
)
from video_commerce.common.telemetry import configure_tracing


class CatalogEventPublisher:
    def __init__(
        self,
        *,
        system_store: Any,
        kafka_manager: Any,
        config: Any,
        worker_id: str,
    ) -> None:
        self.system_store = system_store
        self.kafka_manager = kafka_manager
        self.config = config
        self.worker_id = worker_id

    async def publish_once(self) -> int:
        rows = await self.system_store.claim_catalog_outbox(
            worker_id=self.worker_id,
            batch_size=int(self.config.catalog_outbox_batch_size),
            lease_seconds=int(self.config.catalog_outbox_lease_seconds),
        )
        published = 0
        for event in rows:
            event_id = str(event.get("event_id") or "")
            try:
                success = await self.kafka_manager.send_catalog_feature_event(event)
                if not success:
                    raise RuntimeError("Kafka broker did not acknowledge catalog event")
                await self.system_store.mark_catalog_outbox_published(
                    event_id, worker_id=self.worker_id
                )
                published += 1
            except Exception as exc:
                await self.system_store.mark_catalog_outbox_failed(
                    event_id,
                    str(exc),
                    worker_id=self.worker_id,
                )
        return published


async def main() -> None:
    config = Config()
    configure_logging(config.monitoring_config)
    configure_tracing("catalog-event-publisher", config.monitoring_config)
    observability = ObservabilityManager()
    start_worker_metrics_server(
        observability, config.monitoring_config, default_port=9104
    )
    store = SystemStore(config.database_config, observability=observability)
    await store.initialize()
    kafka = await init_kafka(config.kafka_config)
    worker_id = f"catalog-publisher-{socket.gethostname()}-{os.getpid()}"
    publisher = CatalogEventPublisher(
        system_store=store,
        kafka_manager=kafka,
        config=config.feature_lake_config,
        worker_id=worker_id,
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)
    try:
        last_pruned_at = 0.0
        while not stop_event.is_set():
            published = await publisher.publish_once()
            stats = await store.get_catalog_outbox_stats()
            observability.update_catalog_outbox(**stats)
            if time.monotonic() - last_pruned_at >= 3600.0:
                await store.prune_catalog_outbox(
                    retention_days=config.feature_lake_config.catalog_outbox_retention_days
                )
                last_pruned_at = time.monotonic()
            if published == 0:
                try:
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=float(
                            config.feature_lake_config.catalog_outbox_poll_seconds
                        ),
                    )
                except asyncio.TimeoutError:
                    pass
    finally:
        await close_kafka()
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
