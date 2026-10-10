"""Export durable Postgres PIT run state for CronJob-based deployments."""

from __future__ import annotations

import asyncio
import signal

from video_commerce.common.config import Config
from video_commerce.common.observability import (
    ObservabilityManager,
    configure_logging,
    start_worker_metrics_server,
)
from video_commerce.data_plane.system_store import SystemStore


async def run() -> None:
    config = Config()
    configure_logging(config.monitoring_config)
    observability = ObservabilityManager()
    start_worker_metrics_server(
        observability, config.monitoring_config, default_port=9106
    )
    store = SystemStore(config.database_config, observability=observability)
    await store.initialize()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        while not stop.is_set():
            state = await store.get_pit_operational_metrics()
            observability.update_pit_durable_state(**state)
            release_state = await store.get_model_release_operational_metrics(
                model_name="ranking_model",
                environment=config.model_release_config.environment,
            )
            observability.update_model_release_durable_state(**release_state)
            retrieval_release_state = await store.get_model_release_operational_metrics(
                model_name="two_tower_retrieval",
                environment=config.recommendation_config.retrieval_release_environment,
            )
            observability.update_retrieval_release_durable_state(
                **retrieval_release_state
            )
            try:
                await asyncio.wait_for(stop.wait(), timeout=30)
            except asyncio.TimeoutError:
                pass
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(run())
