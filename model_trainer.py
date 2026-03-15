"""
Offline model trainer service.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from config import Config
from feature_store import FeatureStore
from recommender import RecommendationEngine
from vector_search import VectorSearchEngine

logger = logging.getLogger(__name__)


class ModelTrainerService:
    """Dedicated background service for model retraining."""

    def __init__(self, config: Config):
        self.config = config
        self.feature_store: FeatureStore | None = None
        self.vector_search: VectorSearchEngine | None = None
        self.recommendation_engine: RecommendationEngine | None = None
        self.running = False

    async def initialize(self):
        self.feature_store = FeatureStore(self.config.redis_config, self.config.cache_config)
        await self.feature_store.initialize()

        self.vector_search = VectorSearchEngine(self.config.vector_config)
        await self.vector_search.load_index()

        self.recommendation_engine = RecommendationEngine(
            self.feature_store,
            self.vector_search,
            self.config.recommendation_config,
        )
        await self.recommendation_engine.load_models()

    async def run(self):
        self.running = True
        interval = self.config.service_topology_config.trainer_interval_seconds
        logger.info(f"Model trainer service running with interval={interval}s")
        while self.running:
            try:
                await asyncio.sleep(interval)
                await self.recommendation_engine.update_models()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Model trainer iteration failed: {exc}")
                await asyncio.sleep(60)

    async def shutdown(self):
        self.running = False
        if self.feature_store:
            await self.feature_store.close()


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    config = Config()
    service = ModelTrainerService(config)
    await service.initialize()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def handle_stop():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_stop)

    trainer_task = asyncio.create_task(service.run())
    await stop_event.wait()
    trainer_task.cancel()
    try:
        await trainer_task
    except asyncio.CancelledError:
        pass
    await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
