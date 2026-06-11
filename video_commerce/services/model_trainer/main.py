"""
Offline model trainer service.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket

from video_commerce.common.config import Config
from video_commerce.data_plane.feature_store import FeatureStore
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.ml.ranking import RankingModel
from video_commerce.ml.recommender import RecommendationEngine
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.ml.vector_search import VectorSearchEngine
from video_commerce.common.observability import ObservabilityManager, configure_logging, start_worker_metrics_server
from video_commerce.common.telemetry import configure_tracing

logger = logging.getLogger(__name__)


class ModelTrainerService:
    """Dedicated background service for model retraining."""

    def __init__(self, config: Config):
        self.config = config
        self.feature_store: FeatureStore | None = None
        self.vector_search: VectorSearchEngine | None = None
        self.recommendation_engine: RecommendationEngine | None = None
        self.ranking_model: RankingModel | None = None
        self.system_store: SystemStore | None = None
        self.object_storage: ObjectStorage | None = None
        self.artifact_manager: ModelArtifactManager | None = None
        self.observability = ObservabilityManager()
        self.running = False
        self._heartbeat_task: asyncio.Task | None = None
        self.instance_id = f"model-trainer-{socket.gethostname()}-{os.getpid()}"

    def _ensure_heartbeat_task(self) -> None:
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._publish_heartbeat())

    async def initialize(self):
        configure_logging(self.config.monitoring_config)
        configure_tracing("model-trainer", self.config.monitoring_config)
        metrics_port = start_worker_metrics_server(
            self.observability,
            self.config.monitoring_config,
            default_port=9103,
        )
        if metrics_port:
            logger.info("model_trainer_metrics_server_started", extra={"port": metrics_port})

        self.feature_store = FeatureStore(self.config.redis_config, self.config.cache_config)
        await self.feature_store.initialize()
        self.running = True
        self._ensure_heartbeat_task()

        if self.config.database_config.enable:
            self.system_store = SystemStore(
                self.config.database_config,
                observability=self.observability,
            )
            await self.system_store.initialize()

        self.object_storage = ObjectStorage(self.config.object_storage_config)
        await self.object_storage.initialize()
        self.artifact_manager = ModelArtifactManager(
            system_store=self.system_store,
            object_storage=self.object_storage,
            model_config=self.config.model_config,
            recommendation_config=self.config.recommendation_config,
        )

        self.vector_search = VectorSearchEngine(self.config.vector_config)
        await self.vector_search.load_index()

        self.recommendation_engine = RecommendationEngine(
            self.feature_store,
            self.vector_search,
            self.config.recommendation_config,
            artifact_manager=self.artifact_manager,
            training_sequence_lookback_days=(
                self.config.database_config.training_sequence_lookback_days
            ),
        )
        await self.recommendation_engine.load_models()

        self.ranking_model = RankingModel(self.config.ranking_config)
        ranking_checkpoint = None
        if self.artifact_manager:
            ranking_checkpoint = await self.artifact_manager.sync_latest_ranking_checkpoint()
        await self.ranking_model.load_model(self.config.model_config.ranking_model_path)
        if ranking_checkpoint:
            self.ranking_model.model_version = ranking_checkpoint.model_version

        if self.config.ranking_config.enable_periodic_training:
            await self._train_ranking_model(trigger="startup")
        else:
            logger.info("Ranking periodic training disabled")

    async def _publish_heartbeat(self) -> None:
        interval = self.config.monitoring_config.worker_heartbeat_interval_seconds
        ttl = self.config.monitoring_config.worker_heartbeat_ttl_seconds
        while self.running:
            try:
                await self.feature_store.write_service_heartbeat(
                    "model-trainer",
                    self.instance_id,
                    ttl,
                    {"pid": os.getpid()},
                )
                self.observability.update_worker_heartbeat("model-trainer", self.instance_id)
            except Exception as exc:
                logger.warning(f"Failed to publish trainer heartbeat: {exc}")
            await asyncio.sleep(interval)

    async def run(self):
        self.running = True
        interval = self.config.service_topology_config.trainer_interval_seconds
        logger.info(f"Model trainer service running with interval={interval}s")
        self._ensure_heartbeat_task()
        while self.running:
            try:
                await asyncio.sleep(interval)
                await self.recommendation_engine.update_models()
                await self._train_ranking_model(trigger="scheduled")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Model trainer iteration failed: {exc}")
                await asyncio.sleep(60)

    async def _train_ranking_model(self, trigger: str) -> None:
        """Retrain the ranking model from stored interaction logs and persist a checkpoint."""
        started_at = asyncio.get_running_loop().time()
        if not self.config.ranking_config.enable_periodic_training:
            return
        if not self.feature_store or not self.ranking_model:
            return
        if not self.system_store:
            logger.warning("Skipping ranking retrain because Postgres system store is unavailable")
            return

        training_sample_source = "interaction_events"
        interactions = []
        if (
            getattr(self.config.ranking_config, "ltr_pairwise_enabled", False)
            and hasattr(self.system_store, "get_ltr_training_impressions")
        ):
            impression_samples = await self.system_store.get_ltr_training_impressions(
                limit=50000,
                lookback_days=getattr(
                    self.config.database_config,
                    "ltr_impression_lookback_days",
                    30,
                ),
            )
            if len(impression_samples) >= self.config.ranking_config.training_min_samples:
                interactions = impression_samples
                training_sample_source = "recommendation_impressions"
            else:
                logger.info(
                    "Falling back to event-level ranking training data",
                    extra={
                        "trigger": trigger,
                        "impression_sample_count": len(impression_samples),
                        "min_samples": self.config.ranking_config.training_min_samples,
                    },
                )

        if not interactions:
            interactions = await self.system_store.get_training_interactions(limit=50000)
        if len(interactions) < self.config.ranking_config.training_min_samples:
            self.observability.record_training_run(
                trigger,
                "skipped_insufficient_samples",
                asyncio.get_running_loop().time() - started_at,
            )
            logger.info(
                "Skipping ranking retrain due to insufficient samples",
                extra={
                    "trigger": trigger,
                    "sample_count": len(interactions),
                    "training_sample_source": training_sample_source,
                    "min_samples": self.config.ranking_config.training_min_samples,
                },
            )
            return

        logger.info(
            "Starting ranking retrain",
            extra={
                "trigger": trigger,
                "sample_count": len(interactions),
                "training_sample_source": training_sample_source,
                "model_path": self.config.model_config.ranking_model_path,
            },
        )
        status = "success"
        try:
            user_features_map = await self.feature_store.get_all_user_features_map()
            product_metadata_map = (
                dict(self.vector_search.product_metadata)
                if self.vector_search and self.vector_search.product_metadata
                else {}
            )
            await self.ranking_model.train_model(
                interactions,
                user_features_map=user_features_map,
                product_metadata_map=product_metadata_map,
            )
            if self.ranking_model.is_trained and self.artifact_manager:
                record = await self.artifact_manager.persist_ranking_checkpoint(
                    local_path=self.ranking_model.loaded_model_path or self.config.model_config.ranking_model_path,
                    model_version=self.ranking_model.model_version,
                    payload={
                        "trigger": trigger,
                        "sample_count": len(interactions),
                        "last_training_time": self.ranking_model.last_training_time,
                        "feature_schema_version": self.ranking_model.feature_schema_version,
                        "training_data_source": self.ranking_model.training_data_source,
                        "training_sample_source": training_sample_source,
                    },
                )
                if record:
                    self.ranking_model.model_version = record.model_version
        except Exception:
            status = "error"
            raise
        finally:
            self.observability.record_training_run(
                trigger,
                status,
                asyncio.get_running_loop().time() - started_at,
            )

    async def shutdown(self):
        self.running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self.feature_store:
            await self.feature_store.close()
        if self.system_store:
            await self.system_store.close()


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
