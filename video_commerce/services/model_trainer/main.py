"""
Offline model trainer service.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import os
import signal
import socket
import threading

from video_commerce.common.config import Config
from video_commerce.data_plane.feature_store import FeatureStore
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.ml.ranking import RankingModel
from video_commerce.ml.legacy_training_adapter import LegacyTrainingDatasetAdapter
from video_commerce.ml.pit_training_dataset import (
    PitTrainingDatasetError,
    PitTrainingDatasetReader,
    PitTrainingDatasetUnavailable,
)
from video_commerce.ml.recommender import RecommendationEngine
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.ml.vector_search import VectorSearchEngine
from video_commerce.common.observability import (
    ObservabilityManager,
    configure_logging,
    start_worker_metrics_server,
)
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
        self.pit_dataset_reader: PitTrainingDatasetReader | None = None
        self.last_trained_pit_run_id: str | None = None
        self.legacy_training_adapter: LegacyTrainingDatasetAdapter | None = None
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
            logger.info(
                "model_trainer_metrics_server_started", extra={"port": metrics_port}
            )

        feature_lake_config = getattr(self.config, "feature_lake_config", None)
        use_pit_dataset = (
            getattr(feature_lake_config, "training_source", "legacy") == "pit"
        )
        if not use_pit_dataset:
            self.feature_store = FeatureStore(
                self.config.redis_config, self.config.cache_config
            )
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
        if use_pit_dataset or getattr(feature_lake_config, "pit_shadow_enabled", False):
            self.pit_dataset_reader = PitTrainingDatasetReader(
                self.object_storage,
                expected_feature_definition_version=(
                    feature_lake_config.feature_definition_version
                ),
                observability=self.observability,
            )
        self.artifact_manager = ModelArtifactManager(
            system_store=self.system_store,
            object_storage=self.object_storage,
            model_config=self.config.model_config,
            recommendation_config=self.config.recommendation_config,
        )

        if not use_pit_dataset:
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

        self.ranking_model = RankingModel(
            self.config.ranking_config,
            observability=self.observability,
        )
        if not use_pit_dataset:
            self.legacy_training_adapter = LegacyTrainingDatasetAdapter(
                feature_store=self.feature_store,
                vector_search=self.vector_search,
                ranking_model=self.ranking_model,
                recommendation_engine=self.recommendation_engine,
            )
        ranking_checkpoint = None
        if self.artifact_manager:
            ranking_checkpoint = (
                await self.artifact_manager.sync_latest_ranking_checkpoint()
            )
        loaded_existing_checkpoint = False
        try:
            await self.ranking_model.load_model(
                self.config.model_config.ranking_model_path
            )
            loaded_existing_checkpoint = True
        except RuntimeError:
            if (
                not self.config.ranking_config.history_embeddings_enabled
                or not self.config.ranking_config.enable_periodic_training
            ):
                raise
            logger.warning(
                "ranking_checkpoint_incompatible_for_history_embeddings_fresh_train",
                extra={"model_path": self.config.model_config.ranking_model_path},
            )
            self.ranking_model._initialize_model(
                architecture=self.config.ranking_config.architecture
            )
            if self.ranking_model.model:
                self.ranking_model.model.eval()
            self.ranking_model.is_trained = False
            self.ranking_model.loaded_model_path = (
                self.config.model_config.ranking_model_path
            )
            self.ranking_model.loaded_checkpoint_mtime = 0.0
        if ranking_checkpoint and loaded_existing_checkpoint:
            self.ranking_model.model_version = ranking_checkpoint.model_version
            self.last_trained_pit_run_id = (
                str(
                    ranking_checkpoint.payload.get(
                        "feature_lake_materialization_run_id"
                    )
                    or ""
                ).strip()
                or None
            )

        if self.config.ranking_config.enable_periodic_training:
            await self._train_ranking_model(trigger="startup")
        else:
            logger.info("Ranking periodic training disabled")

    async def _publish_heartbeat(self) -> None:
        interval = self.config.monitoring_config.worker_heartbeat_interval_seconds
        ttl = self.config.monitoring_config.worker_heartbeat_ttl_seconds
        while self.running:
            try:
                if self.feature_store is not None:
                    await self.feature_store.write_service_heartbeat(
                        "model-trainer",
                        self.instance_id,
                        ttl,
                        {"pid": os.getpid()},
                    )
                self.observability.update_worker_heartbeat(
                    "model-trainer", self.instance_id
                )
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
                if self.recommendation_engine is not None:
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
        feature_lake_config = getattr(self.config, "feature_lake_config", None)
        use_pit_dataset = (
            getattr(feature_lake_config, "training_source", "legacy") == "pit"
        )
        if not self.ranking_model:
            return
        if not use_pit_dataset and not self.feature_store:
            return
        if not use_pit_dataset and not self.system_store:
            logger.warning(
                "Skipping ranking retrain because Postgres system store is unavailable"
            )
            return

        training_sample_source = "interaction_events"
        interactions = []
        pit_dataset = None
        if use_pit_dataset:
            if not self.pit_dataset_reader:
                raise RuntimeError(
                    "PIT ranking training is configured but the dataset reader is unavailable"
                )
            try:
                pit_dataset = await self.pit_dataset_reader.read(
                    feature_lake_config.ranking_pit_dataset_uri
                )
            except PitTrainingDatasetUnavailable:
                self.observability.record_training_run(
                    trigger,
                    "waiting_for_pit_manifest",
                    asyncio.get_running_loop().time() - started_at,
                )
                if hasattr(self.observability, "set_pit_trainer_waiting_for_manifest"):
                    self.observability.set_pit_trainer_waiting_for_manifest(True)
                return
            except PitTrainingDatasetError:
                if hasattr(self.observability, "set_pit_trainer_waiting_for_manifest"):
                    self.observability.set_pit_trainer_waiting_for_manifest(False)
                self.observability.record_training_run(
                    trigger,
                    "error",
                    asyncio.get_running_loop().time() - started_at,
                )
                raise
            if hasattr(self.observability, "set_pit_trainer_waiting_for_manifest"):
                self.observability.set_pit_trainer_waiting_for_manifest(False)
            if pit_dataset.materialization_run_id == getattr(
                self, "last_trained_pit_run_id", None
            ):
                self.observability.record_training_run(
                    trigger,
                    "skipped_duplicate_manifest",
                    asyncio.get_running_loop().time() - started_at,
                )
                if hasattr(self.observability, "record_pit_duplicate_manifest_skip"):
                    self.observability.record_pit_duplicate_manifest_skip()
                return
            interactions = pit_dataset.examples
            training_sample_source = "feature_lake_pit"
        elif (
            getattr(self.config.ranking_config, "ltr_pairwise_enabled", False)
            or getattr(self.config.ranking_config, "ltr_listwise_enabled", False)
        ) and hasattr(self.system_store, "get_ltr_training_impressions"):
            impression_samples = await self.system_store.get_ltr_training_impressions(
                limit=50000,
                lookback_days=getattr(
                    self.config.database_config,
                    "ltr_impression_lookback_days",
                    30,
                ),
            )
            if (
                len(impression_samples)
                >= self.config.ranking_config.training_min_samples
            ):
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

        if not interactions and not use_pit_dataset:
            interactions = await self.system_store.get_training_interactions(
                limit=50000
            )
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

        pit_training_claimed = False
        pit_training_heartbeat_task: asyncio.Task | None = None
        pit_training_lease_error: Exception | None = None
        pit_training_cancel_event: threading.Event | None = None
        if use_pit_dataset:
            if not self.system_store or not hasattr(
                self.system_store, "claim_pit_training_run"
            ):
                raise RuntimeError("PIT training requires durable Postgres run claims")
            training_claim = await self.system_store.claim_pit_training_run(
                run_id=pit_dataset.materialization_run_id,
                worker_id=getattr(self, "instance_id", "model-trainer"),
                lease_seconds=getattr(
                    feature_lake_config, "pit_training_lease_seconds", 21600
                ),
            )
            if not training_claim.get("claimed", False):
                self.observability.record_training_run(
                    trigger,
                    "skipped_duplicate_manifest",
                    asyncio.get_running_loop().time() - started_at,
                )
                if hasattr(self.observability, "record_pit_duplicate_manifest_skip"):
                    self.observability.record_pit_duplicate_manifest_skip()
                return
            pit_training_claimed = True
            pit_training_cancel_event = threading.Event()
            training_lease_seconds = int(
                getattr(feature_lake_config, "pit_training_lease_seconds", 21600)
            )
            heartbeat_interval = float(
                getattr(
                    self,
                    "_pit_training_heartbeat_interval_seconds",
                    max(30.0, min(300.0, training_lease_seconds / 3.0)),
                )
            )
            owner_task = asyncio.current_task()

            async def renew_training_lease() -> None:
                nonlocal pit_training_lease_error
                while True:
                    await asyncio.sleep(max(0.001, heartbeat_interval))
                    try:
                        await self.system_store.renew_pit_training_lease(
                            run_id=pit_dataset.materialization_run_id,
                            worker_id=getattr(self, "instance_id", "model-trainer"),
                            lease_seconds=training_lease_seconds,
                        )
                    except Exception as exc:
                        pit_training_lease_error = exc
                        pit_training_cancel_event.set()
                        if owner_task is not None:
                            owner_task.cancel()
                        return

            pit_training_heartbeat_task = asyncio.create_task(
                renew_training_lease(),
                name=f"pit-training-lease-{pit_dataset.materialization_run_id}",
            )

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
            if use_pit_dataset:
                training_examples = pit_dataset.examples
            else:
                adapter = getattr(self, "legacy_training_adapter", None)
                if adapter is None:
                    adapter = LegacyTrainingDatasetAdapter(
                        feature_store=self.feature_store,
                        vector_search=self.vector_search,
                        ranking_model=self.ranking_model,
                        recommendation_engine=getattr(
                            self, "recommendation_engine", None
                        ),
                    )
                    self.legacy_training_adapter = adapter
                training_examples = await adapter.build(
                    interactions,
                    training_sample_source=training_sample_source,
                )
            training_kwargs = {"training_sample_source": training_sample_source}
            if pit_training_cancel_event is not None:
                training_kwargs["cancellation_event"] = pit_training_cancel_event
            await self.ranking_model.train_model(training_examples, **training_kwargs)
            if use_pit_dataset and self.ranking_model.is_trained:
                self.ranking_model.model_version = (
                    f"pit-{pit_dataset.materialization_run_id}"
                )
                await self.ranking_model.save_model(
                    self.ranking_model.loaded_model_path
                    or self.config.model_config.ranking_model_path
                )
            if use_pit_dataset and hasattr(
                self.observability, "update_typed_pit_training_metrics"
            ):
                self.observability.update_typed_pit_training_metrics(
                    assembler_parity_ratio=1.0,
                    label_reconciliation_ratio=1.0,
                    current_state_calls=0,
                    invalid_rows=0,
                    value_mask_coverage=self._value_mask_coverage(training_examples),
                )
            if self.ranking_model.is_trained and self.artifact_manager:
                record = await self.artifact_manager.persist_ranking_checkpoint(
                    local_path=self.ranking_model.loaded_model_path
                    or self.config.model_config.ranking_model_path,
                    model_version=self.ranking_model.model_version,
                    payload={
                        "trigger": trigger,
                        "sample_count": len(interactions),
                        "last_training_time": self.ranking_model.last_training_time,
                        "feature_schema_version": self.ranking_model.feature_schema_version,
                        "training_data_source": self.ranking_model.training_data_source,
                        "training_sample_source": training_sample_source,
                        "feature_lake_dataset_version": (
                            pit_dataset.dataset_version if pit_dataset else None
                        ),
                        "feature_lake_materialization_run_id": (
                            pit_dataset.materialization_run_id if pit_dataset else None
                        ),
                        "feature_definition_version": (
                            getattr(pit_dataset, "feature_definition_version", None)
                            if pit_dataset
                            else None
                        ),
                        "feature_lake_manifest_uri": (
                            getattr(pit_dataset, "manifest_uri", None)
                            if pit_dataset
                            else None
                        ),
                        "feature_lake_iceberg_snapshot_id": (
                            getattr(pit_dataset, "iceberg_snapshot_id", None)
                            if pit_dataset
                            else None
                        ),
                        "feature_lake_schema_hash": (
                            getattr(pit_dataset, "schema_hash", None)
                            if pit_dataset
                            else None
                        ),
                        "label_definition_version": (
                            getattr(pit_dataset, "label_definition_version", None)
                            if pit_dataset
                            else None
                        ),
                        "feature_assembler_version": getattr(
                            getattr(self.ranking_model, "feature_assembler", None),
                            "version",
                            None,
                        ),
                        "training_input_rows": len(training_examples),
                        "training_quarantine_rows": (
                            getattr(pit_dataset, "quarantine_rows", 0)
                            if pit_dataset
                            else 0
                        ),
                        "training_value_mask_coverage": self._value_mask_coverage(
                            training_examples
                        ),
                    },
                )
                if record:
                    self.ranking_model.model_version = record.model_version
                    if use_pit_dataset:
                        self.last_trained_pit_run_id = (
                            pit_dataset.materialization_run_id
                        )
                        await self.system_store.complete_pit_training_run(
                            run_id=pit_dataset.materialization_run_id,
                            worker_id=getattr(self, "instance_id", "model-trainer"),
                            model_version=record.model_version,
                        )
                elif use_pit_dataset:
                    raise RuntimeError("PIT ranking artifact was not persisted")
            if not use_pit_dataset and getattr(
                feature_lake_config, "pit_shadow_enabled", False
            ):
                await self._train_pit_shadow_model(trigger=trigger)
        except asyncio.CancelledError as exc:
            status = "error"
            if pit_training_claimed:
                await self.system_store.fail_pit_training_run(
                    run_id=pit_dataset.materialization_run_id,
                    worker_id=getattr(self, "instance_id", "model-trainer"),
                )
            if pit_training_lease_error is not None:
                raise RuntimeError("PIT training lost its durable lease") from (
                    pit_training_lease_error
                )
            raise exc
        except Exception:
            status = "error"
            if pit_training_claimed:
                await self.system_store.fail_pit_training_run(
                    run_id=pit_dataset.materialization_run_id,
                    worker_id=getattr(self, "instance_id", "model-trainer"),
                )
            raise
        finally:
            if pit_training_heartbeat_task is not None:
                pit_training_heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await pit_training_heartbeat_task
            self.observability.record_training_run(
                trigger,
                status,
                asyncio.get_running_loop().time() - started_at,
            )

    @staticmethod
    def _value_mask_coverage(training_examples) -> float:
        purchases = [
            example.attribution
            for example in training_examples
            if example.attribution.attributed_purchase
        ]
        if not purchases:
            return 0.0
        return sum(fact.attributed_value is not None for fact in purchases) / len(
            purchases
        )

    async def _train_pit_shadow_model(self, *, trigger: str) -> None:
        """Train and persist PIT without replacing the serving ranking artifact."""
        if not self.pit_dataset_reader or not self.artifact_manager:
            raise RuntimeError("PIT shadow training dependencies are unavailable")
        feature_lake = self.config.feature_lake_config
        dataset = await self.pit_dataset_reader.read(
            feature_lake.ranking_pit_dataset_uri
        )
        if len(dataset.examples) < self.config.ranking_config.training_min_samples:
            self.observability.record_training_run(
                f"{trigger}_pit_shadow", "skipped_insufficient_samples", 0.0
            )
            return
        shadow_model = RankingModel(
            self.config.ranking_config,
            observability=self.observability,
        )
        shadow_path = f"{self.config.model_config.ranking_model_path}.pit-shadow"
        shadow_model.loaded_model_path = shadow_path
        shadow_started = asyncio.get_running_loop().time()
        status = "success"
        try:
            await shadow_model.train_model(
                dataset.examples,
                training_sample_source="feature_lake_pit_shadow",
            )
            record = await self.artifact_manager.persist_ranking_shadow_checkpoint(
                local_path=shadow_model.loaded_model_path,
                model_version=shadow_model.model_version,
                payload={
                    "trigger": trigger,
                    "training_sample_source": "feature_lake_pit_shadow",
                    "feature_lake_dataset_version": dataset.dataset_version,
                    "feature_lake_materialization_run_id": (
                        dataset.materialization_run_id
                    ),
                    "feature_lake_manifest_uri": dataset.manifest_uri,
                    "feature_lake_iceberg_snapshot_id": dataset.iceberg_snapshot_id,
                    "feature_lake_schema_hash": dataset.schema_hash,
                    "feature_definition_version": dataset.feature_definition_version,
                    "label_definition_version": dataset.label_definition_version,
                    "feature_assembler_version": shadow_model.feature_assembler.version,
                    "training_input_rows": len(dataset.examples),
                    "training_value_mask_coverage": self._value_mask_coverage(
                        dataset.examples
                    ),
                    "activation_allowed": False,
                },
            )
            if record is None:
                raise RuntimeError("PIT shadow artifact was not persisted")
        except Exception:
            status = "error"
            raise
        finally:
            self.observability.record_training_run(
                f"{trigger}_pit_shadow",
                status,
                asyncio.get_running_loop().time() - shadow_started,
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
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
