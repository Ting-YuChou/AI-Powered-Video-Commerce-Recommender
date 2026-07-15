"""
Video Processor Worker
======================

Kafka consumer worker that processes video content uploaded by users.
Consumes messages from the video-processing-tasks topic and extracts
multi-modal features using CLIP, OCR, and audio analysis.

Usage:
    python -m video_commerce.services.content_worker.video_processor
"""

import asyncio
import logging
import signal
import sys
import os
from pathlib import Path
import socket
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional, List
from video_commerce.data_plane.kafka_client import (
    KafkaConsumerClient,
    KafkaManager,
    get_kafka_manager,
)
from video_commerce.common.config import Config, KafkaConfig
from video_commerce.ml.content_processor import ContentProcessor
from video_commerce.ml.content_artifacts import persist_content_features
from video_commerce.data_plane.feature_store import FeatureStore
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.ml.vector_search import VectorSearchEngine
from video_commerce.common.observability import (
    ObservabilityManager,
    configure_logging,
    start_worker_metrics_server,
)
from video_commerce.common.telemetry import configure_tracing

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VideoProcessorWorker:
    """
    Kafka worker for processing video content.

    Consumes video processing tasks from Kafka and extracts features
    using the ContentProcessor. Results are stored in the FeatureStore
    and vector search index.
    """

    def __init__(self, config: Config):
        """Initialize the video processor worker."""
        self.config = config
        self.kafka_config = config.kafka_config

        # Initialize components
        self.content_processor: Optional[ContentProcessor] = None
        self.feature_store: Optional[FeatureStore] = None
        self.vector_search: Optional[VectorSearchEngine] = None
        self.system_store: Optional[SystemStore] = None
        self.object_storage: Optional[ObjectStorage] = None
        self.consumer: Optional[KafkaConsumerClient] = None
        self.observability = ObservabilityManager()

        self.is_running = False
        self._shutdown_event = asyncio.Event()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self.instance_id = f"content-worker-{socket.gethostname()}-{os.getpid()}"

        logger.info("VideoProcessorWorker initialized")

    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing video processor components...")
        configure_logging(self.config.monitoring_config)
        configure_tracing("content-worker", self.config.monitoring_config)
        metrics_port = start_worker_metrics_server(
            self.observability,
            self.config.monitoring_config,
            default_port=9101,
        )
        if metrics_port:
            logger.info(
                "content_worker_metrics_server_started", extra={"port": metrics_port}
            )

        # Initialize feature store (Redis)
        self.feature_store = FeatureStore(
            self.config.redis_config, self.config.cache_config
        )
        await self.feature_store.initialize()

        if self.config.database_config.enable:
            self.system_store = SystemStore(
                self.config.database_config,
                observability=self.observability,
            )
            await self.system_store.initialize()

        self.object_storage = ObjectStorage(self.config.object_storage_config)
        await self.object_storage.initialize()

        # Initialize content processor
        self.content_processor = ContentProcessor(self.config.model_config)
        await self.content_processor.load_models()

        # Initialize vector search
        self.vector_search = VectorSearchEngine(self.config.vector_config)
        await self.vector_search.load_index()

        # Initialize Kafka consumer
        self.consumer = KafkaConsumerClient(
            self.kafka_config,
            group_id=f"{self.kafka_config.consumer_group_id}-video-processor",
            observability=self.observability,
        )
        self.consumer.register_handler(
            self.kafka_config.video_processing_topic, self._handle_video_task
        )

        await self.consumer.start([self.kafka_config.video_processing_topic])

        logger.info("Video processor components initialized successfully")

    async def _handle_video_task(
        self,
        topic: str,
        key: Optional[str],
        value: Dict[str, Any],
        headers: Optional[List[tuple]],
    ):
        """
        Handle a video processing task from Kafka.

        Args:
            topic: Source topic
            key: Message key (content_id)
            value: Task payload
            headers: Message headers
        """
        content_id = value.get("content_id")
        file_path = value.get("file_path")
        filename = value.get("filename")
        user_id = value.get("user_id")
        priority = value.get("priority", "normal")
        request_id = value.get("request_id")
        started_at = time.perf_counter()
        status = "success"

        logger.info(
            "Processing video task",
            extra={
                "content_id": content_id,
                "priority": priority,
                "filename": filename,
                "request_id": request_id,
            },
        )

        processing_path = file_path
        cleanup_processing_path = False
        try:
            # Update status to processing
            await self.feature_store.update_content_status(content_id, "processing")
            if self.system_store:
                await self.system_store.update_content_job_status(
                    content_id,
                    "processing",
                    storage_path=file_path,
                    payload={
                        "request_id": request_id,
                        "filename": filename,
                    },
                )

            if self.object_storage:
                (
                    processing_path,
                    cleanup_processing_path,
                ) = await self.object_storage.materialize_for_processing(
                    file_path,
                    suggested_suffix=Path(filename or file_path).suffix,
                )

            # Check if file exists
            if not processing_path or not os.path.exists(processing_path):
                logger.error(f"Video file not found: {file_path}")
                await self.feature_store.update_content_status(content_id, "failed")
                if self.system_store:
                    await self.system_store.update_content_job_status(
                        content_id,
                        "failed",
                        error_message="Uploaded file missing before processing",
                        storage_path=file_path,
                        payload={
                            "request_id": request_id,
                            "filename": filename,
                        },
                    )
                return

            # Extract features from video
            features = await self.content_processor.process_video(
                processing_path, content_id
            )

            # Publish immutable training bytes before advancing current pointers.
            features = await persist_content_features(
                features,
                object_storage=self.object_storage,
                system_store=self.system_store,
                feature_store=self.feature_store,
            )

            audio_features = features.audio_features
            transcription_status = (
                audio_features.transcription_status
                if audio_features is not None
                else "not_attempted"
            )
            self.observability.record_asr_transcription(
                transcription_status,
                float(
                    audio_features.transcription_time_seconds
                    if audio_features and audio_features.transcription_time_seconds
                    else 0.0
                ),
            )
            if audio_features is not None:
                self.observability.record_asr_alignment(
                    audio_features.alignment_status,
                    float(audio_features.transcription_time_seconds or 0.0),
                    len(audio_features.asr_segments),
                )

            # Update vector search index
            if features.visual_embedding:
                await self.vector_search.add_content_embedding(
                    content_id, features.visual_embedding
                )

            # Update status to completed
            await self.feature_store.update_content_status(content_id, "completed")
            if self.system_store:
                await self.system_store.update_content_job_status(
                    content_id,
                    "completed",
                    storage_path=file_path,
                    payload={
                        "request_id": request_id,
                        "filename": filename,
                        "processing_time": features.processing_time,
                    },
                )

            logger.info(f"Video processing completed: {content_id}")

            # Send feature update event via Kafka
            kafka_manager = get_kafka_manager(observability=self.observability)
            if kafka_manager:
                await kafka_manager.send_feature_update(
                    entity_type="content",
                    entity_id=content_id,
                    feature_updates={
                        "status": "completed",
                        "has_visual_embedding": features.visual_embedding is not None,
                        "detected_objects": features.detected_objects,
                        "has_transcript": bool(
                            audio_features and audio_features.audio_transcript
                        ),
                        "transcription_status": transcription_status,
                        "has_speech_categories": bool(
                            audio_features and audio_features.speech_categories
                        ),
                        "processing_time": features.processing_time,
                    },
                    request_id=request_id,
                )

        except Exception as e:
            status = "error"
            logger.error(f"Error processing video {content_id}: {e}")
            await self.feature_store.update_content_status(content_id, "failed")
            if self.system_store:
                await self.system_store.update_content_job_status(
                    content_id,
                    "failed",
                    error_message=str(e),
                    storage_path=file_path,
                    payload={
                        "request_id": request_id,
                        "filename": filename,
                    },
                )

        finally:
            self.observability.record_worker_message(
                "content-worker",
                topic,
                status,
                time.perf_counter() - started_at,
            )
            # Cleanup temporary file
            try:
                if (
                    cleanup_processing_path
                    and processing_path
                    and os.path.exists(processing_path)
                ):
                    os.remove(processing_path)
                    logger.debug(f"Cleaned up materialized file: {processing_path}")
                elif (
                    self.config.data_config.cleanup_temp_files
                    and file_path
                    and file_path == processing_path
                    and os.path.exists(file_path)
                ):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup file {processing_path or file_path}: {e}"
                )

    async def _publish_heartbeat(self):
        """Publish worker liveness into Redis for readiness checks."""
        interval = self.config.monitoring_config.worker_heartbeat_interval_seconds
        ttl = self.config.monitoring_config.worker_heartbeat_ttl_seconds
        while self.is_running:
            try:
                await self.feature_store.write_service_heartbeat(
                    "content-worker",
                    self.instance_id,
                    ttl,
                    {"pid": os.getpid()},
                )
                self.observability.update_worker_heartbeat(
                    "content-worker", self.instance_id
                )
            except Exception as exc:
                logger.warning(f"Failed to publish content worker heartbeat: {exc}")
            await asyncio.sleep(interval)

    async def run(self):
        """Run the video processor worker."""
        self.is_running = True
        logger.info("Starting video processor worker...")

        try:
            # Start consuming messages
            self._heartbeat_task = asyncio.create_task(self._publish_heartbeat())
            await self.consumer.consume()

        except asyncio.CancelledError:
            logger.info("Video processor worker cancelled")
        except Exception as e:
            logger.error(f"Error in video processor worker: {e}")
        finally:
            self.is_running = False

    async def shutdown(self):
        """Gracefully shutdown the worker."""
        logger.info("Shutting down video processor worker...")
        self.is_running = False

        if self.consumer:
            await self.consumer.stop()

        if self.feature_store:
            await self.feature_store.close()
        if self.system_store:
            await self.system_store.close()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        self._shutdown_event.set()
        logger.info("Video processor worker shutdown complete")


async def main():
    """Main entry point for the video processor worker."""
    logger.info("Starting Video Processor Worker")

    # Load configuration
    config = Config()

    # Create and initialize worker
    worker = VideoProcessorWorker(config)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(worker.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Initialize components
        await worker.initialize()

        # Run the worker
        await worker.run()

    except Exception as e:
        logger.error(f"Fatal error in video processor worker: {e}")
        raise
    finally:
        await worker.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
