"""
Video Processor Worker
======================

Kafka consumer worker that processes video content uploaded by users.
Consumes messages from the video-processing-tasks topic and extracts
multi-modal features using CLIP, OCR, and audio analysis.

Usage:
    python -m kafka_workers.video_processor
"""

import asyncio
import logging
import signal
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional, List
from kafka_client import KafkaConsumerClient, KafkaManager, get_kafka_manager
from config import Config, KafkaConfig
from content_processor import ContentProcessor
from feature_store import FeatureStore
from vector_search import VectorSearchEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        self.consumer: Optional[KafkaConsumerClient] = None
        
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        
        logger.info("VideoProcessorWorker initialized")
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing video processor components...")
        
        # Initialize feature store (Redis)
        self.feature_store = FeatureStore(self.config.redis_config)
        await self.feature_store.initialize()
        
        # Initialize content processor
        self.content_processor = ContentProcessor(self.config.model_config)
        await self.content_processor.load_models()
        
        # Initialize vector search
        self.vector_search = VectorSearchEngine(self.config.vector_config)
        await self.vector_search.load_index()
        
        # Initialize Kafka consumer
        self.consumer = KafkaConsumerClient(
            self.kafka_config, 
            group_id=f"{self.kafka_config.consumer_group_id}-video-processor"
        )
        self.consumer.register_handler(
            self.kafka_config.video_processing_topic,
            self._handle_video_task
        )
        
        await self.consumer.start([self.kafka_config.video_processing_topic])
        
        logger.info("Video processor components initialized successfully")
    
    async def _handle_video_task(
        self, 
        topic: str, 
        key: Optional[str], 
        value: Dict[str, Any],
        headers: Optional[List[tuple]]
    ):
        """
        Handle a video processing task from Kafka.
        
        Args:
            topic: Source topic
            key: Message key (content_id)
            value: Task payload
            headers: Message headers
        """
        content_id = value.get('content_id')
        file_path = value.get('file_path')
        user_id = value.get('user_id')
        priority = value.get('priority', 'normal')
        
        logger.info(f"Processing video task: {content_id} (priority: {priority})")
        
        try:
            # Update status to processing
            await self.feature_store.update_content_status(content_id, "processing")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"Video file not found: {file_path}")
                await self.feature_store.update_content_status(content_id, "failed")
                return
            
            # Extract features from video
            features = await self.content_processor.process_video(file_path, content_id)
            
            # Store features in Redis
            await self.feature_store.store_content_features(content_id, features)
            
            # Update vector search index
            if features.visual_embedding:
                await self.vector_search.add_content_embedding(
                    content_id, 
                    features.visual_embedding
                )
            
            # Update status to completed
            await self.feature_store.update_content_status(content_id, "completed")
            
            logger.info(f"Video processing completed: {content_id}")
            
            # Send feature update event via Kafka
            kafka_manager = get_kafka_manager()
            await kafka_manager.send_feature_update(
                entity_type='content',
                entity_id=content_id,
                feature_updates={
                    'status': 'completed',
                    'has_visual_embedding': features.visual_embedding is not None,
                    'detected_objects': features.detected_objects,
                    'processing_time': features.processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing video {content_id}: {e}")
            await self.feature_store.update_content_status(content_id, "failed")
        
        finally:
            # Cleanup temporary file
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup file {file_path}: {e}")
    
    async def run(self):
        """Run the video processor worker."""
        self.is_running = True
        logger.info("Starting video processor worker...")
        
        try:
            # Start consuming messages
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
