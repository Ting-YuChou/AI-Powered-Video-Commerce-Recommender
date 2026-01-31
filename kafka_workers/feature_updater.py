"""
Feature Updater Worker
======================

Kafka consumer worker that processes user interaction events and updates
user features in real-time. Consumes messages from the user-interactions
topic and batch updates features in the FeatureStore.

Usage:
    python -m kafka_workers.feature_updater
"""

import asyncio
import logging
import signal
import sys
import os
from collections import defaultdict
from typing import Dict, Any, Optional, List
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kafka_client import KafkaConsumerClient, KafkaManager, get_kafka_manager
from config import Config, KafkaConfig
from feature_store import FeatureStore
from models import InteractionType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureUpdaterWorker:
    """
    Kafka worker for updating user and content features.
    
    Consumes user interaction events from Kafka and updates features
    in the FeatureStore. Supports batch processing for efficiency.
    """
    
    def __init__(self, config: Config):
        """Initialize the feature updater worker."""
        self.config = config
        self.kafka_config = config.kafka_config
        
        # Initialize components
        self.feature_store: Optional[FeatureStore] = None
        self.consumer: Optional[KafkaConsumerClient] = None
        
        # Batch processing settings
        self.batch_size = 100
        self.batch_timeout_seconds = 5.0
        self._pending_updates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._last_flush_time = time.time()
        
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        
        logger.info("FeatureUpdaterWorker initialized")
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing feature updater components...")
        
        # Initialize feature store (Redis)
        self.feature_store = FeatureStore(self.config.redis_config)
        await self.feature_store.initialize()
        
        # Initialize Kafka consumer
        self.consumer = KafkaConsumerClient(
            self.kafka_config,
            group_id=f"{self.kafka_config.consumer_group_id}-feature-updater"
        )
        self.consumer.register_handler(
            self.kafka_config.user_interactions_topic,
            self._handle_user_interaction
        )
        
        await self.consumer.start([self.kafka_config.user_interactions_topic])
        
        logger.info("Feature updater components initialized successfully")
    
    async def _handle_user_interaction(
        self,
        topic: str,
        key: Optional[str],
        value: Dict[str, Any],
        headers: Optional[List[tuple]]
    ):
        """
        Handle a user interaction event from Kafka.
        
        Args:
            topic: Source topic
            key: Message key (user_id)
            value: Interaction payload
            headers: Message headers
        """
        user_id = value.get('user_id')
        product_id = value.get('product_id')
        action = value.get('action')
        context = value.get('context', {})
        timestamp = value.get('timestamp', time.time())
        
        logger.debug(f"Received interaction: {user_id} -> {action} -> {product_id}")
        
        # Add to pending updates
        self._pending_updates[user_id].append({
            'product_id': product_id,
            'action': action,
            'context': context,
            'timestamp': timestamp
        })
        
        # Check if we should flush
        total_pending = sum(len(updates) for updates in self._pending_updates.values())
        time_since_flush = time.time() - self._last_flush_time
        
        if total_pending >= self.batch_size or time_since_flush >= self.batch_timeout_seconds:
            await self._flush_pending_updates()
    
    async def _flush_pending_updates(self):
        """Flush pending updates to the feature store."""
        if not self._pending_updates:
            return
        
        logger.info(f"Flushing {sum(len(u) for u in self._pending_updates.values())} pending updates")
        
        try:
            # Process updates for each user
            for user_id, interactions in self._pending_updates.items():
                await self._process_user_interactions(user_id, interactions)
            
            # Clear pending updates
            self._pending_updates.clear()
            self._last_flush_time = time.time()
            
        except Exception as e:
            logger.error(f"Error flushing updates: {e}")
    
    async def _process_user_interactions(
        self,
        user_id: str,
        interactions: List[Dict[str, Any]]
    ):
        """
        Process a batch of interactions for a single user.
        
        Args:
            user_id: User identifier
            interactions: List of interactions to process
        """
        try:
            # Log interactions to Redis
            for interaction in interactions:
                await self.feature_store.log_user_interaction(
                    user_id=user_id,
                    product_id=interaction['product_id'],
                    action=interaction['action'],
                    context=interaction.get('context', {})
                )
            
            # Aggregate stats for batch update
            action_counts = defaultdict(int)
            categories = set()
            
            for interaction in interactions:
                action_counts[interaction['action']] += 1
                if 'product_category' in interaction.get('context', {}):
                    categories.add(interaction['context']['product_category'])
            
            # Update user features
            for action, count in action_counts.items():
                for _ in range(count):
                    await self.feature_store.update_user_features(
                        user_id=user_id,
                        action=action,
                        context={'batch_update': True}
                    )
            
            logger.debug(f"Processed {len(interactions)} interactions for user {user_id}")
            
            # Send feature update notification
            kafka_manager = get_kafka_manager()
            await kafka_manager.send_feature_update(
                entity_type='user',
                entity_id=user_id,
                feature_updates={
                    'interactions_processed': len(interactions),
                    'action_counts': dict(action_counts),
                    'updated_at': time.time()
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing interactions for user {user_id}: {e}")
    
    async def _periodic_flush(self):
        """Periodically flush pending updates."""
        while self.is_running:
            await asyncio.sleep(self.batch_timeout_seconds)
            if self._pending_updates:
                await self._flush_pending_updates()
    
    async def run(self):
        """Run the feature updater worker."""
        self.is_running = True
        logger.info("Starting feature updater worker...")
        
        try:
            # Start periodic flush task
            flush_task = asyncio.create_task(self._periodic_flush())
            
            # Start consuming messages
            consume_task = asyncio.create_task(self.consumer.consume())
            
            # Wait for either task to complete (or be cancelled)
            done, pending = await asyncio.wait(
                [flush_task, consume_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
        except asyncio.CancelledError:
            logger.info("Feature updater worker cancelled")
        except Exception as e:
            logger.error(f"Error in feature updater worker: {e}")
        finally:
            self.is_running = False
            # Final flush
            await self._flush_pending_updates()
    
    async def shutdown(self):
        """Gracefully shutdown the worker."""
        logger.info("Shutting down feature updater worker...")
        self.is_running = False
        
        # Flush any remaining updates
        await self._flush_pending_updates()
        
        if self.consumer:
            await self.consumer.stop()
        
        if self.feature_store:
            await self.feature_store.close()
        
        self._shutdown_event.set()
        logger.info("Feature updater worker shutdown complete")


class AnalyticsWorker:
    """
    Kafka worker for real-time analytics processing.
    
    Consumes recommendation events and user interactions to calculate
    real-time metrics like CTR, CVR, and trending products.
    """
    
    def __init__(self, config: Config):
        """Initialize the analytics worker."""
        self.config = config
        self.kafka_config = config.kafka_config
        
        self.feature_store: Optional[FeatureStore] = None
        self.consumer: Optional[KafkaConsumerClient] = None
        
        # Real-time metrics
        self._metrics = {
            'total_recommendations': 0,
            'total_clicks': 0,
            'total_purchases': 0,
            'response_times': []
        }
        
        self.is_running = False
        logger.info("AnalyticsWorker initialized")
    
    async def initialize(self):
        """Initialize components."""
        self.feature_store = FeatureStore(self.config.redis_config)
        await self.feature_store.initialize()
        
        self.consumer = KafkaConsumerClient(
            self.kafka_config,
            group_id=f"{self.kafka_config.consumer_group_id}-analytics"
        )
        
        # Register handlers for multiple topics
        self.consumer.register_handler(
            self.kafka_config.recommendation_events_topic,
            self._handle_recommendation_event
        )
        self.consumer.register_handler(
            self.kafka_config.user_interactions_topic,
            self._handle_interaction_for_analytics
        )
        
        await self.consumer.start([
            self.kafka_config.recommendation_events_topic,
            self.kafka_config.user_interactions_topic
        ])
    
    async def _handle_recommendation_event(
        self,
        topic: str,
        key: Optional[str],
        value: Dict[str, Any],
        headers: Optional[List[tuple]]
    ):
        """Handle recommendation events for analytics."""
        self._metrics['total_recommendations'] += value.get('num_recommendations', 0)
        
        response_time = value.get('response_time_ms', 0)
        self._metrics['response_times'].append(response_time)
        
        # Keep only last 1000 response times for rolling average
        if len(self._metrics['response_times']) > 1000:
            self._metrics['response_times'] = self._metrics['response_times'][-1000:]
        
        logger.debug(f"Analytics: recommendation event processed")
    
    async def _handle_interaction_for_analytics(
        self,
        topic: str,
        key: Optional[str],
        value: Dict[str, Any],
        headers: Optional[List[tuple]]
    ):
        """Handle interaction events for analytics."""
        action = value.get('action')
        
        if action == InteractionType.CLICK.value:
            self._metrics['total_clicks'] += 1
        elif action == InteractionType.PURCHASE.value:
            self._metrics['total_purchases'] += 1
        
        logger.debug(f"Analytics: interaction event processed")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current analytics metrics."""
        avg_response_time = (
            sum(self._metrics['response_times']) / len(self._metrics['response_times'])
            if self._metrics['response_times'] else 0
        )
        
        ctr = (
            self._metrics['total_clicks'] / self._metrics['total_recommendations']
            if self._metrics['total_recommendations'] > 0 else 0
        )
        
        cvr = (
            self._metrics['total_purchases'] / self._metrics['total_clicks']
            if self._metrics['total_clicks'] > 0 else 0
        )
        
        return {
            'total_recommendations': self._metrics['total_recommendations'],
            'total_clicks': self._metrics['total_clicks'],
            'total_purchases': self._metrics['total_purchases'],
            'ctr': round(ctr, 4),
            'cvr': round(cvr, 4),
            'avg_response_time_ms': round(avg_response_time, 2)
        }
    
    async def run(self):
        """Run the analytics worker."""
        self.is_running = True
        await self.consumer.consume()
    
    async def shutdown(self):
        """Shutdown the analytics worker."""
        self.is_running = False
        if self.consumer:
            await self.consumer.stop()
        if self.feature_store:
            await self.feature_store.close()


async def main():
    """Main entry point for the feature updater worker."""
    logger.info("Starting Feature Updater Worker")
    
    # Load configuration
    config = Config()
    
    # Create and initialize worker
    worker = FeatureUpdaterWorker(config)
    
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
        logger.error(f"Fatal error in feature updater worker: {e}")
        raise
    finally:
        await worker.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
