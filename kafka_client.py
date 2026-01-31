"""
AI-Powered Video Commerce Recommender - Kafka Client
=====================================================

This module provides async Kafka producer and consumer functionality
for event streaming in the video commerce recommendation system.

Features:
- Async producer for high-throughput message publishing
- Async consumer for processing messages from topics
- Automatic serialization/deserialization of JSON messages
- Connection management and health checks
- Graceful shutdown handling
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import time

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaConnectionError, KafkaError

from config import KafkaConfig

logger = logging.getLogger(__name__)


class KafkaProducerClient:
    """
    Async Kafka producer for publishing messages to topics.
    
    Provides high-throughput, async message publishing with automatic
    JSON serialization and delivery confirmation.
    """
    
    def __init__(self, config: KafkaConfig):
        """Initialize the Kafka producer with configuration."""
        self.config = config
        self.producer: Optional[AIOKafkaProducer] = None
        self.is_connected = False
        self._lock = asyncio.Lock()
        
        logger.info(f"KafkaProducerClient initialized with servers: {config.bootstrap_servers}")
    
    async def start(self):
        """Start the Kafka producer connection."""
        if not self.config.enable:
            logger.info("Kafka is disabled, skipping producer initialization")
            return
        
        async with self._lock:
            if self.is_connected:
                return
            
            try:
                self.producer = AIOKafkaProducer(
                    bootstrap_servers=self.config.bootstrap_servers,
                    acks=self.config.producer_acks,
                    retries=self.config.producer_retries,
                    batch_size=self.config.producer_batch_size,
                    linger_ms=self.config.producer_linger_ms,
                    compression_type=self.config.producer_compression_type,
                    request_timeout_ms=self.config.request_timeout_ms,
                    retry_backoff_ms=self.config.retry_backoff_ms,
                    max_in_flight_requests_per_connection=self.config.max_in_flight_requests,
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                
                await self.producer.start()
                self.is_connected = True
                logger.info("Kafka producer connected successfully")
                
            except KafkaConnectionError as e:
                logger.error(f"Failed to connect Kafka producer: {e}")
                self.is_connected = False
                raise
            except Exception as e:
                logger.error(f"Unexpected error starting Kafka producer: {e}")
                self.is_connected = False
                raise
    
    async def stop(self):
        """Stop the Kafka producer connection."""
        async with self._lock:
            if self.producer:
                try:
                    await self.producer.stop()
                    logger.info("Kafka producer stopped")
                except Exception as e:
                    logger.error(f"Error stopping Kafka producer: {e}")
                finally:
                    self.producer = None
                    self.is_connected = False
    
    async def send(
        self, 
        topic: str, 
        value: Dict[str, Any], 
        key: Optional[str] = None,
        headers: Optional[List[tuple]] = None
    ) -> bool:
        """
        Send a message to a Kafka topic.
        
        Args:
            topic: Target topic name
            value: Message payload (will be JSON serialized)
            key: Optional partition key
            headers: Optional message headers
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.config.enable:
            logger.debug(f"Kafka disabled, skipping message to topic: {topic}")
            return True
        
        if not self.is_connected or not self.producer:
            logger.warning("Kafka producer not connected, attempting to reconnect...")
            try:
                await self.start()
            except Exception as e:
                logger.error(f"Failed to reconnect Kafka producer: {e}")
                return False
        
        try:
            # Add timestamp to message
            value['_kafka_timestamp'] = datetime.utcnow().isoformat()
            value['_kafka_topic'] = topic
            
            # Send message
            await self.producer.send_and_wait(
                topic=topic,
                value=value,
                key=key,
                headers=headers
            )
            
            logger.debug(f"Message sent to topic '{topic}' with key '{key}'")
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to send message to topic '{topic}': {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}")
            return False
    
    async def send_batch(
        self, 
        topic: str, 
        messages: List[Dict[str, Any]],
        key_field: Optional[str] = None
    ) -> int:
        """
        Send multiple messages to a Kafka topic.
        
        Args:
            topic: Target topic name
            messages: List of message payloads
            key_field: Optional field name to use as partition key
            
        Returns:
            Number of successfully sent messages
        """
        if not self.config.enable:
            return len(messages)
        
        success_count = 0
        for msg in messages:
            key = msg.get(key_field) if key_field else None
            if await self.send(topic, msg, key):
                success_count += 1
        
        return success_count
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the Kafka producer."""
        return {
            'status': 'healthy' if self.is_connected else 'unhealthy',
            'connected': self.is_connected,
            'bootstrap_servers': self.config.bootstrap_servers,
            'enabled': self.config.enable
        }


class KafkaConsumerClient:
    """
    Async Kafka consumer for processing messages from topics.
    
    Provides message consumption with automatic JSON deserialization
    and callback-based message handling.
    """
    
    def __init__(self, config: KafkaConfig, group_id: Optional[str] = None):
        """Initialize the Kafka consumer with configuration."""
        self.config = config
        self.group_id = group_id or config.consumer_group_id
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.is_connected = False
        self.is_running = False
        self._lock = asyncio.Lock()
        self._handlers: Dict[str, Callable] = {}
        
        logger.info(f"KafkaConsumerClient initialized with group: {self.group_id}")
    
    def register_handler(self, topic: str, handler: Callable):
        """
        Register a message handler for a topic.
        
        Args:
            topic: Topic to handle
            handler: Async callback function that receives (topic, key, value, headers)
        """
        self._handlers[topic] = handler
        logger.info(f"Registered handler for topic: {topic}")
    
    async def start(self, topics: List[str]):
        """Start the Kafka consumer and subscribe to topics."""
        if not self.config.enable:
            logger.info("Kafka is disabled, skipping consumer initialization")
            return
        
        async with self._lock:
            if self.is_connected:
                return
            
            try:
                self.consumer = AIOKafkaConsumer(
                    *topics,
                    bootstrap_servers=self.config.bootstrap_servers,
                    group_id=self.group_id,
                    auto_offset_reset=self.config.consumer_auto_offset_reset,
                    enable_auto_commit=self.config.consumer_enable_auto_commit,
                    auto_commit_interval_ms=self.config.consumer_auto_commit_interval_ms,
                    max_poll_records=self.config.consumer_max_poll_records,
                    session_timeout_ms=self.config.session_timeout_ms,
                    request_timeout_ms=self.config.request_timeout_ms,
                    key_deserializer=lambda k: k.decode('utf-8') if k else None,
                    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
                )
                
                await self.consumer.start()
                self.is_connected = True
                logger.info(f"Kafka consumer connected, subscribed to topics: {topics}")
                
            except KafkaConnectionError as e:
                logger.error(f"Failed to connect Kafka consumer: {e}")
                self.is_connected = False
                raise
            except Exception as e:
                logger.error(f"Unexpected error starting Kafka consumer: {e}")
                self.is_connected = False
                raise
    
    async def stop(self):
        """Stop the Kafka consumer."""
        self.is_running = False
        
        async with self._lock:
            if self.consumer:
                try:
                    await self.consumer.stop()
                    logger.info("Kafka consumer stopped")
                except Exception as e:
                    logger.error(f"Error stopping Kafka consumer: {e}")
                finally:
                    self.consumer = None
                    self.is_connected = False
    
    async def consume(self):
        """
        Start consuming messages from subscribed topics.
        
        This method runs indefinitely, processing messages as they arrive.
        """
        if not self.config.enable or not self.is_connected:
            logger.warning("Kafka consumer not ready, cannot consume")
            return
        
        self.is_running = True
        logger.info("Starting message consumption loop")
        
        try:
            async for message in self.consumer:
                if not self.is_running:
                    break
                
                try:
                    topic = message.topic
                    key = message.key
                    value = message.value
                    headers = message.headers
                    
                    # Get registered handler for topic
                    handler = self._handlers.get(topic)
                    if handler:
                        await handler(topic, key, value, headers)
                    else:
                        logger.warning(f"No handler registered for topic: {topic}")
                    
                except Exception as e:
                    logger.error(f"Error processing message from {message.topic}: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Consumer loop cancelled")
        except Exception as e:
            logger.error(f"Error in consumer loop: {e}")
        finally:
            self.is_running = False
    
    async def consume_batch(self, timeout_ms: int = 1000) -> List[Dict[str, Any]]:
        """
        Consume a batch of messages.
        
        Args:
            timeout_ms: Timeout for polling in milliseconds
            
        Returns:
            List of consumed messages
        """
        if not self.config.enable or not self.is_connected:
            return []
        
        messages = []
        try:
            data = await self.consumer.getmany(timeout_ms=timeout_ms)
            for tp, records in data.items():
                for record in records:
                    messages.append({
                        'topic': record.topic,
                        'partition': record.partition,
                        'offset': record.offset,
                        'key': record.key,
                        'value': record.value,
                        'timestamp': record.timestamp
                    })
        except Exception as e:
            logger.error(f"Error consuming batch: {e}")
        
        return messages
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the Kafka consumer."""
        return {
            'status': 'healthy' if self.is_connected else 'unhealthy',
            'connected': self.is_connected,
            'running': self.is_running,
            'group_id': self.group_id,
            'topics': list(self._handlers.keys()),
            'enabled': self.config.enable
        }


class KafkaManager:
    """
    High-level Kafka manager for the video commerce system.
    
    Provides convenient methods for common operations like sending
    user interactions, video processing tasks, and recommendation events.
    """
    
    def __init__(self, config: KafkaConfig):
        """Initialize the Kafka manager."""
        self.config = config
        self.producer = KafkaProducerClient(config)
        self._consumers: Dict[str, KafkaConsumerClient] = {}
        
        logger.info("KafkaManager initialized")
    
    async def start(self):
        """Start the Kafka manager (producer only, consumers started separately)."""
        await self.producer.start()
    
    async def stop(self):
        """Stop the Kafka manager and all connections."""
        await self.producer.stop()
        for consumer in self._consumers.values():
            await consumer.stop()
    
    # ==================== User Interaction Events ====================
    
    async def send_user_interaction(
        self,
        user_id: str,
        product_id: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a user interaction event to Kafka.
        
        Args:
            user_id: User identifier
            product_id: Product identifier
            action: Interaction type (view, click, purchase, etc.)
            context: Additional context data
            
        Returns:
            True if sent successfully
        """
        event = {
            'event_type': 'user_interaction',
            'user_id': user_id,
            'product_id': product_id,
            'action': action,
            'context': context or {},
            'timestamp': time.time()
        }
        
        return await self.producer.send(
            topic=self.config.user_interactions_topic,
            value=event,
            key=user_id  # Partition by user for ordering
        )
    
    # ==================== Video Processing Tasks ====================
    
    async def send_video_processing_task(
        self,
        content_id: str,
        file_path: str,
        user_id: Optional[str] = None,
        priority: str = "normal"
    ) -> bool:
        """
        Send a video processing task to Kafka.
        
        Args:
            content_id: Content identifier
            file_path: Path to the video file
            user_id: Optional user who uploaded the content
            priority: Task priority (low, normal, high)
            
        Returns:
            True if sent successfully
        """
        task = {
            'event_type': 'video_processing_task',
            'content_id': content_id,
            'file_path': file_path,
            'user_id': user_id,
            'priority': priority,
            'timestamp': time.time(),
            'status': 'pending'
        }
        
        return await self.producer.send(
            topic=self.config.video_processing_topic,
            value=task,
            key=content_id
        )
    
    # ==================== Recommendation Events ====================
    
    async def send_recommendation_event(
        self,
        user_id: str,
        recommendations: List[str],
        response_time_ms: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a recommendation event to Kafka.
        
        Args:
            user_id: User identifier
            recommendations: List of recommended product IDs
            response_time_ms: Response time in milliseconds
            metadata: Additional metadata
            
        Returns:
            True if sent successfully
        """
        event = {
            'event_type': 'recommendation',
            'user_id': user_id,
            'recommendations': recommendations,
            'num_recommendations': len(recommendations),
            'response_time_ms': response_time_ms,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        return await self.producer.send(
            topic=self.config.recommendation_events_topic,
            value=event,
            key=user_id
        )
    
    # ==================== Feature Update Events ====================
    
    async def send_feature_update(
        self,
        entity_type: str,
        entity_id: str,
        feature_updates: Dict[str, Any]
    ) -> bool:
        """
        Send a feature update event to Kafka.
        
        Args:
            entity_type: Type of entity (user, product, content)
            entity_id: Entity identifier
            feature_updates: Dictionary of feature updates
            
        Returns:
            True if sent successfully
        """
        event = {
            'event_type': 'feature_update',
            'entity_type': entity_type,
            'entity_id': entity_id,
            'updates': feature_updates,
            'timestamp': time.time()
        }
        
        return await self.producer.send(
            topic=self.config.feature_updates_topic,
            value=event,
            key=f"{entity_type}:{entity_id}"
        )
    
    # ==================== Consumer Management ====================
    
    def create_consumer(self, group_id: str) -> KafkaConsumerClient:
        """Create a new consumer with a specific group ID."""
        consumer = KafkaConsumerClient(self.config, group_id)
        self._consumers[group_id] = consumer
        return consumer
    
    # ==================== Health Check ====================
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the Kafka manager."""
        producer_health = await self.producer.health_check()
        consumer_health = {}
        
        for group_id, consumer in self._consumers.items():
            consumer_health[group_id] = await consumer.health_check()
        
        return {
            'producer': producer_health,
            'consumers': consumer_health,
            'enabled': self.config.enable
        }


# Global Kafka manager instance
_kafka_manager: Optional[KafkaManager] = None


def get_kafka_manager(config: Optional[KafkaConfig] = None) -> KafkaManager:
    """Get the global Kafka manager instance."""
    global _kafka_manager
    
    if _kafka_manager is None:
        if config is None:
            from config import get_kafka_config
            config = get_kafka_config()
        _kafka_manager = KafkaManager(config)
    
    return _kafka_manager


async def init_kafka(config: KafkaConfig) -> KafkaManager:
    """Initialize and start the global Kafka manager."""
    manager = get_kafka_manager(config)
    await manager.start()
    return manager


async def close_kafka():
    """Close the global Kafka manager."""
    global _kafka_manager
    if _kafka_manager:
        await _kafka_manager.stop()
        _kafka_manager = None
