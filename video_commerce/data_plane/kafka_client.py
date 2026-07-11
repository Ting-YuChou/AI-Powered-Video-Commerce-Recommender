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
import uuid

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaConnectionError, KafkaError

try:
    from aiokafka import TopicPartition
except ImportError:
    from kafka.structs import TopicPartition

from video_commerce.common.config import KafkaConfig
from video_commerce.common.feature_history_contracts import (
    FEATURE_HISTORY_PAYLOAD_SCHEMA_VERSION,
    RANKING_LTR_FEATURE_DEFINITION_VERSION,
    payload_sha256,
)
from video_commerce.common.observability import request_id_ctx_var
from video_commerce.common.telemetry import inject_trace_headers, kafka_consumer_span

logger = logging.getLogger(__name__)

EVENT_SCHEMA_VERSION = 1


def _build_event_payload(
    event_type: str,
    *,
    request_id: Optional[str] = None,
    occurred_at: Optional[float] = None,
    **payload: Any,
) -> Dict[str, Any]:
    event_time = time.time() if occurred_at is None else float(occurred_at)
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "request_id": request_id,
        "occurred_at": event_time,
        **payload,
    }


def _build_headers(request_id: Optional[str], event_id: str) -> List[tuple]:
    headers = [
        ("x-event-id", event_id.encode("utf-8")),
        ("x-schema-version", str(EVENT_SCHEMA_VERSION).encode("utf-8")),
    ]
    if request_id:
        headers.append(("x-request-id", request_id.encode("utf-8")))
    return inject_trace_headers(headers)


def _add_interaction_history_lineage(
    event: Dict[str, Any],
    *,
    user_id: str,
    product_id: str,
    action: str,
    context: Dict[str, Any],
    event_time: float,
    available_at: float,
) -> None:
    event.update(
        {
            "payload_schema_version": FEATURE_HISTORY_PAYLOAD_SCHEMA_VERSION,
            "feature_definition_version": RANKING_LTR_FEATURE_DEFINITION_VERSION,
            "event_time": event_time,
            "available_at": available_at,
            "source_event_id": event["event_id"],
            "source_version": "interaction-v1",
            "payload_hash": payload_sha256(
                {
                    "user_id": user_id,
                    "product_id": product_id,
                    "action": action,
                    "context": context,
                }
            ),
        }
    )


def _extract_header(headers: Optional[List[tuple]], name: str) -> Optional[str]:
    if not headers:
        return None
    for key, value in headers:
        if key == name and value is not None:
            return value.decode("utf-8")
    return None


def _serialize_headers(headers: Optional[List[tuple]]) -> Dict[str, str]:
    if not headers:
        return {}
    serialized = {}
    for key, value in headers:
        if value is None:
            serialized[key] = ""
        elif isinstance(value, bytes):
            serialized[key] = value.decode("utf-8", errors="replace")
        else:
            serialized[key] = str(value)
    return serialized


class KafkaProducerClient:
    """
    Async Kafka producer for publishing messages to topics.

    Provides high-throughput, async message publishing with automatic
    JSON serialization and delivery confirmation.
    """

    def __init__(self, config: KafkaConfig, observability=None):
        """Initialize the Kafka producer with configuration."""
        self.config = config
        self.observability = observability
        self.producer: Optional[AIOKafkaProducer] = None
        self.is_connected = False
        self._lock = asyncio.Lock()

        logger.info(
            f"KafkaProducerClient initialized with servers: {config.bootstrap_servers}"
        )

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
                    max_batch_size=self.config.producer_batch_size,
                    linger_ms=self.config.producer_linger_ms,
                    compression_type=self.config.producer_compression_type,
                    request_timeout_ms=self.config.request_timeout_ms,
                    retry_backoff_ms=self.config.retry_backoff_ms,
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
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
        headers: Optional[List[tuple]] = None,
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
            self._record_produce(topic, "disabled")
            return True

        if not self.is_connected or not self.producer:
            logger.warning("Kafka producer not connected, attempting to reconnect...")
            try:
                await self.start()
            except Exception as e:
                logger.error(f"Failed to reconnect Kafka producer: {e}")
                self._record_produce(topic, "error")
                return False

        try:
            # Add timestamp to message
            value["_kafka_timestamp"] = datetime.utcnow().isoformat()
            value["_kafka_topic"] = topic

            # Send message
            await self.producer.send_and_wait(
                topic=topic, value=value, key=key, headers=headers
            )

            logger.debug(f"Message sent to topic '{topic}' with key '{key}'")
            self._record_produce(topic, "success")
            return True

        except KafkaError as e:
            logger.error(f"Failed to send message to topic '{topic}': {e}")
            self._record_produce(topic, "error")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}")
            self._record_produce(topic, "error")
            return False

    async def send_nowait(
        self,
        topic: str,
        value: Dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[List[tuple]] = None,
    ) -> bool:
        """Enqueue a message to Kafka without waiting for broker acknowledgment."""
        if not self.config.enable:
            logger.debug(f"Kafka disabled, skipping async enqueue to topic: {topic}")
            self._record_produce(topic, "disabled")
            return True

        if not self.is_connected or not self.producer:
            logger.warning("Kafka producer not connected, attempting to reconnect...")
            try:
                await self.start()
            except Exception as e:
                logger.error(f"Failed to reconnect Kafka producer: {e}")
                self._record_produce(topic, "error")
                return False

        try:
            value["_kafka_timestamp"] = datetime.utcnow().isoformat()
            value["_kafka_topic"] = topic
            await self.producer.send(
                topic=topic,
                value=value,
                key=key,
                headers=headers,
            )
            logger.debug(f"Message enqueued to topic '{topic}' with key '{key}'")
            self._record_produce(topic, "success")
            return True
        except KafkaError as e:
            logger.error(f"Failed to enqueue message to topic '{topic}': {e}")
            self._record_produce(topic, "error")
            return False
        except Exception as e:
            logger.error(f"Unexpected error enqueueing message: {e}")
            self._record_produce(topic, "error")
            return False

    async def send_batch(
        self,
        topic: str,
        messages: List[Dict[str, Any]],
        key_field: Optional[str] = None,
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
            "status": "healthy" if self.is_connected else "unhealthy",
            "connected": self.is_connected,
            "bootstrap_servers": self.config.bootstrap_servers,
            "enabled": self.config.enable,
        }

    def _record_produce(self, topic: str, status: str) -> None:
        if self.observability is not None:
            self.observability.record_kafka_produce(topic, status)


class KafkaConsumerClient:
    """
    Async Kafka consumer for processing messages from topics.

    Provides message consumption with automatic JSON deserialization
    and callback-based message handling.
    """

    def __init__(
        self, config: KafkaConfig, group_id: Optional[str] = None, observability=None
    ):
        """Initialize the Kafka consumer with configuration."""
        self.config = config
        self.group_id = group_id or config.consumer_group_id
        self.observability = observability
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.dlq_producer: Optional[AIOKafkaProducer] = None
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
                    max_poll_interval_ms=self.config.consumer_max_poll_interval_ms,
                    session_timeout_ms=self.config.session_timeout_ms,
                    request_timeout_ms=self.config.request_timeout_ms,
                    key_deserializer=lambda k: k.decode("utf-8") if k else None,
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                )

                await self.consumer.start()
                if self.config.dead_letter_enable:
                    self.dlq_producer = AIOKafkaProducer(
                        bootstrap_servers=self.config.bootstrap_servers,
                        acks=self.config.producer_acks,
                        max_batch_size=self.config.producer_batch_size,
                        linger_ms=self.config.producer_linger_ms,
                        compression_type=self.config.producer_compression_type,
                        request_timeout_ms=self.config.request_timeout_ms,
                        retry_backoff_ms=self.config.retry_backoff_ms,
                        key_serializer=lambda k: k.encode("utf-8") if k else None,
                        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    )
                    await self.dlq_producer.start()
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
            if self.dlq_producer:
                try:
                    await self.dlq_producer.stop()
                    logger.info("Kafka DLQ producer stopped")
                except Exception as e:
                    logger.error(f"Error stopping Kafka DLQ producer: {e}")
                finally:
                    self.dlq_producer = None
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

                handled = await self._process_message_with_retries(message)
                if not handled:
                    raise RuntimeError(
                        f"Message processing failed without DLQ commit: "
                        f"{message.topic}[{message.partition}]@{message.offset}"
                    )
                if not self.config.consumer_enable_auto_commit:
                    await self._commit_message(message)

        except asyncio.CancelledError:
            logger.info("Consumer loop cancelled")
        except Exception as e:
            logger.error(f"Error in consumer loop: {e}")
        finally:
            self.is_running = False

    async def _process_message_with_retries(self, message) -> bool:
        max_attempts = max(1, int(self.config.consumer_handler_retries))
        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                await self._process_message_once(message)
                return True
            except Exception as exc:
                last_error = exc
                if self.observability is not None:
                    self.observability.record_kafka_retry(message.topic, self.group_id)
                logger.error(
                    "Error processing Kafka message",
                    extra={
                        "topic": message.topic,
                        "partition": message.partition,
                        "offset": message.offset,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "error": str(exc),
                    },
                )
                if attempt < max_attempts:
                    backoff_seconds = (
                        max(0, self.config.consumer_handler_retry_backoff_ms) / 1000.0
                    ) * attempt
                    await asyncio.sleep(backoff_seconds)

        if last_error and self.config.dead_letter_enable:
            return await self._publish_dead_letter(message, last_error)
        return False

    async def _process_message_once(self, message) -> None:
        started_at = time.perf_counter()
        topic = message.topic
        key = message.key
        value = message.value
        headers = message.headers
        request_id = _extract_header(headers, "x-request-id")
        if not request_id and isinstance(value, dict):
            request_id = value.get("request_id")
        request_id_token = request_id_ctx_var.set(request_id or "-")
        try:
            with kafka_consumer_span(
                topic=topic, group_id=self.group_id, headers=headers
            ):
                handler = self._handlers.get(topic)
                if not handler:
                    logger.warning(f"No handler registered for topic: {topic}")
                    if self.observability is not None:
                        self.observability.record_kafka_consume(
                            topic,
                            self.group_id,
                            "unhandled",
                            time.perf_counter() - started_at,
                        )
                    return
                await handler(topic, key, value, headers)
            if self.observability is not None:
                self.observability.record_kafka_consume(
                    topic,
                    self.group_id,
                    "success",
                    time.perf_counter() - started_at,
                )
        except Exception:
            if self.observability is not None:
                self.observability.record_kafka_consume(
                    topic,
                    self.group_id,
                    "error",
                    time.perf_counter() - started_at,
                )
            raise
        finally:
            request_id_ctx_var.reset(request_id_token)

    async def _commit_message(self, message) -> None:
        if not self.consumer:
            return
        topic_partition = TopicPartition(message.topic, message.partition)
        await self.consumer.commit({topic_partition: message.offset + 1})

    async def _publish_dead_letter(self, message, exc: Exception) -> bool:
        if not self.dlq_producer:
            return False
        request_id = _extract_header(message.headers, "x-request-id") or (
            message.value.get("request_id") if isinstance(message.value, dict) else None
        )
        payload = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": str(uuid.uuid4()),
            "event_type": "dead_letter",
            "request_id": request_id,
            "source_topic": message.topic,
            "source_partition": message.partition,
            "source_offset": message.offset,
            "source_key": message.key,
            "source_value": message.value,
            "source_headers": _serialize_headers(message.headers),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "occurred_at": time.time(),
        }
        try:
            await self.dlq_producer.send_and_wait(
                topic=self.config.dead_letter_topic,
                value=payload,
                key=f"{message.topic}:{message.partition}:{message.offset}",
                headers=_build_headers(request_id, payload["event_id"]),
            )
            if self.observability is not None:
                self.observability.record_kafka_dead_letter(
                    message.topic,
                    self.config.dead_letter_topic,
                )
            logger.error(
                "Kafka message sent to DLQ",
                extra={
                    "source_topic": message.topic,
                    "source_partition": message.partition,
                    "source_offset": message.offset,
                    "dead_letter_topic": self.config.dead_letter_topic,
                },
            )
            return True
        except Exception as dlq_exc:
            logger.error(f"Failed to publish Kafka message to DLQ: {dlq_exc}")
            return False

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
                    messages.append(
                        {
                            "topic": record.topic,
                            "partition": record.partition,
                            "offset": record.offset,
                            "key": record.key,
                            "value": record.value,
                            "timestamp": record.timestamp,
                        }
                    )
        except Exception as e:
            logger.error(f"Error consuming batch: {e}")

        return messages

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the Kafka consumer."""
        return {
            "status": "healthy" if self.is_connected else "unhealthy",
            "connected": self.is_connected,
            "running": self.is_running,
            "group_id": self.group_id,
            "topics": list(self._handlers.keys()),
            "enabled": self.config.enable,
        }


class KafkaManager:
    """
    High-level Kafka manager for the video commerce system.

    Provides convenient methods for common operations like sending
    user interactions, video processing tasks, and recommendation events.
    """

    def __init__(self, config: KafkaConfig, observability=None):
        """Initialize the Kafka manager."""
        self.config = config
        self.observability = observability
        self.producer = KafkaProducerClient(config, observability=observability)
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
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        event_time: Optional[float] = None,
        server_received_at: Optional[float] = None,
        request_id: Optional[str] = None,
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
        resolved_event_time = (
            float(event_time)
            if event_time is not None
            else (float(timestamp) if timestamp is not None else time.time())
        )
        received_at = (
            time.time() if server_received_at is None else float(server_received_at)
        )
        event = _build_event_payload(
            "user_interaction",
            request_id=request_id,
            occurred_at=resolved_event_time,
            user_id=user_id,
            product_id=product_id,
            action=action,
            context=context or {},
            event_time=resolved_event_time,
            timestamp=resolved_event_time,
            server_received_at=received_at,
        )
        _add_interaction_history_lineage(
            event,
            user_id=user_id,
            product_id=product_id,
            action=action,
            context=context or {},
            event_time=resolved_event_time,
            available_at=received_at,
        )

        return await self.producer.send(
            topic=self.config.user_interactions_topic,
            value=event,
            key=user_id,  # Partition by user for ordering
            headers=_build_headers(request_id, event["event_id"]),
        )

    async def send_user_interaction_async(
        self,
        user_id: str,
        product_id: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        event_time: Optional[float] = None,
        server_received_at: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> bool:
        """Enqueue a user interaction event without waiting for broker ack."""
        resolved_event_time = (
            float(event_time)
            if event_time is not None
            else (float(timestamp) if timestamp is not None else time.time())
        )
        received_at = (
            time.time() if server_received_at is None else float(server_received_at)
        )
        event = _build_event_payload(
            "user_interaction",
            request_id=request_id,
            occurred_at=resolved_event_time,
            user_id=user_id,
            product_id=product_id,
            action=action,
            context=context or {},
            event_time=resolved_event_time,
            timestamp=resolved_event_time,
            server_received_at=received_at,
        )
        _add_interaction_history_lineage(
            event,
            user_id=user_id,
            product_id=product_id,
            action=action,
            context=context or {},
            event_time=resolved_event_time,
            available_at=received_at,
        )

        return await self.producer.send_nowait(
            topic=self.config.user_interactions_topic,
            value=event,
            key=user_id,
            headers=_build_headers(request_id, event["event_id"]),
        )

    # ==================== Video Processing Tasks ====================

    async def send_video_processing_task(
        self,
        content_id: str,
        file_path: str,
        filename: Optional[str] = None,
        user_id: Optional[str] = None,
        priority: str = "normal",
        request_id: Optional[str] = None,
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
        task = _build_event_payload(
            "video_processing_task",
            request_id=request_id,
            content_id=content_id,
            file_path=file_path,
            filename=filename,
            user_id=user_id,
            priority=priority,
            timestamp=time.time(),
            status="pending",
        )

        return await self.producer.send(
            topic=self.config.video_processing_topic,
            value=task,
            key=content_id,
            headers=_build_headers(request_id, task["event_id"]),
        )

    # ==================== Recommendation Events ====================

    async def send_recommendation_event(
        self,
        user_id: str,
        recommendations: List[str],
        response_time_ms: int,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
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
        published_at = time.time()
        event_metadata = dict(metadata or {})
        observation_time = float(event_metadata.get("as_of_ts") or published_at)
        source_version = str(
            event_metadata.get("ranking_model_version")
            or event_metadata.get("model_version")
            or "unversioned"
        )
        event = _build_event_payload(
            "recommendation",
            request_id=request_id,
            occurred_at=published_at,
            user_id=user_id,
            recommendations=recommendations,
            num_recommendations=len(recommendations),
            response_time_ms=response_time_ms,
            metadata=event_metadata,
            timestamp=published_at,
        )
        event.update(
            {
                "payload_schema_version": FEATURE_HISTORY_PAYLOAD_SCHEMA_VERSION,
                "feature_definition_version": RANKING_LTR_FEATURE_DEFINITION_VERSION,
                "event_time": observation_time,
                "available_at": published_at,
                "source_event_id": event["event_id"],
                "source_version": source_version,
                "payload_hash": payload_sha256(
                    {
                        "user_id": user_id,
                        "recommendations": recommendations,
                        "response_time_ms": response_time_ms,
                        "metadata": event_metadata,
                    }
                ),
            }
        )

        return await self.producer.send(
            topic=self.config.recommendation_events_topic,
            value=event,
            key=user_id,
            headers=_build_headers(request_id, event["event_id"]),
        )

    # ==================== Feature Update Events ====================

    async def send_feature_update(
        self,
        entity_type: str,
        entity_id: str,
        feature_updates: Dict[str, Any],
        request_id: Optional[str] = None,
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
        event = _build_event_payload(
            "feature_update",
            request_id=request_id,
            entity_type=entity_type,
            entity_id=entity_id,
            updates=feature_updates,
            timestamp=time.time(),
        )

        return await self.producer.send(
            topic=self.config.feature_updates_topic,
            value=event,
            key=f"{entity_type}:{entity_id}",
            headers=_build_headers(request_id, event["event_id"]),
        )

    async def send_catalog_feature_event(self, event: Dict[str, Any]) -> bool:
        """Publish a pre-built deterministic catalog history event."""
        event_id = str(event.get("event_id") or "")
        entity_id = str(event.get("entity_id") or "")
        if not event_id or not entity_id:
            raise ValueError("catalog feature event requires event_id and entity_id")
        request_id = event.get("request_id")
        return await self.producer.send(
            topic=self.config.catalog_feature_events_topic,
            value=dict(event),
            key=entity_id,
            headers=_build_headers(request_id, event_id),
        )

    async def send_feature_history_backfill_event(
        self,
        *,
        topic: str,
        event: Dict[str, Any],
        key: str,
    ) -> bool:
        """Publish a prebuilt immutable history event to an isolated backfill topic."""
        event_id = str(
            event.get("event_id") or event.get("source_event_id") or ""
        ).strip()
        if not event_id or not key:
            raise ValueError("backfill event requires event_id and key")
        return await self.producer.send(
            topic=topic,
            value=dict(event),
            key=key,
            headers=_build_headers(event.get("request_id"), event_id),
        )

    # ==================== Consumer Management ====================

    def create_consumer(self, group_id: str) -> KafkaConsumerClient:
        """Create a new consumer with a specific group ID."""
        consumer = KafkaConsumerClient(
            self.config, group_id, observability=self.observability
        )
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
            "producer": producer_health,
            "consumers": consumer_health,
            "enabled": self.config.enable,
        }


# Global Kafka manager instance
_kafka_manager: Optional[KafkaManager] = None


def get_kafka_manager(
    config: Optional[KafkaConfig] = None, observability=None
) -> KafkaManager:
    """Get the global Kafka manager instance."""
    global _kafka_manager

    if _kafka_manager is None:
        if config is None:
            from video_commerce.common.config import get_kafka_config

            config = get_kafka_config()
        _kafka_manager = KafkaManager(config, observability=observability)
    elif (
        observability is not None
        and getattr(_kafka_manager, "observability", None) is None
    ):
        _kafka_manager.observability = observability
        _kafka_manager.producer.observability = observability

    return _kafka_manager


async def init_kafka(config: KafkaConfig, observability=None) -> KafkaManager:
    """Initialize and start the global Kafka manager."""
    manager = get_kafka_manager(config, observability=observability)
    await manager.start()
    return manager


async def close_kafka():
    """Close the global Kafka manager."""
    global _kafka_manager
    if _kafka_manager:
        await _kafka_manager.stop()
        _kafka_manager = None
