"""
AI-Powered Video Commerce Recommender - Feature Store
=====================================================

This module implements a Redis-based feature store for caching user features,
content features, and system metrics. It provides real-time feature updates
and intelligent caching strategies for optimal performance.
"""

import redis.asyncio as redis
import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Iterable, Tuple, Set
import numpy as np
from datetime import datetime, timedelta, timezone
import hashlib

# Local imports
from video_commerce.common.models import (
    CandidateProduct,
    UserFeatures,
    ContentFeatures,
    RealtimeWindowFeatures,
    InteractionType,
    SystemMetrics,
)
from video_commerce.common.cache_codec import (
    CacheDecodeError,
    json_dumps,
    json_loads,
    pack_cache_payload,
    unpack_cache_payload,
)
from video_commerce.common.config import RedisConfig, CacheConfig
from video_commerce.ml.din import (
    DIN_ACTIONS,
    DIN_ACTION_MAP,
    DINBehaviorSequences,
    build_din_behavior_sequences,
    build_din_freshness_token,
)

logger = logging.getLogger(__name__)

POSITIVE_SEQUENCE_ACTIONS = {
    InteractionType.VIEW.value,
    InteractionType.CLICK.value,
    InteractionType.ADD_TO_CART.value,
    InteractionType.PURCHASE.value,
}

REALTIME_WINDOW_NAMES = ("5m", "1h", "24h")


class FeatureStore:
    """
    Redis-based feature store for the video commerce recommender.

    Handles:
    - User features and preferences
    - Content features and embeddings
    - Real-time interaction logging
    - System metrics and analytics
    - Intelligent caching with adaptive TTL
    """

    CACHE_KEY_TYPES = {
        "recommendations_cache",
        "candidate_cache",
        "product_metadata",
        "trending_products",
        "category_pool",
        "cluster_pool",
    }

    def __init__(self, redis_config: RedisConfig, cache_config: CacheConfig = None):
        """Initialize the feature store with Redis configuration."""
        self.redis_config = redis_config
        self.cache_config = cache_config or CacheConfig()
        self.redis_client: Optional[redis.Redis] = None
        self.cache_redis_client: Optional[redis.Redis] = None
        self.is_connected = False
        self.din_sequence_read_failures = 0
        self.din_sequence_decode_failures = 0

        # Key prefixes for different data types
        self.prefixes = {
            "user_features": "uf:",
            "content_features": "cf:",
            "content_status": "cs:",
            "user_interactions": "ui:",
            "user_interactions_zset": "uiz:",
            "din_user_interactions_zset": "uiza:",
            "user_sequence_token": "ust:",
            "realtime_window_features": "rtwf:",
            "recommendations_cache": "rc:",
            "candidate_cache": "cc:",
            "product_metadata": "pm:",
            "system_metrics": "sm:",
            "trending_products": "tp:",
            "category_pool": "cp:",
            "cluster_pool": "clp:",
            "product_embeddings": "pe:",
            "analytics": "analytics:",
            "health": "health:",
        }
        self._product_metadata_memory_cache: Dict[str, Dict[str, Any]] = {}
        self._content_features_memory_cache: Dict[str, ContentFeatures] = {}
        self._known_content_feature_ids: Set[str] = set()
        self._content_feature_snapshot_loaded_at: float = 0.0
        self._content_feature_snapshot_enabled_flag: bool = True
        self._content_feature_snapshot_max_items: int = 100000
        self._trending_pool_memory_cache: Dict[str, List[CandidateProduct]] = {}
        self._category_pool_memory_cache: Dict[str, List[CandidateProduct]] = {}
        self._cluster_pool_memory_cache: Dict[str, List[CandidateProduct]] = {}
        self._known_user_ids: Set[str] = set()
        self._known_user_snapshot_loaded_at: float = 0.0
        self._known_user_snapshot_enabled_flag: bool = True
        self._known_user_snapshot_max_users: int = 200000

        logger.info("FeatureStore initialized")

    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = self._build_redis_client(cache_role=False)
            self.cache_redis_client = (
                self._build_redis_client(cache_role=True)
                if self._uses_separate_cache_redis()
                else self.redis_client
            )

            # Test connection
            await self.redis_client.ping()
            if self.cache_redis_client is not self.redis_client:
                await self.cache_redis_client.ping()
            self.is_connected = True

            # Initialize default data if needed
            await self._initialize_default_data()

            logger.info(
                "Redis connection established successfully",
                extra={
                    "separate_cache_redis": self.cache_redis_client
                    is not self.redis_client
                },
            )

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.is_connected = False
            raise

    async def close(self):
        """Close Redis connection."""
        if self.cache_redis_client and self.cache_redis_client is not self.redis_client:
            await self.cache_redis_client.close()
            self.cache_redis_client = None
        if self.redis_client:
            await self.redis_client.close()
            self.is_connected = False
            logger.info("Redis connection closed")

    def _build_redis_client(self, *, cache_role: bool) -> redis.Redis:
        if not cache_role:
            return redis.Redis(
                host=self.redis_config.host,
                port=self.redis_config.port,
                db=self.redis_config.db,
                password=self.redis_config.password,
                decode_responses=False,
                socket_timeout=self.redis_config.socket_timeout,
                socket_connect_timeout=self.redis_config.socket_connect_timeout,
                retry_on_timeout=self.redis_config.retry_on_timeout,
                max_connections=self.redis_config.max_connections,
                health_check_interval=30,
            )

        return redis.Redis(
            host=self._resolve_cache_setting(
                self.redis_config.cache_host, self.redis_config.host
            ),
            port=self._resolve_cache_setting(
                self.redis_config.cache_port, self.redis_config.port
            ),
            db=self.redis_config.cache_db
            if self.redis_config.cache_db is not None
            else self.redis_config.db,
            password=self._resolve_cache_setting(
                self.redis_config.cache_password,
                self.redis_config.password,
            ),
            decode_responses=False,
            socket_timeout=(
                self.redis_config.cache_socket_timeout
                if self.redis_config.cache_socket_timeout is not None
                else self.redis_config.socket_timeout
            ),
            socket_connect_timeout=(
                self.redis_config.cache_socket_connect_timeout
                if self.redis_config.cache_socket_connect_timeout is not None
                else self.redis_config.socket_connect_timeout
            ),
            retry_on_timeout=(
                self.redis_config.cache_retry_on_timeout
                if self.redis_config.cache_retry_on_timeout is not None
                else self.redis_config.retry_on_timeout
            ),
            max_connections=(
                self.redis_config.cache_max_connections
                if self.redis_config.cache_max_connections is not None
                else self.redis_config.max_connections
            ),
            health_check_interval=30,
        )

    def _uses_separate_cache_redis(self) -> bool:
        state_settings = (
            self.redis_config.host,
            self.redis_config.port,
            self.redis_config.db,
            self.redis_config.password,
            self.redis_config.max_connections,
            self.redis_config.socket_timeout,
            self.redis_config.socket_connect_timeout,
            self.redis_config.retry_on_timeout,
        )
        cache_settings = (
            self._resolve_cache_setting(
                self.redis_config.cache_host, self.redis_config.host
            ),
            self._resolve_cache_setting(
                self.redis_config.cache_port, self.redis_config.port
            ),
            self.redis_config.cache_db
            if self.redis_config.cache_db is not None
            else self.redis_config.db,
            self._resolve_cache_setting(
                self.redis_config.cache_password, self.redis_config.password
            ),
            self.redis_config.cache_max_connections
            if self.redis_config.cache_max_connections is not None
            else self.redis_config.max_connections,
            self.redis_config.cache_socket_timeout
            if self.redis_config.cache_socket_timeout is not None
            else self.redis_config.socket_timeout,
            self.redis_config.cache_socket_connect_timeout
            if self.redis_config.cache_socket_connect_timeout is not None
            else self.redis_config.socket_connect_timeout,
            self.redis_config.cache_retry_on_timeout
            if self.redis_config.cache_retry_on_timeout is not None
            else self.redis_config.retry_on_timeout,
        )
        return cache_settings != state_settings

    @staticmethod
    def _resolve_cache_setting(value, fallback):
        if value == "":
            return fallback
        return fallback if value is None else value

    def _client_for_key_type(self, key_type: str):
        if key_type in self.CACHE_KEY_TYPES:
            return self.cache_redis_client or self.redis_client
        return self.redis_client

    def mark_user_known(self, user_id: str) -> None:
        if user_id:
            self._known_user_ids.add(user_id)

    def forget_user_known(self, user_id: str) -> None:
        self._known_user_ids.discard(user_id)

    def known_user_snapshot_ready(self) -> bool:
        return self._known_user_snapshot_loaded_at > 0.0

    def _known_user_snapshot_enabled(self) -> bool:
        return self._known_user_snapshot_enabled_flag

    def configure_known_user_snapshot(
        self,
        *,
        enabled: bool,
        max_users: int,
    ) -> None:
        self._known_user_snapshot_enabled_flag = bool(enabled)
        self._known_user_snapshot_max_users = max(0, int(max_users))

    def _can_skip_user_context_redis(self, user_id: str) -> bool:
        return (
            self._known_user_snapshot_enabled()
            and self.known_user_snapshot_ready()
            and user_id not in self._known_user_ids
        )

    def configure_content_feature_snapshot(
        self,
        *,
        enabled: bool,
        max_items: int,
    ) -> None:
        self._content_feature_snapshot_enabled_flag = bool(enabled)
        self._content_feature_snapshot_max_items = max(0, int(max_items))

    def content_feature_snapshot_ready(self) -> bool:
        return self._content_feature_snapshot_loaded_at > 0.0

    def _content_feature_snapshot_enabled(self) -> bool:
        return self._content_feature_snapshot_enabled_flag

    def mark_content_features_known(
        self, content_id: str, features: Optional[ContentFeatures] = None
    ) -> None:
        if not content_id:
            return
        self._known_content_feature_ids.add(content_id)
        if features is not None:
            self._content_features_memory_cache[
                content_id
            ] = self._clone_content_features(features)

    async def refresh_content_feature_snapshot(self) -> int:
        """Refresh local content-feature IDs and values for request-time miss avoidance."""
        if not self._content_feature_snapshot_enabled():
            return len(self._known_content_feature_ids)

        client = self._client_for_key_type("content_features")
        prefix = self.prefixes["content_features"]
        max_items = self._content_feature_snapshot_max_items
        known_ids: Set[str] = set()
        memory_cache: Dict[str, ContentFeatures] = {}
        key_batch: List[str] = []
        id_batch: List[str] = []

        async def flush_batch() -> None:
            nonlocal key_batch, id_batch
            if not key_batch:
                return
            values = await client.mget(key_batch)
            for content_id, raw_value in zip(id_batch, values):
                if raw_value is None:
                    continue
                try:
                    data = unpack_cache_payload(raw_value, "content_features")
                    memory_cache[content_id] = ContentFeatures(**data)
                    known_ids.add(content_id)
                except (Exception, CacheDecodeError) as exc:
                    logger.warning("content_feature_snapshot_decode_failed: %s", exc)
            key_batch = []
            id_batch = []

        async for raw_key in client.scan_iter(match=f"{prefix}*", count=1000):
            key = raw_key.decode() if isinstance(raw_key, bytes) else str(raw_key)
            if not key.startswith(prefix):
                continue
            content_id = key[len(prefix) :]
            if not content_id:
                continue
            key_batch.append(key)
            id_batch.append(content_id)
            if len(key_batch) >= 250:
                await flush_batch()
            if max_items and len(known_ids) + len(id_batch) >= max_items:
                break
        await flush_batch()

        self._known_content_feature_ids = known_ids
        self._content_features_memory_cache = memory_cache
        self._content_feature_snapshot_loaded_at = time.time()
        logger.info(
            "content_feature_snapshot_refreshed",
            extra={"content_feature_count": len(self._known_content_feature_ids)},
        )
        return len(self._known_content_feature_ids)

    @staticmethod
    def _clone_content_features(features: ContentFeatures) -> ContentFeatures:
        return ContentFeatures(**features.dict())

    async def refresh_known_user_snapshot(self) -> int:
        """Refresh the local set of users known to have serving state."""
        if not self._known_user_snapshot_enabled():
            return len(self._known_user_ids)

        client = self._client_for_key_type("user_features")
        max_users = self._known_user_snapshot_max_users
        known_users: Set[str] = set()

        async def collect(pattern: str, parser) -> None:
            async for raw_key in client.scan_iter(match=pattern, count=1000):
                key = raw_key.decode() if isinstance(raw_key, bytes) else str(raw_key)
                user_id = parser(key)
                if user_id:
                    known_users.add(user_id)
                if max_users and len(known_users) >= max_users:
                    break

        await collect(
            f"{self.prefixes['user_features']}*",
            lambda key: key[len(self.prefixes["user_features"]) :]
            if key.startswith(self.prefixes["user_features"])
            else None,
        )
        if not max_users or len(known_users) < max_users:
            await collect(
                f"{self.prefixes['user_sequence_token']}*",
                self._parse_user_id_from_sequence_token_key,
            )

        self._known_user_ids = known_users
        self._known_user_snapshot_loaded_at = time.time()
        logger.info(
            "known_user_snapshot_refreshed",
            extra={"known_user_count": len(self._known_user_ids)},
        )
        return len(self._known_user_ids)

    def _parse_user_id_from_sequence_token_key(self, key: str) -> Optional[str]:
        prefix = self.prefixes["user_sequence_token"]
        if not key.startswith(prefix):
            return None
        remainder = key[len(prefix) :]
        parts = remainder.split(":", 1)
        if len(parts) != 2:
            return None
        return parts[1] or None

    @staticmethod
    def _feature_namespace_prefix(namespace: str = "official") -> str:
        return "flink:shadow:" if str(namespace or "").lower() == "shadow" else ""

    # User Features Management
    async def get_user_features(
        self, user_id: str, cache_default: bool = True
    ) -> UserFeatures:
        """Get user features from cache or create default."""
        try:
            key = f"{self.prefixes['user_features']}{user_id}"
            client = self._client_for_key_type("user_features")
            cached_data = await client.get(key)

            if cached_data:
                self.mark_user_known(user_id)
                return self._decode_user_features(user_id, cached_data)
            else:
                # Return default user features
                default_features = self._default_user_features(user_id)

                if cache_default:
                    await self._set_user_features(user_id, default_features)
                return default_features

        except Exception as e:
            logger.error(f"Error getting user features for {user_id}: {e}")
            return UserFeatures(user_id=user_id)

    async def get_user_serving_context(
        self,
        user_id: str,
        *,
        sequence_limit: int = 200,
        cache_default: bool = False,
    ) -> Tuple[UserFeatures, Dict[str, Any]]:
        """Read user features and the compact sequence token with one Redis round trip."""
        if self._can_skip_user_context_redis(user_id):
            return (
                self._default_user_features(user_id),
                self._empty_user_sequence_token(),
            )

        try:
            client = self._client_for_key_type("user_features")
            feature_key = f"{self.prefixes['user_features']}{user_id}"
            sequence_key = self._user_sequence_token_key(user_id, sequence_limit)
            pipeline = client.pipeline(transaction=False)
            pipeline.get(feature_key)
            pipeline.get(sequence_key)
            feature_data, token_data = await pipeline.execute()

            if feature_data:
                user_features = self._decode_user_features(user_id, feature_data)
                self.mark_user_known(user_id)
            else:
                user_features = self._default_user_features(user_id)
                if cache_default:
                    await self._set_user_features(user_id, user_features)

            if token_data:
                sequence_token = self._decode_user_sequence_token(token_data)
                self.mark_user_known(user_id)
            elif not feature_data:
                sequence_token = self._empty_user_sequence_token()
            else:
                sequence_token = await self.get_user_sequence_token(
                    user_id,
                    limit=sequence_limit,
                )
            return user_features, sequence_token
        except Exception as e:
            logger.error(f"Error getting user serving context for {user_id}: {e}")
            if self._can_skip_user_context_redis(user_id):
                return (
                    self._default_user_features(user_id),
                    self._empty_user_sequence_token(),
                )
            user_features = await self.get_user_features(
                user_id, cache_default=cache_default
            )
            sequence_token = await self.get_user_sequence_token(
                user_id, limit=sequence_limit
            )
            return user_features, sequence_token

    @staticmethod
    def _default_user_features(user_id: str) -> UserFeatures:
        return UserFeatures(
            user_id=user_id,
            total_interactions=0,
            avg_session_length=0.0,
            preferred_categories=[],
            price_sensitivity=0.5,
            click_through_rate=0.0,
            conversion_rate=0.0,
            last_active=time.time(),
            demographics={},
        )

    @staticmethod
    def _decode_user_features(user_id: str, cached_data: Any) -> UserFeatures:
        data = unpack_cache_payload(cached_data, "user_features")
        if not data.get("user_id"):
            data["user_id"] = user_id
        return UserFeatures(**data)

    async def _set_user_features(self, user_id: str, features: UserFeatures):
        """Set user features in cache."""
        try:
            key = f"{self.prefixes['user_features']}{user_id}"
            data = pack_cache_payload("user_features", features.dict())

            # Use adaptive TTL based on user activity
            ttl = self._calculate_adaptive_ttl(features)

            await self._client_for_key_type("user_features").setex(key, ttl, data)
            self.mark_user_known(user_id)

        except Exception as e:
            logger.error(f"Error setting user features for {user_id}: {e}")

    async def set_user_features_batch(self, features_map: Dict[str, UserFeatures]):
        """Batch-set user features using one Redis pipeline."""
        try:
            if not features_map:
                return

            pipeline = self._client_for_key_type("user_features").pipeline(
                transaction=False
            )
            for user_id, features in features_map.items():
                key = f"{self.prefixes['user_features']}{user_id}"
                ttl = self._calculate_adaptive_ttl(features)
                pipeline.setex(
                    key,
                    ttl,
                    pack_cache_payload("user_features", features.dict()),
                )
            await pipeline.execute()
            for user_id in features_map:
                self.mark_user_known(user_id)
        except Exception as e:
            logger.error(f"Error batch-setting user features: {e}")

    async def update_user_features(
        self, user_id: str, action: str, context: Dict[str, Any] = None
    ):
        """Update user features based on new interaction."""
        try:
            # Get current features
            features = await self.get_user_features(user_id)

            # Update based on action
            features.total_interactions += 1
            features.last_active = time.time()

            # Update action-specific metrics
            if action == InteractionType.CLICK.value:
                # Update CTR (simplified calculation)
                total_views = await self._get_user_stat(user_id, "total_views") or 1
                total_clicks = await self._get_user_stat(user_id, "total_clicks") or 0
                total_clicks += 1
                features.click_through_rate = total_clicks / max(total_views, 1)
                await self._set_user_stat(user_id, "total_clicks", total_clicks)

            elif action == InteractionType.PURCHASE.value:
                # Update conversion rate
                total_clicks = await self._get_user_stat(user_id, "total_clicks") or 1
                total_purchases = (
                    await self._get_user_stat(user_id, "total_purchases") or 0
                )
                total_purchases += 1
                features.conversion_rate = total_purchases / max(total_clicks, 1)
                await self._set_user_stat(user_id, "total_purchases", total_purchases)

            elif action == InteractionType.VIEW.value:
                total_views = await self._get_user_stat(user_id, "total_views") or 0
                total_views += 1
                await self._set_user_stat(user_id, "total_views", total_views)

            # Update session length if provided
            if context and "session_length" in context:
                # Running average of session lengths
                current_avg = features.avg_session_length
                session_count = max(features.total_interactions, 1)
                features.avg_session_length = (
                    current_avg * (session_count - 1) + context["session_length"]
                ) / session_count

            # Update preferred categories if product category provided
            if context and "product_category" in context:
                category = context["product_category"]
                if category not in features.preferred_categories:
                    features.preferred_categories.append(category)
                # Keep only top 10 categories
                features.preferred_categories = features.preferred_categories[:10]

            # Save updated features
            await self._set_user_features(user_id, features)

            logger.debug(f"Updated user features for {user_id}: {action}")

        except Exception as e:
            logger.error(f"Error updating user features for {user_id}: {e}")

    async def _get_user_stat(self, user_id: str, stat_name: str) -> Optional[int]:
        """Get user statistic from cache."""
        try:
            key = f"user_stats:{user_id}:{stat_name}"
            value = await self._client_for_key_type("user_features").get(key)
            return int(value) if value else None
        except:
            return None

    async def _set_user_stat(self, user_id: str, stat_name: str, value: int):
        """Set user statistic in cache."""
        try:
            key = f"user_stats:{user_id}:{stat_name}"
            await self._client_for_key_type("user_features").setex(
                key, 86400 * 7, str(value)
            )
        except Exception as e:
            logger.error(f"Error setting user stat {stat_name} for {user_id}: {e}")

    def _calculate_adaptive_ttl(self, features: UserFeatures) -> int:
        """Calculate adaptive TTL based on user activity."""
        if not self.cache_config.adaptive_ttl:
            return self.cache_config.user_features_ttl

        # Calculate activity score based on recent interactions
        hours_since_active = (time.time() - features.last_active) / 3600

        if hours_since_active < 1:  # Very active user
            return self.cache_config.high_activity_ttl
        elif hours_since_active < 24:  # Active user
            return self.cache_config.user_features_ttl
        else:  # Less active user
            return self.cache_config.low_activity_ttl

    # Content Features Management
    async def store_content_features(self, content_id: str, features: ContentFeatures):
        """Store content features in cache."""
        try:
            key = f"{self.prefixes['content_features']}{content_id}"
            data = pack_cache_payload("content_features", features.dict())
            self.mark_content_features_known(content_id, features)

            await self._client_for_key_type("content_features").setex(
                key, self.cache_config.content_features_ttl, data
            )

            # Also store status
            await self.update_content_status(content_id, "completed")

            logger.info(f"Stored content features for {content_id}")

        except Exception as e:
            logger.error(f"Error storing content features for {content_id}: {e}")

    async def get_content_features(self, content_id: str) -> Optional[ContentFeatures]:
        """Get content features from cache."""
        try:
            cached = self._content_features_memory_cache.get(content_id)
            if cached is not None:
                return self._clone_content_features(cached)
            if (
                self._content_feature_snapshot_enabled()
                and self.content_feature_snapshot_ready()
                and content_id not in self._known_content_feature_ids
            ):
                return None

            key = f"{self.prefixes['content_features']}{content_id}"
            cached_data = await self._client_for_key_type("content_features").get(key)

            if cached_data:
                data = unpack_cache_payload(cached_data, "content_features")
                features = ContentFeatures(**data)
                self.mark_content_features_known(content_id, features)
                return features

            return None

        except Exception as e:
            logger.error(f"Error getting content features for {content_id}: {e}")
            return None

    async def update_content_status(self, content_id: str, status: str):
        """Update content processing status."""
        try:
            key = f"{self.prefixes['content_status']}{content_id}"
            status_data = {"status": status, "updated_at": time.time()}

            await self._client_for_key_type("content_status").setex(
                key, 86400, json_dumps(status_data)  # 24 hours
            )

        except Exception as e:
            logger.error(f"Error updating content status for {content_id}: {e}")

    async def get_content_status(self, content_id: str) -> Optional[str]:
        """Get content processing status."""
        try:
            key = f"{self.prefixes['content_status']}{content_id}"
            data = await self._client_for_key_type("content_status").get(key)

            if data:
                status_data = json_loads(data)
                return status_data.get("status")

            return None

        except Exception as e:
            logger.error(f"Error getting content status for {content_id}: {e}")
            return None

    async def get_content_processed_time(self, content_id: str) -> Optional[float]:
        """Get content processing completion time."""
        try:
            key = f"{self.prefixes['content_status']}{content_id}"
            data = await self._client_for_key_type("content_status").get(key)

            if data:
                status_data = json_loads(data)
                return status_data.get("updated_at")

            return None

        except Exception as e:
            logger.error(f"Error getting content processed time for {content_id}: {e}")
            return None

    async def get_realtime_window_features(
        self,
        entity_type: str,
        entity_id: str,
        *,
        namespace: str = "official",
        windows: Iterable[str] = REALTIME_WINDOW_NAMES,
    ) -> Dict[str, RealtimeWindowFeatures]:
        """Read Flink-produced realtime window features for one entity."""
        result = await self.get_realtime_window_features_batch(
            [(entity_type, entity_id)],
            namespace=namespace,
            windows=windows,
        )
        return result.get(f"{entity_type}:{entity_id}", {})

    async def get_realtime_window_features_batch(
        self,
        entities: Iterable[Tuple[str, str]],
        *,
        namespace: str = "official",
        windows: Iterable[str] = REALTIME_WINDOW_NAMES,
    ) -> Dict[str, Dict[str, RealtimeWindowFeatures]]:
        """Batch-read Flink realtime features keyed by '<entity_type>:<entity_id>'."""
        entity_list = [
            (str(entity_type), str(entity_id))
            for entity_type, entity_id in entities
            if entity_type and entity_id
        ]
        window_list = [str(window) for window in windows]
        if not entity_list or not window_list:
            return {}

        namespace_prefix = self._feature_namespace_prefix(namespace)
        client = self._client_for_key_type("user_features")
        keys: List[str] = []
        key_meta: List[Tuple[str, str, str]] = []
        for entity_type, entity_id in entity_list:
            for window in window_list:
                keys.append(
                    f"{namespace_prefix}{self.prefixes['realtime_window_features']}"
                    f"{entity_type}:{entity_id}:{window}"
                )
                key_meta.append((entity_type, entity_id, window))

        try:
            raw_values = await client.mget(keys)
        except Exception as e:
            logger.error(f"Error reading realtime window features: {e}")
            return {}

        features: Dict[str, Dict[str, RealtimeWindowFeatures]] = {}
        for raw, (entity_type, entity_id, window) in zip(raw_values, key_meta):
            if not raw:
                continue
            try:
                payload = json_loads(raw)
                payload.setdefault("entity_type", entity_type)
                payload.setdefault("entity_id", entity_id)
                payload.setdefault("window", window)
                entity_key = f"{entity_type}:{entity_id}"
                features.setdefault(entity_key, {})[window] = RealtimeWindowFeatures(
                    **payload
                )
            except Exception:
                continue
        return features

    # Interaction Logging
    async def log_user_interaction(
        self,
        user_id: str,
        product_id: str,
        action: str,
        context: Dict[str, Any] = None,
        *,
        event_id: Optional[str] = None,
        schema_version: int = 1,
        occurred_at: Optional[Union[float, datetime]] = None,
        timestamp: Optional[Union[float, datetime]] = None,
    ):
        """Log recent user interaction for online serving state only."""
        try:
            event_timestamp = self._coerce_event_timestamp(
                timestamp, default=time.time()
            )
            event_occurred_at = self._coerce_event_timestamp(
                occurred_at, default=event_timestamp
            )
            interaction_data = {
                "user_id": user_id,
                "product_id": product_id,
                "action": action,
                "timestamp": event_timestamp,
                "occurred_at": event_occurred_at,
                "event_id": event_id,
                "schema_version": schema_version,
                "context": context or {},
            }

            key = f"{self.prefixes['user_interactions']}{user_id}"
            client = self._client_for_key_type("user_interactions")
            await client.lpush(key, json.dumps(interaction_data))

            await client.ltrim(key, 0, 999)
            await client.expire(key, 86400 * 30)  # 30 days
            self.mark_user_known(user_id)
            await self.refresh_user_sequence_token(user_id)

            logger.debug(f"Logged interaction: {user_id} -> {action} -> {product_id}")

        except Exception as e:
            logger.error(f"Error logging interaction: {e}")

    async def log_user_interactions_batch(
        self,
        user_id: str,
        interactions: List[Dict[str, Any]],
    ):
        """Batch-log recent interactions for a single user with one Redis pipeline."""
        try:
            if not interactions:
                return

            serialized = []
            for interaction in interactions:
                event_timestamp = self._coerce_event_timestamp(
                    interaction.get("timestamp"),
                    default=time.time(),
                )
                event_occurred_at = self._coerce_event_timestamp(
                    interaction.get("occurred_at"),
                    default=event_timestamp,
                )
                serialized.append(
                    json.dumps(
                        {
                            "user_id": user_id,
                            "product_id": interaction["product_id"],
                            "action": interaction["action"],
                            "timestamp": event_timestamp,
                            "occurred_at": event_occurred_at,
                            "event_id": interaction.get("event_id"),
                            "schema_version": interaction.get("schema_version", 1),
                            "context": interaction.get("context", {}),
                        }
                    )
                )

            user_key = f"{self.prefixes['user_interactions']}{user_id}"
            pipeline = self._client_for_key_type("user_interactions").pipeline(
                transaction=False
            )
            pipeline.lpush(user_key, *serialized)
            pipeline.ltrim(user_key, 0, 999)
            pipeline.expire(user_key, 86400 * 30)
            for serialized_interaction in serialized:
                payload = json.loads(serialized_interaction)
                action = DIN_ACTION_MAP.get(str(payload.get("action") or "").lower())
                if action is None:
                    continue
                action_key = (
                    f"{self.prefixes['din_user_interactions_zset']}"
                    f"{action}:{user_id}"
                )
                pipeline.zadd(
                    action_key,
                    {serialized_interaction: float(payload["occurred_at"])},
                )
                pipeline.zremrangebyrank(action_key, 0, -201)
                pipeline.expire(action_key, 86400 * 30)
            await pipeline.execute()
            self.mark_user_known(user_id)
            await self.refresh_user_sequence_token(user_id)
        except Exception as e:
            logger.error(f"Error batch-logging interactions for {user_id}: {e}")

    async def get_user_interactions(
        self, user_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent user interactions."""
        try:
            zset_interactions = await self._get_user_interactions_zset(
                user_id,
                limit=limit,
                newest_first=True,
            )
            if zset_interactions:
                return zset_interactions

            key = f"{self.prefixes['user_interactions']}{user_id}"
            interactions_data = await self._client_for_key_type(
                "user_interactions"
            ).lrange(key, 0, limit - 1)

            interactions = []
            for data in interactions_data:
                try:
                    interaction = self._loads_redis_json(data)
                    interactions.append(interaction)
                except Exception:
                    continue

            return interactions

        except Exception as e:
            logger.error(f"Error getting user interactions for {user_id}: {e}")
            return []

    async def get_user_sequence(
        self, user_id: str, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """Get a positive interaction sequence in oldest-to-newest order."""
        try:
            if limit <= 0:
                return []

            zset_sequence = await self._get_user_interactions_zset(
                user_id,
                limit=limit,
                newest_first=False,
            )
            if zset_sequence:
                return [
                    interaction
                    for interaction in (
                        self._normalize_sequence_interaction(user_id, item)
                        for item in zset_sequence
                    )
                    if interaction is not None
                ]

            key = f"{self.prefixes['user_interactions']}{user_id}"
            interactions_data = await self._client_for_key_type(
                "user_interactions"
            ).lrange(
                key,
                0,
                limit - 1,
            )

            sequence: List[Dict[str, Any]] = []
            for data in interactions_data:
                try:
                    interaction = self._loads_redis_json(data)
                    normalized = self._normalize_sequence_interaction(
                        user_id, interaction
                    )
                    if normalized is not None:
                        sequence.append(normalized)
                except Exception:
                    continue

            sequence.reverse()
            return sequence
        except Exception as e:
            logger.error(f"Error getting user sequence for {user_id}: {e}")
            return []

    async def get_din_behavior_sequences(
        self,
        user_id: str,
        *,
        as_of_ts: Optional[float] = None,
        last_n: int = 60,
        namespace: str = "official",
    ) -> DINBehaviorSequences:
        """Read action-specific DIN histories with one Redis round trip."""
        resolved_as_of = float(as_of_ts if as_of_ts is not None else time.time())
        client = self._client_for_key_type("user_interactions")
        events: List[Dict[str, Any]] = []
        try:
            pipeline = client.pipeline(transaction=False)
            for action in DIN_ACTIONS:
                key = (
                    f"{self._feature_namespace_prefix(namespace)}"
                    f"{self.prefixes['din_user_interactions_zset']}{action}:{user_id}"
                )
                pipeline.zrevrangebyscore(
                    key,
                    f"({resolved_as_of}",
                    resolved_as_of - 30 * 86400.0,
                    start=0,
                    num=last_n,
                )
            raw_sequences = await pipeline.execute()
            for raw_values in raw_sequences:
                for raw in raw_values or []:
                    try:
                        events.append(self._loads_redis_json(raw))
                    except Exception as exc:
                        self.din_sequence_decode_failures += 1
                        logger.error(
                            "Invalid official DIN action-history member for %s: %s",
                            user_id,
                            exc,
                        )
                        raise RuntimeError(
                            "official DIN action history contains invalid data"
                        ) from exc
        except RuntimeError:
            raise
        except Exception as exc:
            self.din_sequence_read_failures += 1
            logger.error("Unable to read official DIN action histories: %s", exc)
            raise RuntimeError("official DIN action history read failed") from exc

        if not events and namespace == "legacy":
            events = await self.get_user_sequence(user_id, limit=1000)
        return build_din_behavior_sequences(
            events,
            as_of_ts=resolved_as_of,
            last_n=last_n,
        )

    async def get_din_freshness_token(
        self,
        user_id: str,
        *,
        as_of_ts: Optional[float] = None,
        last_n: int = 60,
    ) -> Dict[str, Any]:
        sequences = await self.get_din_behavior_sequences(
            user_id,
            as_of_ts=as_of_ts,
            last_n=last_n,
        )
        return build_din_freshness_token(sequences)

    async def _get_user_interactions_zset(
        self,
        user_id: str,
        *,
        limit: int,
        newest_first: bool,
        namespace: str = "official",
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        client = self._client_for_key_type("user_interactions")
        if newest_first and not hasattr(client, "zrevrange"):
            return []
        if not newest_first and not hasattr(client, "zrange"):
            return []

        key = (
            f"{self._feature_namespace_prefix(namespace)}"
            f"{self.prefixes['user_interactions_zset']}{user_id}"
        )
        try:
            if newest_first:
                interactions_data = await client.zrevrange(key, 0, limit - 1)
            else:
                interactions_data = await client.zrange(key, -limit, -1)
        except Exception:
            return []

        interactions = []
        for data in interactions_data:
            try:
                interactions.append(self._loads_redis_json(data))
            except Exception:
                continue
        return interactions

    async def get_user_sequence_token(
        self, user_id: str, limit: int = 200
    ) -> Dict[str, Any]:
        """Return a compact freshness token for the user's positive sequence."""
        try:
            token_key = self._user_sequence_token_key(user_id, limit)
            cached = await self._client_for_key_type("user_sequence_token").get(
                token_key
            )
            if cached:
                return self._decode_user_sequence_token(cached)

            sequence = await self.get_user_sequence(user_id, limit=limit)
            token = self._build_user_sequence_token(sequence)
            await self._store_user_sequence_token(user_id, token, limit=limit)
            return token
        except Exception as e:
            logger.error(f"Error getting user sequence token for {user_id}: {e}")
            return self._empty_user_sequence_token()

    async def refresh_user_sequence_token(
        self, user_id: str, limit: int = 200
    ) -> Dict[str, Any]:
        """Recompute and store the compact sequence token after interaction updates."""
        try:
            sequence = await self.get_user_sequence(user_id, limit=limit)
            token = self._build_user_sequence_token(sequence)
            await self._store_user_sequence_token(user_id, token, limit=limit)
            return token
        except Exception as e:
            logger.error(f"Error refreshing user sequence token for {user_id}: {e}")
            return self._empty_user_sequence_token()

    def _user_sequence_token_key(self, user_id: str, limit: int = 200) -> str:
        return f"{self.prefixes['user_sequence_token']}{limit}:{user_id}"

    async def _store_user_sequence_token(
        self,
        user_id: str,
        token: Dict[str, Any],
        *,
        limit: int = 200,
    ) -> None:
        key = self._user_sequence_token_key(user_id, limit)
        await self._client_for_key_type("user_sequence_token").setex(
            key,
            86400 * 30,
            json_dumps(token),
        )
        self.mark_user_known(user_id)

    @classmethod
    def _build_user_sequence_token(
        cls, sequence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not sequence:
            return cls._empty_user_sequence_token()
        latest = sequence[-1]
        return {
            "length": len(sequence),
            "latest_event_id": latest.get("event_id"),
            "latest_occurred_at": round(float(latest.get("occurred_at") or 0.0), 6),
            "latest_product_id": latest.get("product_id"),
            "latest_action": latest.get("action"),
        }

    @classmethod
    def _decode_user_sequence_token(cls, data: Any) -> Dict[str, Any]:
        try:
            decoded = json_loads(data)
        except Exception:
            return cls._empty_user_sequence_token()
        if not isinstance(decoded, dict):
            return cls._empty_user_sequence_token()
        return {
            "length": int(decoded.get("length") or 0),
            "latest_event_id": decoded.get("latest_event_id"),
            "latest_occurred_at": round(
                float(decoded.get("latest_occurred_at") or 0.0), 6
            ),
            "latest_product_id": decoded.get("latest_product_id"),
            "latest_action": decoded.get("latest_action"),
        }

    @staticmethod
    def _empty_user_sequence_token() -> Dict[str, Any]:
        return {
            "length": 0,
            "latest_event_id": None,
            "latest_occurred_at": 0,
            "latest_product_id": None,
            "latest_action": None,
        }

    @staticmethod
    def _coerce_event_timestamp(
        value: Optional[Union[float, int, datetime]],
        *,
        default: float,
    ) -> float:
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    return parsed.timestamp()
                except ValueError:
                    pass
        return float(default)

    @staticmethod
    def _loads_redis_json(data: Any) -> Dict[str, Any]:
        if isinstance(data, bytes):
            data = data.decode()
        return json.loads(data)

    def _normalize_sequence_interaction(
        self,
        user_id: str,
        interaction: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        product_id = interaction.get("product_id")
        action = interaction.get("action")
        if not product_id or action not in POSITIVE_SEQUENCE_ACTIONS:
            return None

        timestamp = self._coerce_event_timestamp(
            interaction.get("timestamp"),
            default=time.time(),
        )
        occurred_at = self._coerce_event_timestamp(
            interaction.get("occurred_at"),
            default=timestamp,
        )
        return {
            "user_id": interaction.get("user_id") or user_id,
            "product_id": product_id,
            "action": action,
            "timestamp": timestamp,
            "occurred_at": occurred_at,
            "event_id": interaction.get("event_id"),
            "schema_version": interaction.get("schema_version", 1),
            "context": interaction.get("context") or {},
        }

    async def apply_user_interaction_batch(
        self,
        user_id: str,
        interactions: List[Dict[str, Any]],
    ) -> UserFeatures:
        """Apply aggregated interaction updates to one user's feature row."""
        try:
            if not interactions:
                return await self.get_user_features(user_id)

            features = await self.get_user_features(user_id)
            action_counts: Dict[str, int] = {}
            categories: List[str] = []
            session_lengths: List[float] = []
            timestamps: List[float] = []

            for interaction in interactions:
                action = interaction.get("action", InteractionType.VIEW.value)
                action_counts[action] = action_counts.get(action, 0) + 1
                context = interaction.get("context") or {}
                category = context.get("product_category")
                if category:
                    categories.append(category)
                if "session_length" in context:
                    try:
                        session_lengths.append(float(context["session_length"]))
                    except (TypeError, ValueError):
                        pass
                try:
                    timestamps.append(float(interaction.get("timestamp", time.time())))
                except (TypeError, ValueError):
                    timestamps.append(time.time())

            previous_interactions = features.total_interactions
            features.total_interactions += len(interactions)
            features.last_active = max([features.last_active, *timestamps])

            if session_lengths:
                previous_total_session = features.avg_session_length * max(
                    previous_interactions, 0
                )
                features.avg_session_length = (
                    previous_total_session + sum(session_lengths)
                ) / max(previous_interactions + len(session_lengths), 1)

            for category in categories:
                if category not in features.preferred_categories:
                    features.preferred_categories.append(category)
            features.preferred_categories = features.preferred_categories[:10]

            stat_names = ("total_views", "total_clicks", "total_purchases")
            stat_keys = [f"user_stats:{user_id}:{name}" for name in stat_names]
            state_client = self._client_for_key_type("user_features")
            stat_values = await state_client.mget(stat_keys)
            stats = {
                name: int(value) if value else 0
                for name, value in zip(stat_names, stat_values)
            }
            stats["total_views"] += action_counts.get(InteractionType.VIEW.value, 0)
            stats["total_clicks"] += action_counts.get(InteractionType.CLICK.value, 0)
            stats["total_purchases"] += action_counts.get(
                InteractionType.PURCHASE.value, 0
            )

            features.click_through_rate = stats["total_clicks"] / max(
                stats["total_views"], 1
            )
            features.conversion_rate = stats["total_purchases"] / max(
                stats["total_clicks"], 1
            )

            pipeline = state_client.pipeline(transaction=False)
            ttl = self._calculate_adaptive_ttl(features)
            pipeline.setex(
                f"{self.prefixes['user_features']}{user_id}",
                ttl,
                pack_cache_payload("user_features", features.dict()),
            )
            for stat_name, stat_value in stats.items():
                pipeline.setex(
                    f"user_stats:{user_id}:{stat_name}", 86400 * 7, str(stat_value)
                )
            await pipeline.execute()
            self.mark_user_known(user_id)
            return features
        except Exception as e:
            logger.error(f"Error applying user interaction batch for {user_id}: {e}")
            return await self.get_user_features(user_id)

    # Recommendation Caching
    async def cache_recommendations(
        self,
        user_id: str,
        context_hash: str,
        recommendations: List[Dict[str, Any]],
        user_features: Optional[UserFeatures] = None,
    ):
        """Cache recommendation results."""
        try:
            if not self.cache_config.enable_caching:
                return

            key = f"{self.prefixes['recommendations_cache']}{user_id}:{context_hash}"
            cache_data = {
                "recommendations": recommendations,
                "cached_at": time.time(),
                "user_id": user_id,
            }

            # Use user-specific TTL
            user_features = user_features or await self.get_user_features(user_id)
            ttl = self._calculate_adaptive_ttl(user_features)

            await self._client_for_key_type("recommendations_cache").setex(
                key,
                min(ttl, self.cache_config.recommendations_ttl),
                pack_cache_payload("recommendations_cache", cache_data),
            )

        except Exception as e:
            logger.error(f"Error caching recommendations for {user_id}: {e}")

    async def cache_candidate_products(
        self,
        user_id: str,
        context_hash: str,
        candidates: List[CandidateProduct],
        user_features: Optional[UserFeatures] = None,
    ):
        """Cache pre-ranked candidate products for reuse across requests."""
        try:
            if not self.cache_config.enable_caching:
                return

            key = f"{self.prefixes['candidate_cache']}{user_id}:{context_hash}"
            user_features = user_features or await self.get_user_features(user_id)
            ttl = min(
                self._calculate_adaptive_ttl(user_features),
                self.cache_config.candidate_ttl,
            )
            payload = {
                "candidates": [candidate.dict() for candidate in candidates],
                "cached_at": time.time(),
                "user_id": user_id,
            }
            await self._client_for_key_type("candidate_cache").setex(
                key,
                ttl,
                pack_cache_payload("candidate_cache", payload),
            )
        except Exception as e:
            logger.error(f"Error caching candidates for {user_id}: {e}")

    async def get_cached_candidate_products(
        self,
        user_id: str,
        context_hash: str,
    ) -> Optional[List[CandidateProduct]]:
        """Return cached candidate products for a user/context pair."""
        try:
            if not self.cache_config.enable_caching:
                return None

            key = f"{self.prefixes['candidate_cache']}{user_id}:{context_hash}"
            cached_data = await self._client_for_key_type("candidate_cache").get(key)
            if not cached_data:
                return None

            cache_data = unpack_cache_payload(cached_data, "candidate_cache")
            return [
                CandidateProduct(**item) for item in cache_data.get("candidates", [])
            ]
        except (Exception, CacheDecodeError) as e:
            logger.error(f"Error getting cached candidates for {user_id}: {e}")
            return None

    async def store_product_metadata(
        self,
        product_id: str,
        metadata: Dict[str, Any],
    ):
        """Store product metadata in the local and Redis-backed metadata cache."""
        await self.store_product_metadata_batch({product_id: metadata})

    def prime_product_metadata_memory_cache(
        self,
        metadata_map: Dict[str, Dict[str, Any]],
    ) -> None:
        """Prime per-process metadata cache without writing Redis."""
        if metadata_map:
            self._product_metadata_memory_cache.update(metadata_map)

    async def store_product_metadata_batch(
        self,
        metadata_map: Dict[str, Dict[str, Any]],
    ):
        """Batch-store product metadata in memory and Redis."""
        try:
            if not metadata_map:
                return

            self._product_metadata_memory_cache.update(metadata_map)
            pipeline = self._client_for_key_type("product_metadata").pipeline(
                transaction=False
            )
            for product_id, metadata in metadata_map.items():
                key = f"{self.prefixes['product_metadata']}{product_id}"
                pipeline.setex(
                    key,
                    self.cache_config.product_metadata_ttl,
                    json_dumps(metadata),
                )
            await pipeline.execute()
        except Exception as e:
            logger.error(f"Error storing product metadata batch: {e}")

    async def get_product_metadata(
        self,
        product_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get cached product metadata for a single product."""
        metadata_map = await self.get_product_metadata_batch([product_id])
        return metadata_map.get(product_id)

    async def get_product_metadata_batch(
        self,
        product_ids: Iterable[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Get cached product metadata for many products with local-memory fast path."""
        try:
            unique_product_ids = list(dict.fromkeys(product_ids))
            if not unique_product_ids:
                return {}

            metadata_map: Dict[str, Dict[str, Any]] = {}
            missing_ids: List[str] = []
            for product_id in unique_product_ids:
                cached = self._product_metadata_memory_cache.get(product_id)
                if cached is not None:
                    metadata_map[product_id] = cached
                else:
                    missing_ids.append(product_id)

            if missing_ids:
                keys = [
                    f"{self.prefixes['product_metadata']}{product_id}"
                    for product_id in missing_ids
                ]
                values = await self._client_for_key_type("product_metadata").mget(keys)
                for product_id, value in zip(missing_ids, values):
                    if not value:
                        continue
                    try:
                        decoded = json_loads(value)
                        metadata_map[product_id] = decoded
                        self._product_metadata_memory_cache[product_id] = decoded
                    except Exception:
                        continue

            return metadata_map
        except Exception as e:
            logger.error(f"Error getting product metadata batch: {e}")
            return {}

    async def store_trending_pool(
        self,
        candidates: List[CandidateProduct],
        pool_name: str = "global",
    ):
        """Store a precomputed trending pool in memory and Redis."""
        try:
            key = f"{self.prefixes['trending_products']}{pool_name}"
            payload = [candidate.dict() for candidate in candidates]
            self._trending_pool_memory_cache[pool_name] = [
                CandidateProduct(**item) for item in payload
            ]
            await self._client_for_key_type("trending_products").setex(
                key,
                self.cache_config.serving_pool_ttl,
                pack_cache_payload("trending_pool", payload),
            )
        except Exception as e:
            logger.error(f"Error storing trending pool {pool_name}: {e}")

    async def get_trending_pool(
        self,
        limit: int,
        pool_name: str = "global",
        exclude_items: Optional[set] = None,
    ) -> List[CandidateProduct]:
        """Get a precomputed trending pool, filtered for excluded items."""
        try:
            exclude_items = exclude_items or set()
            cached = self._trending_pool_memory_cache.get(pool_name)
            if cached is None:
                key = f"{self.prefixes['trending_products']}{pool_name}"
                raw = await self._client_for_key_type("trending_products").get(key)
                if raw:
                    payload = unpack_cache_payload(raw, "trending_pool")
                    cached = [CandidateProduct(**item) for item in payload]
                    self._trending_pool_memory_cache[pool_name] = cached
                else:
                    return []

            return [
                CandidateProduct(**candidate.dict())
                for candidate in cached
                if candidate.product_id not in exclude_items
            ][:limit]
        except (Exception, CacheDecodeError) as e:
            logger.error(f"Error getting trending pool {pool_name}: {e}")
            return []

    async def store_category_pools(
        self,
        pools: Dict[str, List[CandidateProduct]],
    ):
        """Store precomputed category pools in memory and Redis."""
        try:
            if not pools:
                return

            pipeline = self._client_for_key_type("category_pool").pipeline(
                transaction=False
            )
            for category, candidates in pools.items():
                payload = [candidate.dict() for candidate in candidates]
                self._category_pool_memory_cache[category] = [
                    CandidateProduct(**item) for item in payload
                ]
                key = f"{self.prefixes['category_pool']}{category}"
                pipeline.setex(
                    key,
                    self.cache_config.serving_pool_ttl,
                    pack_cache_payload("category_pool", payload),
                )
            await pipeline.execute()
        except Exception as e:
            logger.error(f"Error storing category pools: {e}")

    async def get_category_pool(
        self,
        category: str,
        limit: int,
        exclude_items: Optional[set] = None,
    ) -> List[CandidateProduct]:
        """Get a precomputed category pool, filtered for excluded items."""
        try:
            exclude_items = exclude_items or set()
            cached = self._category_pool_memory_cache.get(category)
            if cached is None:
                key = f"{self.prefixes['category_pool']}{category}"
                raw = await self._client_for_key_type("category_pool").get(key)
                if raw:
                    payload = unpack_cache_payload(raw, "category_pool")
                    cached = [CandidateProduct(**item) for item in payload]
                    self._category_pool_memory_cache[category] = cached
                else:
                    return []

            return [
                CandidateProduct(**candidate.dict())
                for candidate in cached
                if candidate.product_id not in exclude_items
            ][:limit]
        except (Exception, CacheDecodeError) as e:
            logger.error(f"Error getting category pool {category}: {e}")
            return []

    async def store_cluster_pools(
        self,
        pools: Dict[Union[str, int], List[CandidateProduct]],
        pool_version: Optional[str] = None,
    ):
        """Store precomputed content-cluster pools in memory and Redis."""
        try:
            if not pools:
                return

            pipeline = self._client_for_key_type("cluster_pool").pipeline(
                transaction=False
            )
            for cluster_id, candidates in pools.items():
                cluster_key = self._cluster_pool_cache_name(
                    cluster_id,
                    pool_version,
                )
                payload = [candidate.dict() for candidate in candidates]
                self._cluster_pool_memory_cache[cluster_key] = [
                    CandidateProduct(**item) for item in payload
                ]
                key = self._cluster_pool_redis_key(cluster_id, pool_version)
                pipeline.setex(
                    key,
                    self.cache_config.serving_pool_ttl,
                    pack_cache_payload("cluster_pool", payload),
                )
            await pipeline.execute()
        except Exception as e:
            logger.error(f"Error storing cluster pools: {e}")

    async def get_cluster_pool(
        self,
        cluster_id: Union[str, int],
        limit: int,
        exclude_items: Optional[set] = None,
        pool_version: Optional[str] = None,
    ) -> List[CandidateProduct]:
        """Get a precomputed content-cluster pool, filtered for excluded items."""
        try:
            if limit <= 0:
                return []
            exclude_items = exclude_items or set()
            cluster_key = self._cluster_pool_cache_name(
                cluster_id,
                pool_version,
            )
            cached = self._cluster_pool_memory_cache.get(cluster_key)
            if cached is None:
                key = self._cluster_pool_redis_key(cluster_id, pool_version)
                raw = await self._client_for_key_type("cluster_pool").get(key)
                if raw:
                    payload = unpack_cache_payload(raw, "cluster_pool")
                    cached = [CandidateProduct(**item) for item in payload]
                    self._cluster_pool_memory_cache[cluster_key] = cached
                else:
                    return []

            return [
                CandidateProduct(**candidate.dict())
                for candidate in cached
                if candidate.product_id not in exclude_items
            ][:limit]
        except (Exception, CacheDecodeError) as e:
            logger.error(f"Error getting cluster pool {cluster_id}: {e}")
            return []

    @staticmethod
    def _cluster_pool_cache_name(
        cluster_id: Union[str, int],
        pool_version: Optional[str],
    ) -> str:
        cluster_key = str(cluster_id)
        return f"{pool_version}:{cluster_key}" if pool_version else cluster_key

    def _cluster_pool_redis_key(
        self,
        cluster_id: Union[str, int],
        pool_version: Optional[str],
    ) -> str:
        return f"{self.prefixes['cluster_pool']}{self._cluster_pool_cache_name(cluster_id, pool_version)}"

    async def get_cached_recommendations(
        self, user_id: str, context_hash: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached recommendation results."""
        try:
            if not self.cache_config.enable_caching:
                return None

            key = f"{self.prefixes['recommendations_cache']}{user_id}:{context_hash}"
            cached_data = await self._client_for_key_type("recommendations_cache").get(
                key
            )

            if cached_data:
                cache_data = unpack_cache_payload(cached_data, "recommendations_cache")
                return cache_data.get("recommendations")

            return None

        except (Exception, CacheDecodeError) as e:
            logger.error(f"Error getting cached recommendations for {user_id}: {e}")
            return None

    async def invalidate_user_cache(self, user_id: str):
        """Invalidate all cache entries for a user."""
        try:
            cache_patterns = [
                f"{self.prefixes['recommendations_cache']}{user_id}:*",
                f"{self.prefixes['candidate_cache']}{user_id}:*",
            ]
            state_patterns = [f"{self.prefixes['user_features']}{user_id}"]

            await self._delete_matching_keys(
                cache_patterns,
                client=self._client_for_key_type("recommendations_cache"),
            )
            await self._delete_matching_keys(
                state_patterns,
                client=self._client_for_key_type("user_features"),
            )
            self.forget_user_known(user_id)

            logger.info(f"Invalidated cache for user {user_id}")

        except Exception as e:
            logger.error(f"Error invalidating cache for {user_id}: {e}")

    async def invalidate_user_serving_cache(self, user_id: str):
        """Invalidate only user-serving caches, leaving user features intact."""
        try:
            patterns = [
                f"{self.prefixes['recommendations_cache']}{user_id}:*",
                f"{self.prefixes['candidate_cache']}{user_id}:*",
            ]
            await self._delete_matching_keys(
                patterns,
                client=self._client_for_key_type("recommendations_cache"),
            )
        except Exception as e:
            logger.error(f"Error invalidating serving cache for {user_id}: {e}")

    # System Metrics and Analytics
    async def log_recommendation_request(
        self, user_id: str, num_recommendations: int, response_time: float
    ):
        """Log recommendation request metrics."""
        try:
            current_minute = int(time.time() // 60)
            key = f"{self.prefixes['system_metrics']}requests:{current_minute}"
            pipeline = self._client_for_key_type("system_metrics").pipeline(
                transaction=False
            )
            pipeline.hincrby(key, "count", 1)
            pipeline.hincrby(key, "total_recommendations", num_recommendations)
            pipeline.hincrby(key, "total_response_time", int(response_time * 1000))
            pipeline.expire(key, 86400)  # 24 hours
            await pipeline.execute()

        except Exception as e:
            logger.error(f"Error logging recommendation request: {e}")

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            current_time = int(time.time())
            metrics = {
                "timestamp": current_time,
                "requests_last_hour": 0,
                "avg_response_time_ms": 0,
                "total_users": 0,
                "cache_hit_rate": 0.0,
            }

            # Get request metrics for last hour
            total_requests = 0
            total_response_time = 0

            for minutes_ago in range(60):
                minute = (current_time // 60) - minutes_ago
                key = f"{self.prefixes['system_metrics']}requests:{minute}"

                minute_data = await self._client_for_key_type("system_metrics").hgetall(
                    key
                )
                if minute_data:
                    count = int(minute_data.get(b"count", 0))
                    response_time = int(minute_data.get(b"total_response_time", 0))

                    total_requests += count
                    total_response_time += response_time

            metrics["requests_last_hour"] = total_requests
            if total_requests > 0:
                metrics["avg_response_time_ms"] = total_response_time / total_requests

            # Count total users (approximate)
            user_pattern = f"{self.prefixes['user_features']}*"
            metrics["total_users"] = await self._count_matching_keys(
                user_pattern,
                client=self._client_for_key_type("user_features"),
            )

            return metrics

        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {"error": str(e), "timestamp": time.time()}

    async def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics data."""
        try:
            # Get recent interactions
            interactions_data = await self._client_for_key_type("analytics").lrange(
                "global_interactions", 0, 999
            )

            interactions = []
            for data in interactions_data:
                try:
                    interaction = json.loads(data.decode())
                    interactions.append(interaction)
                except:
                    continue

            # Calculate analytics
            total_interactions = len(interactions)
            unique_users = len(set(i.get("user_id") for i in interactions))
            unique_products = len(set(i.get("product_id") for i in interactions))

            # Calculate action counts
            action_counts = {}
            for interaction in interactions:
                action = interaction.get("action", "unknown")
                action_counts[action] = action_counts.get(action, 0) + 1

            # Calculate conversion metrics
            clicks = action_counts.get("click", 0)
            purchases = action_counts.get("purchase", 0)
            views = action_counts.get("view", 0)

            ctr = clicks / max(views, 1)
            conversion_rate = purchases / max(clicks, 1)

            return {
                "total_interactions": total_interactions,
                "unique_users": unique_users,
                "unique_products": unique_products,
                "action_counts": action_counts,
                "ctr": round(ctr, 4),
                "conversion_rate": round(conversion_rate, 4),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {"error": str(e)}

    async def write_service_heartbeat(
        self,
        service_name: str,
        instance_id: str,
        ttl_seconds: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish a worker heartbeat entry for readiness checks."""
        payload = {
            "service": service_name,
            "instance_id": instance_id,
            "updated_at": time.time(),
            "metadata": metadata or {},
        }
        key = f"{self.prefixes['health']}heartbeat:{service_name}:{instance_id}"
        await self._client_for_key_type("health").setex(
            key, ttl_seconds, json_dumps(payload)
        )

    async def get_service_heartbeat_status(
        self,
        service_name: str,
    ) -> Dict[str, Any]:
        """Get readiness summary for all live heartbeat keys of a service."""
        pattern = f"{self.prefixes['health']}heartbeat:{service_name}:*"
        heartbeats: List[Dict[str, Any]] = []
        client = self._client_for_key_type("health")
        async for key in client.scan_iter(match=pattern, count=100):
            raw = await client.get(key)
            if not raw:
                continue
            try:
                heartbeats.append(json_loads(raw))
            except Exception:
                continue

        if not heartbeats:
            return {
                "status": "unhealthy",
                "live_instances": 0,
                "service": service_name,
            }

        latest = max(heartbeat.get("updated_at", 0.0) for heartbeat in heartbeats)
        return {
            "status": "healthy",
            "live_instances": len(heartbeats),
            "latest_heartbeat_at": latest,
            "service": service_name,
        }

    # Health Check
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the feature store."""
        try:
            start_time = time.time()

            # Test basic operations
            test_key = "health_check_test"
            test_value = "test_value"
            state_client = self._client_for_key_type("health")
            cache_client = self._client_for_key_type("recommendations_cache")

            # Test write
            await state_client.set(test_key, test_value, ex=60)

            # Test read
            retrieved_value = await state_client.get(test_key)

            # Test delete
            await state_client.delete(test_key)

            response_time = (time.time() - start_time) * 1000

            # Get Redis info
            redis_info = await state_client.info()
            cache_info = (
                await cache_client.info()
                if cache_client is not state_client
                else redis_info
            )

            return {
                "status": "healthy",
                "connected": self.is_connected,
                "response_time_ms": round(response_time, 2),
                "redis_version": redis_info.get("redis_version", "unknown"),
                "used_memory": redis_info.get("used_memory_human", "unknown"),
                "connected_clients": redis_info.get("connected_clients", 0),
                "cache_redis_separate": cache_client is not state_client,
                "cache_used_memory": cache_info.get("used_memory_human", "unknown"),
                "cache_connected_clients": cache_info.get("connected_clients", 0),
                "test_passed": retrieved_value.decode() == test_value
                if retrieved_value
                else False,
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "connected": False, "error": str(e)}

    # Two-Tower Training Data Persistence
    async def store_training_interaction(self, interaction: Dict[str, Any]):
        """Store interaction in a high-capacity list for Two-Tower model training.

        Retains up to 100K interactions over 30 days, providing a much larger
        training window than the 10K global_interactions list.
        """
        try:
            key = "tt_training_interactions"
            client = self._client_for_key_type("analytics")
            await client.lpush(key, json.dumps(interaction))
            await client.ltrim(key, 0, 99999)  # Keep 100K interactions
            await client.expire(key, 86400 * 30)  # 30 days
        except Exception as e:
            logger.error(f"Error storing training interaction: {e}")

    async def get_training_interactions(
        self, limit: int = 50000
    ) -> List[Dict[str, Any]]:
        """Get training interactions for the Two-Tower model.

        Returns up to *limit* recent interactions from the dedicated training list.
        """
        try:
            key = "tt_training_interactions"
            data = await self._client_for_key_type("analytics").lrange(
                key, 0, limit - 1
            )
            interactions = []
            for item in data:
                try:
                    interactions.append(json.loads(item.decode()))
                except Exception:
                    continue
            return interactions
        except Exception as e:
            logger.error(f"Error getting training interactions: {e}")
            return []

    async def get_all_user_features_map(self) -> Dict[str, Dict[str, Any]]:
        """Batch-get user features for all known users.

        Returns a dict mapping user_id -> user features dict.
        Used by the Two-Tower trainer to build side-feature tensors.
        """
        try:
            pattern = f"{self.prefixes['user_features']}*"
            client = self._client_for_key_type("user_features")
            keys = await self._collect_matching_keys(pattern, client=client)
            if not keys:
                return {}

            result: Dict[str, Dict[str, Any]] = {}
            prefix_len = len(self.prefixes["user_features"])

            for key in keys:
                try:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    user_id = key_str[prefix_len:]
                    cached = await client.get(key)
                    if cached:
                        data = unpack_cache_payload(cached, "user_features")
                        result[user_id] = data
                except Exception:
                    continue

            return result
        except Exception as e:
            logger.error(f"Error getting all user features: {e}")
            return {}

    async def _initialize_default_data(self):
        """Initialize any default data needed for the system."""
        try:
            # Initialize system counters if they don't exist
            counters_key = "system_counters"
            client = self._client_for_key_type("system_metrics")
            if not await client.exists(counters_key):
                await client.hset(
                    counters_key,
                    mapping={
                        "total_users": 0,
                        "total_recommendations": 0,
                        "total_interactions": 0,
                    },
                )

            logger.info("Default data initialized")

        except Exception as e:
            logger.error(f"Error initializing default data: {e}")

    async def _collect_matching_keys(
        self, pattern: str, client=None
    ) -> List[Union[str, bytes]]:
        """Collect Redis keys with SCAN to avoid blocking KEYS."""
        client = client or self.redis_client
        results: List[Union[str, bytes]] = []
        async for key in client.scan_iter(match=pattern, count=500):
            results.append(key)
        return results

    async def _count_matching_keys(self, pattern: str, client=None) -> int:
        """Count Redis keys with SCAN to avoid blocking KEYS."""
        client = client or self.redis_client
        count = 0
        async for _ in client.scan_iter(match=pattern, count=500):
            count += 1
        return count

    async def _delete_matching_keys(self, patterns: List[str], client=None):
        """Delete groups of keys discovered via SCAN."""
        client = client or self.redis_client
        for pattern in patterns:
            keys = await self._collect_matching_keys(pattern, client=client)
            if keys:
                await client.delete(*keys)

    # Utility methods
    def generate_context_hash(self, context: Dict[str, Any]) -> str:
        """Generate hash for context to use as cache key."""
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()[:16]
