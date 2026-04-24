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
from typing import Dict, List, Any, Optional, Union, Iterable
import numpy as np
from datetime import datetime, timedelta
import hashlib

# Local imports
from models import (
    CandidateProduct,
    UserFeatures,
    ContentFeatures,
    InteractionType,
    SystemMetrics,
)
from cache_codec import (
    CacheDecodeError,
    json_dumps,
    json_loads,
    pack_cache_payload,
    unpack_cache_payload,
)
from config import RedisConfig, CacheConfig

logger = logging.getLogger(__name__)

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
    
    def __init__(self, redis_config: RedisConfig, cache_config: CacheConfig = None):
        """Initialize the feature store with Redis configuration."""
        self.redis_config = redis_config
        self.cache_config = cache_config or CacheConfig()
        self.redis_client: Optional[redis.Redis] = None
        self.is_connected = False
        
        # Key prefixes for different data types
        self.prefixes = {
            'user_features': 'uf:',
            'content_features': 'cf:',
            'content_status': 'cs:',
            'user_interactions': 'ui:',
            'recommendations_cache': 'rc:',
            'candidate_cache': 'cc:',
            'product_metadata': 'pm:',
            'system_metrics': 'sm:',
            'trending_products': 'tp:',
            'category_pool': 'cp:',
            'product_embeddings': 'pe:',
            'analytics': 'analytics:',
            'health': 'health:'
        }
        self._product_metadata_memory_cache: Dict[str, Dict[str, Any]] = {}
        self._trending_pool_memory_cache: Dict[str, List[CandidateProduct]] = {}
        self._category_pool_memory_cache: Dict[str, List[CandidateProduct]] = {}
        
        logger.info("FeatureStore initialized")
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_config.host,
                port=self.redis_config.port,
                db=self.redis_config.db,
                password=self.redis_config.password,
                decode_responses=False,  # We handle encoding manually
                socket_timeout=self.redis_config.socket_timeout,
                socket_connect_timeout=self.redis_config.socket_connect_timeout,
                retry_on_timeout=self.redis_config.retry_on_timeout,
                max_connections=self.redis_config.max_connections,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            self.is_connected = True
            
            # Initialize default data if needed
            await self._initialize_default_data()
            
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.is_connected = False
            raise
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.is_connected = False
            logger.info("Redis connection closed")
    
    # User Features Management
    async def get_user_features(self, user_id: str) -> UserFeatures:
        """Get user features from cache or create default."""
        try:
            key = f"{self.prefixes['user_features']}{user_id}"
            cached_data = await self.redis_client.get(key)
            
            if cached_data:
                data = unpack_cache_payload(cached_data, "user_features")
                return UserFeatures(**data)
            else:
                # Return default user features
                default_features = UserFeatures(
                    user_id=user_id,
                    total_interactions=0,
                    avg_session_length=0.0,
                    preferred_categories=[],
                    price_sensitivity=0.5,
                    click_through_rate=0.0,
                    conversion_rate=0.0,
                    last_active=time.time(),
                    demographics={}
                )
                
                # Cache default features
                await self._set_user_features(user_id, default_features)
                return default_features
                
        except Exception as e:
            logger.error(f"Error getting user features for {user_id}: {e}")
            return UserFeatures(user_id=user_id)
    
    async def _set_user_features(self, user_id: str, features: UserFeatures):
        """Set user features in cache."""
        try:
            key = f"{self.prefixes['user_features']}{user_id}"
            data = pack_cache_payload("user_features", features.dict())
            
            # Use adaptive TTL based on user activity
            ttl = self._calculate_adaptive_ttl(features)
            
            await self.redis_client.setex(key, ttl, data)
            
        except Exception as e:
            logger.error(f"Error setting user features for {user_id}: {e}")

    async def set_user_features_batch(self, features_map: Dict[str, UserFeatures]):
        """Batch-set user features using one Redis pipeline."""
        try:
            if not features_map:
                return

            pipeline = self.redis_client.pipeline(transaction=False)
            for user_id, features in features_map.items():
                key = f"{self.prefixes['user_features']}{user_id}"
                ttl = self._calculate_adaptive_ttl(features)
                pipeline.setex(
                    key,
                    ttl,
                    pack_cache_payload("user_features", features.dict()),
                )
            await pipeline.execute()
        except Exception as e:
            logger.error(f"Error batch-setting user features: {e}")
    
    async def update_user_features(self, user_id: str, action: str, context: Dict[str, Any] = None):
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
                total_views = await self._get_user_stat(user_id, 'total_views') or 1
                total_clicks = await self._get_user_stat(user_id, 'total_clicks') or 0
                total_clicks += 1
                features.click_through_rate = total_clicks / max(total_views, 1)
                await self._set_user_stat(user_id, 'total_clicks', total_clicks)
                
            elif action == InteractionType.PURCHASE.value:
                # Update conversion rate
                total_clicks = await self._get_user_stat(user_id, 'total_clicks') or 1
                total_purchases = await self._get_user_stat(user_id, 'total_purchases') or 0
                total_purchases += 1
                features.conversion_rate = total_purchases / max(total_clicks, 1)
                await self._set_user_stat(user_id, 'total_purchases', total_purchases)
            
            elif action == InteractionType.VIEW.value:
                total_views = await self._get_user_stat(user_id, 'total_views') or 0
                total_views += 1
                await self._set_user_stat(user_id, 'total_views', total_views)
            
            # Update session length if provided
            if context and 'session_length' in context:
                # Running average of session lengths
                current_avg = features.avg_session_length
                session_count = max(features.total_interactions, 1)
                features.avg_session_length = (
                    (current_avg * (session_count - 1) + context['session_length']) / session_count
                )
            
            # Update preferred categories if product category provided
            if context and 'product_category' in context:
                category = context['product_category']
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
            value = await self.redis_client.get(key)
            return int(value) if value else None
        except:
            return None
    
    async def _set_user_stat(self, user_id: str, stat_name: str, value: int):
        """Set user statistic in cache."""
        try:
            key = f"user_stats:{user_id}:{stat_name}"
            await self.redis_client.setex(key, 86400 * 7, str(value))  # 1 week TTL
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
            
            await self.redis_client.setex(
                key, 
                self.cache_config.content_features_ttl, 
                data
            )
            
            # Also store status
            await self.update_content_status(content_id, "completed")
            
            logger.info(f"Stored content features for {content_id}")
            
        except Exception as e:
            logger.error(f"Error storing content features for {content_id}: {e}")
    
    async def get_content_features(self, content_id: str) -> Optional[ContentFeatures]:
        """Get content features from cache."""
        try:
            key = f"{self.prefixes['content_features']}{content_id}"
            cached_data = await self.redis_client.get(key)
            
            if cached_data:
                data = unpack_cache_payload(cached_data, "content_features")
                return ContentFeatures(**data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting content features for {content_id}: {e}")
            return None
    
    async def update_content_status(self, content_id: str, status: str):
        """Update content processing status."""
        try:
            key = f"{self.prefixes['content_status']}{content_id}"
            status_data = {
                'status': status,
                'updated_at': time.time()
            }
            
            await self.redis_client.setex(
                key, 
                86400,  # 24 hours
                json_dumps(status_data)
            )
            
        except Exception as e:
            logger.error(f"Error updating content status for {content_id}: {e}")
    
    async def get_content_status(self, content_id: str) -> Optional[str]:
        """Get content processing status."""
        try:
            key = f"{self.prefixes['content_status']}{content_id}"
            data = await self.redis_client.get(key)
            
            if data:
                status_data = json_loads(data)
                return status_data.get('status')
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting content status for {content_id}: {e}")
            return None
    
    async def get_content_processed_time(self, content_id: str) -> Optional[float]:
        """Get content processing completion time."""
        try:
            key = f"{self.prefixes['content_status']}{content_id}"
            data = await self.redis_client.get(key)
            
            if data:
                status_data = json_loads(data)
                return status_data.get('updated_at')
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting content processed time for {content_id}: {e}")
            return None
    
    # Interaction Logging
    async def log_user_interaction(
        self, 
        user_id: str, 
        product_id: str, 
        action: str, 
        context: Dict[str, Any] = None
    ):
        """Log recent user interaction for online serving state only."""
        try:
            interaction_data = {
                'user_id': user_id,
                'product_id': product_id,
                'action': action,
                'timestamp': time.time(),
                'context': context or {}
            }
            
            key = f"{self.prefixes['user_interactions']}{user_id}"
            await self.redis_client.lpush(key, json.dumps(interaction_data))
            
            await self.redis_client.ltrim(key, 0, 999)
            await self.redis_client.expire(key, 86400 * 30)  # 30 days

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

            serialized = [
                json.dumps(
                    {
                        "user_id": user_id,
                        "product_id": interaction["product_id"],
                        "action": interaction["action"],
                        "timestamp": interaction.get("timestamp", time.time()),
                        "context": interaction.get("context", {}),
                    }
                )
                for interaction in interactions
            ]

            user_key = f"{self.prefixes['user_interactions']}{user_id}"
            pipeline = self.redis_client.pipeline(transaction=False)
            pipeline.lpush(user_key, *serialized)
            pipeline.ltrim(user_key, 0, 999)
            pipeline.expire(user_key, 86400 * 30)
            await pipeline.execute()
        except Exception as e:
            logger.error(f"Error batch-logging interactions for {user_id}: {e}")

    async def get_user_interactions(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent user interactions."""
        try:
            key = f"{self.prefixes['user_interactions']}{user_id}"
            interactions_data = await self.redis_client.lrange(key, 0, limit - 1)
            
            interactions = []
            for data in interactions_data:
                try:
                    interaction = json.loads(data.decode())
                    interactions.append(interaction)
                except:
                    continue
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error getting user interactions for {user_id}: {e}")
            return []

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
                previous_total_session = features.avg_session_length * max(previous_interactions, 0)
                features.avg_session_length = (
                    previous_total_session + sum(session_lengths)
                ) / max(previous_interactions + len(session_lengths), 1)

            for category in categories:
                if category not in features.preferred_categories:
                    features.preferred_categories.append(category)
            features.preferred_categories = features.preferred_categories[:10]

            stat_names = ("total_views", "total_clicks", "total_purchases")
            stat_keys = [f"user_stats:{user_id}:{name}" for name in stat_names]
            stat_values = await self.redis_client.mget(stat_keys)
            stats = {
                name: int(value) if value else 0
                for name, value in zip(stat_names, stat_values)
            }
            stats["total_views"] += action_counts.get(InteractionType.VIEW.value, 0)
            stats["total_clicks"] += action_counts.get(InteractionType.CLICK.value, 0)
            stats["total_purchases"] += action_counts.get(InteractionType.PURCHASE.value, 0)

            features.click_through_rate = stats["total_clicks"] / max(stats["total_views"], 1)
            features.conversion_rate = stats["total_purchases"] / max(stats["total_clicks"], 1)

            pipeline = self.redis_client.pipeline(transaction=False)
            ttl = self._calculate_adaptive_ttl(features)
            pipeline.setex(
                f"{self.prefixes['user_features']}{user_id}",
                ttl,
                pack_cache_payload("user_features", features.dict()),
            )
            for stat_name, stat_value in stats.items():
                pipeline.setex(f"user_stats:{user_id}:{stat_name}", 86400 * 7, str(stat_value))
            await pipeline.execute()
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
                'recommendations': recommendations,
                'cached_at': time.time(),
                'user_id': user_id
            }
            
            # Use user-specific TTL
            user_features = user_features or await self.get_user_features(user_id)
            ttl = self._calculate_adaptive_ttl(user_features)
            
            await self.redis_client.setex(
                key, 
                min(ttl, self.cache_config.recommendations_ttl),
                pack_cache_payload("recommendations_cache", cache_data)
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
            await self.redis_client.setex(
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
            cached_data = await self.redis_client.get(key)
            if not cached_data:
                return None

            cache_data = unpack_cache_payload(cached_data, "candidate_cache")
            return [CandidateProduct(**item) for item in cache_data.get("candidates", [])]
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

    async def store_product_metadata_batch(
        self,
        metadata_map: Dict[str, Dict[str, Any]],
    ):
        """Batch-store product metadata in memory and Redis."""
        try:
            if not metadata_map:
                return

            self._product_metadata_memory_cache.update(metadata_map)
            pipeline = self.redis_client.pipeline(transaction=False)
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
                keys = [f"{self.prefixes['product_metadata']}{product_id}" for product_id in missing_ids]
                values = await self.redis_client.mget(keys)
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
            self._trending_pool_memory_cache[pool_name] = [CandidateProduct(**item) for item in payload]
            await self.redis_client.setex(
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
                raw = await self.redis_client.get(key)
                if raw:
                    payload = unpack_cache_payload(raw, "trending_pool")
                    cached = [CandidateProduct(**item) for item in payload]
                    self._trending_pool_memory_cache[pool_name] = cached
                else:
                    return []

            return [
                candidate
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

            pipeline = self.redis_client.pipeline(transaction=False)
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
                raw = await self.redis_client.get(key)
                if raw:
                    payload = unpack_cache_payload(raw, "category_pool")
                    cached = [CandidateProduct(**item) for item in payload]
                    self._category_pool_memory_cache[category] = cached
                else:
                    return []

            return [
                candidate
                for candidate in cached
                if candidate.product_id not in exclude_items
            ][:limit]
        except (Exception, CacheDecodeError) as e:
            logger.error(f"Error getting category pool {category}: {e}")
            return []
    
    async def get_cached_recommendations(
        self, 
        user_id: str, 
        context_hash: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached recommendation results."""
        try:
            if not self.cache_config.enable_caching:
                return None
            
            key = f"{self.prefixes['recommendations_cache']}{user_id}:{context_hash}"
            cached_data = await self.redis_client.get(key)
            
            if cached_data:
                cache_data = unpack_cache_payload(cached_data, "recommendations_cache")
                return cache_data.get('recommendations')
            
            return None
            
        except (Exception, CacheDecodeError) as e:
            logger.error(f"Error getting cached recommendations for {user_id}: {e}")
            return None
    
    async def invalidate_user_cache(self, user_id: str):
        """Invalidate all cache entries for a user."""
        try:
            patterns = [
                f"{self.prefixes['recommendations_cache']}{user_id}:*",
                f"{self.prefixes['candidate_cache']}{user_id}:*",
                f"{self.prefixes['user_features']}{user_id}"
            ]

            await self._delete_matching_keys(patterns)
            
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
            await self._delete_matching_keys(patterns)
        except Exception as e:
            logger.error(f"Error invalidating serving cache for {user_id}: {e}")
    
    # System Metrics and Analytics
    async def log_recommendation_request(self, user_id: str, num_recommendations: int, response_time: float):
        """Log recommendation request metrics."""
        try:
            current_minute = int(time.time() // 60)
            key = f"{self.prefixes['system_metrics']}requests:{current_minute}"
            pipeline = self.redis_client.pipeline(transaction=False)
            pipeline.hincrby(key, 'count', 1)
            pipeline.hincrby(key, 'total_recommendations', num_recommendations)
            pipeline.hincrby(key, 'total_response_time', int(response_time * 1000))
            pipeline.expire(key, 86400)  # 24 hours
            await pipeline.execute()
            
        except Exception as e:
            logger.error(f"Error logging recommendation request: {e}")
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            current_time = int(time.time())
            metrics = {
                'timestamp': current_time,
                'requests_last_hour': 0,
                'avg_response_time_ms': 0,
                'total_users': 0,
                'cache_hit_rate': 0.0
            }
            
            # Get request metrics for last hour
            total_requests = 0
            total_response_time = 0
            
            for minutes_ago in range(60):
                minute = (current_time // 60) - minutes_ago
                key = f"{self.prefixes['system_metrics']}requests:{minute}"
                
                minute_data = await self.redis_client.hgetall(key)
                if minute_data:
                    count = int(minute_data.get(b'count', 0))
                    response_time = int(minute_data.get(b'total_response_time', 0))
                    
                    total_requests += count
                    total_response_time += response_time
            
            metrics['requests_last_hour'] = total_requests
            if total_requests > 0:
                metrics['avg_response_time_ms'] = total_response_time / total_requests
            
            # Count total users (approximate)
            user_pattern = f"{self.prefixes['user_features']}*"
            metrics['total_users'] = await self._count_matching_keys(user_pattern)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics data."""
        try:
            # Get recent interactions
            interactions_data = await self.redis_client.lrange("global_interactions", 0, 999)
            
            interactions = []
            for data in interactions_data:
                try:
                    interaction = json.loads(data.decode())
                    interactions.append(interaction)
                except:
                    continue
            
            # Calculate analytics
            total_interactions = len(interactions)
            unique_users = len(set(i.get('user_id') for i in interactions))
            unique_products = len(set(i.get('product_id') for i in interactions))
            
            # Calculate action counts
            action_counts = {}
            for interaction in interactions:
                action = interaction.get('action', 'unknown')
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Calculate conversion metrics
            clicks = action_counts.get('click', 0)
            purchases = action_counts.get('purchase', 0)
            views = action_counts.get('view', 0)
            
            ctr = clicks / max(views, 1)
            conversion_rate = purchases / max(clicks, 1)
            
            return {
                'total_interactions': total_interactions,
                'unique_users': unique_users,
                'unique_products': unique_products,
                'action_counts': action_counts,
                'ctr': round(ctr, 4),
                'conversion_rate': round(conversion_rate, 4),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {'error': str(e)}

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
        await self.redis_client.setex(key, ttl_seconds, json_dumps(payload))

    async def get_service_heartbeat_status(
        self,
        service_name: str,
    ) -> Dict[str, Any]:
        """Get readiness summary for all live heartbeat keys of a service."""
        pattern = f"{self.prefixes['health']}heartbeat:{service_name}:*"
        heartbeats: List[Dict[str, Any]] = []
        async for key in self.redis_client.scan_iter(match=pattern, count=100):
            raw = await self.redis_client.get(key)
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
            
            # Test write
            await self.redis_client.set(test_key, test_value, ex=60)
            
            # Test read
            retrieved_value = await self.redis_client.get(test_key)
            
            # Test delete
            await self.redis_client.delete(test_key)
            
            response_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            redis_info = await self.redis_client.info()
            
            return {
                'status': 'healthy',
                'connected': self.is_connected,
                'response_time_ms': round(response_time, 2),
                'redis_version': redis_info.get('redis_version', 'unknown'),
                'used_memory': redis_info.get('used_memory_human', 'unknown'),
                'connected_clients': redis_info.get('connected_clients', 0),
                'test_passed': retrieved_value.decode() == test_value if retrieved_value else False
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'connected': False,
                'error': str(e)
            }
    
    # Two-Tower Training Data Persistence
    async def store_training_interaction(self, interaction: Dict[str, Any]):
        """Store interaction in a high-capacity list for Two-Tower model training.

        Retains up to 100K interactions over 30 days, providing a much larger
        training window than the 10K global_interactions list.
        """
        try:
            key = "tt_training_interactions"
            await self.redis_client.lpush(key, json.dumps(interaction))
            await self.redis_client.ltrim(key, 0, 99999)  # Keep 100K interactions
            await self.redis_client.expire(key, 86400 * 30)  # 30 days
        except Exception as e:
            logger.error(f"Error storing training interaction: {e}")

    async def get_training_interactions(self, limit: int = 50000) -> List[Dict[str, Any]]:
        """Get training interactions for the Two-Tower model.

        Returns up to *limit* recent interactions from the dedicated training list.
        """
        try:
            key = "tt_training_interactions"
            data = await self.redis_client.lrange(key, 0, limit - 1)
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
            keys = await self._collect_matching_keys(pattern)
            if not keys:
                return {}

            result: Dict[str, Dict[str, Any]] = {}
            prefix_len = len(self.prefixes['user_features'])

            for key in keys:
                try:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    user_id = key_str[prefix_len:]
                    cached = await self.redis_client.get(key)
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
            if not await self.redis_client.exists(counters_key):
                await self.redis_client.hset(counters_key, mapping={
                    'total_users': 0,
                    'total_recommendations': 0,
                    'total_interactions': 0
                })
            
            logger.info("Default data initialized")
            
        except Exception as e:
            logger.error(f"Error initializing default data: {e}")

    async def _collect_matching_keys(self, pattern: str) -> List[Union[str, bytes]]:
        """Collect Redis keys with SCAN to avoid blocking KEYS."""
        results: List[Union[str, bytes]] = []
        async for key in self.redis_client.scan_iter(match=pattern, count=500):
            results.append(key)
        return results

    async def _count_matching_keys(self, pattern: str) -> int:
        """Count Redis keys with SCAN to avoid blocking KEYS."""
        count = 0
        async for _ in self.redis_client.scan_iter(match=pattern, count=500):
            count += 1
        return count

    async def _delete_matching_keys(self, patterns: List[str]):
        """Delete groups of keys discovered via SCAN."""
        for pattern in patterns:
            keys = await self._collect_matching_keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
    
    # Utility methods
    def generate_context_hash(self, context: Dict[str, Any]) -> str:
        """Generate hash for context to use as cache key."""
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()[:16]
