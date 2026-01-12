"""
AI-Powered Video Commerce Recommender - Feature Store
=====================================================

This module implements a Redis-based feature store for caching user features,
content features, and system metrics. It provides real-time feature updates
and intelligent caching strategies for optimal performance.
"""

import redis.asyncio as redis
import json
import pickle
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from datetime import datetime, timedelta
import hashlib

# Local imports
from models import UserFeatures, ContentFeatures, InteractionType, SystemMetrics
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
            'system_metrics': 'sm:',
            'trending_products': 'tp:',
            'product_embeddings': 'pe:',
            'analytics': 'analytics:',
            'health': 'health:'
        }
        
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
                data = pickle.loads(cached_data)
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
            data = pickle.dumps(features.dict())
            
            # Use adaptive TTL based on user activity
            ttl = self._calculate_adaptive_ttl(features)
            
            await self.redis_client.setex(key, ttl, data)
            
        except Exception as e:
            logger.error(f"Error setting user features for {user_id}: {e}")
    
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
            data = pickle.dumps(features.dict())
            
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
                data = pickle.loads(cached_data)
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
                json.dumps(status_data)
            )
            
        except Exception as e:
            logger.error(f"Error updating content status for {content_id}: {e}")
    
    async def get_content_status(self, content_id: str) -> Optional[str]:
        """Get content processing status."""
        try:
            key = f"{self.prefixes['content_status']}{content_id}"
            data = await self.redis_client.get(key)
            
            if data:
                status_data = json.loads(data.decode())
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
                status_data = json.loads(data.decode())
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
        """Log user interaction for analytics and model training."""
        try:
            interaction_data = {
                'user_id': user_id,
                'product_id': product_id,
                'action': action,
                'timestamp': time.time(),
                'context': context or {}
            }
            
            # Store in time-ordered list for analytics
            key = f"{self.prefixes['user_interactions']}{user_id}"
            await self.redis_client.lpush(key, json.dumps(interaction_data))
            
            # Keep only recent interactions (last 1000)
            await self.redis_client.ltrim(key, 0, 999)
            await self.redis_client.expire(key, 86400 * 30)  # 30 days
            
            # Also store globally for system analytics
            global_key = "global_interactions"
            await self.redis_client.lpush(global_key, json.dumps(interaction_data))
            await self.redis_client.ltrim(global_key, 0, 9999)  # Keep last 10k
            await self.redis_client.expire(global_key, 86400 * 7)  # 7 days
            
            logger.debug(f"Logged interaction: {user_id} -> {action} -> {product_id}")
            
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")
    
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
    
    # Recommendation Caching
    async def cache_recommendations(
        self, 
        user_id: str, 
        context_hash: str, 
        recommendations: List[Dict[str, Any]]
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
            user_features = await self.get_user_features(user_id)
            ttl = self._calculate_adaptive_ttl(user_features)
            
            await self.redis_client.setex(
                key, 
                min(ttl, self.cache_config.recommendations_ttl),
                pickle.dumps(cache_data)
            )
            
        except Exception as e:
            logger.error(f"Error caching recommendations for {user_id}: {e}")
    
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
                cache_data = pickle.loads(cached_data)
                return cache_data.get('recommendations')
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached recommendations for {user_id}: {e}")
            return None
    
    async def invalidate_user_cache(self, user_id: str):
        """Invalidate all cache entries for a user."""
        try:
            # Find all cache keys for this user
            patterns = [
                f"{self.prefixes['recommendations_cache']}{user_id}:*",
                f"{self.prefixes['user_features']}{user_id}"
            ]
            
            for pattern in patterns:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            
            logger.info(f"Invalidated cache for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error invalidating cache for {user_id}: {e}")
    
    # System Metrics and Analytics
    async def log_recommendation_request(self, user_id: str, num_recommendations: int, response_time: float):
        """Log recommendation request metrics."""
        try:
            current_minute = int(time.time() // 60)
            key = f"{self.prefixes['system_metrics']}requests:{current_minute}"
            
            # Increment request count
            await self.redis_client.hincrby(key, 'count', 1)
            await self.redis_client.hincrby(key, 'total_recommendations', num_recommendations)
            await self.redis_client.hincrby(key, 'total_response_time', int(response_time * 1000))
            
            # Set expiration for cleanup
            await self.redis_client.expire(key, 86400)  # 24 hours
            
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
            user_keys = await self.redis_client.keys(user_pattern)
            metrics['total_users'] = len(user_keys) if user_keys else 0
            
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
    
    # Utility methods
    def generate_context_hash(self, context: Dict[str, Any]) -> str:
        """Generate hash for context to use as cache key."""
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()[:16]