"""
AI-Powered Video Commerce Recommender - Main Recommendation Engine
==================================================================

This module implements the core recommendation logic that combines multiple
recommendation sources: collaborative filtering, content-based matching,
and trending/popularity algorithms to generate diverse candidate products.
"""

import numpy as np
import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import time
import json
from collections import defaultdict, Counter
import random

# Local imports
from models import (
    UserFeatures, ContentFeatures, CandidateProduct, 
    ProductData, InteractionType
)
from feature_store import FeatureStore
from vector_search import VectorSearchEngine
from config import RecommendationConfig

logger = logging.getLogger(__name__)

class CollaborativeFilteringEngine:
    """
    Collaborative filtering using Non-negative Matrix Factorization (NMF).
    Learns user-item interactions to recommend products based on similar users.
    """
    
    def __init__(self, config: RecommendationConfig):
        self.config = config
        self.model = NMF(
            n_components=config.cf_factors,
            init='random',
            random_state=42,
            max_iter=config.cf_iterations,
            alpha=config.cf_regularization
        )
        
        # User-item interaction matrix
        self.user_item_matrix: Optional[csr_matrix] = None
        self.user_mapping: Dict[str, int] = {}  # user_id -> matrix index
        self.item_mapping: Dict[str, int] = {}  # product_id -> matrix index
        self.reverse_item_mapping: Dict[int, str] = {}  # matrix index -> product_id
        
        # Model components
        self.user_features_matrix: Optional[np.ndarray] = None
        self.item_features_matrix: Optional[np.ndarray] = None
        
        self.is_trained = False
        self.last_training_time = 0
        
        logger.info("Collaborative filtering engine initialized")
    
    async def train_model(self, interactions: List[Dict[str, Any]]):
        """Train the collaborative filtering model on user interactions."""
        try:
            logger.info(f"Training CF model on {len(interactions)} interactions")
            
            if len(interactions) < 10:
                logger.warning("Too few interactions to train CF model")
                return
            
            # Build user-item interaction matrix
            await self._build_interaction_matrix(interactions)
            
            if self.user_item_matrix is None or self.user_item_matrix.nnz == 0:
                logger.warning("Empty interaction matrix, cannot train CF model")
                return
            
            # Train NMF model
            start_time = time.time()
            self.user_features_matrix = self.model.fit_transform(self.user_item_matrix)
            self.item_features_matrix = self.model.components_.T
            
            training_time = time.time() - start_time
            self.last_training_time = time.time()
            self.is_trained = True
            
            logger.info(f"CF model trained in {training_time:.2f}s with {self.config.cf_factors} factors")
            
        except Exception as e:
            logger.error(f"Error training CF model: {e}")
            self.is_trained = False
    
    async def _build_interaction_matrix(self, interactions: List[Dict[str, Any]]):
        """Build sparse user-item interaction matrix from interactions."""
        try:
            # Count interactions and create mappings
            user_item_counts = defaultdict(lambda: defaultdict(float))
            users = set()
            items = set()
            
            # Weight different interaction types
            interaction_weights = {
                InteractionType.VIEW.value: 1.0,
                InteractionType.CLICK.value: 2.0,
                InteractionType.ADD_TO_CART.value: 3.0,
                InteractionType.PURCHASE.value: 5.0,
                InteractionType.FAVORITE.value: 2.5,
                InteractionType.SHARE.value: 1.5
            }
            
            for interaction in interactions:
                user_id = interaction.get('user_id')
                product_id = interaction.get('product_id')
                action = interaction.get('action', 'view')
                
                if user_id and product_id:
                    weight = interaction_weights.get(action, 1.0)
                    user_item_counts[user_id][product_id] += weight
                    users.add(user_id)
                    items.add(product_id)
            
            if not users or not items:
                logger.warning("No valid user-item pairs found")
                return
            
            # Create mappings
            self.user_mapping = {user: idx for idx, user in enumerate(sorted(users))}
            self.item_mapping = {item: idx for idx, item in enumerate(sorted(items))}
            self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
            
            # Build sparse matrix
            rows, cols, data = [], [], []
            
            for user_id, items_dict in user_item_counts.items():
                user_idx = self.user_mapping[user_id]
                for product_id, weight in items_dict.items():
                    if product_id in self.item_mapping:
                        item_idx = self.item_mapping[product_id]
                        rows.append(user_idx)
                        cols.append(item_idx)
                        data.append(weight)
            
            if rows:
                self.user_item_matrix = csr_matrix(
                    (data, (rows, cols)),
                    shape=(len(users), len(items)),
                    dtype=np.float32
                )
                
                logger.info(f"Built interaction matrix: {self.user_item_matrix.shape} with {self.user_item_matrix.nnz} interactions")
            
        except Exception as e:
            logger.error(f"Error building interaction matrix: {e}")
    
    async def get_user_recommendations(
        self, 
        user_id: str, 
        k: int = 100,
        exclude_items: Set[str] = None
    ) -> List[CandidateProduct]:
        """Get collaborative filtering recommendations for a user."""
        try:
            if not self.is_trained or self.user_features_matrix is None:
                logger.warning("CF model not trained, returning empty recommendations")
                return []
            
            exclude_items = exclude_items or set()
            user_idx = self.user_mapping.get(user_id)
            
            if user_idx is None:
                # New user - return popular items
                return await self._get_popular_items_fallback(k, exclude_items)
            
            # Get user's latent features
            user_features = self.user_features_matrix[user_idx]
            
            # Compute scores for all items
            item_scores = np.dot(self.item_features_matrix, user_features)
            
            # Get top-k items (excluding already interacted items)
            user_items = set(self.user_item_matrix[user_idx].nonzero()[1])
            
            candidates = []
            sorted_indices = np.argsort(item_scores)[::-1]
            
            for item_idx in sorted_indices:
                if len(candidates) >= k:
                    break
                
                product_id = self.reverse_item_mapping.get(item_idx)
                if (product_id and 
                    item_idx not in user_items and 
                    product_id not in exclude_items):
                    
                    score = float(item_scores[item_idx])
                    if score > 0:  # Only positive scores
                        candidate = CandidateProduct(
                            product_id=product_id,
                            collaborative_score=score,
                            combined_score=score,
                            source="collaborative_filtering"
                        )
                        candidates.append(candidate)
            
            logger.debug(f"Generated {len(candidates)} CF recommendations for user {user_id}")
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting CF recommendations for {user_id}: {e}")
            return []
    
    async def _get_popular_items_fallback(
        self, 
        k: int, 
        exclude_items: Set[str] = None
    ) -> List[CandidateProduct]:
        """Fallback to popular items for new users."""
        try:
            if self.user_item_matrix is None:
                return []
            
            exclude_items = exclude_items or set()
            
            # Calculate item popularity (sum of interactions)
            item_popularity = np.array(self.user_item_matrix.sum(axis=0)).flatten()
            
            candidates = []
            sorted_indices = np.argsort(item_popularity)[::-1]
            
            for item_idx in sorted_indices:
                if len(candidates) >= k:
                    break
                
                product_id = self.reverse_item_mapping.get(item_idx)
                if product_id and product_id not in exclude_items:
                    score = float(item_popularity[item_idx]) / max(item_popularity.max(), 1)
                    
                    candidate = CandidateProduct(
                        product_id=product_id,
                        collaborative_score=score,
                        combined_score=score,
                        source="popular_fallback"
                    )
                    candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting popular items fallback: {e}")
            return []

class TrendingEngine:
    """
    Trending/popularity engine that identifies hot products based on
    recent user interactions with time decay.
    """
    
    def __init__(self, config: RecommendationConfig):
        self.config = config
        self.trending_scores: Dict[str, float] = {}
        self.interaction_counts: Dict[str, List[Tuple[float, float]]] = {}  # product_id -> [(timestamp, weight)]
        self.last_updated = 0
        
        logger.info("Trending engine initialized")
    
    async def update_trending_scores(self, recent_interactions: List[Dict[str, Any]]):
        """Update trending scores based on recent interactions."""
        try:
            current_time = time.time()
            window_start = current_time - (self.config.trending_window_hours * 3600)
            
            # Weight different interaction types
            interaction_weights = {
                InteractionType.VIEW.value: 1.0,
                InteractionType.CLICK.value: 3.0,
                InteractionType.ADD_TO_CART.value: 5.0,
                InteractionType.PURCHASE.value: 10.0,
                InteractionType.FAVORITE.value: 4.0,
                InteractionType.SHARE.value: 6.0
            }
            
            # Process recent interactions
            for interaction in recent_interactions:
                product_id = interaction.get('product_id')
                action = interaction.get('action', 'view')
                timestamp = interaction.get('timestamp', current_time)
                
                if product_id and timestamp >= window_start:
                    weight = interaction_weights.get(action, 1.0)
                    
                    if product_id not in self.interaction_counts:
                        self.interaction_counts[product_id] = []
                    
                    self.interaction_counts[product_id].append((timestamp, weight))
            
            # Calculate trending scores with time decay
            self.trending_scores = {}
            
            for product_id, interactions in self.interaction_counts.items():
                # Filter interactions within window
                recent_interactions_list = [
                    (ts, weight) for ts, weight in interactions 
                    if ts >= window_start
                ]
                
                if not recent_interactions_list:
                    continue
                
                # Calculate score with time decay
                total_score = 0.0
                for timestamp, weight in recent_interactions_list:
                    # Time decay: more recent interactions have higher weight
                    hours_ago = (current_time - timestamp) / 3600
                    decay_factor = self.config.trending_decay_factor ** hours_ago
                    total_score += weight * decay_factor
                
                self.trending_scores[product_id] = total_score
            
            # Clean up old interaction data
            cutoff_time = current_time - (self.config.trending_window_hours * 2 * 3600)  # Keep 2x window
            for product_id in list(self.interaction_counts.keys()):
                self.interaction_counts[product_id] = [
                    (ts, weight) for ts, weight in self.interaction_counts[product_id]
                    if ts >= cutoff_time
                ]
                
                # Remove products with no recent interactions
                if not self.interaction_counts[product_id]:
                    del self.interaction_counts[product_id]
            
            self.last_updated = current_time
            
            logger.debug(f"Updated trending scores for {len(self.trending_scores)} products")
            
        except Exception as e:
            logger.error(f"Error updating trending scores: {e}")
    
    async def get_trending_recommendations(
        self, 
        k: int = 100,
        category_filter: str = None,
        exclude_items: Set[str] = None
    ) -> List[CandidateProduct]:
        """Get trending product recommendations."""
        try:
            exclude_items = exclude_items or set()
            
            if not self.trending_scores:
                logger.warning("No trending scores available")
                return []
            
            # Sort products by trending score
            sorted_products = sorted(
                self.trending_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            candidates = []
            max_score = max(self.trending_scores.values()) if self.trending_scores else 1.0
            
            for product_id, score in sorted_products:
                if len(candidates) >= k:
                    break
                
                if product_id not in exclude_items:
                    normalized_score = score / max_score
                    
                    candidate = CandidateProduct(
                        product_id=product_id,
                        popularity_score=normalized_score,
                        combined_score=normalized_score,
                        source="trending"
                    )
                    candidates.append(candidate)
            
            logger.debug(f"Generated {len(candidates)} trending recommendations")
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []

class RecommendationEngine:
    """
    Main recommendation engine that orchestrates multiple recommendation sources
    and combines them to generate diverse, high-quality product recommendations.
    """
    
    def __init__(
        self, 
        feature_store: FeatureStore, 
        vector_search: VectorSearchEngine,
        config: RecommendationConfig
    ):
        self.feature_store = feature_store
        self.vector_search = vector_search
        self.config = config
        
        # Recommendation engines
        self.cf_engine = CollaborativeFilteringEngine(config)
        self.trending_engine = TrendingEngine(config)
        
        # Model state
        self.is_initialized = False
        self.last_model_update = 0
        
        logger.info("Recommendation engine initialized")
    
    async def load_models(self):
        """Load and initialize all recommendation models."""
        try:
            logger.info("Loading recommendation models")
            
            # Load recent interactions for model training
            await self._update_models_from_interactions()
            
            self.is_initialized = True
            logger.info("Recommendation models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading recommendation models: {e}")
            raise
    
    async def _update_models_from_interactions(self):
        """Update models using recent interaction data."""
        try:
            # Get recent interactions from feature store
            interactions_data = await self.feature_store.redis_client.lrange("global_interactions", 0, 9999)
            
            interactions = []
            for data in interactions_data:
                try:
                    interaction = json.loads(data.decode())
                    interactions.append(interaction)
                except:
                    continue
            
            if interactions:
                # Update collaborative filtering
                await self.cf_engine.train_model(interactions)
                
                # Update trending scores
                await self.trending_engine.update_trending_scores(interactions[-1000:])  # Recent 1k interactions
                
                self.last_model_update = time.time()
                logger.info(f"Updated models with {len(interactions)} interactions")
            else:
                logger.warning("No interactions found for model training")
            
        except Exception as e:
            logger.error(f"Error updating models from interactions: {e}")
    
    async def generate_candidates(
        self,
        user_id: str,
        content_features: Optional[ContentFeatures] = None,
        context: Dict[str, Any] = None,
        k_per_source: int = 100
    ) -> List[CandidateProduct]:
        """
        Generate candidate products from multiple recommendation sources.
        
        Args:
            user_id: User identifier
            content_features: Optional content features for content-based recommendations
            context: Additional context information
            k_per_source: Number of candidates per recommendation source
            
        Returns:
            Combined list of candidate products with scores from different sources
        """
        try:
            logger.debug(f"Generating candidates for user {user_id}")
            
            context = context or {}
            all_candidates: Dict[str, CandidateProduct] = {}
            
            # Get user's interaction history to exclude already seen items
            user_interactions = await self.feature_store.get_user_interactions(user_id, limit=500)
            exclude_items = {interaction['product_id'] for interaction in user_interactions}
            
            # 1. Collaborative Filtering Recommendations
            if self.cf_engine.is_trained:
                cf_candidates = await self.cf_engine.get_user_recommendations(
                    user_id, k_per_source, exclude_items
                )
                
                for candidate in cf_candidates:
                    if candidate.product_id not in all_candidates:
                        all_candidates[candidate.product_id] = candidate
                    else:
                        # Merge scores if product already exists from another source
                        all_candidates[candidate.product_id].collaborative_score = candidate.collaborative_score
            
            # 2. Content-Based Recommendations (if content provided)
            if content_features and content_features.visual_embedding:
                try:
                    content_candidates = await self.vector_search.search_similar_products(
                        np.array(content_features.visual_embedding),
                        k=k_per_source
                    )
                    
                    for candidate in content_candidates:
                        if candidate.product_id not in exclude_items:
                            if candidate.product_id not in all_candidates:
                                all_candidates[candidate.product_id] = candidate
                            else:
                                all_candidates[candidate.product_id].content_similarity_score = candidate.content_similarity_score
                                
                except Exception as e:
                    logger.warning(f"Error getting content-based recommendations: {e}")
            
            # 3. Trending/Popular Recommendations
            trending_candidates = await self.trending_engine.get_trending_recommendations(
                k=k_per_source,
                exclude_items=exclude_items
            )
            
            for candidate in trending_candidates:
                if candidate.product_id not in all_candidates:
                    all_candidates[candidate.product_id] = candidate
                else:
                    all_candidates[candidate.product_id].popularity_score = candidate.popularity_score
            
            # 4. Random/Diversity Recommendations (for exploration)
            if len(all_candidates) < k_per_source:
                random_candidates = await self.vector_search.get_random_products(
                    k=k_per_source - len(all_candidates)
                )
                
                for candidate in random_candidates:
                    if (candidate.product_id not in exclude_items and 
                        candidate.product_id not in all_candidates):
                        all_candidates[candidate.product_id] = candidate
            
            # Combine scores for final ranking
            final_candidates = []
            for candidate in all_candidates.values():
                # Calculate combined score using weighted average
                cf_score = candidate.collaborative_score or 0.0
                content_score = candidate.content_similarity_score or 0.0
                popularity_score = candidate.popularity_score or 0.0
                
                combined_score = (
                    cf_score * self.config.cf_weight +
                    content_score * self.config.content_weight +
                    popularity_score * self.config.popularity_weight
                )
                
                candidate.combined_score = combined_score
                final_candidates.append(candidate)
            
            # Apply diversity if enabled
            if self.config.enable_diversity:
                final_candidates = await self._apply_diversity_filter(final_candidates, context)
            
            # Sort by combined score
            final_candidates.sort(key=lambda x: x.combined_score, reverse=True)
            
            # Limit to maximum candidates
            final_candidates = final_candidates[:self.config.max_total_candidates]
            
            logger.debug(f"Generated {len(final_candidates)} candidates from multiple sources")
            return final_candidates
            
        except Exception as e:
            logger.error(f"Error generating candidates for {user_id}: {e}")
            return []
    
    async def _apply_diversity_filter(
        self, 
        candidates: List[CandidateProduct], 
        context: Dict[str, Any]
    ) -> List[CandidateProduct]:
        """Apply diversity filter to avoid over-concentration in specific categories."""
        try:
            if not self.config.enable_diversity or len(candidates) <= 10:
                return candidates
            
            # Group candidates by category (using vector search metadata)
            category_groups = defaultdict(list)
            ungrouped_candidates = []
            
            for candidate in candidates:
                metadata = await self.vector_search.get_product_metadata(candidate.product_id)
                if metadata and 'category' in metadata:
                    category = metadata['category']
                    category_groups[category].append(candidate)
                else:
                    ungrouped_candidates.append(candidate)
            
            # Select diverse candidates
            diverse_candidates = []
            max_per_category = self.config.max_items_per_category
            
            # Interleave candidates from different categories
            while len(diverse_candidates) < len(candidates) and (category_groups or ungrouped_candidates):
                added_in_round = 0
                
                # Add one from each category
                for category in list(category_groups.keys()):
                    if category_groups[category]:
                        # Get highest scoring candidate from this category
                        category_groups[category].sort(key=lambda x: x.combined_score, reverse=True)
                        candidate = category_groups[category].pop(0)
                        diverse_candidates.append(candidate)
                        added_in_round += 1
                        
                        # Remove category if we've hit the limit
                        category_count = sum(1 for c in diverse_candidates 
                                           if self._get_candidate_category(c) == category)
                        if category_count >= max_per_category:
                            del category_groups[category]
                
                # Add ungrouped candidates
                if ungrouped_candidates and added_in_round < 3:
                    diverse_candidates.extend(ungrouped_candidates[:3-added_in_round])
                    ungrouped_candidates = ungrouped_candidates[3-added_in_round:]
                
                if added_in_round == 0:
                    break
            
            logger.debug(f"Applied diversity filter: {len(candidates)} -> {len(diverse_candidates)} candidates")
            return diverse_candidates
            
        except Exception as e:
            logger.error(f"Error applying diversity filter: {e}")
            return candidates
    
    def _get_candidate_category(self, candidate: CandidateProduct) -> str:
        """Get category for a candidate (cached lookup)."""
        # This would typically be cached or pre-computed
        return "unknown"
    
    async def get_trending_recommendations(self, k: int = 10) -> List[Dict[str, Any]]:
        """Get trending recommendations for fallback scenarios."""
        try:
            trending_candidates = await self.trending_engine.get_trending_recommendations(k=k)
            
            recommendations = []
            for candidate in trending_candidates:
                # Get product metadata
                metadata = await self.vector_search.get_product_metadata(candidate.product_id)
                
                if metadata:
                    recommendation = {
                        'product_id': candidate.product_id,
                        'title': metadata.get('title', 'Product'),
                        'price': metadata.get('price', 0.0),
                        'category': metadata.get('category', 'unknown'),
                        'confidence_score': candidate.popularity_score,
                        'ranking_score': candidate.combined_score,
                        'reason': 'Trending product'
                    }
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    async def update_models(self):
        """Periodically update recommendation models with new data."""
        try:
            current_time = time.time()
            
            # Update models every hour
            if current_time - self.last_model_update > 3600:
                await self._update_models_from_interactions()
                logger.info("Recommendation models updated")
            
        except Exception as e:
            logger.error(f"Error updating models: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recommendation engine statistics."""
        return {
            'is_initialized': self.is_initialized,
            'last_model_update': self.last_model_update,
            'cf_trained': self.cf_engine.is_trained,
            'cf_users': len(self.cf_engine.user_mapping),
            'cf_items': len(self.cf_engine.item_mapping),
            'trending_products': len(self.trending_engine.trending_scores),
            'config': {
                'cf_factors': self.config.cf_factors,
                'cf_weight': self.config.cf_weight,
                'content_weight': self.config.content_weight,
                'popularity_weight': self.config.popularity_weight,
                'enable_diversity': self.config.enable_diversity
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the recommendation engine."""
        try:
            return {
                'status': 'healthy' if self.is_initialized else 'unhealthy',
                'initialized': self.is_initialized,
                'cf_model_trained': self.cf_engine.is_trained,
                'trending_data_available': len(self.trending_engine.trending_scores) > 0,
                'last_model_update': self.last_model_update,
                'stats': self.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'initialized': False
            }