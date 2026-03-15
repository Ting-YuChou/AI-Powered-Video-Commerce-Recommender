"""
AI-Powered Video Commerce Recommender - Main Recommendation Engine
==================================================================

This module implements the core recommendation logic that combines multiple
recommendation sources: Two-Tower collaborative filtering with ANN retrieval,
content-based matching, and trending/popularity algorithms to generate diverse
candidate products.
"""

import numpy as np
import faiss
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import time
import json
from collections import defaultdict
from pathlib import Path

# Local imports
from models import (
    UserFeatures, ContentFeatures, CandidateProduct,
    InteractionType
)
from feature_store import FeatureStore
from vector_search import VectorSearchEngine
from config import RecommendationConfig
from two_tower import TwoTowerTrainer

logger = logging.getLogger(__name__)


class TwoTowerRetrievalEngine:
    """Two-Tower neural retrieval engine for collaborative filtering.

    Replaces the legacy NMF-based approach with a dual-encoder architecture:
      - UserTower encodes user_id + side features into a 128-dim embedding.
      - ItemTower encodes item_id + CLIP embedding + side features into a 128-dim embedding.
      - Training uses InfoNCE loss with hard/mixed negative sampling.
      - Serving uses FAISS HNSW ANN search for O(log N) retrieval.

    External interface is kept identical to the old CollaborativeFilteringEngine
    so that RecommendationEngine.generate_candidates() works without changes.
    """

    def __init__(self, config: RecommendationConfig, vector_search: VectorSearchEngine):
        self.config = config
        self.vector_search = vector_search

        # Trainer (handles model, training, encoding)
        self.trainer = TwoTowerTrainer(
            clip_dim=vector_search.embedding_dim,
            output_dim=config.tt_embedding_dim,
            temperature=config.tt_temperature,
            learning_rate=config.tt_learning_rate,
            batch_size=config.tt_batch_size,
            epochs=config.tt_epochs,
            num_hard_negatives=config.tt_num_hard_negatives,
            num_random_negatives=config.tt_num_random_negatives,
            hard_ratio_start=config.tt_hard_negative_ratio_start,
            hard_ratio_end=config.tt_hard_negative_ratio_end,
            user_hidden_dims=config.tt_user_hidden_dims,
            item_hidden_dims=config.tt_item_hidden_dims,
        )

        # CF FAISS index (populated after training)
        self.cf_index: Optional[faiss.Index] = None
        self.cf_index_map: Dict[int, str] = {}  # faiss_idx -> product_id

        # Backward-compatible attributes consumed by RecommendationEngine.get_stats()
        self.user_mapping: Dict[str, int] = {}
        self.item_mapping: Dict[str, int] = {}

        self.is_trained = False
        self.last_training_time: float = 0
        self._item_popularity: Dict[str, float] = {}

        logger.info("Two-Tower retrieval engine initialized")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    async def train_model(
        self,
        interactions: List[Dict[str, Any]],
        user_features_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Train the Two-Tower model on user interaction data.

        Args:
            interactions: list of {user_id, product_id, action, timestamp, ...}
            user_features_map: optional mapping of user_id -> user features dict
        """
        try:
            logger.info(f"Training Two-Tower model on {len(interactions)} interactions")

            if len(interactions) < 10:
                logger.warning("Too few interactions to train Two-Tower model")
                return

            # Gather product metadata and CLIP embeddings from VectorSearchEngine
            product_metadata = dict(self.vector_search.product_metadata)
            product_clip_embeddings = self.vector_search.get_all_product_embeddings()
            user_features_map = user_features_map or {}

            # Build item popularity counts for cold-start fallback
            self._item_popularity = defaultdict(float)
            interaction_weights = {
                InteractionType.VIEW.value: 1.0,
                InteractionType.CLICK.value: 2.0,
                InteractionType.ADD_TO_CART.value: 3.0,
                InteractionType.PURCHASE.value: 5.0,
                InteractionType.FAVORITE.value: 2.5,
                InteractionType.SHARE.value: 1.5,
            }
            for inter in interactions:
                pid = inter.get("product_id")
                action = inter.get("action", "view")
                if pid:
                    self._item_popularity[pid] += interaction_weights.get(action, 1.0)

            # Prepare data
            self.trainer.prepare(
                interactions=interactions,
                product_metadata=product_metadata,
                product_clip_embeddings=product_clip_embeddings,
                user_features_map=user_features_map,
            )

            # Try to load existing checkpoint for warm-start
            existing_index = self.cf_index
            checkpoint_path = self.config.cf_index_path.replace(".faiss", ".pt")
            if Path(checkpoint_path).exists():
                self.trainer.load_checkpoint(checkpoint_path)

            # Run training
            start_time = time.time()
            stats = self.trainer.train(existing_cf_index=existing_index)
            training_time = time.time() - start_time

            # Build FAISS index from trained item embeddings
            self.cf_index, idx_map = self.trainer.build_item_index()

            # Build reverse mapping: faiss_idx -> product_id
            self.cf_index_map = {}
            for faiss_idx, item_idx in idx_map.items():
                product_id = self.trainer.reverse_item_mapping.get(item_idx)
                if product_id:
                    self.cf_index_map[faiss_idx] = product_id

            # Sync backward-compat attributes
            self.user_mapping = dict(self.trainer.user_mapping)
            self.item_mapping = dict(self.trainer.item_mapping)

            # Save checkpoint and index
            self.trainer.save_checkpoint(checkpoint_path)
            VectorSearchEngine.save_cf_index(
                self.cf_index,
                self.config.cf_index_path,
                metadata={
                    "num_items": len(self.item_mapping),
                    "embedding_dim": self.config.tt_embedding_dim,
                    "index_map": {str(k): v for k, v in self.cf_index_map.items()},
                },
            )

            self.is_trained = True
            self.last_training_time = time.time()

            logger.info(
                f"Two-Tower model trained in {training_time:.1f}s — "
                f"{len(self.user_mapping)} users, {len(self.item_mapping)} items, "
                f"index size={self.cf_index.ntotal}"
            )

        except Exception as e:
            logger.error(f"Error training Two-Tower model: {e}")
            self.is_trained = False

    # ------------------------------------------------------------------
    # Recommendation serving
    # ------------------------------------------------------------------

    async def get_user_recommendations(
        self,
        user_id: str,
        k: int = 100,
        exclude_items: Optional[Set[str]] = None,
        user_features: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateProduct]:
        """Get collaborative-filtering recommendations via Two-Tower ANN retrieval.

        Args:
            user_id: user identifier
            k: number of candidates to return
            exclude_items: product_ids to exclude
            user_features: optional user features dict (avoids redundant lookups)
        """
        try:
            if not self.is_trained or self.cf_index is None or self.cf_index.ntotal == 0:
                logger.warning("Two-Tower model not trained, returning empty recommendations")
                return []

            exclude_items = exclude_items or set()
            user_features = user_features or {}

            # Encode user via UserTower
            user_embedding = self.trainer.encode_user(user_id, user_features)

            if user_embedding is None:
                return await self._get_popular_items_fallback(k, exclude_items)

            # ANN search
            query = user_embedding.reshape(1, -1).astype(np.float32)
            search_k = min(k * 3, self.cf_index.ntotal)
            scores, indices = self.cf_index.search(query, search_k)

            candidates: List[CandidateProduct] = []
            for score, idx in zip(scores[0], indices[0]):
                if len(candidates) >= k:
                    break
                if idx == -1:
                    continue
                product_id = self.cf_index_map.get(int(idx))
                if product_id and product_id not in exclude_items:
                    # FAISS inner-product scores from HNSW; normalise to [0, 1]
                    norm_score = float(max(min((score + 1.0) / 2.0, 1.0), 0.0))
                    candidate = CandidateProduct(
                        product_id=product_id,
                        collaborative_score=norm_score,
                        combined_score=norm_score,
                        source="collaborative_filtering",
                    )
                    candidates.append(candidate)

            logger.debug(f"Generated {len(candidates)} Two-Tower recommendations for user {user_id}")
            return candidates

        except Exception as e:
            logger.error(f"Error getting Two-Tower recommendations for {user_id}: {e}")
            return []

    async def _get_popular_items_fallback(
        self, k: int, exclude_items: Optional[Set[str]] = None
    ) -> List[CandidateProduct]:
        """Fallback to popular items for cold-start users."""
        try:
            exclude_items = exclude_items or set()
            if not self._item_popularity:
                return []

            max_pop = max(self._item_popularity.values()) if self._item_popularity else 1.0
            sorted_items = sorted(self._item_popularity.items(), key=lambda x: x[1], reverse=True)

            candidates: List[CandidateProduct] = []
            for product_id, pop_score in sorted_items:
                if len(candidates) >= k:
                    break
                if product_id not in exclude_items:
                    score = pop_score / max(max_pop, 1.0)
                    candidates.append(
                        CandidateProduct(
                            product_id=product_id,
                            collaborative_score=score,
                            combined_score=score,
                            source="popular_fallback",
                        )
                    )
            return candidates

        except Exception as e:
            logger.error(f"Error in popular items fallback: {e}")
            return []


class TrendingEngine:
    """
    Trending/popularity engine that identifies hot products based on
    recent user interactions with time decay.
    """

    def __init__(self, config: RecommendationConfig):
        self.config = config
        self.trending_scores: Dict[str, float] = {}
        self.interaction_counts: Dict[str, List[Tuple[float, float]]] = {}
        self.last_updated = 0

        logger.info("Trending engine initialized")

    async def update_trending_scores(self, recent_interactions: List[Dict[str, Any]]):
        """Update trending scores based on recent interactions."""
        try:
            current_time = time.time()
            window_start = current_time - (self.config.trending_window_hours * 3600)

            interaction_weights = {
                InteractionType.VIEW.value: 1.0,
                InteractionType.CLICK.value: 3.0,
                InteractionType.ADD_TO_CART.value: 5.0,
                InteractionType.PURCHASE.value: 10.0,
                InteractionType.FAVORITE.value: 4.0,
                InteractionType.SHARE.value: 6.0,
            }

            for interaction in recent_interactions:
                product_id = interaction.get("product_id")
                action = interaction.get("action", "view")
                timestamp = interaction.get("timestamp", current_time)

                if product_id and timestamp >= window_start:
                    weight = interaction_weights.get(action, 1.0)
                    if product_id not in self.interaction_counts:
                        self.interaction_counts[product_id] = []
                    self.interaction_counts[product_id].append((timestamp, weight))

            self.trending_scores = {}
            for product_id, interactions in self.interaction_counts.items():
                recent = [(ts, w) for ts, w in interactions if ts >= window_start]
                if not recent:
                    continue
                total_score = 0.0
                for timestamp, weight in recent:
                    hours_ago = (current_time - timestamp) / 3600
                    decay_factor = self.config.trending_decay_factor ** hours_ago
                    total_score += weight * decay_factor
                self.trending_scores[product_id] = total_score

            cutoff_time = current_time - (self.config.trending_window_hours * 2 * 3600)
            for product_id in list(self.interaction_counts.keys()):
                self.interaction_counts[product_id] = [
                    (ts, w) for ts, w in self.interaction_counts[product_id] if ts >= cutoff_time
                ]
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
        exclude_items: Optional[Set[str]] = None,
    ) -> List[CandidateProduct]:
        """Get trending product recommendations."""
        try:
            exclude_items = exclude_items or set()
            if not self.trending_scores:
                logger.warning("No trending scores available")
                return []

            sorted_products = sorted(self.trending_scores.items(), key=lambda x: x[1], reverse=True)
            max_score = max(self.trending_scores.values()) if self.trending_scores else 1.0

            candidates: List[CandidateProduct] = []
            for product_id, score in sorted_products:
                if len(candidates) >= k:
                    break
                if product_id not in exclude_items:
                    normalized_score = score / max_score
                    candidates.append(
                        CandidateProduct(
                            product_id=product_id,
                            popularity_score=normalized_score,
                            combined_score=normalized_score,
                            source="trending",
                        )
                    )

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
        config: RecommendationConfig,
    ):
        self.feature_store = feature_store
        self.vector_search = vector_search
        self.config = config

        # Recommendation engines
        self.cf_engine = TwoTowerRetrievalEngine(config, vector_search)
        self.trending_engine = TrendingEngine(config)

        # Model state
        self.is_initialized = False
        self.last_model_update = 0

        logger.info("Recommendation engine initialized (Two-Tower retrieval)")

    async def load_models(self):
        """Load and initialize all recommendation models."""
        try:
            logger.info("Loading recommendation models")

            # Try to load a pre-existing CF index for fast startup
            await self._try_load_cf_index()

            # Train / update with latest interaction data
            await self._update_models_from_interactions()

            self.is_initialized = True
            logger.info("Recommendation models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading recommendation models: {e}")
            raise

    async def load_serving_state(self):
        """Load only the artifacts needed for online serving.

        This path avoids any retraining work in the serving process. If no
        retrieval index exists yet, the service still starts in degraded mode
        and can serve cached or trending-only responses.
        """
        try:
            logger.info("Loading recommendation serving state")
            await self._try_load_cf_index()
            self.is_initialized = True
            if not self.cf_engine.is_trained:
                logger.warning("Two-Tower index not available; serving will fall back to non-CF sources")
            logger.info("Recommendation serving state ready")
        except Exception as e:
            logger.error(f"Error loading recommendation serving state: {e}")
            raise

    async def _try_load_cf_index(self):
        """Attempt to load a previously saved CF FAISS index for fast cold start."""
        try:
            result = VectorSearchEngine.load_cf_index(self.config.cf_index_path)
            if result is not None:
                index, metadata = result
                # Restore index map
                index_map_raw = metadata.get("index_map", {})
                self.cf_engine.cf_index = index
                self.cf_engine.cf_index_map = {int(k): v for k, v in index_map_raw.items()}

                # Load trainer checkpoint
                checkpoint_path = self.config.cf_index_path.replace(".faiss", ".pt")
                if self.cf_engine.trainer.load_checkpoint(checkpoint_path):
                    self.cf_engine.user_mapping = dict(self.cf_engine.trainer.user_mapping)
                    self.cf_engine.item_mapping = dict(self.cf_engine.trainer.item_mapping)
                    self.cf_engine.is_trained = True
                    logger.info("Loaded pre-existing Two-Tower model and CF index")
        except Exception as e:
            logger.warning(f"Could not load pre-existing CF index: {e}")

    async def _update_models_from_interactions(self):
        """Update models using recent interaction data."""
        try:
            # Prefer the larger training-data list; fall back to global_interactions
            interactions = await self.feature_store.get_training_interactions(limit=50000)
            if not interactions:
                interactions_data = await self.feature_store.redis_client.lrange(
                    "global_interactions", 0, 9999
                )
                interactions = []
                for data in interactions_data:
                    try:
                        interactions.append(json.loads(data.decode()))
                    except Exception:
                        continue

            if interactions:
                # Gather user features for the trainer
                user_features_map = await self.feature_store.get_all_user_features_map()

                # Train Two-Tower model
                await self.cf_engine.train_model(
                    interactions, user_features_map=user_features_map
                )

                # Update trending scores (use last 1K for recency)
                await self.trending_engine.update_trending_scores(interactions[-1000:])

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
        context: Optional[Dict[str, Any]] = None,
        k_per_source: int = 100,
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
            exclude_items = {interaction["product_id"] for interaction in user_interactions}

            # Get user features once for reuse
            user_features_obj = await self.feature_store.get_user_features(user_id)
            user_features_dict = user_features_obj.dict() if user_features_obj else {}

            # 1. Collaborative Filtering via Two-Tower ANN Retrieval
            if self.cf_engine.is_trained:
                cf_candidates = await self.cf_engine.get_user_recommendations(
                    user_id, k_per_source, exclude_items, user_features=user_features_dict
                )
                for candidate in cf_candidates:
                    if candidate.product_id not in all_candidates:
                        all_candidates[candidate.product_id] = candidate
                    else:
                        all_candidates[candidate.product_id].collaborative_score = (
                            candidate.collaborative_score
                        )

            # 2. Content-Based Recommendations (if content provided)
            if content_features and content_features.visual_embedding:
                try:
                    content_candidates = await self.vector_search.search_similar_products(
                        np.array(content_features.visual_embedding), k=k_per_source
                    )
                    for candidate in content_candidates:
                        if candidate.product_id not in exclude_items:
                            if candidate.product_id not in all_candidates:
                                all_candidates[candidate.product_id] = candidate
                            else:
                                all_candidates[
                                    candidate.product_id
                                ].content_similarity_score = candidate.content_similarity_score
                except Exception as e:
                    logger.warning(f"Error getting content-based recommendations: {e}")

            # 3. Trending/Popular Recommendations
            trending_candidates = await self.trending_engine.get_trending_recommendations(
                k=k_per_source, exclude_items=exclude_items
            )
            for candidate in trending_candidates:
                if candidate.product_id not in all_candidates:
                    all_candidates[candidate.product_id] = candidate
                else:
                    all_candidates[candidate.product_id].popularity_score = (
                        candidate.popularity_score
                    )

            # 4. Random/Diversity Recommendations (for exploration)
            if len(all_candidates) < k_per_source:
                random_candidates = await self.vector_search.get_random_products(
                    k=k_per_source - len(all_candidates)
                )
                for candidate in random_candidates:
                    if (
                        candidate.product_id not in exclude_items
                        and candidate.product_id not in all_candidates
                    ):
                        all_candidates[candidate.product_id] = candidate

            # Combine scores for final ranking
            final_candidates: List[CandidateProduct] = []
            for candidate in all_candidates.values():
                cf_score = candidate.collaborative_score or 0.0
                content_score = candidate.content_similarity_score or 0.0
                popularity_score = candidate.popularity_score or 0.0

                combined_score = (
                    cf_score * self.config.cf_weight
                    + content_score * self.config.content_weight
                    + popularity_score * self.config.popularity_weight
                )
                candidate.combined_score = combined_score
                final_candidates.append(candidate)

            # Apply diversity if enabled
            if self.config.enable_diversity:
                final_candidates = await self._apply_diversity_filter(final_candidates, context)

            # Sort by combined score
            final_candidates.sort(key=lambda x: x.combined_score, reverse=True)
            final_candidates = final_candidates[: self.config.max_total_candidates]

            logger.debug(f"Generated {len(final_candidates)} candidates from multiple sources")
            return final_candidates

        except Exception as e:
            logger.error(f"Error generating candidates for {user_id}: {e}")
            return []

    async def _apply_diversity_filter(
        self, candidates: List[CandidateProduct], context: Dict[str, Any]
    ) -> List[CandidateProduct]:
        """Apply diversity filter to avoid over-concentration in specific categories."""
        try:
            if not self.config.enable_diversity or len(candidates) <= 10:
                return candidates

            category_groups: Dict[str, List[CandidateProduct]] = defaultdict(list)
            ungrouped_candidates: List[CandidateProduct] = []

            for candidate in candidates:
                metadata = await self.vector_search.get_product_metadata(candidate.product_id)
                if metadata and "category" in metadata:
                    category_groups[metadata["category"]].append(candidate)
                else:
                    ungrouped_candidates.append(candidate)

            diverse_candidates: List[CandidateProduct] = []
            max_per_category = self.config.max_items_per_category

            while len(diverse_candidates) < len(candidates) and (
                category_groups or ungrouped_candidates
            ):
                added_in_round = 0
                for category in list(category_groups.keys()):
                    if category_groups[category]:
                        category_groups[category].sort(
                            key=lambda x: x.combined_score, reverse=True
                        )
                        c = category_groups[category].pop(0)
                        diverse_candidates.append(c)
                        added_in_round += 1

                        category_count = sum(
                            1
                            for dc in diverse_candidates
                            if self._get_candidate_category(dc) == category
                        )
                        if category_count >= max_per_category:
                            del category_groups[category]

                if ungrouped_candidates and added_in_round < 3:
                    diverse_candidates.extend(ungrouped_candidates[: 3 - added_in_round])
                    ungrouped_candidates = ungrouped_candidates[3 - added_in_round :]

                if added_in_round == 0:
                    break

            logger.debug(
                f"Applied diversity filter: {len(candidates)} -> {len(diverse_candidates)} candidates"
            )
            return diverse_candidates

        except Exception as e:
            logger.error(f"Error applying diversity filter: {e}")
            return candidates

    def _get_candidate_category(self, candidate: CandidateProduct) -> str:
        """Get category for a candidate (cached lookup)."""
        return "unknown"

    async def get_trending_recommendations(self, k: int = 10) -> List[Dict[str, Any]]:
        """Get trending recommendations for fallback scenarios."""
        try:
            trending_candidates = await self.trending_engine.get_trending_recommendations(k=k)
            recommendations = []
            for candidate in trending_candidates:
                metadata = await self.vector_search.get_product_metadata(candidate.product_id)
                if metadata:
                    recommendations.append(
                        {
                            "product_id": candidate.product_id,
                            "title": metadata.get("title", "Product"),
                            "price": metadata.get("price", 0.0),
                            "category": metadata.get("category", "unknown"),
                            "confidence_score": candidate.popularity_score,
                            "ranking_score": candidate.combined_score,
                            "reason": "Trending product",
                        }
                    )
            return recommendations

        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []

    async def update_models(self):
        """Periodically update recommendation models with new data."""
        try:
            current_time = time.time()
            if current_time - self.last_model_update > 3600:
                await self._update_models_from_interactions()
                logger.info("Recommendation models updated")

        except Exception as e:
            logger.error(f"Error updating models: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get recommendation engine statistics."""
        model_params = 0
        if self.cf_engine.trainer.model is not None:
            model_params = sum(
                p.numel() for p in self.cf_engine.trainer.model.parameters()
            )

        return {
            "is_initialized": self.is_initialized,
            "last_model_update": self.last_model_update,
            "cf_trained": self.cf_engine.is_trained,
            "cf_users": len(self.cf_engine.user_mapping),
            "cf_items": len(self.cf_engine.item_mapping),
            "cf_embedding_dim": self.config.tt_embedding_dim,
            "cf_index_size": self.cf_engine.cf_index.ntotal if self.cf_engine.cf_index else 0,
            "cf_model_parameters": model_params,
            "trending_products": len(self.trending_engine.trending_scores),
            "config": {
                "tt_embedding_dim": self.config.tt_embedding_dim,
                "tt_temperature": self.config.tt_temperature,
                "tt_num_hard_negatives": self.config.tt_num_hard_negatives,
                "tt_num_random_negatives": self.config.tt_num_random_negatives,
                "cf_weight": self.config.cf_weight,
                "content_weight": self.config.content_weight,
                "popularity_weight": self.config.popularity_weight,
                "enable_diversity": self.config.enable_diversity,
            },
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the recommendation engine."""
        try:
            return {
                "status": "healthy" if self.is_initialized else "unhealthy",
                "initialized": self.is_initialized,
                "cf_model_trained": self.cf_engine.is_trained,
                "cf_index_size": (
                    self.cf_engine.cf_index.ntotal if self.cf_engine.cf_index else 0
                ),
                "trending_data_available": len(self.trending_engine.trending_scores) > 0,
                "last_model_update": self.last_model_update,
                "stats": self.get_stats(),
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": False,
            }
