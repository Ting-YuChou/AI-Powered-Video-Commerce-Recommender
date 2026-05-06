"""
AI-Powered Video Commerce Recommender - Main Recommendation Engine
==================================================================

This module implements the core recommendation logic that combines multiple
recommendation sources: Two-Tower collaborative filtering with ANN retrieval,
content-based matching, and trending/popularity algorithms to generate diverse
candidate products.
"""

import asyncio
import numpy as np
import faiss
import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Set, Tuple
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
import threading

# Local imports
from models import (
    UserFeatures, ContentFeatures, CandidateProduct,
    InteractionType
)
from feature_store import FeatureStore
from model_artifacts import ModelArtifactManager
from vector_search import VectorSearchEngine
from config import RecommendationConfig
from two_tower import TwoTowerTrainer
from sasrec import SASRecCandidateEngine

logger = logging.getLogger(__name__)

POSITIVE_SEQUENCE_ACTIONS = {
    InteractionType.VIEW.value,
    InteractionType.CLICK.value,
    InteractionType.ADD_TO_CART.value,
    InteractionType.PURCHASE.value,
}


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
            architecture=config.tt_architecture,
            cross_layers=config.tt_cross_layers,
        )

        # CF FAISS index (populated after training)
        self.cf_index: Optional[faiss.Index] = None
        self.cf_index_map: Dict[int, str] = {}  # faiss_idx -> product_id

        # Backward-compatible attributes consumed by RecommendationEngine.get_stats()
        self.user_mapping: Dict[str, int] = {}
        self.item_mapping: Dict[str, int] = {}

        self.is_trained = False
        self.last_training_time: float = 0
        self.model_version: Optional[str] = None
        self._item_popularity: Dict[str, float] = {}
        self._user_embedding_cache_max_size = max(
            0,
            int(getattr(config, "user_embedding_cache_size", 20000)),
        )
        self._user_embedding_cache_time_bucket_seconds = max(
            0.001,
            float(getattr(config, "user_embedding_cache_time_bucket_seconds", 1.0)),
        )
        self._user_embedding_cache: OrderedDict[
            Tuple[Any, ...],
            np.ndarray,
        ] = OrderedDict()
        self._user_embedding_cache_lock = threading.RLock()

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
            await asyncio.to_thread(
                self.trainer.prepare,
                interactions=interactions,
                product_metadata=product_metadata,
                product_clip_embeddings=product_clip_embeddings,
                user_features_map=user_features_map,
            )

            # Try to load existing checkpoint for warm-start
            existing_index = self.cf_index
            checkpoint_path = self.config.cf_index_path.replace(".faiss", ".pt")
            if Path(checkpoint_path).exists():
                await asyncio.to_thread(self.trainer.warm_start_from_checkpoint, checkpoint_path)

            # Run training
            start_time = time.time()
            stats = await asyncio.to_thread(
                self.trainer.train,
                existing_cf_index=existing_index,
            )
            training_time = time.time() - start_time

            # Build FAISS index from trained item embeddings
            self.cf_index, idx_map = await asyncio.to_thread(self.trainer.build_item_index)

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
            await asyncio.to_thread(self.trainer.save_checkpoint, checkpoint_path)
            await asyncio.to_thread(
                VectorSearchEngine.save_cf_index,
                self.cf_index,
                self.config.cf_index_path,
                {
                    "num_items": len(self.item_mapping),
                    "embedding_dim": self.config.tt_embedding_dim,
                    "index_map": {str(k): v for k, v in self.cf_index_map.items()},
                },
            )

            self.is_trained = True
            self.last_training_time = time.time()
            self.model_version = f"two-tower-{int(self.last_training_time)}"
            self.clear_user_embedding_cache()

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
        return await asyncio.to_thread(
            self._get_user_recommendations_sync,
            user_id,
            k,
            exclude_items,
            user_features,
        )

    def _get_user_recommendations_sync(
        self,
        user_id: str,
        k: int = 100,
        exclude_items: Optional[Set[str]] = None,
        user_features: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateProduct]:
        try:
            if not self.is_trained or self.cf_index is None or self.cf_index.ntotal == 0:
                logger.warning("Two-Tower model not trained, returning empty recommendations")
                return []

            exclude_items = exclude_items or set()
            user_features = user_features or {}
            current_time = time.time()
            user_embedding = self._get_user_embedding(
                user_id,
                user_features,
                current_time,
            )

            if user_embedding is None:
                return self._get_popular_items_fallback_sync(k, exclude_items)

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
                    norm_score = float(max(min((score + 1.0) / 2.0, 1.0), 0.0))
                    candidates.append(
                        CandidateProduct(
                            product_id=product_id,
                            collaborative_score=norm_score,
                            combined_score=norm_score,
                            source="collaborative_filtering",
                        )
                    )

            logger.debug(f"Generated {len(candidates)} Two-Tower recommendations for user {user_id}")
            return candidates

        except Exception as e:
            logger.error(f"Error getting Two-Tower recommendations for {user_id}: {e}")
            return []

    def clear_user_embedding_cache(self) -> None:
        with self._user_embedding_cache_lock:
            self._user_embedding_cache.clear()

    def _get_user_embedding(
        self,
        user_id: str,
        user_features: Dict[str, Any],
        current_time: float,
    ) -> Optional[np.ndarray]:
        cache_key = self._user_embedding_cache_key(user_id, user_features, current_time)
        with self._user_embedding_cache_lock:
            cached = self._user_embedding_cache.get(cache_key)
            if cached is not None:
                self._user_embedding_cache.move_to_end(cache_key)
                return cached.copy()

        user_embedding = self.trainer.encode_user(
            user_id,
            user_features,
            current_time=current_time,
        )
        if user_embedding is None or self._user_embedding_cache_max_size <= 0:
            return user_embedding

        with self._user_embedding_cache_lock:
            self._user_embedding_cache[cache_key] = user_embedding.copy()
            if len(self._user_embedding_cache) > self._user_embedding_cache_max_size:
                self._user_embedding_cache.popitem(last=False)
        return user_embedding

    def _user_embedding_cache_key(
        self,
        user_id: str,
        user_features: Dict[str, Any],
        current_time: float,
    ) -> Tuple[Any, ...]:
        time_bucket = int(
            current_time / self._user_embedding_cache_time_bucket_seconds
        )
        normalized_features = {
            "total_interactions": int(user_features.get("total_interactions", 0)),
            "avg_session_length": round(float(user_features.get("avg_session_length", 0.0)), 3),
            "preferred_categories": list(user_features.get("preferred_categories", [])),
            "price_sensitivity": round(float(user_features.get("price_sensitivity", 0.5)), 6),
            "click_through_rate": round(float(user_features.get("click_through_rate", 0.0)), 6),
            "conversion_rate": round(float(user_features.get("conversion_rate", 0.0)), 6),
            "last_active": round(float(user_features.get("last_active", current_time)), 3),
        }
        feature_hash = hashlib.sha256(
            json.dumps(normalized_features, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return (
            self.model_version,
            user_id,
            time_bucket,
            feature_hash,
        )

    async def _get_popular_items_fallback(
        self, k: int, exclude_items: Optional[Set[str]] = None
    ) -> List[CandidateProduct]:
        """Fallback to popular items for cold-start users."""
        return self._get_popular_items_fallback_sync(k, exclude_items)

    def _get_popular_items_fallback_sync(
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
        artifact_manager: Optional[ModelArtifactManager] = None,
    ):
        self.feature_store = feature_store
        self.vector_search = vector_search
        self.config = config
        self.artifact_manager = artifact_manager

        # Recommendation engines
        self.cf_engine = TwoTowerRetrievalEngine(config, vector_search)
        self.sasrec_engine = SASRecCandidateEngine(config)
        self.trending_engine = TrendingEngine(config)

        # Model state
        self.is_initialized = False
        self.last_model_update = 0
        self.loaded_two_tower_version: Optional[str] = None
        self.loaded_sasrec_version: Optional[str] = None

        logger.info("Recommendation engine initialized (Two-Tower + SASRec retrieval)")

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
            await self._try_load_sasrec_artifacts()
            await self.refresh_serving_pools()
            self.is_initialized = True
            if not self.cf_engine.is_trained:
                logger.warning("Two-Tower index not available; serving will fall back to non-CF sources")
            if self.config.enable_sasrec and not self.sasrec_engine.is_trained:
                logger.warning("SASRec artifacts not available; serving will skip sequential candidates")
            logger.info("Recommendation serving state ready")
        except Exception as e:
            logger.error(f"Error loading recommendation serving state: {e}")
            raise

    async def _try_load_cf_index(self):
        """Attempt to load a previously saved CF FAISS index for fast cold start."""
        try:
            checkpoint_record = None
            if self.artifact_manager:
                checkpoint_record = await self.artifact_manager.sync_latest_two_tower_artifacts()

            result = await asyncio.to_thread(
                VectorSearchEngine.load_cf_index,
                self.config.cf_index_path,
            )
            if result is not None:
                index, metadata = result
                # Restore index map
                index_map_raw = metadata.get("index_map", {})
                self.cf_engine.cf_index = index
                self.cf_engine.cf_index_map = {int(k): v for k, v in index_map_raw.items()}

                # Load trainer checkpoint
                checkpoint_path = (
                    self.artifact_manager.two_tower_local_checkpoint_path
                    if self.artifact_manager
                    else self.config.cf_index_path.replace(".faiss", ".pt")
                )
                if await asyncio.to_thread(self.cf_engine.trainer.load_checkpoint, checkpoint_path):
                    self.cf_engine.user_mapping = dict(self.cf_engine.trainer.user_mapping)
                    self.cf_engine.item_mapping = dict(self.cf_engine.trainer.item_mapping)
                    self.cf_engine.is_trained = True
                    if checkpoint_record:
                        self.cf_engine.model_version = checkpoint_record.model_version
                        self.loaded_two_tower_version = checkpoint_record.model_version
                    self.cf_engine.clear_user_embedding_cache()
                    logger.info("Loaded pre-existing Two-Tower model and CF index")
        except Exception as e:
            logger.warning(f"Could not load pre-existing CF index: {e}")

    async def _try_load_sasrec_artifacts(self) -> bool:
        """Load SASRec serving artifacts if the sequential source is enabled."""
        if not self.config.enable_sasrec:
            return False
        try:
            checkpoint_record = None
            if self.artifact_manager:
                checkpoint_record = await self.artifact_manager.sync_latest_sasrec_artifacts()

            checkpoint_path, vocab_path, metadata_path = self._sasrec_artifact_paths()
            loaded = await asyncio.to_thread(
                self.sasrec_engine.load_artifacts,
                checkpoint_path,
                vocab_path,
                metadata_path,
            )
            if loaded:
                if checkpoint_record:
                    self.sasrec_engine.model_version = checkpoint_record.model_version
                self.loaded_sasrec_version = self.sasrec_engine.model_version
                logger.info("Loaded SASRec serving artifacts")
                return True
        except Exception as e:
            logger.warning(f"Could not load SASRec artifacts: {e}")
        return False

    async def _update_models_from_interactions(self):
        """Update models using recent interaction data."""
        try:
            if not self.artifact_manager or not self.artifact_manager.system_store:
                await self.refresh_serving_pools()
                logger.warning("Skipping Two-Tower retraining because Postgres system store is unavailable")
                return

            interactions = await self.artifact_manager.system_store.get_training_interactions(limit=50000)

            if interactions:
                # Gather user features for the trainer
                user_features_map = await self.feature_store.get_all_user_features_map()

                # Train Two-Tower model
                await self.cf_engine.train_model(
                    interactions, user_features_map=user_features_map
                )
                if self.cf_engine.is_trained:
                    checkpoint_path = self.artifact_manager.two_tower_local_checkpoint_path
                    index_path = self.artifact_manager.two_tower_local_index_path
                    metadata_path = self.artifact_manager.two_tower_local_metadata_path
                    model_version = self.cf_engine.model_version or f"two-tower-{int(time.time())}"
                    artifact_record = await self.artifact_manager.persist_two_tower_artifacts(
                        checkpoint_path=checkpoint_path,
                        index_path=index_path,
                        metadata_path=metadata_path,
                        model_version=model_version,
                        payload={
                            "sample_count": len(interactions),
                            "last_training_time": self.cf_engine.last_training_time,
                        },
                    )
                    self.loaded_two_tower_version = artifact_record.model_version if artifact_record else model_version

                await self._update_sasrec_from_sequences()

                # Update trending scores (use last 1K for recency)
                await self.trending_engine.update_trending_scores(interactions[:1000])
                await self.refresh_serving_pools()

                self.last_model_update = time.time()
                logger.info(f"Updated models with {len(interactions)} interactions")
            else:
                await self.refresh_serving_pools()
                logger.warning("No interactions found for model training")

        except Exception as e:
            logger.error(f"Error updating models from interactions: {e}")

    async def _update_sasrec_from_sequences(self) -> None:
        if not self.config.enable_sasrec:
            return
        if not self.artifact_manager or not self.artifact_manager.system_store:
            logger.warning("Skipping SASRec training because Postgres system store is unavailable")
            return

        max_events = max(
            int(self.config.sasrec_min_sequence_length),
            int(self.config.sasrec_max_sequence_length) + 1,
        )
        sequences = await self.artifact_manager.system_store.get_user_training_sequences(
            max_events_per_user=max_events,
            min_sequence_length=self.config.sasrec_min_sequence_length,
        )
        if not sequences:
            logger.info("Skipping SASRec training because no positive user sequences are available")
            return

        trained = await self.sasrec_engine.train_model(
            sequences,
            catalog_product_ids=self.vector_search.product_metadata.keys(),
        )
        if not trained or not self.sasrec_engine.is_trained:
            return

        checkpoint_path, vocab_path, metadata_path = self._sasrec_artifact_paths()
        await asyncio.to_thread(
            self.sasrec_engine.save_artifacts,
            checkpoint_path,
            vocab_path,
            metadata_path,
        )
        model_version = self.sasrec_engine.model_version or f"sasrec-{int(time.time())}"
        artifact_record = await self.artifact_manager.persist_sasrec_artifacts(
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            metadata_path=metadata_path,
            model_version=model_version,
            payload={
                "sequence_count": len(sequences),
                "training_sample_count": self.sasrec_engine.training_sample_count,
                "last_training_time": self.sasrec_engine.last_training_time,
            },
        )
        self.loaded_sasrec_version = artifact_record.model_version if artifact_record else model_version

    def _sasrec_artifact_paths(self) -> Tuple[str, str, str]:
        if self.artifact_manager:
            return (
                self.artifact_manager.sasrec_local_checkpoint_path,
                self.artifact_manager.sasrec_local_vocab_path,
                self.artifact_manager.sasrec_local_metadata_path,
            )
        base_dir = Path(self.config.cf_index_path).parent
        return (
            self.config.sasrec_checkpoint_path or str(base_dir / "sasrec_model.pt"),
            self.config.sasrec_vocab_path or str(base_dir / "sasrec_vocab.json"),
            self.config.sasrec_metadata_path or str(base_dir / "sasrec_metadata.json"),
        )

    async def sync_serving_artifacts_if_updated(self) -> bool:
        """Reload retrieval serving artifacts when newer checkpoints are available."""
        if not self.artifact_manager:
            return False

        updated = False
        latest_two_tower = await self.artifact_manager.get_latest_model_checkpoint(
            ModelArtifactManager.TWO_TOWER_MODEL_NAME
        )
        if latest_two_tower and latest_two_tower.model_version != self.loaded_two_tower_version:
            await self._try_load_cf_index()
            updated = updated or self.loaded_two_tower_version == latest_two_tower.model_version

        if self.config.enable_sasrec:
            latest_sasrec = await self.artifact_manager.get_latest_model_checkpoint(
                ModelArtifactManager.SASREC_MODEL_NAME
            )
            if latest_sasrec and latest_sasrec.model_version != self.loaded_sasrec_version:
                await self._try_load_sasrec_artifacts()
                updated = updated or self.loaded_sasrec_version == latest_sasrec.model_version

        return updated

    def _compute_serving_pool_score(
        self,
        product_id: str,
        metadata: Dict[str, Any],
        current_time: Optional[float] = None,
    ) -> float:
        """Compute a stable heuristic score for serving pools."""
        current_time = current_time or time.time()
        trending_score = float(self.trending_engine.trending_scores.get(product_id, 0.0))
        rating_score = float(metadata.get("rating", 0.0)) / 5.0
        review_score = np.log1p(float(metadata.get("num_reviews", 0.0))) / 5.0
        age_days = max((current_time - float(metadata.get("created_at", current_time))) / 86400.0, 0.0)
        freshness_score = 1.0 / (1.0 + age_days / 30.0)
        return trending_score + rating_score * 0.6 + review_score * 0.2 + freshness_score * 0.2

    async def refresh_serving_pools(self):
        """Precompute global trending and per-category serving pools."""
        try:
            if not self.vector_search.product_metadata:
                return

            current_time = time.time()
            global_scored: List[Tuple[str, float]] = []
            category_scored: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

            for product_id, metadata in self.vector_search.product_metadata.items():
                score = self._compute_serving_pool_score(product_id, metadata, current_time)
                global_scored.append((product_id, score))
                category = metadata.get("category", "unknown")
                category_scored[category].append((product_id, score))

            global_scored.sort(key=lambda item: item[1], reverse=True)
            global_top = global_scored[: self.config.serving_trending_pool_size]
            global_max = max((score for _, score in global_top), default=1.0) or 1.0
            trending_pool = [
                CandidateProduct(
                    product_id=product_id,
                    popularity_score=float(score / global_max),
                    combined_score=float(score / global_max),
                    source="trending_pool",
                )
                for product_id, score in global_top
            ]
            await self.feature_store.store_trending_pool(trending_pool)

            category_pools: Dict[str, List[CandidateProduct]] = {}
            for category, scored_items in category_scored.items():
                scored_items.sort(key=lambda item: item[1], reverse=True)
                category_top = scored_items[: self.config.serving_category_pool_size]
                category_max = max((score for _, score in category_top), default=1.0) or 1.0
                category_pools[category] = [
                    CandidateProduct(
                        product_id=product_id,
                        popularity_score=float(score / category_max),
                        combined_score=float(score / category_max),
                        source="category_pool",
                    )
                    for product_id, score in category_top
                ]

            await self.feature_store.store_category_pools(category_pools)
            logger.info(
                "Refreshed serving pools: %s global products, %s categories",
                len(trending_pool),
                len(category_pools),
            )
        except Exception as e:
            logger.error(f"Error refreshing serving pools: {e}")

    def _merge_candidate(
        self,
        all_candidates: Dict[str, CandidateProduct],
        candidate: CandidateProduct,
    ) -> bool:
        """Merge candidate scores without expanding the candidate set excessively."""
        existing = all_candidates.get(candidate.product_id)
        if existing is None:
            all_candidates[candidate.product_id] = candidate
            return False

        if candidate.collaborative_score is not None:
            existing.collaborative_score = max(
                existing.collaborative_score or 0.0,
                candidate.collaborative_score,
            )
        if candidate.content_similarity_score is not None:
            existing.content_similarity_score = max(
                existing.content_similarity_score or 0.0,
                candidate.content_similarity_score,
            )
        if candidate.popularity_score is not None:
            existing.popularity_score = max(
                existing.popularity_score or 0.0,
                candidate.popularity_score,
            )
        if candidate.source not in existing.source.split("+"):
            existing.source = f"{existing.source}+{candidate.source}"
        return True

    def _build_sasrec_sequence_from_interactions(
        self,
        user_interactions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalize Redis newest-first interactions into SASRec chronological positives."""
        sequence: List[Dict[str, Any]] = []
        for interaction in reversed(user_interactions):
            product_id = interaction.get("product_id")
            action = interaction.get("action")
            if not product_id or action not in POSITIVE_SEQUENCE_ACTIONS:
                continue
            sequence.append(
                {
                    "user_id": interaction.get("user_id"),
                    "product_id": product_id,
                    "action": action,
                    "timestamp": interaction.get("timestamp"),
                    "occurred_at": interaction.get("occurred_at", interaction.get("timestamp")),
                    "event_id": interaction.get("event_id"),
                    "schema_version": interaction.get("schema_version", 1),
                    "context": interaction.get("context") or {},
                }
            )
        return sequence[-self.config.sasrec_max_sequence_length :]

    @staticmethod
    def _count_source_tokens(candidates: List[CandidateProduct]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for candidate in candidates:
            for source in str(candidate.source or "unknown").split("+"):
                source = source or "unknown"
                counts[source] = counts.get(source, 0) + 1
        return counts

    def _resolve_preferred_categories(
        self,
        user_features: Optional[UserFeatures],
        context: Dict[str, Any],
    ) -> List[str]:
        """Resolve the small set of categories to pull from precomputed pools."""
        categories: List[str] = []
        for key in ("product_category", "category"):
            value = context.get(key)
            if isinstance(value, str) and value:
                categories.append(value)

        if user_features:
            categories.extend(user_features.preferred_categories)

        deduped: List[str] = []
        for category in categories:
            if category and category not in deduped:
                deduped.append(category)
        return deduped[: self.config.preferred_category_pool_count]

    async def generate_candidates(
        self,
        user_id: str,
        content_features: Optional[ContentFeatures] = None,
        context: Optional[Dict[str, Any]] = None,
        k_per_source: int = 100,
        include_profile: bool = False,
        user_features: Optional[UserFeatures] = None,
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
            started_at = time.perf_counter()
            profile = {
                "user_interactions_ms": 0.0,
                "user_features_ms": 0.0,
                "cf_candidates_ms": 0.0,
                "sasrec_candidates_ms": 0.0,
                "content_candidates_ms": 0.0,
                "trending_candidates_ms": 0.0,
                "category_pool_ms": 0.0,
                "random_candidates_ms": 0.0,
                "score_merge_ms": 0.0,
                "total_ms": 0.0,
                "candidate_count": 0,
                "preferred_categories": [],
                "source_counts": {},
                "source_overlap_counts": {},
                "merged_source_counts": {},
                "sasrec_sequence_length": 0,
            }

            async def timed_stage(
                metric_name: str,
                awaitable,
                fallback,
                timeout_ms: Optional[float],
            ):
                stage_started = time.perf_counter()
                try:
                    if timeout_ms and timeout_ms > 0:
                        return await asyncio.wait_for(awaitable, timeout=timeout_ms / 1000.0)
                    return await awaitable
                except asyncio.TimeoutError:
                    logger.warning("%s_timed_out", metric_name)
                    return fallback
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("%s_failed: %s", metric_name, exc)
                    return fallback
                finally:
                    profile[metric_name] = round((time.perf_counter() - stage_started) * 1000, 2)

            interaction_task = asyncio.create_task(
                timed_stage(
                    "user_interactions_ms",
                    self.feature_store.get_user_interactions(user_id, limit=200),
                    [],
                    self.config.interaction_history_timeout_ms,
                ),
                name="candidate-user-interactions",
            )
            if user_features is not None:
                user_features_obj = user_features
            else:
                user_features_obj = await timed_stage(
                    "user_features_ms",
                    self.feature_store.get_user_features(user_id, cache_default=False),
                    UserFeatures(user_id=user_id),
                    self.feature_store.cache_config.hot_path_read_timeout_ms,
                )
            if user_features is not None:
                profile["user_features_ms"] = 0.0

            user_interactions = await interaction_task
            exclude_items = {
                interaction.get("product_id")
                for interaction in user_interactions
                if interaction.get("product_id")
            }
            sasrec_sequence = self._build_sasrec_sequence_from_interactions(user_interactions)
            profile["sasrec_sequence_length"] = len(sasrec_sequence)
            user_features_dict = user_features_obj.dict() if user_features_obj else {}
            preferred_categories = self._resolve_preferred_categories(user_features_obj, context)
            profile["preferred_categories"] = preferred_categories
            target_candidates = min(
                max(k_per_source, self.config.candidates_per_source),
                self.config.max_total_candidates,
            )

            async def fetch_cf_candidates() -> List[CandidateProduct]:
                if not self.cf_engine.is_trained:
                    return []
                return await self.cf_engine.get_user_recommendations(
                    user_id,
                    min(target_candidates, self.config.max_live_cf_candidates),
                    exclude_items,
                    user_features=user_features_dict,
                )

            async def fetch_sasrec_candidates() -> List[CandidateProduct]:
                if not self.config.enable_sasrec or not self.sasrec_engine.is_trained:
                    return []
                return await self.sasrec_engine.get_candidates(
                    sasrec_sequence,
                    k=min(target_candidates, self.config.max_live_sasrec_candidates),
                    exclude_items=exclude_items,
                    catalog_product_ids=self.vector_search.product_metadata.keys(),
                )

            async def fetch_content_candidates() -> List[CandidateProduct]:
                if not content_features or not content_features.visual_embedding:
                    return []
                candidates = await self.vector_search.search_similar_products(
                    np.array(content_features.visual_embedding),
                    k=min(target_candidates, self.config.max_live_content_candidates),
                )
                return [
                    candidate
                    for candidate in candidates
                    if candidate.product_id not in exclude_items
                ]

            async def fetch_trending_candidates() -> List[CandidateProduct]:
                return await self.feature_store.get_trending_pool(
                    min(target_candidates, self.config.max_pool_trending_candidates),
                    exclude_items=exclude_items,
                )

            async def fetch_category_candidates() -> List[CandidateProduct]:
                if not preferred_categories:
                    return []
                per_category_limit = max(
                    1,
                    self.config.max_pool_category_candidates // max(len(preferred_categories), 1),
                )
                category_tasks = [
                    self.feature_store.get_category_pool(
                        category,
                        per_category_limit,
                        exclude_items=exclude_items,
                    )
                    for category in preferred_categories
                ]
                category_results = await asyncio.gather(*category_tasks, return_exceptions=True)
                category_candidates: List[CandidateProduct] = []
                for result in category_results:
                    if isinstance(result, Exception):
                        logger.warning("category_pool_fetch_failed: %s", result)
                        continue
                    category_candidates.extend(result)
                return category_candidates

            source_timeout_ms = self.config.candidate_source_timeout_ms
            cf_task = asyncio.create_task(
                timed_stage("cf_candidates_ms", fetch_cf_candidates(), [], source_timeout_ms),
                name="candidate-source-cf",
            )
            sasrec_task = asyncio.create_task(
                timed_stage("sasrec_candidates_ms", fetch_sasrec_candidates(), [], source_timeout_ms),
                name="candidate-source-sasrec",
            )
            content_task = asyncio.create_task(
                timed_stage("content_candidates_ms", fetch_content_candidates(), [], source_timeout_ms),
                name="candidate-source-content",
            )
            trending_task = asyncio.create_task(
                timed_stage("trending_candidates_ms", fetch_trending_candidates(), [], source_timeout_ms),
                name="candidate-source-trending",
            )
            category_task = asyncio.create_task(
                timed_stage("category_pool_ms", fetch_category_candidates(), [], source_timeout_ms),
                name="candidate-source-category",
            )

            cf_candidates, sasrec_candidates, content_candidates, trending_candidates, category_candidates = (
                await asyncio.gather(cf_task, sasrec_task, content_task, trending_task, category_task)
            )

            def merge_source(source_name: str, candidates: List[CandidateProduct]) -> None:
                overlaps = 0
                for candidate in candidates:
                    if self._merge_candidate(all_candidates, candidate):
                        overlaps += 1
                profile["source_counts"][source_name] = len(candidates)
                profile["source_overlap_counts"][source_name] = overlaps

            merge_source("cf", cf_candidates)
            merge_source("sasrec", sasrec_candidates)
            merge_source("content", content_candidates)
            merge_source("trending_pool", trending_candidates)
            merge_source("category_pool", category_candidates)

            # 5. Small random fallback only if stronger sources are still thin
            stage_started = time.perf_counter()
            if len(all_candidates) < target_candidates:
                random_target = min(
                    target_candidates - len(all_candidates),
                    self.config.max_random_candidates,
                )
                if not all_candidates:
                    random_target = min(
                        random_target,
                        self.config.cold_start_random_candidate_cap,
                    )
                random_candidates = []
                if random_target > 0:
                    random_candidates = await timed_stage(
                        "random_candidates_ms",
                        self.vector_search.get_random_products(k=random_target),
                        [],
                        source_timeout_ms,
                    )
                for candidate in random_candidates:
                    if (
                        candidate.product_id not in exclude_items
                        and candidate.product_id not in all_candidates
                    ):
                        all_candidates[candidate.product_id] = candidate
                    elif candidate.product_id in all_candidates:
                        profile["source_overlap_counts"]["random"] = (
                            profile["source_overlap_counts"].get("random", 0) + 1
                        )
                profile["source_counts"]["random"] = len(random_candidates)
                profile["source_overlap_counts"].setdefault("random", 0)
            elif profile["random_candidates_ms"] == 0.0:
                profile["random_candidates_ms"] = round(
                    (time.perf_counter() - stage_started) * 1000,
                    2,
                )

            # Combine scores for final ranking
            stage_started = time.perf_counter()
            profile["merged_source_counts"] = self._count_source_tokens(list(all_candidates.values()))
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
            profile["score_merge_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
            profile["candidate_count"] = len(final_candidates)
            profile["total_ms"] = round((time.perf_counter() - started_at) * 1000, 2)

            logger.debug(f"Generated {len(final_candidates)} candidates from multiple sources")
            if include_profile:
                return final_candidates, profile
            return final_candidates

        except Exception as e:
            logger.error(f"Error generating candidates for {user_id}: {e}")
            if include_profile:
                return [], {
                    "user_interactions_ms": 0.0,
                    "user_features_ms": 0.0,
                    "cf_candidates_ms": 0.0,
                    "sasrec_candidates_ms": 0.0,
                    "content_candidates_ms": 0.0,
                    "trending_candidates_ms": 0.0,
                    "category_pool_ms": 0.0,
                    "random_candidates_ms": 0.0,
                    "score_merge_ms": 0.0,
                    "total_ms": 0.0,
                    "candidate_count": 0,
                    "preferred_categories": [],
                    "source_counts": {},
                    "source_overlap_counts": {},
                    "merged_source_counts": {},
                    "sasrec_sequence_length": 0,
                    "error": str(e),
                }
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
                metadata = self.vector_search.product_metadata.get(candidate.product_id)
                if metadata and "category" in metadata:
                    category_groups[metadata["category"]].append(candidate)
                else:
                    ungrouped_candidates.append(candidate)

            diverse_candidates: List[CandidateProduct] = []
            max_per_category = self.config.max_items_per_category
            for group in category_groups.values():
                group.sort(key=lambda x: x.combined_score, reverse=True)
            ungrouped_candidates.sort(key=lambda x: x.combined_score, reverse=True)
            category_counts: Dict[str, int] = defaultdict(int)

            while len(diverse_candidates) < len(candidates) and (
                category_groups or ungrouped_candidates
            ):
                added_in_round = 0
                for category in list(category_groups.keys()):
                    if category_groups[category]:
                        c = category_groups[category].pop(0)
                        diverse_candidates.append(c)
                        added_in_round += 1
                        category_counts[category] += 1
                        if category_counts[category] >= max_per_category:
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
        metadata = self.vector_search.product_metadata.get(candidate.product_id, {})
        return metadata.get("category", "unknown")

    async def get_trending_recommendations(self, k: int = 10) -> List[Dict[str, Any]]:
        """Get trending recommendations for fallback scenarios."""
        try:
            trending_candidates = await self.feature_store.get_trending_pool(k)
            recommendations = []
            for candidate in trending_candidates:
                metadata = self.vector_search.product_metadata.get(candidate.product_id)
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
        sasrec_params = 0
        if self.sasrec_engine.model is not None:
            sasrec_params = sum(p.numel() for p in self.sasrec_engine.model.parameters())

        return {
            "is_initialized": self.is_initialized,
            "last_model_update": self.last_model_update,
            "cf_trained": self.cf_engine.is_trained,
            "cf_model_version": self.cf_engine.model_version,
            "cf_users": len(self.cf_engine.user_mapping),
            "cf_items": len(self.cf_engine.item_mapping),
            "cf_embedding_dim": self.config.tt_embedding_dim,
            "cf_index_size": self.cf_engine.cf_index.ntotal if self.cf_engine.cf_index else 0,
            "cf_model_parameters": model_params,
            "sasrec_enabled": self.config.enable_sasrec,
            "sasrec_trained": self.sasrec_engine.is_trained,
            "sasrec_model_version": self.sasrec_engine.model_version,
            "sasrec_items": len(self.sasrec_engine.product_to_id),
            "sasrec_model_parameters": sasrec_params,
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
