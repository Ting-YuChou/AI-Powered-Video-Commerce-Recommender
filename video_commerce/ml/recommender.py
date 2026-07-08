"""
AI-Powered Video Commerce Recommender - Main Recommendation Engine
==================================================================

This module implements the core recommendation logic that combines multiple
recommendation sources: Two-Tower collaborative filtering with ANN retrieval,
content-based matching, and trending/popularity algorithms to generate diverse
candidate products.
"""

import asyncio
import functools
import numpy as np
import faiss
import logging
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Set, Tuple
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
import threading

# Local imports
from video_commerce.common.models import UserFeatures, ContentFeatures, CandidateProduct, InteractionType
from video_commerce.data_plane.feature_store import FeatureStore
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.ml.vector_search import VectorSearchEngine
from video_commerce.common.config import RecommendationConfig
from video_commerce.ml.two_tower import TwoTowerTrainer
from video_commerce.ml.sasrec import SASRecCandidateEngine
from video_commerce.ml.swing_itemcf import (
    SwingItemCFCandidateEngine,
    SwingItemCFTrainer,
)
from video_commerce.ml.cf_cold_start import (
    ContentToCFAdapter,
    build_content_feature,
    build_hybrid_synthetic_embedding,
    is_cold_start_eligible,
    load_item_embedding_sidecar,
    normalize_vector,
    save_item_embedding_sidecar,
)
from video_commerce.ml.content_clusters import (
    build_content_cluster_artifact,
    save_content_cluster_artifact,
)

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
        self._base_cf_index: Optional[faiss.Index] = None
        self._base_cf_index_map: Dict[int, str] = {}

        # Backward-compatible attributes consumed by RecommendationEngine.get_stats()
        self.user_mapping: Dict[str, int] = {}
        self.item_mapping: Dict[str, int] = {}

        self.is_trained = False
        self.last_training_time: float = 0
        self.model_version: Optional[str] = None
        self._item_popularity: Dict[str, float] = {}
        self.trained_item_embeddings: Dict[str, np.ndarray] = {}
        self.trained_item_clip_available: Dict[str, bool] = {}
        self.trained_item_features: Dict[str, np.ndarray] = {}
        self.content_to_cf_adapter: Optional[ContentToCFAdapter] = None
        self.synthetic_item_embeddings: Dict[str, np.ndarray] = {}
        self.synthetic_item_metadata: Dict[str, Dict[str, Any]] = {}
        self.cold_start_overlay_version: Optional[str] = None
        self._new_item_candidates: List[CandidateProduct] = []
        self.new_item_pool_version: Optional[str] = None
        self.new_item_pool_refreshed_at: float = 0.0
        self._new_item_pool_catalog_token: Optional[str] = None
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
        self._unknown_user_candidate_cache: OrderedDict[
            Tuple[Any, ...],
            List[CandidateProduct],
        ] = OrderedDict()
        self._user_embedding_cache_lock = threading.RLock()
        retrieval_workers = max(
            1,
            int(getattr(config, "retrieval_executor_workers", 1) or 1),
        )
        self._retrieval_executor = ThreadPoolExecutor(
            max_workers=retrieval_workers,
            thread_name_prefix="two-tower-retrieval",
        )

        logger.info("Two-Tower retrieval engine initialized")

    async def run_in_retrieval_executor(self, func, *args, **kwargs):
        """Run synchronous retrieval CPU work on the bounded retrieval executor."""
        loop = asyncio.get_running_loop()
        if kwargs:
            return await loop.run_in_executor(
                self._retrieval_executor,
                functools.partial(func, *args, **kwargs),
            )
        return await loop.run_in_executor(self._retrieval_executor, func, *args)

    def prime_default_unknown_candidates(self, k: int) -> None:
        """Warm the shared cold-user CF candidate cache for common live requests."""
        if k <= 0 or not self.is_trained:
            return
        self._get_user_recommendations_sync(
            "__default_unknown_user__",
            k,
            set(),
            {
                "total_interactions": 0,
                "avg_session_length": 0.0,
                "preferred_categories": [],
                "price_sensitivity": 0.5,
                "click_through_rate": 0.0,
                "conversion_rate": 0.0,
            },
        )

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
            existing_index = self._base_cf_index or self.cf_index
            checkpoint_path = self.config.cf_index_path.replace(".faiss", ".pt")
            if Path(checkpoint_path).exists():
                await asyncio.to_thread(
                    self.trainer.warm_start_from_checkpoint, checkpoint_path
                )

            # Run training
            start_time = time.time()
            stats = await asyncio.to_thread(
                self.trainer.train,
                existing_cf_index=existing_index,
            )
            training_time = time.time() - start_time

            # Build FAISS index from trained item embeddings
            self.cf_index, idx_map = await asyncio.to_thread(
                self.trainer.build_item_index
            )

            # Build reverse mapping: faiss_idx -> product_id
            self.cf_index_map = {}
            for faiss_idx, item_idx in idx_map.items():
                product_id = self.trainer.reverse_item_mapping.get(item_idx)
                if product_id:
                    self.cf_index_map[faiss_idx] = product_id
            self._base_cf_index = self.cf_index
            self._base_cf_index_map = dict(self.cf_index_map)

            # Sync backward-compat attributes
            self.user_mapping = dict(self.trainer.user_mapping)
            self.item_mapping = dict(self.trainer.item_mapping)
            self.trained_item_embeddings = self.trainer.get_item_embedding_map()
            self.trained_item_clip_available = (
                self.trainer.get_item_clip_available_map()
            )
            self.trained_item_features = self.trainer.get_item_side_feature_map()

            # Save checkpoint and index
            await asyncio.to_thread(self.trainer.save_checkpoint, checkpoint_path)
            model_version = f"two-tower-{int(time.time())}"
            self.model_version = model_version
            await asyncio.to_thread(
                self._save_cold_start_sidecars,
                model_version,
                product_clip_embeddings,
                product_metadata,
            )
            await asyncio.to_thread(
                VectorSearchEngine.save_cf_index,
                self.cf_index,
                self.config.cf_index_path,
                {
                    "num_items": len(self.item_mapping),
                    "embedding_dim": self.config.tt_embedding_dim,
                    "index_map": {str(k): v for k, v in self.cf_index_map.items()},
                    "cf_embedding_sidecar_path": self._embedding_sidecar_path(),
                    "cf_adapter_path": self._adapter_path(),
                },
            )

            self.is_trained = True
            self.last_training_time = time.time()
            await self.refresh_cold_start_overlay()
            await asyncio.to_thread(self.refresh_new_item_candidates)
            self.clear_user_embedding_cache()

            logger.info(
                f"Two-Tower model trained in {training_time:.1f}s — "
                f"{len(self.user_mapping)} users, {len(self.item_mapping)} items, "
                f"index size={self.cf_index.ntotal}"
            )

        except Exception as e:
            logger.error(f"Error training Two-Tower model: {e}")
            self.is_trained = False

    def _embedding_sidecar_path(self) -> str:
        return str(Path(self.config.cf_index_path).with_suffix(".cf_embeddings.npz"))

    def _adapter_path(self) -> str:
        return str(Path(self.config.cf_index_path).with_suffix(".cf_adapter.npz"))

    def _save_cold_start_sidecars(
        self,
        model_version: str,
        product_clip_embeddings: Dict[str, np.ndarray],
        product_metadata: Dict[str, Dict[str, Any]],
    ) -> None:
        self.content_to_cf_adapter = None
        self._remove_cold_start_artifact(self._adapter_path())
        if not self.trained_item_embeddings:
            self._remove_cold_start_artifact(self._embedding_sidecar_path())
            return

        save_item_embedding_sidecar(
            self._embedding_sidecar_path(),
            embedding_map=self.trained_item_embeddings,
            clip_available=self.trained_item_clip_available,
            item_features=self.trained_item_features,
            model_version=model_version,
        )
        adapter = self._fit_content_to_cf_adapter(
            product_clip_embeddings, product_metadata
        )
        if adapter is not None:
            adapter.save(
                self._adapter_path(),
                metadata={
                    "two_tower_model_version": model_version,
                    "trained_item_count": len(self.trained_item_embeddings),
                },
            )
            self.content_to_cf_adapter = adapter

    @staticmethod
    def _remove_cold_start_artifact(path: str) -> None:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "Failed to remove stale CF cold-start artifact %s: %s", path, exc
            )

    def _fit_content_to_cf_adapter(
        self,
        product_clip_embeddings: Dict[str, np.ndarray],
        product_metadata: Dict[str, Dict[str, Any]],
    ) -> Optional[ContentToCFAdapter]:
        features: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        for product_id, embedding in self.trained_item_embeddings.items():
            clip_embedding = product_clip_embeddings.get(product_id)
            if clip_embedding is None:
                continue
            try:
                features.append(
                    build_content_feature(
                        clip_embedding,
                        product_metadata.get(product_id, {}),
                        clip_dim=self.vector_search.embedding_dim,
                    )
                )
                targets.append(normalize_vector(embedding))
            except ValueError:
                continue

        if len(features) < 2:
            logger.warning(
                "Skipping content-to-CF adapter fit; not enough CLIP-backed items"
            )
            return None
        try:
            return ContentToCFAdapter.fit(np.vstack(features), np.vstack(targets))
        except Exception as exc:
            logger.warning("Skipping content-to-CF adapter fit: %s", exc)
            return None

    def _clear_loaded_cold_start_sidecars(self) -> None:
        self.trained_item_embeddings = {}
        self.trained_item_clip_available = {}
        self.trained_item_features = {}
        self.content_to_cf_adapter = None
        self.synthetic_item_embeddings = {}
        self.synthetic_item_metadata = {}
        self.cold_start_overlay_version = None
        if self._base_cf_index is not None:
            self.cf_index = self._base_cf_index
            self.cf_index_map = dict(self._base_cf_index_map)

    def _load_cold_start_sidecars(self) -> None:
        (
            embeddings,
            clip_available,
            item_features,
            sidecar_version,
        ) = load_item_embedding_sidecar(self._embedding_sidecar_path())
        expected_model_version = self.model_version
        if not embeddings:
            self._clear_loaded_cold_start_sidecars()
            return
        if expected_model_version and sidecar_version != expected_model_version:
            logger.warning(
                "Skipping CF cold-start sidecars because embedding sidecar version %s "
                "does not match checkpoint version %s",
                sidecar_version,
                expected_model_version,
            )
            self._clear_loaded_cold_start_sidecars()
            return
        if sidecar_version and not self.model_version:
            self.model_version = sidecar_version

        adapter = ContentToCFAdapter.load(self._adapter_path())
        adapter_model_version = (
            adapter.metadata.get("two_tower_model_version")
            if adapter is not None
            else None
        )
        expected_model_version = self.model_version
        if adapter is None or adapter_model_version != expected_model_version:
            logger.warning(
                "Skipping CF cold-start sidecars because adapter version %s "
                "does not match checkpoint version %s",
                adapter_model_version,
                expected_model_version,
            )
            self._clear_loaded_cold_start_sidecars()
            return

        self.trained_item_embeddings = embeddings
        self.trained_item_clip_available = clip_available
        self.trained_item_features = item_features
        self.content_to_cf_adapter = adapter

    async def refresh_cold_start_overlay(self) -> None:
        if not self.config.enable_cf_cold_start_bootstrap:
            self.synthetic_item_embeddings = {}
            self.synthetic_item_metadata = {}
            self.cold_start_overlay_version = None
            if self._base_cf_index is not None:
                self.cf_index = self._base_cf_index
                self.cf_index_map = dict(self._base_cf_index_map)
            return

        if not self.trained_item_embeddings or self.content_to_cf_adapter is None:
            logger.info(
                "Skipping CF cold-start overlay; trained embeddings or adapter unavailable"
            )
            self.synthetic_item_embeddings = {}
            self.synthetic_item_metadata = {}
            self.cold_start_overlay_version = None
            if self._base_cf_index is not None:
                self.cf_index = self._base_cf_index
                self.cf_index_map = dict(self._base_cf_index_map)
            return

        (
            synthetic_embeddings,
            synthetic_metadata,
            index,
            index_map,
            overlay_version,
        ) = await asyncio.to_thread(self._build_cold_start_overlay_snapshot)
        self.synthetic_item_embeddings = synthetic_embeddings
        self.synthetic_item_metadata = synthetic_metadata
        if index is not None:
            self.cf_index = index
            self.cf_index_map = index_map
        self.cold_start_overlay_version = overlay_version

    def _build_cold_start_overlay_snapshot(
        self,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, Dict[str, Any]],
        Optional[faiss.Index],
        Dict[int, str],
        str,
    ]:
        (
            synthetic_embeddings,
            synthetic_metadata,
        ) = self._build_synthetic_item_embeddings()
        index, index_map = self._build_serving_cf_index_snapshot(synthetic_embeddings)
        overlay_version = self._build_cold_start_overlay_version(synthetic_embeddings)
        return (
            synthetic_embeddings,
            synthetic_metadata,
            index,
            index_map,
            overlay_version,
        )

    def _build_synthetic_item_embeddings(
        self,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        current_time = time.time()
        trained_item_ids = set(self.item_mapping.keys())
        synthetic_embeddings: Dict[str, np.ndarray] = {}
        synthetic_metadata: Dict[str, Dict[str, Any]] = {}
        eligible_product_ids = self._eligible_cold_start_product_ids(current_time)

        for product_id in eligible_product_ids:
            if len(synthetic_embeddings) >= self.config.cf_cold_start_max_items:
                break
            metadata = self.vector_search.product_metadata.get(product_id, {})
            clip_embedding = self._get_product_clip_embedding(product_id)
            if not is_cold_start_eligible(
                product_id=product_id,
                metadata=metadata,
                clip_embedding=clip_embedding,
                trained_item_ids=trained_item_ids,
                current_time=current_time,
                max_age_days=self.config.cf_cold_start_max_age_days,
                max_interactions=self.config.cf_cold_start_max_interactions,
            ):
                continue

            synthetic = self._build_one_synthetic_embedding(
                product_id,
                clip_embedding,
                metadata,
            )
            if synthetic is None:
                continue
            embedding, metadata_payload = synthetic
            synthetic_embeddings[product_id] = embedding
            synthetic_metadata[product_id] = metadata_payload

        return synthetic_embeddings, synthetic_metadata

    def _eligible_cold_start_product_ids(self, current_time: float) -> List[str]:
        if not hasattr(self.vector_search, "get_product_embedding"):
            return []
        trained_item_ids = set(self.item_mapping.keys())
        candidates: List[Tuple[str, float]] = []
        for product_id, metadata in self.vector_search.product_metadata.items():
            clip_embedding = self._get_product_clip_embedding(product_id)
            if not is_cold_start_eligible(
                product_id=product_id,
                metadata=metadata,
                clip_embedding=clip_embedding,
                trained_item_ids=trained_item_ids,
                current_time=current_time,
                max_age_days=self.config.cf_cold_start_max_age_days,
                max_interactions=self.config.cf_cold_start_max_interactions,
            ):
                continue
            created_at = float(metadata.get("created_at", current_time))
            candidates.append((product_id, created_at))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return [
            product_id
            for product_id, _ in candidates[: self.config.cf_cold_start_max_items]
        ]

    def _get_product_clip_embedding(self, product_id: str) -> Optional[np.ndarray]:
        if not hasattr(self.vector_search, "get_product_embedding"):
            return None
        return self.vector_search.get_product_embedding(product_id)

    def _build_one_synthetic_embedding(
        self,
        product_id: str,
        clip_embedding: np.ndarray,
        metadata: Dict[str, Any],
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        assert self.content_to_cf_adapter is not None
        neighbors = self._get_clip_neighbors(
            product_id,
            clip_embedding,
            max(
                self.config.cf_cold_start_neighbors * 2,
                self.config.cf_cold_start_neighbors,
            ),
        )
        neighbor_embeddings: List[np.ndarray] = []
        neighbor_similarities: List[float] = []
        neighbor_metadatas: List[Dict[str, Any]] = []
        neighbor_ids: List[str] = []

        for neighbor in neighbors:
            if neighbor.product_id == product_id:
                continue
            trained_embedding = self.trained_item_embeddings.get(neighbor.product_id)
            if trained_embedding is None:
                continue
            similarity = float(neighbor.content_similarity_score or 0.0)
            if similarity < self.config.cf_cold_start_min_clip_similarity:
                continue
            neighbor_embeddings.append(trained_embedding)
            neighbor_similarities.append(similarity)
            neighbor_metadatas.append(
                self.vector_search.product_metadata.get(neighbor.product_id, {})
            )
            neighbor_ids.append(neighbor.product_id)
            if len(neighbor_embeddings) >= self.config.cf_cold_start_neighbors:
                break

        if len(neighbor_embeddings) < self.config.cf_cold_start_min_valid_neighbors:
            return None
        if (
            float(np.mean(neighbor_similarities))
            < self.config.cf_cold_start_min_clip_similarity
        ):
            return None

        try:
            content_feature = build_content_feature(
                clip_embedding,
                metadata,
                clip_dim=self.vector_search.embedding_dim,
            )
            adapter_embedding = self.content_to_cf_adapter.predict(content_feature)[0]
            (
                synthetic_embedding,
                confidence,
                neighbor_weights,
            ) = build_hybrid_synthetic_embedding(
                neighbor_embeddings=neighbor_embeddings,
                neighbor_similarities=neighbor_similarities,
                adapter_embedding=adapter_embedding,
                query_metadata=metadata,
                neighbor_metadatas=neighbor_metadatas,
                neighbor_weight=self.config.cf_cold_start_neighbor_weight,
                softmax_temperature=self.config.cf_cold_start_softmax_temperature,
                configured_neighbor_count=self.config.cf_cold_start_neighbors,
            )
        except Exception as exc:
            logger.warning("synthetic_cf_embedding_failed: %s", exc)
            return None

        metadata_payload = {
            "source": "synthetic_clip_neighbor_adapter",
            "confidence": confidence,
            "neighbor_product_ids": neighbor_ids,
            "neighbor_similarities": neighbor_similarities,
            "neighbor_weights": neighbor_weights,
            "adapter_version": self.content_to_cf_adapter.version,
            "base_two_tower_version": self.model_version,
            "created_at": time.time(),
        }
        return synthetic_embedding, metadata_payload

    def _get_clip_neighbors(
        self,
        product_id: str,
        clip_embedding: np.ndarray,
        k: int,
    ) -> List[CandidateProduct]:
        if k <= 0 or not hasattr(self.vector_search, "get_all_product_embeddings"):
            return []
        query = normalize_vector(clip_embedding)
        if not np.any(query):
            return []
        scored: List[Tuple[str, float]] = []
        for (
            neighbor_id,
            embedding,
        ) in self.vector_search.get_all_product_embeddings().items():
            if neighbor_id == product_id:
                continue
            score = float(np.dot(query, normalize_vector(embedding)))
            if np.isfinite(score):
                scored.append((neighbor_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            CandidateProduct(
                product_id=neighbor_id,
                content_similarity_score=score,
                combined_score=score,
                source="content_similarity",
            )
            for neighbor_id, score in scored[:k]
        ]

    def _rebuild_serving_cf_index(self) -> None:
        index, index_map = self._build_serving_cf_index_snapshot(
            self.synthetic_item_embeddings
        )
        if index is None:
            return
        self.cf_index = index
        self.cf_index_map = index_map

    def _build_serving_cf_index_snapshot(
        self,
        synthetic_item_embeddings: Dict[str, np.ndarray],
    ) -> Tuple[Optional[faiss.Index], Dict[int, str]]:
        if not self.trained_item_embeddings:
            return None, {}
        product_ids = list(self.trained_item_embeddings.keys()) + [
            product_id
            for product_id in synthetic_item_embeddings.keys()
            if product_id not in self.item_mapping
        ]
        if not product_ids:
            return None, {}
        embeddings = []
        for product_id in product_ids:
            embedding = (
                self.trained_item_embeddings.get(product_id)
                if product_id in self.trained_item_embeddings
                else synthetic_item_embeddings[product_id]
            )
            embeddings.append(normalize_vector(embedding))
        index = self.vector_search.create_cf_index(self.config.tt_embedding_dim)
        index.add(np.vstack(embeddings).astype(np.float32))
        return index, {idx: product_id for idx, product_id in enumerate(product_ids)}

    def _build_cold_start_overlay_version(
        self,
        synthetic_item_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> str:
        synthetic_item_embeddings = (
            self.synthetic_item_embeddings
            if synthetic_item_embeddings is None
            else synthetic_item_embeddings
        )
        payload = {
            "base_model": self.model_version,
            "synthetic_count": len(synthetic_item_embeddings),
            "synthetic_ids": sorted(synthetic_item_embeddings.keys()),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]

    def refresh_new_item_candidates(self) -> None:
        refreshed_at = time.time()
        candidates = self._build_new_item_candidates_snapshot(refreshed_at)
        self._new_item_candidates = candidates
        self.new_item_pool_refreshed_at = refreshed_at
        self._new_item_pool_catalog_token = self._new_item_catalog_token()
        self.new_item_pool_version = self._build_new_item_pool_version(candidates)

    def should_refresh_new_item_candidates(
        self, current_time: Optional[float] = None
    ) -> bool:
        if self.config.max_new_item_candidates <= 0:
            return bool(self._new_item_candidates)
        if self.new_item_pool_refreshed_at <= 0.0:
            return True
        current_time = current_time or time.time()
        interval_seconds = float(
            getattr(self.config, "new_item_pool_refresh_interval_seconds", 300.0)
        )
        if (
            interval_seconds > 0
            and current_time - self.new_item_pool_refreshed_at >= interval_seconds
        ):
            return True
        return self._new_item_catalog_token() != self._new_item_pool_catalog_token

    def _new_item_catalog_token(self) -> str:
        if hasattr(self.vector_search, "get_catalog_version_context"):
            context = self.vector_search.get_catalog_version_context()
        else:
            context = {
                "last_updated": getattr(self.vector_search, "last_updated", None),
                "product_count": len(
                    getattr(self.vector_search, "product_metadata", {}) or {}
                ),
            }
        return hashlib.sha256(
            json.dumps(
                context, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        ).hexdigest()[:16]

    def _build_new_item_candidates_snapshot(
        self, current_time: float
    ) -> List[CandidateProduct]:
        if self.config.max_new_item_candidates <= 0:
            return []
        candidates: List[CandidateProduct] = []
        for product_id in self._eligible_cold_start_product_ids(current_time):
            metadata = self.vector_search.product_metadata.get(product_id, {})
            age_days = max(
                (current_time - float(metadata.get("created_at", current_time)))
                / 86400.0,
                0.0,
            )
            freshness_score = 1.0 / (1.0 + age_days / 30.0)
            bootstrap_confidence = float(
                self.synthetic_item_metadata.get(product_id, {}).get("confidence", 0.5)
            )
            score = float(
                min(1.0, max(0.0, 0.5 * freshness_score + 0.5 * bootstrap_confidence))
            )
            candidates.append(
                CandidateProduct(
                    product_id=product_id,
                    popularity_score=score,
                    combined_score=score,
                    source="new_item",
                )
            )
        return candidates

    @staticmethod
    def _build_new_item_pool_version(candidates: List[CandidateProduct]) -> str:
        payload = [
            {
                "product_id": candidate.product_id,
                "score": round(
                    float(
                        candidate.popularity_score or candidate.combined_score or 0.0
                    ),
                    6,
                ),
            }
            for candidate in candidates
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]

    def get_new_item_candidates(
        self,
        k: int,
        exclude_items: Optional[Set[str]] = None,
    ) -> List[CandidateProduct]:
        exclude_items = exclude_items or set()
        if k <= 0:
            return []
        current_time = time.time()
        candidates: List[CandidateProduct] = []
        for candidate in self._new_item_candidates:
            if len(candidates) >= k:
                break
            if candidate.product_id in exclude_items:
                continue
            if not self._new_item_metadata_still_eligible(
                candidate.product_id, current_time
            ):
                continue
            candidates.append(CandidateProduct(**candidate.dict()))
        return candidates

    def _new_item_metadata_still_eligible(
        self, product_id: str, current_time: float
    ) -> bool:
        metadata = self.vector_search.product_metadata.get(product_id)
        if not metadata or product_id in self.item_mapping:
            return False
        if metadata.get("active") is False or metadata.get("in_stock") is False:
            return False
        if metadata.get("deleted") is True or metadata.get("is_deleted") is True:
            return False
        try:
            created_at = float(metadata.get("created_at", current_time))
        except (TypeError, ValueError):
            created_at = current_time
        age_days = max((current_time - created_at) / 86400.0, 0.0)
        if age_days > float(self.config.cf_cold_start_max_age_days):
            return False
        interaction_count = self._metadata_interaction_count(metadata)
        return interaction_count <= float(self.config.cf_cold_start_max_interactions)

    @staticmethod
    def _metadata_interaction_count(metadata: Dict[str, Any]) -> float:
        for key in ("interaction_count", "num_interactions", "views", "view_count"):
            if key in metadata:
                try:
                    return max(0.0, float(metadata.get(key) or 0.0))
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._retrieval_executor,
            self._get_user_recommendations_sync,
            user_id,
            k,
            exclude_items,
            user_features,
        )

    def close(self) -> None:
        self._retrieval_executor.shutdown(wait=False, cancel_futures=True)

    def _get_user_recommendations_sync(
        self,
        user_id: str,
        k: int = 100,
        exclude_items: Optional[Set[str]] = None,
        user_features: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateProduct]:
        try:
            if (
                not self.is_trained
                or self.cf_index is None
                or self.cf_index.ntotal == 0
            ):
                logger.warning(
                    "Two-Tower model not trained, returning empty recommendations"
                )
                return []

            exclude_items = exclude_items or set()
            user_features = user_features or {}
            current_time = time.time()
            unknown_cache_key = self._unknown_user_candidate_cache_key(
                user_id,
                user_features,
                current_time,
                k,
                exclude_items,
            )
            if unknown_cache_key is not None:
                cached_candidates = self._get_unknown_user_candidate_cache(
                    unknown_cache_key
                )
                if cached_candidates is not None:
                    return cached_candidates

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
                    norm_score = self._cf_search_score_to_similarity(
                        self.cf_index, score
                    )
                    source = "collaborative_filtering"
                    if (
                        product_id in self.synthetic_item_metadata
                        and product_id not in self.item_mapping
                    ):
                        confidence = float(
                            self.synthetic_item_metadata[product_id].get(
                                "confidence", 0.0
                            )
                        )
                        norm_score *= max(0.0, min(confidence, 1.0))
                        source = "cf_cold_start"
                    candidates.append(
                        CandidateProduct(
                            product_id=product_id,
                            collaborative_score=norm_score,
                            combined_score=norm_score,
                            source=source,
                        )
                    )

            logger.debug(
                f"Generated {len(candidates)} Two-Tower recommendations for user {user_id}"
            )
            if unknown_cache_key is not None:
                self._set_unknown_user_candidate_cache(unknown_cache_key, candidates)
            return candidates

        except Exception as e:
            logger.error(f"Error getting Two-Tower recommendations for {user_id}: {e}")
            return []

    def _unknown_user_candidate_cache_key(
        self,
        user_id: str,
        user_features: Dict[str, Any],
        current_time: float,
        k: int,
        exclude_items: Set[str],
    ) -> Optional[Tuple[Any, ...]]:
        if user_id in self.trainer.user_mapping or exclude_items:
            return None
        if self._is_default_unknown_user_features(user_features):
            return (
                self.model_version,
                getattr(self.cf_index, "ntotal", 0),
                k,
                "default_unknown",
            )
        time_bucket = int(current_time / self._user_embedding_cache_time_bucket_seconds)
        last_active = float(user_features.get("last_active", current_time))
        hours_since_active = max((current_time - last_active) / 3600.0, 0.0)
        normalized_features = {
            "total_interactions": int(user_features.get("total_interactions", 0)),
            "avg_session_length": round(
                float(user_features.get("avg_session_length", 0.0)), 3
            ),
            "preferred_categories": list(user_features.get("preferred_categories", [])),
            "price_sensitivity": round(
                float(user_features.get("price_sensitivity", 0.5)), 6
            ),
            "click_through_rate": round(
                float(user_features.get("click_through_rate", 0.0)), 6
            ),
            "conversion_rate": round(
                float(user_features.get("conversion_rate", 0.0)), 6
            ),
            "hours_since_active": round(hours_since_active, 3),
        }
        feature_hash = hashlib.sha256(
            json.dumps(
                normalized_features, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()[:16]
        return (
            self.model_version,
            getattr(self.cf_index, "ntotal", 0),
            k,
            time_bucket,
            feature_hash,
        )

    @staticmethod
    def _is_default_unknown_user_features(user_features: Dict[str, Any]) -> bool:
        preferred_categories = user_features.get("preferred_categories", [])
        return (
            int(user_features.get("total_interactions", 0) or 0) == 0
            and abs(float(user_features.get("avg_session_length", 0.0) or 0.0)) < 1e-9
            and list(preferred_categories or []) == []
            and abs(float(user_features.get("price_sensitivity", 0.5) or 0.5) - 0.5)
            < 1e-9
            and abs(float(user_features.get("click_through_rate", 0.0) or 0.0))
            < 1e-9
            and abs(float(user_features.get("conversion_rate", 0.0) or 0.0)) < 1e-9
        )

    def _get_unknown_user_candidate_cache(
        self, cache_key: Tuple[Any, ...]
    ) -> Optional[List[CandidateProduct]]:
        with self._user_embedding_cache_lock:
            cached = self._unknown_user_candidate_cache.get(cache_key)
            if cached is None:
                return None
            self._unknown_user_candidate_cache.move_to_end(cache_key)
            return [CandidateProduct(**candidate.dict()) for candidate in cached]

    def _set_unknown_user_candidate_cache(
        self, cache_key: Tuple[Any, ...], candidates: List[CandidateProduct]
    ) -> None:
        if self._user_embedding_cache_max_size <= 0:
            return
        with self._user_embedding_cache_lock:
            self._unknown_user_candidate_cache[cache_key] = [
                CandidateProduct(**candidate.dict()) for candidate in candidates
            ]
            if len(self._unknown_user_candidate_cache) > self._user_embedding_cache_max_size:
                self._unknown_user_candidate_cache.popitem(last=False)

    @staticmethod
    def _cf_search_score_to_similarity(index: faiss.Index, score: float) -> float:
        raw_score = float(score)
        if not np.isfinite(raw_score):
            return 0.0
        metric_type = getattr(index, "metric_type", faiss.METRIC_L2)
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            return float(max(min((raw_score + 1.0) / 2.0, 1.0), 0.0))
        # FAISS L2 indexes return squared distance. For normalized vectors,
        # squared L2 is in [0, 4], and 1 - distance / 4 maps nearest to 1.
        distance = max(raw_score, 0.0)
        return float(max(min(1.0 - distance / 4.0, 1.0), 0.0))

    def clear_user_embedding_cache(self) -> None:
        with self._user_embedding_cache_lock:
            self._user_embedding_cache.clear()
            self._unknown_user_candidate_cache.clear()

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
        time_bucket = int(current_time / self._user_embedding_cache_time_bucket_seconds)
        normalized_features = {
            "total_interactions": int(user_features.get("total_interactions", 0)),
            "avg_session_length": round(
                float(user_features.get("avg_session_length", 0.0)), 3
            ),
            "preferred_categories": list(user_features.get("preferred_categories", [])),
            "price_sensitivity": round(
                float(user_features.get("price_sensitivity", 0.5)), 6
            ),
            "click_through_rate": round(
                float(user_features.get("click_through_rate", 0.0)), 6
            ),
            "conversion_rate": round(
                float(user_features.get("conversion_rate", 0.0)), 6
            ),
            "last_active": round(
                float(user_features.get("last_active", current_time)), 3
            ),
        }
        feature_hash = hashlib.sha256(
            json.dumps(
                normalized_features, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
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

            max_pop = (
                max(self._item_popularity.values()) if self._item_popularity else 1.0
            )
            sorted_items = sorted(
                self._item_popularity.items(), key=lambda x: x[1], reverse=True
            )

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
                    decay_factor = self.config.trending_decay_factor**hours_ago
                    total_score += weight * decay_factor
                self.trending_scores[product_id] = total_score

            cutoff_time = current_time - (self.config.trending_window_hours * 2 * 3600)
            for product_id in list(self.interaction_counts.keys()):
                self.interaction_counts[product_id] = [
                    (ts, w)
                    for ts, w in self.interaction_counts[product_id]
                    if ts >= cutoff_time
                ]
                if not self.interaction_counts[product_id]:
                    del self.interaction_counts[product_id]

            self.last_updated = current_time
            logger.debug(
                f"Updated trending scores for {len(self.trending_scores)} products"
            )

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

            sorted_products = sorted(
                self.trending_scores.items(), key=lambda x: x[1], reverse=True
            )
            max_score = (
                max(self.trending_scores.values()) if self.trending_scores else 1.0
            )

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
        training_sequence_lookback_days: Optional[int] = None,
    ):
        self.feature_store = feature_store
        self.vector_search = vector_search
        self.config = config
        self.artifact_manager = artifact_manager
        self.training_sequence_lookback_days = training_sequence_lookback_days

        # Recommendation engines
        self.cf_engine = TwoTowerRetrievalEngine(config, vector_search)
        self.sasrec_engine = SASRecCandidateEngine(config)
        self.swing_itemcf_engine = SwingItemCFCandidateEngine(
            max_seed_items=config.swing_itemcf_max_seed_items,
            score_weight=config.swing_itemcf_score_weight,
        )
        self.trending_engine = TrendingEngine(config)

        # Model state
        self.is_initialized = False
        self.last_model_update = 0
        self.loaded_two_tower_version: Optional[str] = None
        self.loaded_sasrec_version: Optional[str] = None
        self.loaded_swing_itemcf_version: Optional[str] = None
        self.loaded_content_cluster_version: Optional[str] = None

        logger.info("Recommendation engine initialized (Two-Tower + SASRec + Swing retrieval)")

    def close(self) -> None:
        self.cf_engine.close()

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
            await self._try_load_swing_itemcf_artifact()
            await self._try_load_content_cluster_artifacts()
            await self.refresh_serving_pools()
            await self._prime_hot_serving_caches()
            self.is_initialized = True
            if not self.cf_engine.is_trained:
                logger.warning(
                    "Two-Tower index not available; serving will fall back to non-CF sources"
                )
            if self.config.enable_sasrec and not self.sasrec_engine.is_trained:
                logger.warning(
                    "SASRec artifacts not available; serving will skip sequential candidates"
                )
            if (
                self.config.enable_swing_itemcf
                and not self.swing_itemcf_engine.is_trained
            ):
                logger.warning(
                    "Swing ItemCF artifact not available; serving will skip Swing candidates"
                )
            logger.info("Recommendation serving state ready")
        except Exception as e:
            logger.error(f"Error loading recommendation serving state: {e}")
            raise

    async def _prime_hot_serving_caches(self) -> None:
        if not self.cf_engine.is_trained:
            return
        default_k = min(
            max(200, self.config.candidates_per_source),
            self.config.max_total_candidates,
            self.config.max_live_cf_candidates,
        )
        await asyncio.to_thread(
            self.cf_engine.prime_default_unknown_candidates,
            default_k,
        )

    async def _try_load_cf_index(self):
        """Attempt to load a previously saved CF FAISS index for fast cold start."""
        try:
            checkpoint_record = None
            if self.artifact_manager:
                checkpoint_record = (
                    await self.artifact_manager.sync_latest_two_tower_artifacts()
                )

            result = await asyncio.to_thread(
                VectorSearchEngine.load_cf_index,
                self.config.cf_index_path,
            )
            if result is not None:
                index, metadata = result
                index_map_raw = metadata.get("index_map", {})
                index_map = {int(k): v for k, v in index_map_raw.items()}

                checkpoint_path = (
                    self.artifact_manager.two_tower_local_checkpoint_path
                    if self.artifact_manager
                    else self.config.cf_index_path.replace(".faiss", ".pt")
                )
                loaded_engine = TwoTowerRetrievalEngine(self.config, self.vector_search)
                loaded_engine.cf_index = index
                loaded_engine.cf_index_map = index_map
                loaded_engine._base_cf_index = index
                loaded_engine._base_cf_index_map = dict(index_map)

                if not await asyncio.to_thread(
                    loaded_engine.trainer.load_checkpoint, checkpoint_path
                ):
                    loaded_engine.close()
                    return False

                loaded_engine.user_mapping = dict(loaded_engine.trainer.user_mapping)
                loaded_engine.item_mapping = dict(loaded_engine.trainer.item_mapping)
                loaded_engine.is_trained = True
                if checkpoint_record:
                    loaded_engine.model_version = checkpoint_record.model_version
                loaded_engine._load_cold_start_sidecars()
                await loaded_engine.refresh_cold_start_overlay()
                await asyncio.to_thread(loaded_engine.refresh_new_item_candidates)
                loaded_engine.clear_user_embedding_cache()

                self.cf_engine.close()
                self.cf_engine = loaded_engine
                if checkpoint_record:
                    self.loaded_two_tower_version = checkpoint_record.model_version
                elif loaded_engine.model_version:
                    self.loaded_two_tower_version = loaded_engine.model_version
                logger.info("Loaded pre-existing Two-Tower model and CF index")
                return True
        except Exception as e:
            logger.warning(f"Could not load pre-existing CF index: {e}")
        return False

    async def _try_load_sasrec_artifacts(self) -> bool:
        """Load SASRec serving artifacts if the sequential source is enabled."""
        if not self.config.enable_sasrec:
            return False
        try:
            checkpoint_record = None
            if self.artifact_manager:
                checkpoint_record = (
                    await self.artifact_manager.sync_latest_sasrec_artifacts()
                )

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

    async def _try_load_swing_itemcf_artifact(self) -> bool:
        """Load Swing ItemCF serving artifact if the source is enabled."""
        if not self.config.enable_swing_itemcf:
            return False
        try:
            checkpoint_record = None
            if self.artifact_manager:
                checkpoint_record = (
                    await self.artifact_manager.sync_latest_swing_itemcf_artifact()
                )

            artifact_path = self._swing_itemcf_artifact_path()
            loaded_engine = SwingItemCFCandidateEngine(
                max_seed_items=self.config.swing_itemcf_max_seed_items,
                score_weight=self.config.swing_itemcf_score_weight,
            )
            loaded = await asyncio.to_thread(loaded_engine.load_artifact, artifact_path)
            if loaded:
                if checkpoint_record:
                    loaded_engine.model_version = checkpoint_record.model_version
                self.swing_itemcf_engine = loaded_engine
                self.loaded_swing_itemcf_version = loaded_engine.model_version
                logger.info("Loaded Swing ItemCF serving artifact")
                return True
        except Exception as e:
            logger.warning(f"Could not load Swing ItemCF artifact: {e}")
        return False

    async def _try_load_content_cluster_artifacts(self) -> bool:
        """Load content-cluster serving artifacts if the source is enabled."""
        if not self.config.enable_content_cluster_pools:
            self.loaded_content_cluster_version = None
            return False
        load_artifact = getattr(
            self.vector_search,
            "load_content_cluster_artifact",
            None,
        )
        if not callable(load_artifact):
            self.loaded_content_cluster_version = None
            return False
        metadata_path, centroids_path = self._content_cluster_artifact_paths()
        loaded = await asyncio.to_thread(
            load_artifact,
            metadata_path=metadata_path,
            centroids_path=centroids_path,
        )
        if loaded:
            self.loaded_content_cluster_version = (
                self.vector_search.content_cluster_model_version
            )
            return True
        self.loaded_content_cluster_version = None
        return False

    async def _update_content_cluster_artifacts(self) -> bool:
        """Build and persist content-cluster artifacts on the offline update path."""
        if not self.config.enable_content_cluster_pools:
            return False
        get_embeddings = getattr(self.vector_search, "get_all_product_embeddings", None)
        if not callable(get_embeddings):
            logger.warning(
                "Skipping content cluster build because vector backend does not expose embeddings"
            )
            return False
        product_embeddings = get_embeddings()
        if not product_embeddings:
            logger.warning("Skipping content cluster build because product embeddings are empty")
            return False

        metadata_path, centroids_path = self._content_cluster_artifact_paths()
        try:
            catalog_context = (
                self.vector_search.get_catalog_version_context()
                if hasattr(self.vector_search, "get_catalog_version_context")
                else {}
            )
            artifact = await asyncio.to_thread(
                build_content_cluster_artifact,
                product_embeddings,
                num_clusters=self.config.content_cluster_count,
                source_catalog_context=catalog_context,
            )
            await asyncio.to_thread(
                save_content_cluster_artifact,
                artifact,
                metadata_path=metadata_path,
                centroids_path=centroids_path,
            )
            self.vector_search.apply_content_cluster_artifact(artifact)
            self.loaded_content_cluster_version = artifact.cluster_model_version
            logger.info(
                "Built content cluster artifact %s with %s clusters and %s products",
                artifact.cluster_model_version,
                artifact.num_clusters,
                artifact.product_count,
            )
            return True
        except Exception as exc:
            logger.warning("Content cluster artifact update failed: %s", exc)
            return False

    async def _update_models_from_interactions(self):
        """Update models using recent interaction data."""
        try:
            if not self.artifact_manager or not self.artifact_manager.system_store:
                await self._update_content_cluster_artifacts()
                await self.refresh_serving_pools()
                logger.warning(
                    "Skipping Two-Tower retraining because Postgres system store is unavailable"
                )
                return

            interactions = (
                await self.artifact_manager.system_store.get_training_interactions(
                    limit=50000
                )
            )

            if interactions:
                # Gather user features for the trainer
                user_features_map = await self.feature_store.get_all_user_features_map()

                # Train Two-Tower model
                await self.cf_engine.train_model(
                    interactions, user_features_map=user_features_map
                )
                if self.cf_engine.is_trained:
                    checkpoint_path = (
                        self.artifact_manager.two_tower_local_checkpoint_path
                    )
                    index_path = self.artifact_manager.two_tower_local_index_path
                    metadata_path = self.artifact_manager.two_tower_local_metadata_path
                    model_version = (
                        self.cf_engine.model_version or f"two-tower-{int(time.time())}"
                    )
                    artifact_record = await self.artifact_manager.persist_two_tower_artifacts(
                        checkpoint_path=checkpoint_path,
                        index_path=index_path,
                        metadata_path=metadata_path,
                        model_version=model_version,
                        embedding_sidecar_path=self.artifact_manager.two_tower_local_embedding_sidecar_path,
                        adapter_path=self.artifact_manager.two_tower_local_adapter_path,
                        payload={
                            "sample_count": len(interactions),
                            "last_training_time": self.cf_engine.last_training_time,
                        },
                    )
                    self.loaded_two_tower_version = (
                        artifact_record.model_version
                        if artifact_record
                        else model_version
                    )

                await self._update_sasrec_from_sequences()
                await self._update_swing_itemcf_from_sequences()
                await self._update_content_cluster_artifacts()

                # Update trending scores (use last 1K for recency)
                await self.trending_engine.update_trending_scores(interactions[:1000])
                await self.refresh_serving_pools()

                self.last_model_update = time.time()
                logger.info(f"Updated models with {len(interactions)} interactions")
            else:
                await self._update_content_cluster_artifacts()
                await self.refresh_serving_pools()
                logger.warning("No interactions found for model training")

        except Exception as e:
            logger.error(f"Error updating models from interactions: {e}")

    async def _update_sasrec_from_sequences(self) -> None:
        if not self.config.enable_sasrec:
            return
        if not self.artifact_manager or not self.artifact_manager.system_store:
            logger.warning(
                "Skipping SASRec training because Postgres system store is unavailable"
            )
            return

        try:
            max_events = max(
                int(self.config.sasrec_min_sequence_length),
                int(self.config.sasrec_max_sequence_length) + 1,
            )
            lookback_days = max(0, int(self.training_sequence_lookback_days or 0))
            since = time.time() - (lookback_days * 86400) if lookback_days > 0 else None
            sequences = (
                await self.artifact_manager.system_store.get_user_training_sequences(
                    max_events_per_user=max_events,
                    min_sequence_length=self.config.sasrec_min_sequence_length,
                    since=since,
                )
            )
            if not sequences:
                logger.info(
                    "Skipping SASRec training because no positive user sequences are available"
                )
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
            self.loaded_sasrec_version = (
                artifact_record.model_version if artifact_record else model_version
            )
        except Exception as exc:
            logger.warning(
                "SASRec training update failed; continuing without new sequential artifacts: %s",
                exc,
            )

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

    async def _update_swing_itemcf_from_sequences(self) -> None:
        if not self.config.enable_swing_itemcf:
            return
        if not self.artifact_manager or not self.artifact_manager.system_store:
            logger.warning(
                "Skipping Swing ItemCF training because Postgres system store is unavailable"
            )
            return

        try:
            lookback_days = max(0, int(self.training_sequence_lookback_days or 0))
            since = time.time() - (lookback_days * 86400) if lookback_days > 0 else None
            sequences = (
                await self.artifact_manager.system_store.get_user_training_sequences(
                    max_users=self.config.swing_itemcf_training_max_users,
                    max_events_per_user=self.config.swing_itemcf_training_max_events_per_user,
                    min_sequence_length=2,
                    since=since,
                    actions=POSITIVE_SEQUENCE_ACTIONS,
                )
            )
            if not sequences:
                logger.info(
                    "Skipping Swing ItemCF training because no positive user sequences are available"
                )
                return

            trainer = SwingItemCFTrainer(
                alpha=self.config.swing_itemcf_alpha,
                max_neighbors_per_item=self.config.swing_itemcf_max_neighbors_per_item,
                max_items_per_user=self.config.swing_itemcf_max_items_per_user,
                max_users_per_item=self.config.swing_itemcf_max_users_per_item,
            )
            model_version = f"swing-itemcf-{int(time.time())}"
            index = await asyncio.to_thread(
                trainer.fit,
                sequences,
                model_version=model_version,
            )
            if not index.is_trained:
                logger.info("Skipping Swing ItemCF artifact persistence; index is empty")
                return

            artifact_path = self._swing_itemcf_artifact_path()
            await asyncio.to_thread(index.save, artifact_path)
            artifact_record = await self.artifact_manager.persist_swing_itemcf_artifact(
                index_path=artifact_path,
                model_version=index.model_version or model_version,
                payload={
                    "sequence_count": len(sequences),
                    **dict(index.metadata),
                },
            )
            if artifact_record:
                index.model_version = artifact_record.model_version
            self.swing_itemcf_engine = SwingItemCFCandidateEngine(
                index,
                max_seed_items=self.config.swing_itemcf_max_seed_items,
                score_weight=self.config.swing_itemcf_score_weight,
            )
            self.loaded_swing_itemcf_version = (
                artifact_record.model_version
                if artifact_record
                else self.swing_itemcf_engine.model_version
            )
        except Exception as exc:
            logger.warning(
                "Swing ItemCF training update failed; continuing without new Swing artifacts: %s",
                exc,
            )

    def _swing_itemcf_artifact_path(self) -> str:
        if self.artifact_manager:
            return self.artifact_manager.swing_itemcf_local_index_path
        base_dir = Path(self.config.cf_index_path).parent
        return self.config.swing_itemcf_index_path or str(
            base_dir / "swing_itemcf.json.gz"
        )

    def _content_cluster_artifact_paths(self) -> Tuple[str, str]:
        base_dir = Path(self.config.cf_index_path).parent
        return (
            self.config.content_cluster_metadata_path
            or str(base_dir / "content_clusters.metadata.json"),
            self.config.content_cluster_centroids_path
            or str(base_dir / "content_clusters.centroids.npz"),
        )

    async def sync_serving_artifacts_if_updated(self) -> bool:
        """Reload retrieval serving artifacts when newer checkpoints are available."""
        updated = False
        pools_refreshed = False
        if self.config.enable_content_cluster_pools:
            previous_cluster_version = self.loaded_content_cluster_version
            await self._try_load_content_cluster_artifacts()
            if previous_cluster_version != self.loaded_content_cluster_version:
                await self.refresh_serving_pools()
                updated = True
                pools_refreshed = True

        if not self.artifact_manager:
            return updated

        latest_two_tower = await self.artifact_manager.get_latest_model_checkpoint(
            ModelArtifactManager.TWO_TOWER_MODEL_NAME
        )
        if (
            latest_two_tower
            and latest_two_tower.model_version != self.loaded_two_tower_version
        ):
            await self._try_load_cf_index()
            updated = (
                updated
                or self.loaded_two_tower_version == latest_two_tower.model_version
            )

        if self.config.enable_sasrec:
            latest_sasrec = await self.artifact_manager.get_latest_model_checkpoint(
                ModelArtifactManager.SASREC_MODEL_NAME
            )
            if (
                latest_sasrec
                and latest_sasrec.model_version != self.loaded_sasrec_version
            ):
                await self._try_load_sasrec_artifacts()
                updated = (
                    updated or self.loaded_sasrec_version == latest_sasrec.model_version
                )

        if self.config.enable_swing_itemcf:
            latest_swing = await self.artifact_manager.get_latest_model_checkpoint(
                ModelArtifactManager.SWING_ITEMCF_MODEL_NAME
            )
            if (
                latest_swing
                and latest_swing.model_version != self.loaded_swing_itemcf_version
            ):
                await self._try_load_swing_itemcf_artifact()
                updated = (
                    updated
                    or self.loaded_swing_itemcf_version == latest_swing.model_version
                )

        if self.cf_engine.should_refresh_new_item_candidates() and not pools_refreshed:
            await asyncio.to_thread(self.cf_engine.refresh_new_item_candidates)
            updated = True

        return updated

    def _content_cluster_pool_version(self) -> Optional[str]:
        if not self.config.enable_content_cluster_pools:
            return None
        return self.loaded_content_cluster_version or getattr(
            self.vector_search,
            "content_cluster_model_version",
            None,
        )

    def _compute_serving_pool_score(
        self,
        product_id: str,
        metadata: Dict[str, Any],
        current_time: Optional[float] = None,
    ) -> float:
        """Compute a stable heuristic score for serving pools."""
        current_time = current_time or time.time()
        trending_score = float(
            self.trending_engine.trending_scores.get(product_id, 0.0)
        )
        rating_score = float(metadata.get("rating", 0.0)) / 5.0
        review_score = np.log1p(float(metadata.get("num_reviews", 0.0))) / 5.0
        age_days = max(
            (current_time - float(metadata.get("created_at", current_time))) / 86400.0,
            0.0,
        )
        freshness_score = 1.0 / (1.0 + age_days / 30.0)
        return (
            trending_score
            + rating_score * 0.6
            + review_score * 0.2
            + freshness_score * 0.2
        )

    async def refresh_serving_pools(self):
        """Precompute global trending and per-category serving pools."""
        try:
            if not self.vector_search.product_metadata:
                await asyncio.to_thread(self.cf_engine.refresh_new_item_candidates)
                return

            current_time = time.time()
            global_scored: List[Tuple[str, float]] = []
            category_scored: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
            cluster_scored: Dict[int, List[Tuple[str, float]]] = defaultdict(list)

            for product_id, metadata in self.vector_search.product_metadata.items():
                score = self._compute_serving_pool_score(
                    product_id, metadata, current_time
                )
                global_scored.append((product_id, score))
                category = metadata.get("category", "unknown")
                category_scored[category].append((product_id, score))
                get_cluster_id = getattr(
                    self.vector_search,
                    "get_product_cluster_id",
                    None,
                )
                if self.config.enable_content_cluster_pools and callable(get_cluster_id):
                    cluster_id = get_cluster_id(product_id)
                    if cluster_id is not None:
                        cluster_scored[int(cluster_id)].append((product_id, score))

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
                category_max = (
                    max((score for _, score in category_top), default=1.0) or 1.0
                )
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
            cluster_pools: Dict[int, List[CandidateProduct]] = {}
            cluster_pool_version = self._content_cluster_pool_version()
            if cluster_pool_version and cluster_scored:
                for cluster_id, scored_items in cluster_scored.items():
                    scored_items.sort(key=lambda item: item[1], reverse=True)
                    cluster_top = scored_items[: self.config.serving_cluster_pool_size]
                    cluster_max = (
                        max((score for _, score in cluster_top), default=1.0) or 1.0
                    )
                    cluster_pools[cluster_id] = [
                        CandidateProduct(
                            product_id=product_id,
                            popularity_score=float(score / cluster_max),
                            combined_score=float(score / cluster_max),
                            source="cluster_pool",
                        )
                        for product_id, score in cluster_top
                    ]
                await self.feature_store.store_cluster_pools(
                    cluster_pools,
                    pool_version=cluster_pool_version,
                )
            await asyncio.to_thread(self.cf_engine.refresh_new_item_candidates)
            logger.info(
                "Refreshed serving pools: %s global products, %s categories, %s clusters",
                len(trending_pool),
                len(category_pools),
                len(cluster_pools),
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
                    "occurred_at": interaction.get(
                        "occurred_at", interaction.get("timestamp")
                    ),
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
        content_features: Optional[ContentFeatures] = None,
    ) -> List[str]:
        """Resolve the small set of categories to pull from precomputed pools."""
        categories: List[str] = []
        for key in ("product_category", "category"):
            value = context.get(key)
            if isinstance(value, str) and value:
                categories.append(value)

        audio_features = content_features.audio_features if content_features else None
        if (
            self.config.speech_category_candidates_enabled
            and audio_features
            and audio_features.transcription_status == "completed"
        ):
            categories.extend(audio_features.speech_categories)

        if user_features:
            categories.extend(user_features.preferred_categories)

        deduped: List[str] = []
        for category in categories:
            if category and category not in deduped:
                deduped.append(category)
        return deduped[: self.config.preferred_category_pool_count]

    def _resolve_preferred_content_clusters(
        self,
        content_features: Optional[ContentFeatures],
        user_interactions: List[Dict[str, Any]],
    ) -> List[int]:
        """Resolve content-cluster pools from current content and recent positives."""
        if not self._content_cluster_pool_version():
            return []
        cluster_limit = max(1, int(self.config.preferred_cluster_pool_count))
        clusters: List[int] = []

        if content_features and content_features.visual_embedding:
            assign_clusters = getattr(
                self.vector_search,
                "assign_content_embedding_clusters",
                None,
            )
            if callable(assign_clusters):
                clusters.extend(
                    assign_clusters(
                        content_features.visual_embedding,
                        limit=cluster_limit,
                    )
                )

        for interaction in reversed(user_interactions):
            if len(clusters) >= cluster_limit:
                break
            product_id = interaction.get("product_id")
            action = interaction.get("action")
            if not product_id or action not in POSITIVE_SEQUENCE_ACTIONS:
                continue
            get_cluster_id = getattr(self.vector_search, "get_product_cluster_id", None)
            if not callable(get_cluster_id):
                break
            cluster_id = get_cluster_id(product_id)
            if cluster_id is not None:
                clusters.append(int(cluster_id))

        deduped: List[int] = []
        for cluster_id in clusters:
            if cluster_id not in deduped:
                deduped.append(cluster_id)
        return deduped[:cluster_limit]

    async def generate_candidates(
        self,
        user_id: str,
        content_features: Optional[ContentFeatures] = None,
        context: Optional[Dict[str, Any]] = None,
        k_per_source: int = 100,
        include_profile: bool = False,
        user_features: Optional[UserFeatures] = None,
        user_interactions: Optional[List[Dict[str, Any]]] = None,
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
                "swing_itemcf_candidates_ms": 0.0,
                "content_candidates_ms": 0.0,
                "trending_candidates_ms": 0.0,
                "category_pool_ms": 0.0,
                "cluster_pool_ms": 0.0,
                "new_item_candidates_ms": 0.0,
                "random_candidates_ms": 0.0,
                "score_merge_ms": 0.0,
                "total_ms": 0.0,
                "candidate_count": 0,
                "preferred_categories": [],
                "preferred_clusters": [],
                "source_counts": {},
                "source_overlap_counts": {},
                "merged_source_counts": {},
                "sasrec_sequence_length": 0,
                "swing_itemcf_seed_count": 0,
                "has_transcript": False,
                "speech_category_candidates_used": False,
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
                        return await asyncio.wait_for(
                            awaitable, timeout=timeout_ms / 1000.0
                        )
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
                    profile[metric_name] = round(
                        (time.perf_counter() - stage_started) * 1000, 2
                    )

            interaction_task = None
            sasrec_needs_interactions = (
                bool(self.config.enable_sasrec)
                and bool(getattr(self.sasrec_engine, "is_trained", False))
            )
            swing_needs_interactions = (
                bool(self.config.enable_swing_itemcf)
                and bool(getattr(self.swing_itemcf_engine, "is_trained", False))
            )
            serving_recent_limit = int(
                getattr(self.config, "serving_recent_interaction_limit", 0) or 0
            )
            interaction_limit = 0
            if user_interactions is None:
                requested_limits: List[int] = []
                if sasrec_needs_interactions:
                    requested_limits.append(
                        int(getattr(self.config, "sasrec_max_sequence_length", 0) or 0)
                    )
                if swing_needs_interactions:
                    swing_read_limit = int(
                        getattr(
                            self.config,
                            "swing_itemcf_serving_interaction_limit",
                            0,
                        )
                        or 0
                    )
                    swing_seed_limit = int(
                        getattr(self.config, "swing_itemcf_max_seed_items", 0) or 0
                    )
                    requested_limits.append(
                        max(swing_read_limit, swing_seed_limit)
                    )
                if serving_recent_limit > 0:
                    requested_limits.append(serving_recent_limit)
                if requested_limits:
                    interaction_limit = max(requested_limits)
            if user_interactions is None and interaction_limit > 0:
                interaction_task = asyncio.create_task(
                    timed_stage(
                        "user_interactions_ms",
                        self.feature_store.get_user_interactions(
                            user_id,
                            limit=interaction_limit,
                        ),
                        [],
                        self.config.interaction_history_timeout_ms,
                    ),
                    name="candidate-user-interactions",
                )
            else:
                profile["user_interactions_ms"] = 0.0
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

            resolved_user_interactions = (
                await interaction_task
                if interaction_task is not None
                else (user_interactions or [])
            )
            exclude_items = {
                interaction.get("product_id")
                for interaction in resolved_user_interactions
                if interaction.get("product_id")
            }
            sasrec_sequence = self._build_sasrec_sequence_from_interactions(
                resolved_user_interactions
            )
            profile["sasrec_sequence_length"] = len(sasrec_sequence)
            profile["swing_itemcf_seed_count"] = len(
                {
                    interaction.get("product_id")
                    for interaction in resolved_user_interactions
                    if interaction.get("product_id")
                    and interaction.get("action") in POSITIVE_SEQUENCE_ACTIONS
                }
            )
            user_features_dict = user_features_obj.dict() if user_features_obj else {}
            preferred_categories = self._resolve_preferred_categories(
                user_features_obj, context, content_features
            )
            profile["preferred_categories"] = preferred_categories
            preferred_clusters = self._resolve_preferred_content_clusters(
                content_features,
                resolved_user_interactions,
            )
            profile["preferred_clusters"] = preferred_clusters
            audio_features = content_features.audio_features if content_features else None
            speech_categories = (
                list(audio_features.speech_categories)
                if audio_features is not None
                else []
            )
            profile["has_transcript"] = bool(
                audio_features and audio_features.audio_transcript
            )
            profile["speech_category_candidates_used"] = bool(
                self.config.speech_category_candidates_enabled
                and any(category in preferred_categories for category in speech_categories)
            )
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

            async def fetch_swing_itemcf_candidates() -> List[CandidateProduct]:
                if (
                    not self.config.enable_swing_itemcf
                    or not self.swing_itemcf_engine.is_trained
                ):
                    return []
                return self.swing_itemcf_engine.get_candidates(
                    resolved_user_interactions,
                    k=min(target_candidates, self.config.max_live_swing_itemcf_candidates),
                    exclude_items=exclude_items,
                    current_time=time.time(),
                )

            async def fetch_content_candidates() -> List[CandidateProduct]:
                if not content_features or not content_features.visual_embedding:
                    return []
                query_embedding = np.array(content_features.visual_embedding)
                content_k = min(
                    target_candidates,
                    self.config.max_live_content_candidates,
                )
                sync_search = getattr(
                    self.vector_search,
                    "search_similar_products_sync",
                    None,
                )
                if callable(sync_search):
                    candidates = await self.cf_engine.run_in_retrieval_executor(
                        sync_search,
                        query_embedding,
                        content_k,
                    )
                else:
                    candidates = await self.vector_search.search_similar_products(
                        query_embedding,
                        k=content_k,
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
                    self.config.max_pool_category_candidates
                    // max(len(preferred_categories), 1),
                )
                category_tasks = [
                    self.feature_store.get_category_pool(
                        category,
                        per_category_limit,
                        exclude_items=exclude_items,
                    )
                    for category in preferred_categories
                ]
                category_results = await asyncio.gather(
                    *category_tasks, return_exceptions=True
                )
                category_candidates: List[CandidateProduct] = []
                for result in category_results:
                    if isinstance(result, Exception):
                        logger.warning("category_pool_fetch_failed: %s", result)
                        continue
                    category_candidates.extend(result)
                return category_candidates

            async def fetch_cluster_candidates() -> List[CandidateProduct]:
                cluster_pool_version = self._content_cluster_pool_version()
                if (
                    not cluster_pool_version
                    or not preferred_clusters
                    or self.config.max_pool_cluster_candidates <= 0
                ):
                    return []
                per_cluster_limit = max(
                    1,
                    self.config.max_pool_cluster_candidates
                    // max(len(preferred_clusters), 1),
                )
                cluster_tasks = [
                    self.feature_store.get_cluster_pool(
                        cluster_id,
                        per_cluster_limit,
                        exclude_items=exclude_items,
                        pool_version=cluster_pool_version,
                    )
                    for cluster_id in preferred_clusters
                ]
                cluster_results = await asyncio.gather(
                    *cluster_tasks, return_exceptions=True
                )
                cluster_candidates: List[CandidateProduct] = []
                for result in cluster_results:
                    if isinstance(result, Exception):
                        logger.warning("cluster_pool_fetch_failed: %s", result)
                        continue
                    cluster_candidates.extend(result)
                return cluster_candidates

            async def fetch_new_item_candidates() -> List[CandidateProduct]:
                if not hasattr(self.cf_engine, "get_new_item_candidates"):
                    return []
                return self.cf_engine.get_new_item_candidates(
                    min(target_candidates, self.config.max_new_item_candidates),
                    exclude_items,
                )

            source_timeout_ms = self.config.candidate_source_timeout_ms
            cf_task = asyncio.create_task(
                timed_stage(
                    "cf_candidates_ms", fetch_cf_candidates(), [], source_timeout_ms
                ),
                name="candidate-source-cf",
            )
            sasrec_task = asyncio.create_task(
                timed_stage(
                    "sasrec_candidates_ms",
                    fetch_sasrec_candidates(),
                    [],
                    source_timeout_ms,
                ),
                name="candidate-source-sasrec",
            )
            swing_itemcf_task = asyncio.create_task(
                timed_stage(
                    "swing_itemcf_candidates_ms",
                    fetch_swing_itemcf_candidates(),
                    [],
                    source_timeout_ms,
                ),
                name="candidate-source-swing-itemcf",
            )
            content_task = asyncio.create_task(
                timed_stage(
                    "content_candidates_ms",
                    fetch_content_candidates(),
                    [],
                    source_timeout_ms,
                ),
                name="candidate-source-content",
            )
            trending_task = asyncio.create_task(
                timed_stage(
                    "trending_candidates_ms",
                    fetch_trending_candidates(),
                    [],
                    source_timeout_ms,
                ),
                name="candidate-source-trending",
            )
            category_task = asyncio.create_task(
                timed_stage(
                    "category_pool_ms",
                    fetch_category_candidates(),
                    [],
                    source_timeout_ms,
                ),
                name="candidate-source-category",
            )
            cluster_task = asyncio.create_task(
                timed_stage(
                    "cluster_pool_ms",
                    fetch_cluster_candidates(),
                    [],
                    source_timeout_ms,
                ),
                name="candidate-source-cluster",
            )
            new_item_task = asyncio.create_task(
                timed_stage(
                    "new_item_candidates_ms",
                    fetch_new_item_candidates(),
                    [],
                    source_timeout_ms,
                ),
                name="candidate-source-new-item",
            )

            (
                cf_candidates,
                sasrec_candidates,
                swing_itemcf_candidates,
                content_candidates,
                trending_candidates,
                category_candidates,
                cluster_candidates,
                new_item_candidates,
            ) = await asyncio.gather(
                cf_task,
                sasrec_task,
                swing_itemcf_task,
                content_task,
                trending_task,
                category_task,
                cluster_task,
                new_item_task,
            )

            def merge_source(
                source_name: str, candidates: List[CandidateProduct]
            ) -> None:
                overlaps = 0
                for candidate in candidates:
                    if self._merge_candidate(all_candidates, candidate):
                        overlaps += 1
                profile["source_counts"][source_name] = len(candidates)
                profile["source_overlap_counts"][source_name] = overlaps

            merge_source("cf", cf_candidates)
            merge_source("sasrec", sasrec_candidates)
            merge_source("swing_itemcf", swing_itemcf_candidates)
            merge_source("content", content_candidates)
            merge_source("trending_pool", trending_candidates)
            merge_source("category_pool", category_candidates)
            merge_source("cluster_pool", cluster_candidates)
            merge_source("new_item", new_item_candidates)

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
                    sync_random = getattr(
                        self.vector_search,
                        "get_random_products_sync",
                        None,
                    )
                    random_awaitable = (
                        self.cf_engine.run_in_retrieval_executor(
                            sync_random,
                            random_target,
                        )
                        if callable(sync_random)
                        else self.vector_search.get_random_products(k=random_target)
                    )
                    random_candidates = await timed_stage(
                        "random_candidates_ms",
                        random_awaitable,
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
            profile["merged_source_counts"] = self._count_source_tokens(
                list(all_candidates.values())
            )
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
                final_candidates = await self._apply_diversity_filter(
                    final_candidates, context
                )

            # Sort by combined score
            final_candidates.sort(key=lambda x: x.combined_score, reverse=True)
            final_candidates = final_candidates[: self.config.max_total_candidates]
            profile["score_merge_ms"] = round(
                (time.perf_counter() - stage_started) * 1000, 2
            )
            profile["candidate_count"] = len(final_candidates)
            profile["total_ms"] = round((time.perf_counter() - started_at) * 1000, 2)

            logger.debug(
                f"Generated {len(final_candidates)} candidates from multiple sources"
            )
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
                    "swing_itemcf_candidates_ms": 0.0,
                    "content_candidates_ms": 0.0,
                    "trending_candidates_ms": 0.0,
                    "category_pool_ms": 0.0,
                    "cluster_pool_ms": 0.0,
                    "new_item_candidates_ms": 0.0,
                    "random_candidates_ms": 0.0,
                    "score_merge_ms": 0.0,
                    "total_ms": 0.0,
                    "candidate_count": 0,
                    "preferred_categories": [],
                    "preferred_clusters": [],
                    "source_counts": {},
                    "source_overlap_counts": {},
                    "merged_source_counts": {},
                    "sasrec_sequence_length": 0,
                    "swing_itemcf_seed_count": 0,
                    "has_transcript": False,
                    "speech_category_candidates_used": False,
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
                    diverse_candidates.extend(
                        ungrouped_candidates[: 3 - added_in_round]
                    )
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
            sasrec_params = sum(
                p.numel() for p in self.sasrec_engine.model.parameters()
            )

        return {
            "is_initialized": self.is_initialized,
            "last_model_update": self.last_model_update,
            "cf_trained": self.cf_engine.is_trained,
            "cf_model_version": self.cf_engine.model_version,
            "cf_users": len(self.cf_engine.user_mapping),
            "cf_items": len(self.cf_engine.item_mapping),
            "cf_embedding_dim": self.config.tt_embedding_dim,
            "cf_index_size": self.cf_engine.cf_index.ntotal
            if self.cf_engine.cf_index
            else 0,
            "cf_cold_start_enabled": self.config.enable_cf_cold_start_bootstrap,
            "cf_cold_start_synthetic_items": len(
                self.cf_engine.synthetic_item_embeddings
            ),
            "cf_cold_start_overlay_version": self.cf_engine.cold_start_overlay_version,
            "cf_model_parameters": model_params,
            "sasrec_enabled": self.config.enable_sasrec,
            "sasrec_trained": self.sasrec_engine.is_trained,
            "sasrec_model_version": self.sasrec_engine.model_version,
            "sasrec_items": len(self.sasrec_engine.product_to_id),
            "sasrec_model_parameters": sasrec_params,
            "content_cluster_pools_enabled": self.config.enable_content_cluster_pools,
            "content_cluster_model_version": self.loaded_content_cluster_version,
            "content_cluster_assignments": len(
                getattr(self.vector_search, "content_cluster_assignments", {}) or {}
            ),
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
                "trending_data_available": len(self.trending_engine.trending_scores)
                > 0,
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
