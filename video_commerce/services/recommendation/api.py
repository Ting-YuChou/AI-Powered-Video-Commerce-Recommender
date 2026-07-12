"""
Dedicated recommendation-serving API.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import time
import traceback
import uuid
from collections import ChainMap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import httpx
from fastapi import Body, HTTPException, Request, Response
import torch

from video_commerce.common.cache_codec import json_dumps, json_loads
from video_commerce.common.config import Config
from video_commerce.common.feature_history_contracts import (
    RANKING_LTR_FEATURE_DEFINITION_VERSION,
    payload_sha256,
)
from video_commerce.data_plane.feature_store import FeatureStore
from video_commerce.data_plane.kafka_client import close_kafka, init_kafka
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.common.models import (
    CandidateProduct,
    ProductRecommendation,
    RecommendationRequest,
    UserFeatures,
)
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.ranking_runtime.ranking_batcher import (
    RankingBatcher,
    RankingQueueFullError,
    RankingQueueTimeoutError,
)
from video_commerce.ranking_runtime.ranking_coordinator_client import (
    RankingCoordinatorClientPool,
    RankingCoordinatorError,
)
from video_commerce.ml.ranking import RankingModel
from video_commerce.ml.ranking_history import (
    RANKING_HISTORY_CONTEXT_KEY,
    build_ranking_history_context,
    history_context_profile,
    ranking_history_config_from_settings,
)
from video_commerce.ml.recommender import RecommendationEngine
from video_commerce.common.service_common import (
    build_health_response,
    build_liveness_payload,
    build_metrics_response,
    build_readiness_response,
    component_health,
    configure_service_logging,
    create_service_app,
    require_internal_service_auth,
    RoundRobinAsyncClientPool,
)
from video_commerce.ml.slate_diversity import select_mmr_recommendations
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.ml.vector_search import VectorSearchEngine
from video_commerce.common.telemetry import inject_http_headers

logger = logging.getLogger(__name__)

app = create_service_app(
    title="Recommendation Service",
    description="Dedicated online recommendation serving service",
    service_name="recommendation-service",
)

feature_store: Optional[FeatureStore] = None
vector_search: Optional[VectorSearchEngine] = None
recommendation_engine: Optional[RecommendationEngine] = None
ranking_model: Optional[RankingModel] = None
ranking_batcher: Optional[RankingBatcher] = None
ranking_client_pool: Optional[RoundRobinAsyncClientPool] = None
ranking_coordinator_client_pool: Optional[RankingCoordinatorClientPool] = None
kafka_manager = None
system_store: Optional[SystemStore] = None
ranking_checkpoint_sync_task: Optional[asyncio.Task] = None
known_user_snapshot_refresh_task: Optional[asyncio.Task] = None
content_feature_snapshot_refresh_task: Optional[asyncio.Task] = None
object_storage: Optional[ObjectStorage] = None
artifact_manager: Optional[ModelArtifactManager] = None
best_effort_task_queue = None
_recommendation_singleflight: Dict[str, asyncio.Future] = {}
_recommendation_singleflight_lock = asyncio.Lock()
_serving_version_context_cache: Optional[Dict[str, Any]] = None
_serving_version_context_paths: Optional[
    Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]
] = None
_IMPRESSION_CONTEXT_KEYS = {
    "session_id",
    "surface",
    "device",
    "time_of_day",
    "location",
    "priority",
    "demo_mode",
    "content_id",
    "category",
    "request_category",
    "recommendation_count",
}


class _BestEffortTaskQueue:
    def __init__(self, runtime) -> None:
        cache_config = runtime.config.cache_config
        self.runtime = runtime
        self.queue: asyncio.Queue[Optional[Tuple[str, Any, float]]] = asyncio.Queue(
            maxsize=max(
                1, int(getattr(cache_config, "background_task_queue_size", 8192))
            )
        )
        self.worker_count = max(
            1, int(getattr(cache_config, "background_task_worker_count", 4))
        )
        self._workers: List[asyncio.Task] = []
        self._closing = False
        self._last_warning_at: Dict[Tuple[str, str], float] = {}

    async def start(self) -> None:
        if self._workers:
            return
        self._workers = [
            asyncio.create_task(
                self._run_worker(worker_id),
                name=f"recommendation-best-effort-{worker_id}",
            )
            for worker_id in range(self.worker_count)
        ]
        self._record_depth()

    async def close(self) -> None:
        self._closing = True
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not None:
                _close_awaitable(item[1])
            self.queue.task_done()
        self._record_depth()

    def enqueue(self, task_name: str, awaitable, timeout_seconds: float) -> bool:
        if self._closing:
            _close_awaitable(awaitable)
            self._record(task_name, "dropped_closing")
            return False
        try:
            self.queue.put_nowait((task_name, awaitable, timeout_seconds))
        except asyncio.QueueFull:
            _close_awaitable(awaitable)
            self._record(task_name, "dropped_queue_full")
            self._log_warning(task_name, "queue_full", "best_effort_queue_full")
            return False
        self._record(task_name, "enqueued")
        self._record_depth()
        return True

    async def _run_worker(self, worker_id: int) -> None:
        while True:
            item = await self.queue.get()
            self._record_depth()
            if item is None:
                self.queue.task_done()
                break
            task_name, awaitable, timeout_seconds = item
            try:
                await asyncio.wait_for(awaitable, timeout=timeout_seconds)
                self._record(task_name, "success")
            except asyncio.TimeoutError:
                self._record(task_name, "timeout")
                self._log_warning(task_name, "timeout", f"{task_name}_timed_out")
            except asyncio.CancelledError:
                _close_awaitable(awaitable)
                raise
            except Exception as exc:
                self._record(task_name, "error")
                self._log_warning(task_name, "error", f"{task_name}_failed: {exc}")
            finally:
                self.queue.task_done()
                self._record_depth()

    def _record(self, task_name: str, status: str) -> None:
        observability = getattr(self.runtime, "observability", None)
        if observability and hasattr(observability, "record_best_effort_task"):
            observability.record_best_effort_task(task_name, status)

    def _record_depth(self) -> None:
        observability = getattr(self.runtime, "observability", None)
        if observability and hasattr(observability, "set_best_effort_queue_depth"):
            observability.set_best_effort_queue_depth(self.queue.qsize())

    def _log_warning(self, task_name: str, reason: str, message: str) -> None:
        key = (task_name, reason)
        now = time.monotonic()
        last = self._last_warning_at.get(key, 0.0)
        if now - last < 30.0:
            return
        self._last_warning_at[key] = now
        logger.warning(message)


def _close_awaitable(awaitable) -> None:
    close = getattr(awaitable, "close", None)
    if callable(close):
        close()


@app.middleware("http")
async def internal_auth(request: Request, call_next):
    require_internal_service_auth(request, app.state.runtime)
    return await call_next(request)


def _build_candidate_cache_context(
    payload: RecommendationRequest, k_per_source: int
) -> dict:
    """Build a coarse retrieval context so candidate cache can be reused."""
    context = payload.context or {}
    return {
        "content_id": payload.content_id,
        "device": context.get("device"),
        "page": context.get("page"),
        "category": context.get("product_category") or context.get("category"),
        "k_per_source": k_per_source,
    }


def _build_recommendation_cache_context(
    payload: RecommendationRequest,
    *,
    current_time: Optional[float] = None,
    recommendation_config: Optional[Any] = None,
) -> dict:
    """Build an exact-enough cache context from fields used by serving logic."""
    context = payload.context or {}
    current_time = current_time or time.time()
    return {
        **_build_candidate_cache_context(payload, min(payload.k * 10, 500)),
        "k": payload.k,
        "device": context.get("device"),
        "session_position": context.get("session_position", 1),
        "time_on_page": context.get("time_on_page", 0),
        "ranking_time_hour": int(current_time // 3600),
        "slate_diversity": _build_slate_diversity_cache_context(recommendation_config),
    }


def _build_slate_diversity_cache_context(
    recommendation_config: Optional[Any],
) -> Dict[str, Any]:
    if not _is_mmr_slate_diversity_enabled(recommendation_config):
        return {"enabled": False}
    return {
        "enabled": True,
        "method": "mmr",
        "lambda": round(float(getattr(recommendation_config, "mmr_lambda", 0.8)), 6),
        "pool_multiplier": int(
            getattr(recommendation_config, "mmr_rerank_pool_multiplier", 5)
        ),
        "min_pool_size": int(
            getattr(recommendation_config, "mmr_min_rerank_pool_size", 50)
        ),
        "max_pool_size": int(
            getattr(recommendation_config, "mmr_max_rerank_pool_size", 100)
        ),
    }


def _is_mmr_slate_diversity_enabled(recommendation_config: Optional[Any]) -> bool:
    if recommendation_config is None:
        return False
    method = str(
        getattr(recommendation_config, "slate_diversity_method", "mmr") or ""
    ).lower()
    return (
        bool(getattr(recommendation_config, "enable_slate_diversity", False))
        and method == "mmr"
    )


def _is_realtime_window_features_enabled(runtime: Any) -> bool:
    config = getattr(runtime, "config", None)
    ranking_config = getattr(config, "ranking_config", None)
    return bool(getattr(ranking_config, "realtime_window_features_enabled", False))


def _is_ranking_history_embeddings_enabled(runtime: Any) -> bool:
    config = getattr(runtime, "config", None)
    ranking_config = getattr(config, "ranking_config", None)
    return bool(getattr(ranking_config, "history_embeddings_enabled", False))


def _ranking_history_item_embedding_map() -> Mapping[str, Any]:
    if recommendation_engine is None:
        return {}
    cf_engine = getattr(recommendation_engine, "cf_engine", None)
    if cf_engine is None:
        return {}
    trained_embeddings = getattr(cf_engine, "trained_item_embeddings", None)
    synthetic_embeddings = getattr(cf_engine, "synthetic_item_embeddings", None)
    maps = [
        item_map
        for item_map in (synthetic_embeddings, trained_embeddings)
        if isinstance(item_map, Mapping)
    ]
    if not maps:
        return {}
    return ChainMap(*maps)


def _ranking_history_serving_context(runtime: Any) -> Dict[str, Any]:
    ranking_config = getattr(getattr(runtime, "config", None), "ranking_config", None)
    if ranking_config is None or not getattr(
        ranking_config,
        "history_embeddings_enabled",
        False,
    ):
        return {"enabled": False}
    return {
        "enabled": True,
        "click_last_n": int(getattr(ranking_config, "history_click_last_n", 20)),
        "cart_last_n": int(getattr(ranking_config, "history_cart_last_n", 20)),
        "purchase_last_n": int(getattr(ranking_config, "history_purchase_last_n", 20)),
        "embedding_dim": int(getattr(ranking_config, "history_embedding_dim", 128)),
        "click_scale": round(
            float(getattr(ranking_config, "history_click_scale", 1.0)), 6
        ),
        "cart_scale": round(
            float(getattr(ranking_config, "history_cart_scale", 1.25)), 6
        ),
        "purchase_scale": round(
            float(getattr(ranking_config, "history_purchase_scale", 1.75)),
            6,
        ),
    }


def _should_initialize_local_ranker(runtime) -> bool:
    topology = runtime.config.service_topology_config
    has_remote_ranker = bool(ranking_coordinator_client_pool or ranking_client_pool)
    return not has_remote_ranker or bool(topology.ranking_service_fallback_enabled)


def _calculate_mmr_rerank_pool_size(
    *,
    requested_k: int,
    candidate_count: int,
    recommendation_config: Any,
) -> int:
    multiplier = max(
        1, int(getattr(recommendation_config, "mmr_rerank_pool_multiplier", 5))
    )
    min_pool_size = max(
        1, int(getattr(recommendation_config, "mmr_min_rerank_pool_size", 50))
    )
    max_pool_size = max(
        min_pool_size,
        int(getattr(recommendation_config, "mmr_max_rerank_pool_size", 100)),
    )
    desired_pool_size = max(requested_k * multiplier, min_pool_size)
    return min(desired_pool_size, max_pool_size, candidate_count)


def _user_feature_cache_token(user_features: UserFeatures) -> Dict[str, Any]:
    """Return the personalization inputs that make a cached result fresh enough."""
    has_personalization_signal = (
        user_features.total_interactions > 0
        or bool(user_features.preferred_categories)
        or user_features.click_through_rate > 0
        or user_features.conversion_rate > 0
    )
    return {
        "total_interactions": int(user_features.total_interactions),
        "avg_session_length": round(float(user_features.avg_session_length), 3),
        "preferred_categories": list(user_features.preferred_categories),
        "price_sensitivity": round(float(user_features.price_sensitivity), 6),
        "click_through_rate": round(float(user_features.click_through_rate), 6),
        "conversion_rate": round(float(user_features.conversion_rate), 6),
        "last_active": int(user_features.last_active)
        if has_personalization_signal
        else 0,
    }


def _default_user_sequence_token() -> Dict[str, Any]:
    return {
        "length": 0,
        "latest_event_id": None,
        "latest_occurred_at": 0,
        "latest_product_id": None,
        "latest_action": None,
    }


def _build_cache_freshness_context(
    serving_versions: Dict[str, Any],
    user_feature_token: Dict[str, Any],
    user_sequence_token: Optional[Dict[str, Any]],
    *,
    content_feature_token: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "serving_versions": serving_versions,
        "user_feature_token": user_feature_token,
        "user_sequence_token": user_sequence_token or _default_user_sequence_token(),
        "content_feature_token": content_feature_token
        or _content_feature_cache_token(
            None,
            None,
            None,
        ),
    }


def _content_feature_cache_token(
    content_id: Optional[str],
    content_features: Optional[Any],
    content_processed_at: Optional[float],
) -> Dict[str, Any]:
    if not content_id:
        return {
            "content_id": None,
            "content_features_present": False,
            "content_features_created_at": 0,
            "content_status_updated_at": 0,
        }
    return {
        "content_id": content_id,
        "content_features_present": content_features is not None,
        "content_features_created_at": round(
            float(getattr(content_features, "created_at", 0.0) or 0.0),
            6,
        ),
        "content_status_updated_at": round(float(content_processed_at or 0.0), 6),
    }


def _serving_version_context(runtime) -> Dict[str, Any]:
    global _serving_version_context_cache, _serving_version_context_paths

    paths = _serving_version_paths(runtime)
    if (
        _serving_version_context_cache is None
        or _serving_version_context_paths != paths
    ):
        _refresh_serving_version_context(runtime)
    return _serving_version_context_cache or {}


def _refresh_serving_version_context(runtime) -> Dict[str, Any]:
    global _serving_version_context_cache, _serving_version_context_paths

    _serving_version_context_paths = _serving_version_paths(runtime)
    _serving_version_context_cache = _build_serving_version_context_uncached(runtime)
    return _serving_version_context_cache


def _serving_version_paths(
    runtime,
) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
]:
    cluster_metadata_path, cluster_centroids_path = _content_cluster_artifact_paths(
        runtime
    )
    return (
        runtime.config.model_config.ranking_model_path,
        runtime.config.vector_config.index_path,
        runtime.config.recommendation_config.sasrec_checkpoint_path,
        runtime.config.recommendation_config.sasrec_vocab_path,
        getattr(runtime.config.recommendation_config, "swing_itemcf_index_path", None),
        cluster_metadata_path,
        cluster_centroids_path,
    )


def _build_serving_version_context_uncached(runtime) -> Dict[str, Any]:
    ranking_path = runtime.config.model_config.ranking_model_path
    vector_path = runtime.config.vector_config.index_path
    swing_itemcf_index_path = getattr(
        runtime.config.recommendation_config,
        "swing_itemcf_index_path",
        None,
    )
    sasrec_checkpoint_path = runtime.config.recommendation_config.sasrec_checkpoint_path
    sasrec_vocab_path = runtime.config.recommendation_config.sasrec_vocab_path
    cluster_metadata_path, cluster_centroids_path = _content_cluster_artifact_paths(
        runtime
    )
    return {
        "ranking_model": ranking_model.model_version if ranking_model else None,
        "ranking_checkpoint_mtime": _safe_file_mtime(ranking_path),
        "two_tower_model": (
            recommendation_engine.loaded_two_tower_version
            if recommendation_engine
            else None
        ),
        "cf_model": (
            recommendation_engine.cf_engine.model_version
            if recommendation_engine
            else None
        ),
        "cf_cold_start_overlay": (
            getattr(recommendation_engine.cf_engine, "cold_start_overlay_version", None)
            if recommendation_engine
            else None
        ),
        "cf_new_item_pool": (
            getattr(recommendation_engine.cf_engine, "new_item_pool_version", None)
            if recommendation_engine
            else None
        ),
        "sasrec_model": (
            getattr(recommendation_engine, "loaded_sasrec_version", None)
            if recommendation_engine
            else None
        ),
        "sasrec_checkpoint_mtime": _safe_file_mtime(sasrec_checkpoint_path),
        "sasrec_vocab_mtime": _safe_file_mtime(sasrec_vocab_path),
        "swing_itemcf_model": (
            getattr(recommendation_engine, "loaded_swing_itemcf_version", None)
            if recommendation_engine
            else None
        ),
        "swing_itemcf_index_mtime": _safe_file_mtime(swing_itemcf_index_path),
        "content_cluster_model": (
            getattr(recommendation_engine, "loaded_content_cluster_version", None)
            if recommendation_engine
            else None
        ),
        "content_cluster_metadata_mtime": (
            _safe_file_mtime(cluster_metadata_path) if cluster_metadata_path else None
        ),
        "content_cluster_centroids_mtime": (
            _safe_file_mtime(cluster_centroids_path) if cluster_centroids_path else None
        ),
        "content_cluster": _content_cluster_serving_version_context(),
        "vector_index_mtime": _safe_file_mtime(vector_path),
        "catalog": _catalog_serving_version_context(),
        "ranking_history_embeddings": _ranking_history_serving_context(runtime),
    }


def _content_cluster_artifact_paths(runtime) -> Tuple[Optional[str], Optional[str]]:
    recommendation_config = runtime.config.recommendation_config
    metadata_path = getattr(
        recommendation_config,
        "content_cluster_metadata_path",
        None,
    )
    centroids_path = getattr(
        recommendation_config,
        "content_cluster_centroids_path",
        None,
    )
    if metadata_path and centroids_path:
        return metadata_path, centroids_path
    cf_index_path = getattr(recommendation_config, "cf_index_path", None)
    if not cf_index_path:
        return metadata_path, centroids_path
    base_dir = Path(cf_index_path).parent
    return (
        metadata_path or str(base_dir / "content_clusters.metadata.json"),
        centroids_path or str(base_dir / "content_clusters.centroids.npz"),
    )


def _content_cluster_serving_version_context() -> Dict[str, Any]:
    if vector_search is None:
        return {
            "cluster_model_version": None,
            "num_clusters": 0,
            "assignment_count": 0,
            "product_count": 0,
            "created_at": 0,
        }
    if hasattr(vector_search, "get_content_cluster_version_context"):
        return vector_search.get_content_cluster_version_context()
    return {
        "cluster_model_version": getattr(
            vector_search, "content_cluster_model_version", None
        ),
        "num_clusters": 0,
        "assignment_count": len(
            getattr(vector_search, "content_cluster_assignments", {}) or {}
        ),
        "product_count": 0,
        "created_at": 0,
    }


def _candidate_generation_interaction_hint(
    engine: Any,
    user_features: UserFeatures,
    user_sequence_token: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    if int((user_sequence_token or {}).get("length", 0) or 0) != 0:
        return None

    config = getattr(engine, "config", None)
    sasrec_requires_interactions = bool(
        getattr(config, "enable_sasrec", False)
        and getattr(getattr(engine, "sasrec_engine", None), "is_trained", False)
    )
    swing_requires_interactions = bool(
        getattr(config, "enable_swing_itemcf", False)
        and getattr(getattr(engine, "swing_itemcf_engine", None), "is_trained", False)
    )
    if sasrec_requires_interactions or swing_requires_interactions:
        return None

    if int(getattr(user_features, "total_interactions", 0) or 0) <= 0:
        return []
    return None


def _catalog_serving_version_context() -> Dict[str, Any]:
    if vector_search is None:
        return {"catalog_version": None, "last_updated": None, "product_count": 0}
    if hasattr(vector_search, "get_catalog_version_context"):
        return vector_search.get_catalog_version_context()
    return {
        "catalog_version": int(getattr(vector_search, "last_updated", 0) or 0),
        "last_updated": int(getattr(vector_search, "last_updated", 0) or 0),
        "product_count": len(getattr(vector_search, "product_metadata", {}) or {}),
    }


def _safe_file_mtime(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    try:
        return int(Path(path).stat().st_mtime)
    except OSError:
        return None


async def _join_or_create_recommendation_singleflight(
    key: str,
) -> Tuple[asyncio.Future, bool]:
    loop = asyncio.get_running_loop()
    async with _recommendation_singleflight_lock:
        existing = _recommendation_singleflight.get(key)
        if existing is not None:
            return existing, False
        future = loop.create_future()
        _recommendation_singleflight[key] = future
        return future, True


async def _resolve_recommendation_singleflight(
    key: Optional[str],
    future: Optional[asyncio.Future],
    *,
    result: Optional[Dict[str, Any]] = None,
    exception: Optional[BaseException] = None,
) -> None:
    if key is None or future is None:
        return
    async with _recommendation_singleflight_lock:
        if _recommendation_singleflight.get(key) is future:
            _recommendation_singleflight.pop(key, None)
    if future.done():
        return
    if exception is not None:
        future.set_exception(exception)
        future.add_done_callback(_consume_singleflight_exception)
    else:
        future.set_result(result or {})


def _build_singleflight_result(
    recommendations: List[Any],
    *,
    total_candidates: int,
    content_processed: bool,
    fallback: bool = False,
    fallback_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "recommendations": recommendations,
        "total_candidates": total_candidates,
        "content_processed": content_processed,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
    }


def _consume_singleflight_exception(future: asyncio.Future) -> None:
    try:
        future.exception()
    except (asyncio.CancelledError, asyncio.InvalidStateError):
        pass


def _is_recommendable_product(metadata: Optional[Dict[str, Any]]) -> bool:
    if not metadata:
        return True
    return not (
        metadata.get("active") is False
        or metadata.get("in_stock") is False
        or metadata.get("deleted") is True
        or metadata.get("is_deleted") is True
    )


def _filter_recommendable_candidates(
    candidates: List[Any],
    metadata_map: Dict[str, Dict[str, Any]],
) -> List[Any]:
    return [
        candidate
        for candidate in candidates
        if _is_recommendable_product(metadata_map.get(candidate.product_id))
    ]


def _count_candidate_source_tokens(candidates: List[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for candidate in candidates:
        for source in str(getattr(candidate, "source", None) or "unknown").split("+"):
            source = source or "unknown"
            counts[source] = counts.get(source, 0) + 1
    return counts


def _count_ranked_source_tokens(
    recommendations: List[Any],
    candidate_source_by_product: Dict[str, str],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for recommendation in recommendations:
        product_id = (
            recommendation.get("product_id")
            if isinstance(recommendation, dict)
            else getattr(recommendation, "product_id", None)
        )
        source_value = candidate_source_by_product.get(product_id, "unknown")
        for source in str(source_value or "unknown").split("+"):
            source = source or "unknown"
            counts[source] = counts.get(source, 0) + 1
    return counts


def _recommendation_field(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _bounded_impression_context_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value[:512] if isinstance(value, str) else value
    if isinstance(value, list):
        bounded = []
        for item in value[:20]:
            if isinstance(item, (str, int, float, bool)) or item is None:
                bounded.append(item[:512] if isinstance(item, str) else item)
        return bounded
    return None


def _build_impression_context_snapshot(
    context: Optional[Dict[str, Any]],
    *,
    content_id: Optional[str],
) -> Dict[str, Any]:
    source = context or {}
    snapshot: Dict[str, Any] = {}
    for key in _IMPRESSION_CONTEXT_KEYS:
        if key not in source:
            continue
        bounded = _bounded_impression_context_value(source.get(key))
        if bounded is not None:
            snapshot[key] = bounded
    if content_id is not None:
        snapshot["content_id"] = content_id
    return snapshot


def _candidate_score_snapshot(candidate: Optional[CandidateProduct]) -> Dict[str, Any]:
    if candidate is None:
        return {}
    return {
        "collaborative_score": candidate.collaborative_score,
        "content_similarity_score": candidate.content_similarity_score,
        "popularity_score": candidate.popularity_score,
        "combined_score": candidate.combined_score,
    }


def _build_displayed_item_snapshots(
    recommendations: List[Any],
    *,
    candidate_by_product: Dict[str, CandidateProduct],
    product_metadata_map: Dict[str, Dict[str, Any]],
    max_items: int,
    user_id: Optional[str] = None,
    user_features: Optional[UserFeatures] = None,
    observation_context: Optional[Dict[str, Any]] = None,
    as_of_ts: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build bounded top-k item snapshots for impression-backed LTR."""
    if max_items <= 0:
        return []

    displayed_items: List[Dict[str, Any]] = []
    for position, recommendation in enumerate(recommendations[:max_items], start=1):
        product_id = _recommendation_field(recommendation, "product_id")
        if not product_id:
            continue
        product_id = str(product_id)
        candidate = candidate_by_product.get(product_id)
        metadata = product_metadata_map.get(product_id, {})
        source = (
            getattr(candidate, "source", None)
            or _recommendation_field(recommendation, "source")
            or _recommendation_field(recommendation, "candidate_source")
        )
        scores = _candidate_score_snapshot(candidate)
        cached_scores = _recommendation_field(recommendation, "candidate_scores", {})
        if isinstance(cached_scores, dict):
            scores.update({key: value for key, value in cached_scores.items()})
        scores.update(
            {
                "ranking_score": _recommendation_field(
                    recommendation,
                    "ranking_score",
                    scores.get("combined_score"),
                ),
                "confidence_score": _recommendation_field(
                    recommendation,
                    "confidence_score",
                    None,
                ),
            }
        )
        feature_snapshot = {
            **dict(metadata),
            "price": _recommendation_field(
                recommendation,
                "price",
                metadata.get("price"),
            ),
            "category": _recommendation_field(
                recommendation,
                "category",
                metadata.get("category"),
            ),
            "brand": _recommendation_field(
                recommendation,
                "brand",
                metadata.get("brand"),
            ),
            "candidate_source": source,
        }
        item_snapshot = {
            "product_id": product_id,
            "position": position,
            "candidate_source": source,
            "source": source,
            "collaborative_score": scores.get("collaborative_score"),
            "content_similarity_score": scores.get("content_similarity_score"),
            "popularity_score": scores.get("popularity_score"),
            "combined_score": scores.get("combined_score"),
            "ranking_score": scores.get("ranking_score"),
            "confidence_score": scores.get("confidence_score"),
            "price": _recommendation_field(
                recommendation,
                "price",
                metadata.get("price"),
            ),
            "category": _recommendation_field(
                recommendation,
                "category",
                metadata.get("category"),
            ),
            "brand": _recommendation_field(
                recommendation,
                "brand",
                metadata.get("brand"),
            ),
            "feature_snapshot": feature_snapshot,
            "item_snapshot_complete": bool(metadata),
            "scores": scores,
        }
        if as_of_ts is not None and user_id and user_features is not None:
            user_feature_snapshot = _recommendation_item_payload(user_features)
            bundle = {
                "as_of_ts": float(as_of_ts),
                "candidate_features": scores,
                "context": dict(observation_context or {}),
                "feature_definition_version": RANKING_LTR_FEATURE_DEFINITION_VERSION,
                "product_id": product_id,
                "product_metadata": feature_snapshot,
                "user_features": user_feature_snapshot,
                "user_id": user_id,
            }
            item_snapshot.update(
                {
                    "as_of_ts": float(as_of_ts),
                    "feature_definition_version": RANKING_LTR_FEATURE_DEFINITION_VERSION,
                    "feature_bundle_hash": payload_sha256(bundle),
                }
            )
        displayed_items.append(item_snapshot)
    return displayed_items


def _build_rejected_candidate_snapshots(
    candidates: List[CandidateProduct],
    recommendations: List[Any],
    *,
    product_metadata_map: Dict[str, Dict[str, Any]],
    max_items: int,
) -> List[Dict[str, Any]]:
    """Build bounded ranker-rejected snapshots for weak retrieval negatives."""
    if max_items <= 0:
        return []

    returned_product_ids = {
        str(product_id)
        for product_id in (
            _recommendation_field(recommendation, "product_id")
            for recommendation in recommendations
        )
        if product_id
    }
    rejected_items: List[Dict[str, Any]] = []
    for position, candidate in enumerate(candidates, start=1):
        if len(rejected_items) >= max_items:
            break
        product_id = str(candidate.product_id or "")
        if not product_id or product_id in returned_product_ids:
            continue
        metadata = product_metadata_map.get(product_id, {})
        scores = _candidate_score_snapshot(candidate)
        rejected_items.append(
            {
                "product_id": product_id,
                "position": position,
                "candidate_source": candidate.source,
                "source": candidate.source,
                "feature_snapshot": {
                    "price": metadata.get("price"),
                    "category": metadata.get("category"),
                    "brand": metadata.get("brand"),
                    "candidate_source": candidate.source,
                },
                "scores": scores,
            }
        )
    return rejected_items


def _attach_displayed_item_snapshot_metadata(
    recommendation_payloads: List[Dict[str, Any]],
    displayed_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    snapshots_by_product = {
        item.get("product_id"): item
        for item in displayed_items
        if item.get("product_id")
    }
    enriched_payloads: List[Dict[str, Any]] = []
    for payload in recommendation_payloads:
        enriched = dict(payload)
        snapshot = snapshots_by_product.get(enriched.get("product_id"))
        if snapshot:
            source = snapshot.get("source") or snapshot.get("candidate_source")
            if source is not None:
                enriched.setdefault("source", source)
                enriched.setdefault("candidate_source", source)
            enriched.setdefault("candidate_scores", snapshot.get("scores") or {})
        enriched_payloads.append(enriched)
    return enriched_payloads


def _recommendation_product_ids(recommendations: List[Any]) -> List[str]:
    product_ids = []
    for item in recommendations:
        product_id = _recommendation_field(item, "product_id")
        if product_id:
            product_ids.append(str(product_id))
    return product_ids


@app.on_event("startup")
async def startup_event():
    global feature_store, vector_search, recommendation_engine, ranking_model, ranking_batcher, ranking_client_pool, ranking_coordinator_client_pool, kafka_manager
    global system_store
    global ranking_checkpoint_sync_task, known_user_snapshot_refresh_task
    global content_feature_snapshot_refresh_task
    global content_feature_snapshot_refresh_task
    global object_storage, artifact_manager
    global best_effort_task_queue
    global _serving_version_context_cache, _serving_version_context_paths

    runtime = app.state.runtime
    _serving_version_context_cache = None
    _serving_version_context_paths = None
    runtime.config = Config()
    configure_service_logging(runtime)
    best_effort_task_queue = _BestEffortTaskQueue(runtime)
    await best_effort_task_queue.start()
    _configure_torch_runtime(runtime)
    ranking_urls = RoundRobinAsyncClientPool.parse_urls(
        runtime.config.service_topology_config.ranking_service_urls,
        fallback=runtime.config.service_topology_config.ranking_service_url or None,
    )
    if (
        runtime.config.service_topology_config.ranking_single_coordinator_enabled
        and len(ranking_urls) > 1
    ):
        logger.info(
            "ranking_single_coordinator_enabled",
            extra={
                "configured_urls": len(ranking_urls),
                "selected_url": ranking_urls[0],
            },
        )
        ranking_urls = ranking_urls[:1]
    if ranking_urls:
        topology = runtime.config.service_topology_config
        ranking_client_pool = RoundRobinAsyncClientPool(
            ranking_urls,
            timeout=httpx.Timeout(
                connect=topology.proxy_connect_timeout_seconds,
                read=topology.proxy_read_timeout_seconds,
                write=topology.proxy_write_timeout_seconds,
                pool=topology.proxy_pool_timeout_seconds,
            ),
            limits=httpx.Limits(
                max_connections=topology.proxy_max_connections,
                max_keepalive_connections=topology.proxy_max_keepalive_connections,
                keepalive_expiry=topology.proxy_keepalive_expiry_seconds,
            ),
        )
    topology = runtime.config.service_topology_config
    if (
        topology.ranking_coordinator_direct_enabled
        and topology.ranking_coordinator_host
    ):
        ranking_coordinator_client_pool = RankingCoordinatorClientPool(
            topology.ranking_coordinator_host,
            topology.ranking_coordinator_port,
            pool_size=topology.ranking_coordinator_client_pool_size,
            connect_timeout_seconds=topology.ranking_coordinator_connect_timeout_seconds,
            request_timeout_seconds=topology.ranking_coordinator_request_timeout_seconds,
        )

    feature_store = FeatureStore(
        runtime.config.redis_config, runtime.config.cache_config
    )
    feature_store.configure_known_user_snapshot(
        enabled=runtime.config.recommendation_config.known_user_snapshot_enabled,
        max_users=runtime.config.recommendation_config.known_user_snapshot_max_users,
    )
    feature_store.configure_content_feature_snapshot(
        enabled=runtime.config.recommendation_config.content_features_snapshot_enabled,
        max_items=runtime.config.recommendation_config.content_features_snapshot_max_items,
    )
    await feature_store.initialize()
    if runtime.config.recommendation_config.known_user_snapshot_enabled:
        await feature_store.refresh_known_user_snapshot()
        known_user_snapshot_refresh_task = asyncio.create_task(
            _periodic_known_user_snapshot_refresh(runtime),
            name="known-user-snapshot-refresh",
        )
    if runtime.config.recommendation_config.content_features_snapshot_enabled:
        await feature_store.refresh_content_feature_snapshot()
        content_feature_snapshot_refresh_task = asyncio.create_task(
            _periodic_content_feature_snapshot_refresh(runtime),
            name="content-feature-snapshot-refresh",
        )

    if runtime.config.database_config.enable:
        system_store = SystemStore(
            runtime.config.database_config,
            observability=runtime.observability,
        )
        await system_store.initialize()

    object_storage = ObjectStorage(runtime.config.object_storage_config)
    await object_storage.initialize()
    artifact_manager = ModelArtifactManager(
        system_store=system_store,
        object_storage=object_storage,
        model_config=runtime.config.model_config,
        recommendation_config=runtime.config.recommendation_config,
    )

    vector_search = VectorSearchEngine(runtime.config.vector_config)
    await vector_search.load_index()
    feature_store.prime_product_metadata_memory_cache(vector_search.product_metadata)
    if runtime.config.recommendation_config.preload_product_metadata_on_startup:
        await feature_store.store_product_metadata_batch(vector_search.product_metadata)
    recommendation_engine = RecommendationEngine(
        feature_store,
        vector_search,
        runtime.config.recommendation_config,
        artifact_manager=artifact_manager,
        training_sequence_lookback_days=(
            runtime.config.database_config.training_sequence_lookback_days
        ),
    )
    await recommendation_engine.load_serving_state()

    if _should_initialize_local_ranker(runtime):
        ranking_model = RankingModel(
            runtime.config.ranking_config,
            observability=runtime.observability,
        )
        ranking_checkpoint = None
        if artifact_manager:
            ranking_checkpoint = await artifact_manager.sync_latest_ranking_checkpoint()
        await ranking_model.load_model(runtime.config.model_config.ranking_model_path)
        if ranking_checkpoint:
            ranking_model.model_version = ranking_checkpoint.model_version
        ranking_model.enable_profiling_logs = (
            runtime.config.monitoring_config.enable_profiling_logs
        )
        ranking_model.profiling_log_min_duration_ms = (
            runtime.config.monitoring_config.profiling_log_min_duration_ms
        )
    else:
        ranking_model = None
    _refresh_serving_version_context(runtime)
    if ranking_model is not None and (
        not ranking_client_pool
        or runtime.config.service_topology_config.ranking_service_fallback_enabled
    ):
        ranking_batcher = RankingBatcher(
            ranking_model,
            runtime.config.ranking_config,
            observability=runtime.observability,
        )
        await ranking_batcher.start()
    if (
        ranking_model is not None
        and runtime.config.ranking_config.checkpoint_sync_interval_seconds > 0
    ):
        ranking_checkpoint_sync_task = asyncio.create_task(
            _periodic_ranking_checkpoint_sync(runtime),
            name="ranking-checkpoint-sync",
        )

    if runtime.config.kafka_config.enable:
        try:
            kafka_manager = await init_kafka(
                runtime.config.kafka_config,
                observability=runtime.observability,
            )
        except Exception as exc:
            logger.warning(f"Recommendation service Kafka init failed: {exc}")
            kafka_manager = None

    logger.info(
        "recommendation_service_worker_started",
        extra={
            "service": runtime.service_name,
            "process_id": os.getpid(),
            "configured_workers": runtime.config.service_topology_config.recommendation_workers,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
    )


@app.on_event("shutdown")
async def shutdown_event():
    global ranking_checkpoint_sync_task, known_user_snapshot_refresh_task
    global best_effort_task_queue, ranking_coordinator_client_pool

    if ranking_checkpoint_sync_task:
        ranking_checkpoint_sync_task.cancel()
        try:
            await ranking_checkpoint_sync_task
        except asyncio.CancelledError:
            pass
        ranking_checkpoint_sync_task = None
    if known_user_snapshot_refresh_task:
        known_user_snapshot_refresh_task.cancel()
        try:
            await known_user_snapshot_refresh_task
        except asyncio.CancelledError:
            pass
        known_user_snapshot_refresh_task = None
    if content_feature_snapshot_refresh_task:
        content_feature_snapshot_refresh_task.cancel()
        try:
            await content_feature_snapshot_refresh_task
        except asyncio.CancelledError:
            pass
        content_feature_snapshot_refresh_task = None
    if best_effort_task_queue:
        await best_effort_task_queue.close()
        best_effort_task_queue = None
    if ranking_batcher:
        await ranking_batcher.close()
    if ranking_coordinator_client_pool:
        await ranking_coordinator_client_pool.aclose()
        ranking_coordinator_client_pool = None
    if ranking_client_pool:
        await ranking_client_pool.aclose()
    if recommendation_engine:
        recommendation_engine.close()
    if feature_store:
        await feature_store.close()
    if system_store:
        await system_store.close()
    if kafka_manager:
        await close_kafka()


@app.get("/")
async def root():
    return {
        "service": "recommendation-service",
        "version": "1.0.0",
        "health": "/health",
        "livez": "/livez",
        "readyz": "/readyz",
    }


@app.get("/livez")
async def livez():
    return build_liveness_payload(app.state.runtime)


@app.get("/readyz")
async def readyz():
    runtime = app.state.runtime
    feature_store_health = await feature_store.health_check()
    ranking_health = (
        ranking_model.health_check()
        if ranking_model is not None
        else {"status": "healthy", "mode": "remote"}
    )
    vector_health = vector_search.health_check()
    database_health = {"status": "healthy", "response_time_ms": 0.0}
    if system_store:
        database_status = await system_store.health_check()
        database_health = {
            "status": database_status.status,
            "response_time_ms": database_status.response_time_ms,
            "error": database_status.error,
        }

    kafka_health = {"status": "healthy", "response_time_ms": 0.0}
    if kafka_manager:
        kafka_status = await kafka_manager.health_check()
        producer_health = kafka_status.get("producer", {})
        kafka_health = {
            "status": producer_health.get("status", "healthy"),
            "response_time_ms": 0.0,
            "error": None
            if producer_health.get("connected")
            else "Kafka producer unavailable",
        }
    elif runtime.config.kafka_config.enable:
        kafka_health = {"status": "degraded", "error": "Kafka producer unavailable"}
    ranking_service_health = (
        await _probe_ranking_service_ready(runtime)
        if ranking_client_pool
        else {"status": "healthy", "response_time_ms": 0.0}
    )

    return build_readiness_response(
        runtime,
        {
            "redis": feature_store_health,
            "database": database_health,
            "vector_search": vector_health,
            "ranking_model": ranking_health,
            "ranking_service": ranking_service_health,
            "kafka": kafka_health,
        },
    )


@app.post("/api/recommendations")
async def get_recommendations(
    http_request: Request,
    response: Response,
    payload: RecommendationRequest = Body(...),
):
    start_time = time.time()
    started_at = time.perf_counter()
    runtime = app.state.runtime
    service_topology_config = getattr(
        runtime.config,
        "service_topology_config",
        None,
    )
    max_inflight = int(
        getattr(service_topology_config, "max_inflight_recommendations", 0) or 0
    )
    if max_inflight > 0 and runtime.active_requests >= max_inflight:
        raise HTTPException(status_code=503, detail="Recommendation service overloaded")
    profile = {
        "process_id": getattr(http_request.state, "worker_process_id", os.getpid()),
        "worker_active_requests_at_entry": getattr(
            http_request.state,
            "worker_active_requests_at_entry",
            runtime.active_requests,
        ),
        "worker_handled_requests": getattr(
            http_request.state,
            "worker_handled_requests",
            runtime.handled_requests,
        ),
        "worker_max_active_requests_seen": runtime.max_active_requests,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "handler_queue_ms": round(
            (started_at - getattr(http_request.state, "request_started_at", started_at))
            * 1000,
            2,
        ),
        "cache_lookup_ms": 0.0,
        "candidate_cache_lookup_ms": 0.0,
        "content_features_ms": 0.0,
        "content_status_ms": 0.0,
        "user_features_ms": 0.0,
        "user_sequence_token_ms": 0.0,
        "candidate_generation_ms": 0.0,
        "candidate_cache_write_ms": 0.0,
        "metadata_lookup_ms": 0.0,
        "ranking_ms": 0.0,
        "history_embeddings_ms": 0.0,
        "history_embeddings_status": (
            "enabled" if _is_ranking_history_embeddings_enabled(runtime) else "disabled"
        ),
        "slate_diversity_enabled": _is_mmr_slate_diversity_enabled(
            runtime.config.recommendation_config
        ),
        "slate_diversity_method": getattr(
            runtime.config.recommendation_config,
            "slate_diversity_method",
            "mmr",
        ),
        "slate_diversity_ms": 0.0,
        "slate_diversity_pool_size": 0,
        "slate_diversity_selected_count": 0,
        "cache_write_ms": 0.0,
        "analytics_log_ms": 0.0,
        "kafka_schedule_ms": 0.0,
        "total_ms": 0.0,
        "candidate_count": 0,
        "ranked_count": 0,
        "candidate_source_counts": {},
        "candidate_source_counts_before_filter": {},
        "ranked_source_counts": {},
        "ranked_sasrec_count": 0,
        "cache_hit": False,
        "candidate_cache_hit": False,
        "metadata_cache_miss_count": 0,
        "filtered_unavailable_candidates": 0,
        "singleflight_joined": False,
        "singleflight_wait_ms": 0.0,
        "serving_path": "live_candidates_then_rank",
    }
    singleflight_key: Optional[str] = None
    singleflight_future: Optional[asyncio.Future] = None
    owns_singleflight = False

    try:
        content_features = None
        content_processed_at = None
        content_features_task = None
        content_processed_at_task = None
        if payload.content_id:
            content_features_task = asyncio.create_task(
                _timed_awaitable(
                    _bounded_hot_path_read(
                        runtime,
                        "content_features",
                        feature_store.get_content_features(payload.content_id),
                        None,
                    )
                ),
                name="recommendation-content-features",
            )
            content_processed_at_task = asyncio.create_task(
                _timed_awaitable(
                    _bounded_hot_path_read(
                        runtime,
                        "content_processed_at",
                        feature_store.get_content_processed_time(payload.content_id),
                        None,
                    )
                ),
                name="recommendation-content-processed-at",
            )

        stage_started = time.perf_counter()
        user_features, user_sequence_token = await _bounded_hot_path_read(
            runtime,
            "user_serving_context",
            feature_store.get_user_serving_context(
                payload.user_id,
                sequence_limit=200,
                cache_default=False,
            ),
            (
                _default_user_features(payload.user_id),
                _default_user_sequence_token(),
            ),
        )
        profile["user_features_ms"] = round(
            (time.perf_counter() - stage_started) * 1000, 2
        )
        profile["user_sequence_token_ms"] = 0.0

        if content_features_task:
            (
                content_features,
                profile["content_features_ms"],
            ) = await content_features_task
        if content_processed_at_task:
            (
                content_processed_at,
                profile["content_status_ms"],
            ) = await content_processed_at_task
        content_cache_writes_allowed = (
            payload.content_id is None or content_features is not None
        )
        serving_versions = _serving_version_context(runtime)
        user_feature_token = _user_feature_cache_token(user_features)
        freshness_context = _build_cache_freshness_context(
            serving_versions,
            user_feature_token,
            user_sequence_token,
            content_feature_token=_content_feature_cache_token(
                payload.content_id,
                content_features,
                content_processed_at,
            ),
        )
        cache_key = feature_store.generate_context_hash(
            {
                **_build_recommendation_cache_context(
                    payload,
                    current_time=start_time,
                    recommendation_config=runtime.config.recommendation_config,
                ),
                **freshness_context,
            }
        )
        recommendation_cache_task = asyncio.create_task(
            _timed_awaitable(
                _bounded_hot_path_read(
                    runtime,
                    "cached_recommendations",
                    feature_store.get_cached_recommendations(
                        payload.user_id, cache_key
                    ),
                    None,
                )
            ),
            name="recommendation-cache",
        )
        cached, profile["cache_lookup_ms"] = await _await_recommendation_cache_race(
            runtime,
            recommendation_cache_task,
        )
        if cached:
            profile["cache_hit"] = True
            profile["serving_path"] = "recommendation_cache"
            profile["ranked_count"] = len(cached)
            profile["total_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            cache_impression_id = None
            cached_displayed_items: List[Dict[str, Any]] = []
            if runtime.config.recommendation_config.impression_logging_enabled:
                cache_impression_id = uuid.uuid4().hex
                cached_displayed_items = _build_displayed_item_snapshots(
                    cached,
                    candidate_by_product={},
                    product_metadata_map={},
                    max_items=int(
                        runtime.config.recommendation_config.impression_max_items
                    ),
                    user_id=payload.user_id,
                    user_features=user_features,
                    observation_context=dict(payload.context or {}),
                    as_of_ts=start_time,
                )
            client_connected = not await _is_request_disconnected(http_request)
            if kafka_manager and client_connected:
                _schedule_best_effort_task(
                    "send_recommendation_event",
                    kafka_manager.send_recommendation_event(
                        user_id=payload.user_id,
                        recommendations=_recommendation_product_ids(cached),
                        response_time_ms=int((time.time() - start_time) * 1000),
                        request_id=getattr(http_request.state, "request_id", None),
                        metadata={
                            **(
                                {"impression_id": cache_impression_id}
                                if cache_impression_id
                                else {}
                            ),
                            "request_id": getattr(
                                http_request.state, "request_id", None
                            ),
                            "session_id": (payload.context or {}).get("session_id"),
                            "content_id": payload.content_id,
                            "model_version": "v1.0.0",
                            "ranking_model_version": serving_versions.get(
                                "ranking_model"
                            ),
                            "as_of_ts": start_time,
                            "feature_definition_version": RANKING_LTR_FEATURE_DEFINITION_VERSION,
                            "user_feature_snapshot": _recommendation_item_payload(
                                user_features
                            ),
                            "feature_context": dict(payload.context or {}),
                            "context": _build_impression_context_snapshot(
                                payload.context,
                                content_id=payload.content_id,
                            ),
                            "candidate_count": len(cached),
                            "candidate_source_counts": {},
                            "ranked_source_counts": {},
                            "item_snapshot_scope": "returned_top_k",
                            "displayed_items": cached_displayed_items,
                        },
                    ),
                    timeout_seconds=runtime.config.cache_config.background_write_timeout_ms
                    / 1000.0,
                )
            _attach_profile_headers(response, profile)
            _log_recommendation_profile(runtime, payload, profile)
            return _recommendation_json_response(
                _recommendation_payload(
                    user_id=payload.user_id,
                    recommendations=cached,
                    metadata={
                        "total_candidates": len(cached),
                        "response_time_ms": int((time.time() - start_time) * 1000),
                        "model_version": "v1.0.0",
                        "cache_hit": True,
                        **(
                            {"impression_id": cache_impression_id}
                            if cache_impression_id
                            else {}
                        ),
                        "cache_freshness": "model_user_sequence_and_catalog_versioned",
                        "content_processed": payload.content_id is not None,
                        **(
                            {"profile": profile}
                            if runtime.config.monitoring_config.enable_profiling_logs
                            else {}
                        ),
                    },
                ),
                profile,
            )

        singleflight_key = f"{payload.user_id}:{cache_key}"
        (
            singleflight_future,
            owns_singleflight,
        ) = await _join_or_create_recommendation_singleflight(singleflight_key)
        if not owns_singleflight:
            stage_started = time.perf_counter()
            shared_result = await asyncio.shield(singleflight_future)
            profile["singleflight_joined"] = True
            profile["singleflight_wait_ms"] = round(
                (time.perf_counter() - stage_started) * 1000,
                2,
            )
            profile["serving_path"] = "singleflight_joined"
            profile["ranked_count"] = len(shared_result.get("recommendations", []))
            profile["candidate_count"] = shared_result.get("total_candidates", 0)
            profile["total_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            shared_recommendations = shared_result.get("recommendations", [])
            join_impression_id = None
            join_displayed_items: List[Dict[str, Any]] = []
            if runtime.config.recommendation_config.impression_logging_enabled:
                join_impression_id = uuid.uuid4().hex
                join_displayed_items = _build_displayed_item_snapshots(
                    shared_recommendations,
                    candidate_by_product={},
                    product_metadata_map={},
                    max_items=int(
                        runtime.config.recommendation_config.impression_max_items
                    ),
                    user_id=payload.user_id,
                    user_features=user_features,
                    observation_context=dict(payload.context or {}),
                    as_of_ts=start_time,
                )
            if kafka_manager and not await _is_request_disconnected(http_request):
                _schedule_best_effort_task(
                    "send_recommendation_event",
                    kafka_manager.send_recommendation_event(
                        user_id=payload.user_id,
                        recommendations=_recommendation_product_ids(
                            shared_recommendations
                        ),
                        response_time_ms=int((time.time() - start_time) * 1000),
                        request_id=getattr(http_request.state, "request_id", None),
                        metadata={
                            **(
                                {"impression_id": join_impression_id}
                                if join_impression_id
                                else {}
                            ),
                            "request_id": getattr(
                                http_request.state, "request_id", None
                            ),
                            "session_id": (payload.context or {}).get("session_id"),
                            "content_id": payload.content_id,
                            "model_version": "v1.0.0",
                            "ranking_model_version": serving_versions.get(
                                "ranking_model"
                            ),
                            "as_of_ts": start_time,
                            "feature_definition_version": RANKING_LTR_FEATURE_DEFINITION_VERSION,
                            "user_feature_snapshot": _recommendation_item_payload(
                                user_features
                            ),
                            "feature_context": dict(payload.context or {}),
                            "context": _build_impression_context_snapshot(
                                payload.context,
                                content_id=payload.content_id,
                            ),
                            "candidate_count": shared_result.get(
                                "total_candidates",
                                len(shared_recommendations),
                            ),
                            "candidate_source_counts": {},
                            "ranked_source_counts": {},
                            "item_snapshot_scope": "returned_top_k",
                            "displayed_items": join_displayed_items,
                        },
                    ),
                    timeout_seconds=runtime.config.cache_config.background_write_timeout_ms
                    / 1000.0,
                )
            _attach_profile_headers(response, profile)
            _log_recommendation_profile(runtime, payload, profile)
            return _recommendation_json_response(
                _recommendation_payload(
                    user_id=payload.user_id,
                    recommendations=shared_recommendations,
                    metadata={
                        "total_candidates": shared_result.get("total_candidates", 0),
                        "response_time_ms": int((time.time() - start_time) * 1000),
                        "model_version": "v1.0.0",
                        "cache_hit": False,
                        "cache_freshness": "model_user_sequence_and_catalog_versioned",
                        "singleflight_joined": True,
                        **(
                            {"impression_id": join_impression_id}
                            if join_impression_id
                            else {}
                        ),
                        "content_processed": shared_result.get(
                            "content_processed",
                            payload.content_id is not None,
                        ),
                        **({"fallback": True} if shared_result.get("fallback") else {}),
                        **(
                            {"fallback_reason": shared_result["fallback_reason"]}
                            if shared_result.get("fallback_reason")
                            else {}
                        ),
                        **(
                            {"profile": profile}
                            if runtime.config.monitoring_config.enable_profiling_logs
                            else {}
                        ),
                    },
                ),
                profile,
            )

        k_per_source = min(payload.k * 10, 500)
        candidate_cache_key = feature_store.generate_context_hash(
            {
                **_build_candidate_cache_context(payload, k_per_source),
                **freshness_context,
            }
        )
        candidate_cache_task = asyncio.create_task(
            _timed_awaitable(
                _bounded_hot_path_read(
                    runtime,
                    "candidate_cache",
                    feature_store.get_cached_candidate_products(
                        payload.user_id,
                        candidate_cache_key,
                    ),
                    None,
                )
            ),
            name="recommendation-candidate-cache",
        )

        (
            candidates,
            profile["candidate_cache_lookup_ms"],
        ) = await _await_candidate_cache_race(runtime, candidate_cache_task)
        candidate_profile = {
            "path": "candidate_cache",
            "candidate_count": len(candidates or []),
        }
        if candidates is not None:
            profile["candidate_cache_hit"] = True
            profile["serving_path"] = "candidate_cache_then_rank"
        else:
            stage_started = time.perf_counter()
            (
                candidates,
                candidate_profile,
            ) = await recommendation_engine.generate_candidates(
                user_id=payload.user_id,
                content_features=content_features,
                context=payload.context,
                k_per_source=k_per_source,
                include_profile=True,
                user_features=user_features,
                user_interactions=_candidate_generation_interaction_hint(
                    recommendation_engine,
                    user_features,
                    user_sequence_token,
                ),
            )
            profile["candidate_generation_ms"] = round(
                (time.perf_counter() - stage_started) * 1000, 2
            )
            stage_started = time.perf_counter()
            if content_cache_writes_allowed and not await _is_request_disconnected(
                http_request
            ):
                _schedule_best_effort_task(
                    "cache_candidate_products",
                    feature_store.cache_candidate_products(
                        payload.user_id,
                        candidate_cache_key,
                        candidates,
                        user_features=user_features,
                    ),
                    timeout_seconds=runtime.config.cache_config.background_write_timeout_ms
                    / 1000.0,
                )
            profile["candidate_cache_write_ms"] = round(
                (time.perf_counter() - stage_started) * 1000, 2
            )
        profile["candidate_count"] = len(candidates)
        profile["candidate_profile"] = candidate_profile

        if not candidates:
            profile["total_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            _attach_profile_headers(response, profile)
            _log_recommendation_profile(runtime, payload, profile)
            await _resolve_recommendation_singleflight(
                singleflight_key,
                singleflight_future if owns_singleflight else None,
                result=_build_singleflight_result(
                    [],
                    total_candidates=0,
                    content_processed=payload.content_id is not None,
                    fallback=True,
                    fallback_reason="no_candidates",
                ),
            )
            return _recommendation_json_response(
                _recommendation_payload(
                    user_id=payload.user_id,
                    recommendations=[],
                    metadata={
                        "total_candidates": 0,
                        "response_time_ms": int((time.time() - start_time) * 1000),
                        "fallback_reason": "no_candidates",
                        "cache_hit": False,
                        **(
                            {"profile": profile}
                            if runtime.config.monitoring_config.enable_profiling_logs
                            else {}
                        ),
                    },
                ),
                profile,
            )

        stage_started = time.perf_counter()
        product_ids = [candidate.product_id for candidate in candidates]
        product_metadata_map = await _bounded_hot_path_read(
            runtime,
            "product_metadata_batch",
            feature_store.get_product_metadata_batch(product_ids),
            {},
        )
        missing_product_ids = [
            product_id
            for product_id in product_ids
            if product_id not in product_metadata_map
        ]
        if missing_product_ids:
            fetched_metadata = await vector_search.get_product_metadata_batch(
                missing_product_ids
            )
            if fetched_metadata:
                feature_store.prime_product_metadata_memory_cache(fetched_metadata)
                _schedule_best_effort_task(
                    "store_product_metadata_batch",
                    feature_store.store_product_metadata_batch(fetched_metadata),
                    timeout_seconds=runtime.config.cache_config.background_write_timeout_ms
                    / 1000.0,
                )
                product_metadata_map.update(fetched_metadata)
            profile["metadata_cache_miss_count"] = len(missing_product_ids) - len(
                fetched_metadata
            )
        profile["metadata_lookup_ms"] = round(
            (time.perf_counter() - stage_started) * 1000, 2
        )
        ranked_candidate_count = len(candidates)
        profile[
            "candidate_source_counts_before_filter"
        ] = _count_candidate_source_tokens(candidates)
        candidates = _filter_recommendable_candidates(candidates, product_metadata_map)
        candidate_source_by_product = {
            candidate.product_id: candidate.source for candidate in candidates
        }
        candidate_by_product = {
            candidate.product_id: candidate for candidate in candidates
        }
        profile["candidate_source_counts"] = _count_candidate_source_tokens(candidates)
        profile["candidate_count_before_eligibility_filter"] = ranked_candidate_count
        profile["candidate_count"] = len(candidates)
        profile["filtered_unavailable_candidates"] = ranked_candidate_count - len(
            candidates
        )

        if not candidates:
            profile["total_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            _attach_profile_headers(response, profile)
            _log_recommendation_profile(runtime, payload, profile)
            await _resolve_recommendation_singleflight(
                singleflight_key,
                singleflight_future if owns_singleflight else None,
                result=_build_singleflight_result(
                    [],
                    total_candidates=ranked_candidate_count,
                    content_processed=payload.content_id is not None,
                    fallback=True,
                    fallback_reason="no_eligible_candidates",
                ),
            )
            return _recommendation_json_response(
                _recommendation_payload(
                    user_id=payload.user_id,
                    recommendations=[],
                    metadata={
                        "total_candidates": ranked_candidate_count,
                        "response_time_ms": int((time.time() - start_time) * 1000),
                        "fallback_reason": "no_eligible_candidates",
                        "cache_hit": False,
                        **(
                            {"profile": profile}
                            if runtime.config.monitoring_config.enable_profiling_logs
                            else {}
                        ),
                    },
                ),
                profile,
            )

        slate_diversity_enabled = _is_mmr_slate_diversity_enabled(
            runtime.config.recommendation_config
        )
        ranker_k = payload.k
        if slate_diversity_enabled:
            ranker_k = _calculate_mmr_rerank_pool_size(
                requested_k=payload.k,
                candidate_count=len(candidates),
                recommendation_config=runtime.config.recommendation_config,
            )

        if await _is_request_disconnected(http_request):
            raise HTTPException(status_code=499, detail="Client disconnected")

        ranking_context = {
            **dict(payload.context or {}),
            "_feature_as_of_ts": start_time,
        }
        if _is_realtime_window_features_enabled(runtime):
            stage_started = time.perf_counter()
            ranking_context = await _attach_realtime_window_features(
                ranking_context,
                payload.user_id,
                candidates,
                product_metadata_map,
            )
            profile["realtime_window_features_ms"] = round(
                (time.perf_counter() - stage_started) * 1000,
                2,
            )
        if _is_ranking_history_embeddings_enabled(runtime):
            stage_started = time.perf_counter()
            ranking_context, history_profile = await _attach_ranking_history_embeddings(
                runtime,
                ranking_context,
                payload.user_id,
                candidates,
                serving_versions,
            )
            profile["history_embeddings_ms"] = round(
                (time.perf_counter() - stage_started) * 1000,
                2,
            )
            profile.update(history_profile)

        stage_started = time.perf_counter()
        ranked_recommendations, ranking_profile = await _rank_candidates_for_request(
            candidates=candidates,
            user_features=user_features,
            context=ranking_context,
            product_metadata_map=product_metadata_map,
            k=ranker_k,
            http_request=http_request,
        )
        profile["ranking_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
        profile["ranking_profile"] = ranking_profile
        profile["slate_diversity_pool_size"] = len(ranked_recommendations)

        if slate_diversity_enabled:
            stage_started = time.perf_counter()
            ranked_recommendations = select_mmr_recommendations(
                ranked_recommendations,
                k=payload.k,
                embedding_lookup=vector_search.get_product_embedding,
                lambda_weight=runtime.config.recommendation_config.mmr_lambda,
            )
            profile["slate_diversity_ms"] = round(
                (time.perf_counter() - stage_started) * 1000,
                2,
            )
        profile["ranked_count"] = len(ranked_recommendations)
        profile["slate_diversity_selected_count"] = len(ranked_recommendations)
        ranked_source_counts = _count_ranked_source_tokens(
            ranked_recommendations,
            candidate_source_by_product,
        )
        profile["ranked_source_counts"] = ranked_source_counts
        profile["ranked_sasrec_count"] = ranked_source_counts.get("sasrec", 0)
        response_time = time.time() - start_time
        ranked_recommendation_payloads = [
            recommendation.dict() for recommendation in ranked_recommendations
        ]
        impression_id = None
        displayed_items: List[Dict[str, Any]] = []
        rejected_candidate_items: List[Dict[str, Any]] = []
        if runtime.config.recommendation_config.impression_logging_enabled:
            impression_id = uuid.uuid4().hex
            displayed_items = _build_displayed_item_snapshots(
                ranked_recommendations,
                candidate_by_product=candidate_by_product,
                product_metadata_map=product_metadata_map,
                max_items=int(
                    runtime.config.recommendation_config.impression_max_items
                ),
                user_id=payload.user_id,
                user_features=user_features,
                observation_context=ranking_context,
                as_of_ts=start_time,
            )
            ranked_recommendation_payloads = _attach_displayed_item_snapshot_metadata(
                ranked_recommendation_payloads,
                displayed_items,
            )
            rejected_candidate_items = _build_rejected_candidate_snapshots(
                candidates,
                ranked_recommendations,
                product_metadata_map=product_metadata_map,
                max_items=int(
                    runtime.config.recommendation_config.ranker_rejected_logging_max_items
                ),
            )

        stage_started = time.perf_counter()
        client_connected = not await _is_request_disconnected(http_request)
        if client_connected and content_cache_writes_allowed:
            _schedule_best_effort_task(
                "cache_recommendations",
                feature_store.cache_recommendations(
                    payload.user_id,
                    cache_key,
                    ranked_recommendation_payloads,
                    user_features=user_features,
                ),
                timeout_seconds=runtime.config.cache_config.background_write_timeout_ms
                / 1000.0,
            )
        profile["cache_write_ms"] = round(
            (time.perf_counter() - stage_started) * 1000, 2
        )

        stage_started = time.perf_counter()
        if client_connected:
            _schedule_best_effort_task(
                "log_recommendation_request",
                feature_store.log_recommendation_request(
                    payload.user_id,
                    len(ranked_recommendations),
                    response_time,
                ),
                timeout_seconds=runtime.config.cache_config.background_write_timeout_ms
                / 1000.0,
            )
        profile["analytics_log_ms"] = round(
            (time.perf_counter() - stage_started) * 1000, 2
        )

        if kafka_manager and client_connected:
            stage_started = time.perf_counter()
            _schedule_best_effort_task(
                "send_recommendation_event",
                kafka_manager.send_recommendation_event(
                    user_id=payload.user_id,
                    recommendations=[
                        item.product_id for item in ranked_recommendations
                    ],
                    response_time_ms=int(response_time * 1000),
                    request_id=getattr(http_request.state, "request_id", None),
                    metadata={
                        **({"impression_id": impression_id} if impression_id else {}),
                        "request_id": getattr(http_request.state, "request_id", None),
                        "session_id": (payload.context or {}).get("session_id"),
                        "content_id": payload.content_id,
                        "model_version": "v1.0.0",
                        "ranking_model_version": serving_versions.get("ranking_model"),
                        "as_of_ts": start_time,
                        "feature_definition_version": RANKING_LTR_FEATURE_DEFINITION_VERSION,
                        "user_feature_snapshot": _recommendation_item_payload(
                            user_features
                        ),
                        "feature_context": ranking_context,
                        "context": _build_impression_context_snapshot(
                            payload.context,
                            content_id=payload.content_id,
                        ),
                        "candidate_count": len(candidates),
                        "candidate_source_counts": profile.get(
                            "candidate_source_counts", {}
                        ),
                        "ranked_source_counts": profile.get("ranked_source_counts", {}),
                        "item_snapshot_scope": "returned_top_k",
                        "displayed_items": displayed_items,
                        "rejected_candidate_items": rejected_candidate_items,
                    },
                ),
                timeout_seconds=runtime.config.cache_config.background_write_timeout_ms
                / 1000.0,
            )
            profile["kafka_schedule_ms"] = round(
                (time.perf_counter() - stage_started) * 1000, 2
            )

        profile["total_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        _attach_profile_headers(response, profile)
        _log_recommendation_profile(runtime, payload, profile)
        await _resolve_recommendation_singleflight(
            singleflight_key,
            singleflight_future if owns_singleflight else None,
            result=_build_singleflight_result(
                ranked_recommendation_payloads,
                total_candidates=len(candidates),
                content_processed=payload.content_id is not None,
            ),
        )

        return _recommendation_json_response(
            _recommendation_payload(
                user_id=payload.user_id,
                recommendations=ranked_recommendation_payloads,
                metadata={
                    "total_candidates": len(candidates),
                    "response_time_ms": int(response_time * 1000),
                    "model_version": "v1.0.0",
                    **({"impression_id": impression_id} if impression_id else {}),
                    "cache_freshness": "model_user_sequence_and_catalog_versioned",
                    "cache_hit": False,
                    "content_processed": payload.content_id is not None,
                    **(
                        {"profile": profile}
                        if runtime.config.monitoring_config.enable_profiling_logs
                        else {}
                    ),
                },
            ),
            profile,
        )

    except asyncio.CancelledError as exc:
        await _resolve_recommendation_singleflight(
            singleflight_key,
            singleflight_future if owns_singleflight else None,
            exception=exc,
        )
        raise
    except HTTPException as exc:
        await _resolve_recommendation_singleflight(
            singleflight_key,
            singleflight_future if owns_singleflight else None,
            exception=exc,
        )
        profile["error"] = str(exc.detail)
        profile["total_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        _attach_profile_headers(response, profile)
        _log_recommendation_profile(runtime, payload, profile, level="error")
        raise
    except Exception as exc:
        logger.error(f"Recommendation request failed: {exc}")
        logger.error(traceback.format_exc())
        profile["error"] = str(exc)
        profile["total_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        _attach_profile_headers(response, profile)
        _log_recommendation_profile(runtime, payload, profile, level="error")
        try:
            trending_recommendations = (
                await recommendation_engine.get_trending_recommendations(payload.k)
            )
            trending_payloads = [
                _recommendation_item_payload(item) for item in trending_recommendations
            ]
            await _resolve_recommendation_singleflight(
                singleflight_key,
                singleflight_future if owns_singleflight else None,
                result=_build_singleflight_result(
                    trending_payloads,
                    total_candidates=len(trending_recommendations),
                    content_processed=payload.content_id is not None,
                    fallback=True,
                    fallback_reason="serving_error",
                ),
            )
            return _recommendation_json_response(
                _recommendation_payload(
                    user_id=payload.user_id,
                    recommendations=trending_payloads,
                    metadata={
                        "total_candidates": len(trending_recommendations),
                        "response_time_ms": int((time.time() - start_time) * 1000),
                        "fallback": True,
                        "fallback_reason": "serving_error",
                        **(
                            {"profile": profile}
                            if runtime.config.monitoring_config.enable_profiling_logs
                            else {}
                        ),
                    },
                ),
                profile,
            )
        except Exception as fallback_exc:
            await _resolve_recommendation_singleflight(
                singleflight_key,
                singleflight_future if owns_singleflight else None,
                exception=fallback_exc,
            )
            raise HTTPException(
                status_code=503,
                detail="Recommendation service unavailable",
            ) from fallback_exc


def _profile_headers(profile: dict) -> Dict[str, str]:
    return {
        "X-Service-Process-Pid": str(profile["process_id"]),
        "X-Worker-Active-Requests": str(profile["worker_active_requests_at_entry"]),
        "X-Worker-Handled-Requests": str(profile["worker_handled_requests"]),
        "X-Handler-Queue-Ms": str(profile["handler_queue_ms"]),
        "X-Recommendation-Total-Ms": str(profile["total_ms"]),
        "X-Torch-Num-Threads": str(profile["torch_num_threads"]),
        "X-Torch-Num-Interop-Threads": str(profile["torch_num_interop_threads"]),
    }


def _attach_profile_headers(response: Response, profile: dict) -> None:
    for name, value in _profile_headers(profile).items():
        response.headers[name] = value


def _recommendation_payload(
    *,
    user_id: str,
    recommendations: List[Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "recommendations": [
            _recommendation_item_payload(item) for item in recommendations
        ],
        "metadata": metadata,
    }


def _recommendation_item_payload(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return item
    raw = getattr(item, "__dict__", None)
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(item, "dict"):
        return item.dict()
    return dict(item)


def _recommendation_json_response(payload: Dict[str, Any], profile: dict) -> Response:
    return Response(
        content=json_dumps(payload),
        media_type="application/json",
        headers=_profile_headers(profile),
    )


async def _rank_candidates_for_request(
    *,
    candidates: List[CandidateProduct],
    user_features: UserFeatures,
    context: Dict[str, Any],
    product_metadata_map: Dict[str, Dict[str, Any]],
    k: int,
    http_request: Request,
) -> Tuple[List[ProductRecommendation], Dict[str, Any]]:
    runtime = app.state.runtime
    deadline_unix_seconds = _ranking_deadline_unix_seconds(runtime)
    if ranking_coordinator_client_pool:
        try:
            return await _rank_candidates_coordinator(
                candidates=candidates,
                user_features=user_features,
                context=context,
                product_metadata_map=product_metadata_map,
                k=k,
                http_request=http_request,
                deadline_unix_seconds=deadline_unix_seconds,
            )
        except (RankingCoordinatorError, ValueError, HTTPException) as exc:
            if (
                not runtime.config.service_topology_config.ranking_service_fallback_enabled
                or ranking_batcher is None
            ):
                if isinstance(exc, HTTPException):
                    raise
                raise HTTPException(
                    status_code=503,
                    detail="Ranking service unavailable",
                ) from exc
            logger.warning(
                "coordinator_ranking_failed_local_same_model_fallback: %s",
                exc,
            )

    if ranking_client_pool:
        try:
            return await _rank_candidates_remote(
                candidates=candidates,
                user_features=user_features,
                context=context,
                product_metadata_map=product_metadata_map,
                k=k,
                http_request=http_request,
                deadline_unix_seconds=deadline_unix_seconds,
            )
        except (httpx.HTTPError, ValueError, HTTPException) as exc:
            if (
                not runtime.config.service_topology_config.ranking_service_fallback_enabled
                or ranking_batcher is None
            ):
                if isinstance(exc, HTTPException):
                    raise
                raise HTTPException(
                    status_code=503,
                    detail="Ranking service unavailable",
                ) from exc
            logger.warning("remote_ranking_failed_local_same_model_fallback: %s", exc)

    if ranking_batcher is None:
        raise HTTPException(status_code=503, detail="Ranking service unavailable")
    try:
        return await ranking_batcher.rank_candidates(
            candidates=candidates,
            user_features=user_features,
            context=context,
            product_metadata_map=product_metadata_map,
            k=k,
            include_profile=True,
            deadline_unix_seconds=deadline_unix_seconds,
        )
    except (RankingQueueFullError, RankingQueueTimeoutError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def _attach_realtime_window_features(
    context: Dict[str, Any],
    user_id: str,
    candidates: List[CandidateProduct],
    product_metadata_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Attach Flink realtime window features to the ranking context."""
    if feature_store is None:
        return context

    entities: Set[Tuple[str, str]] = {("user", user_id)}
    for candidate in candidates:
        product_id = getattr(candidate, "product_id", None)
        if product_id:
            entities.add(("product", str(product_id)))
            category = (product_metadata_map.get(str(product_id)) or {}).get("category")
            if category:
                entities.add(("category", str(category)))

    feature_map = await _bounded_hot_path_read(
        app.state.runtime,
        "realtime_window_features",
        feature_store.get_realtime_window_features_batch(entities),
        {},
    )
    if not feature_map:
        return context

    serialized = {
        entity_key: {window: features.dict() for window, features in windows.items()}
        for entity_key, windows in feature_map.items()
    }
    return {
        **(context or {}),
        "_realtime_window_features": serialized,
    }


async def _attach_ranking_history_embeddings(
    runtime: Any,
    context: Optional[Dict[str, Any]],
    user_id: str,
    candidates: List[CandidateProduct],
    serving_versions: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Attach compact last-N history embedding context for the ranker."""
    base_context = dict(context or {})
    if feature_store is None:
        return base_context, {"history_embeddings_status": "feature_store_unavailable"}

    history_config = ranking_history_config_from_settings(runtime.config.ranking_config)
    sequence_limit = max(
        200,
        history_config.click_last_n
        + history_config.cart_last_n
        + history_config.purchase_last_n,
    )
    sequence = await _bounded_hot_path_read(
        runtime,
        "ranking_history_sequence",
        feature_store.get_user_sequence(user_id, limit=sequence_limit),
        [],
    )
    candidate_ids = [str(candidate.product_id) for candidate in candidates]
    history_context = build_ranking_history_context(
        sequence or [],
        _ranking_history_item_embedding_map(),
        config=history_config,
        candidate_product_ids=candidate_ids,
        current_time=time.time(),
        two_tower_model_version=serving_versions.get("two_tower_model"),
    )
    return {
        **base_context,
        RANKING_HISTORY_CONTEXT_KEY: history_context,
    }, {
        "history_embeddings_status": "attached",
        **history_context_profile(history_context),
    }


async def _rank_candidates_remote(
    *,
    candidates: List[CandidateProduct],
    user_features: UserFeatures,
    context: Dict[str, Any],
    product_metadata_map: Dict[str, Dict[str, Any]],
    k: int,
    http_request: Request,
    deadline_unix_seconds: float,
) -> Tuple[List[ProductRecommendation], Dict[str, Any]]:
    runtime = app.state.runtime
    headers = {"Content-Type": "application/json"}
    request_id_header = runtime.config.monitoring_config.request_id_header
    request_id = getattr(http_request.state, "request_id", None)
    if request_id:
        headers[request_id_header] = request_id
    internal_header = runtime.config.security_config.internal_service_header
    internal_key = runtime.config.security_config.internal_service_key
    if internal_key:
        headers[internal_header] = internal_key
    inject_http_headers(headers)

    runtime.observability.inc_upstream_inflight("ranking-service")
    try:
        upstream = await ranking_client_pool.post(
            "/internal/rank",
            content=_ranking_request_body(
                candidates=candidates,
                user_features=user_features,
                context=context,
                product_metadata_map=product_metadata_map,
                k=k,
                request_id=request_id,
                deadline_unix_seconds=deadline_unix_seconds,
            ),
            headers=headers,
        )
        runtime.observability.record_upstream_request(
            "ranking-service", upstream.status_code
        )
    except httpx.HTTPError:
        runtime.observability.record_upstream_request("ranking-service", "error")
        raise
    finally:
        runtime.observability.dec_upstream_inflight("ranking-service")

    if upstream.status_code >= 500:
        raise HTTPException(status_code=503, detail="Ranking service unavailable")
    if upstream.status_code >= 400:
        raise HTTPException(status_code=upstream.status_code, detail=upstream.text)

    payload = upstream.json()
    recommendations = [
        ProductRecommendation.construct(**item)
        for item in payload.get("recommendations", [])
    ]
    profile = payload.get("profile") or {}
    profile["path"] = f"remote_{profile.get('path', 'ranking')}"
    return recommendations, profile


async def _rank_candidates_coordinator(
    *,
    candidates: List[CandidateProduct],
    user_features: UserFeatures,
    context: Dict[str, Any],
    product_metadata_map: Dict[str, Dict[str, Any]],
    k: int,
    http_request: Request,
    deadline_unix_seconds: float,
) -> Tuple[List[ProductRecommendation], Dict[str, Any]]:
    runtime = app.state.runtime
    request_id = getattr(http_request.state, "request_id", None)
    body = _ranking_request_body(
        candidates=candidates,
        user_features=user_features,
        context=context,
        product_metadata_map=product_metadata_map,
        k=k,
        request_id=request_id,
        deadline_unix_seconds=deadline_unix_seconds,
    )
    runtime.observability.inc_upstream_inflight("ranking-coordinator")
    try:
        upstream = await ranking_coordinator_client_pool.rank(body)
        runtime.observability.record_upstream_request(
            "ranking-coordinator",
            upstream.status_code,
        )
    except RankingCoordinatorError:
        runtime.observability.record_upstream_request("ranking-coordinator", "error")
        raise
    finally:
        runtime.observability.dec_upstream_inflight("ranking-coordinator")

    if upstream.status_code >= 500:
        raise HTTPException(status_code=503, detail="Ranking service unavailable")
    if upstream.status_code >= 400:
        raise HTTPException(
            status_code=upstream.status_code,
            detail=upstream.body.decode("utf-8", errors="replace"),
        )

    payload = json_loads(upstream.body)
    recommendations = [
        ProductRecommendation.construct(**item)
        for item in payload.get("recommendations", [])
    ]
    profile = payload.get("profile") or {}
    profile["path"] = f"coordinator_{profile.get('path', 'ranking')}"
    return recommendations, profile


def _ranking_request_body(
    *,
    candidates: List[CandidateProduct],
    user_features: UserFeatures,
    context: Dict[str, Any],
    product_metadata_map: Dict[str, Dict[str, Any]],
    k: int,
    request_id: Optional[str],
    deadline_unix_seconds: Optional[float] = None,
) -> bytes:
    payload = {
        "request_id": request_id,
        "candidates": [
            _recommendation_item_payload(candidate) for candidate in candidates
        ],
        "user_features": _recommendation_item_payload(user_features),
        "context": context,
        "product_metadata_map": product_metadata_map,
        "k": k,
    }
    if deadline_unix_seconds is not None:
        payload["deadline_unix_seconds"] = float(deadline_unix_seconds)
    return json_dumps(payload)


def _ranking_deadline_unix_seconds(runtime) -> float:
    topology = getattr(
        getattr(runtime, "config", None), "service_topology_config", None
    )
    request_timeout = getattr(
        topology,
        "ranking_coordinator_request_timeout_seconds",
        1.0,
    )
    return time.time() + max(0.05, float(request_timeout) - 0.1)


async def _probe_ranking_service_ready(runtime) -> Dict[str, Any]:
    headers = {}
    internal_header = runtime.config.security_config.internal_service_header
    internal_key = runtime.config.security_config.internal_service_key
    if internal_key:
        headers[internal_header] = internal_key
    inject_http_headers(headers)
    started_at = time.perf_counter()
    try:
        upstream = await ranking_client_pool.get("/readyz", headers=headers)
        if upstream.status_code != 200:
            return {
                "status": "unhealthy",
                "response_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "error": f"ranking-service returned {upstream.status_code}",
            }
        return {
            "status": "healthy",
            "response_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "response_time_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "error": str(exc),
        }


async def _timed_awaitable(awaitable):
    started_at = time.perf_counter()
    result = await awaitable
    return result, round((time.perf_counter() - started_at) * 1000, 2)


async def _await_candidate_cache_race(
    runtime,
    candidate_cache_task: asyncio.Task,
) -> Tuple[Optional[List[CandidateProduct]], float]:
    timeout_seconds = (
        getattr(runtime.config.cache_config, "candidate_cache_race_timeout_ms", 5.0)
        / 1000.0
    )
    started_at = time.perf_counter()
    if timeout_seconds <= 0:
        return await candidate_cache_task
    try:
        return await asyncio.wait_for(candidate_cache_task, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        _cancel_optional_task(candidate_cache_task)
        return None, round((time.perf_counter() - started_at) * 1000, 2)


async def _await_recommendation_cache_race(
    runtime,
    recommendation_cache_task: asyncio.Task,
) -> Tuple[Optional[List[Dict[str, Any]]], float]:
    timeout_seconds = (
        getattr(
            runtime.config.cache_config, "recommendation_cache_race_timeout_ms", 5.0
        )
        / 1000.0
    )
    started_at = time.perf_counter()
    if timeout_seconds <= 0:
        return await recommendation_cache_task
    try:
        return await asyncio.wait_for(
            recommendation_cache_task, timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        _cancel_optional_task(recommendation_cache_task)
        return None, round((time.perf_counter() - started_at) * 1000, 2)


def _cancel_optional_task(task: Optional[asyncio.Task]) -> None:
    if task is not None and not task.done():
        task.cancel()


async def _is_request_disconnected(http_request: Request) -> bool:
    is_disconnected = getattr(http_request, "is_disconnected", None)
    if is_disconnected is None:
        return False
    return await is_disconnected()


async def _bounded_hot_path_read(runtime, operation_name: str, awaitable, fallback):
    timeout_seconds = runtime.config.cache_config.hot_path_read_timeout_ms / 1000.0
    if timeout_seconds <= 0:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning("%s_hot_path_read_timed_out", operation_name)
        return fallback
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("%s_hot_path_read_failed: %s", operation_name, exc)
        return fallback


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


def _schedule_best_effort_task(
    task_name: str,
    awaitable,
    timeout_seconds: float = 0.25,
) -> None:
    if best_effort_task_queue is not None:
        best_effort_task_queue.enqueue(task_name, awaitable, timeout_seconds)
        return
    asyncio.create_task(
        _run_best_effort_task(task_name, awaitable, timeout_seconds),
        name=f"best-effort-{task_name}",
    )


async def _run_best_effort_task(
    task_name: str, awaitable, timeout_seconds: float
) -> None:
    try:
        await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning("%s_timed_out", task_name)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("%s_failed: %s", task_name, exc)


def _configure_torch_runtime(runtime) -> None:
    cpu_limit = _detect_cgroup_cpu_limit()
    configured_threads = runtime.config.model_config.torch_num_threads
    configured_interop_threads = runtime.config.model_config.torch_num_interop_threads

    intra_threads = configured_threads or cpu_limit
    if cpu_limit:
        intra_threads = min(intra_threads, cpu_limit)
    intra_threads = max(1, intra_threads)

    interop_threads = configured_interop_threads or cpu_limit
    interop_threads = max(1, min(interop_threads, intra_threads))

    torch.set_num_threads(intra_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError as exc:
        logger.warning(
            "torch_interop_threads_not_updated",
            extra={
                "service": runtime.service_name,
                "process_id": os.getpid(),
                "exception_type": type(exc).__name__,
                "exception_repr": repr(exc),
            },
        )

    logger.info(
        "torch_runtime_configured",
        extra={
            "service": runtime.service_name,
            "process_id": os.getpid(),
            "cpu_limit": cpu_limit,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
    )


async def _periodic_ranking_checkpoint_sync(runtime) -> None:
    interval_seconds = max(
        1, int(runtime.config.ranking_config.checkpoint_sync_interval_seconds)
    )
    model_path = runtime.config.model_config.ranking_model_path
    last_ranking_version: Optional[str] = (
        ranking_model.model_version if ranking_model else None
    )
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            if ranking_model and model_path and artifact_manager:
                latest_ranking = await artifact_manager.get_latest_model_checkpoint(
                    ModelArtifactManager.RANKING_MODEL_NAME
                )
                if (
                    latest_ranking
                    and latest_ranking.model_version != last_ranking_version
                ):
                    await artifact_manager.sync_latest_ranking_checkpoint()
                    if await ranking_model.reload_model_if_updated(model_path):
                        ranking_model.model_version = latest_ranking.model_version
                        last_ranking_version = latest_ranking.model_version
            if recommendation_engine:
                if await recommendation_engine.sync_serving_artifacts_if_updated():
                    _refresh_serving_version_context(runtime)
            if ranking_model:
                _refresh_serving_version_context(runtime)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(f"Ranking checkpoint sync failed: {exc}")
            await asyncio.sleep(min(interval_seconds, 60))


async def _periodic_known_user_snapshot_refresh(runtime) -> None:
    interval_seconds = max(
        1.0,
        float(
            runtime.config.recommendation_config.known_user_snapshot_refresh_interval_seconds
        ),
    )
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            if feature_store:
                await feature_store.refresh_known_user_snapshot()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("known_user_snapshot_refresh_failed: %s", exc)
            await asyncio.sleep(min(interval_seconds, 60.0))


async def _periodic_content_feature_snapshot_refresh(runtime) -> None:
    interval_seconds = max(
        1.0,
        float(
            runtime.config.recommendation_config.content_features_snapshot_refresh_interval_seconds
        ),
    )
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            if feature_store:
                await feature_store.refresh_content_feature_snapshot()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("content_feature_snapshot_refresh_failed: %s", exc)
            await asyncio.sleep(min(interval_seconds, 60.0))


def _detect_cgroup_cpu_limit() -> int:
    cpu_max_path = "/sys/fs/cgroup/cpu.max"
    if os.path.exists(cpu_max_path):
        try:
            quota, period = (
                open(cpu_max_path, "r", encoding="utf-8").read().strip().split()
            )
            if quota != "max":
                return max(1, math.ceil(int(quota) / int(period)))
        except (OSError, ValueError, ZeroDivisionError):
            pass

    quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
    period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
    if os.path.exists(quota_path) and os.path.exists(period_path):
        try:
            quota = int(open(quota_path, "r", encoding="utf-8").read().strip())
            period = int(open(period_path, "r", encoding="utf-8").read().strip())
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))
        except (OSError, ValueError, ZeroDivisionError):
            pass

    return max(1, os.cpu_count() or 1)


def _log_recommendation_profile(
    runtime, request, profile: dict, level: str = "info"
) -> None:
    runtime.observability.record_recommendation(
        result="error" if profile.get("error") else "success",
        cache_hit=bool(profile.get("cache_hit")),
        serving_path=str(profile.get("serving_path", "unknown")),
        candidate_count=int(profile.get("candidate_count") or 0),
        ranked_count=int(profile.get("ranked_count") or 0),
    )
    if (
        level != "error"
        and not runtime.config.monitoring_config.enable_profiling_logs
        and profile["total_ms"]
        < runtime.config.monitoring_config.profiling_log_min_duration_ms
    ):
        return
    if level != "error":
        sample_rate = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        runtime.config.monitoring_config,
                        "profiling_log_sample_rate",
                        1.0,
                    )
                ),
            ),
        )
        if sample_rate <= 0.0 or random.random() >= sample_rate:
            return

    log_fn = logger.error if level == "error" else logger.info
    log_fn(
        "recommendation_request_profile",
        extra={
            "service": runtime.service_name,
            "user_id": request.user_id,
            "content_id": request.content_id,
            **profile,
        },
    )


@app.get("/health")
async def health_check():
    feature_store_health = await feature_store.health_check()
    recommendation_health = recommendation_engine.health_check()
    ranking_health = ranking_model.health_check()
    vector_health = vector_search.health_check()
    database_health = {"status": "healthy", "response_time_ms": 0.0}
    if system_store:
        database_status = await system_store.health_check()
        database_health = {
            "status": database_status.status,
            "response_time_ms": database_status.response_time_ms,
            "error": database_status.error,
        }
    kafka_health = {"status": "healthy", "response_time_ms": 0.0}
    if kafka_manager:
        kafka_status = await kafka_manager.health_check()
        producer_health = kafka_status.get("producer", {})
        kafka_health = {
            "status": producer_health.get("status", "healthy"),
            "response_time_ms": 0.0,
            "error": None
            if producer_health.get("connected")
            else "Kafka producer unavailable",
        }
    elif app.state.runtime.config.kafka_config.enable:
        kafka_health = {"status": "degraded", "error": "Kafka producer unavailable"}

    return build_health_response(
        {
            "feature_store": component_health(
                feature_store_health.get("status", "unhealthy"),
                feature_store_health.get("response_time_ms"),
                feature_store_health.get("error"),
            ),
            "recommendation_engine": component_health(
                recommendation_health.get("status", "unhealthy"),
                error_message=recommendation_health.get("error"),
            ),
            "ranking_model": component_health(
                ranking_health.get("status", "unhealthy"),
                error_message=ranking_health.get("error"),
            ),
            "database": component_health(
                database_health.get("status", "unhealthy"),
                database_health.get("response_time_ms"),
                database_health.get("error"),
            ),
            "vector_search": component_health(
                vector_health.get("status", "unhealthy"),
                error_message=vector_health.get("error"),
            ),
            "kafka": component_health(
                kafka_health.get("status", "healthy"),
                kafka_health.get("response_time_ms"),
                kafka_health.get("error"),
            ),
        },
        app.state.runtime.started_at,
    )


@app.get("/metrics")
async def metrics():
    return await build_metrics_response(
        app.state.runtime,
        feature_store=feature_store,
        kafka_manager=kafka_manager,
        system_store=system_store,
    )
