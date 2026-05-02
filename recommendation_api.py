"""
Dedicated recommendation-serving API.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, HTTPException, Request, Response
import torch

from config import Config
from feature_store import FeatureStore
from kafka_client import close_kafka, init_kafka
from model_artifacts import ModelArtifactManager
from models import RecommendationRequest, RecommendationResponse, UserFeatures
from object_storage import ObjectStorage
from ranking_batcher import RankingBatcher
from ranking import RankingModel
from recommender import RecommendationEngine
from service_common import (
    build_health_response,
    build_liveness_payload,
    build_metrics_response,
    build_readiness_response,
    component_health,
    configure_service_logging,
    create_service_app,
    require_internal_service_auth,
)
from system_store import SystemStore
from vector_search import VectorSearchEngine

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
kafka_manager = None
system_store: Optional[SystemStore] = None
ranking_checkpoint_sync_task: Optional[asyncio.Task] = None
object_storage: Optional[ObjectStorage] = None
artifact_manager: Optional[ModelArtifactManager] = None
_recommendation_singleflight: Dict[str, asyncio.Future] = {}
_recommendation_singleflight_lock = asyncio.Lock()


@app.middleware("http")
async def internal_auth(request: Request, call_next):
    require_internal_service_auth(request, app.state.runtime)
    return await call_next(request)


def _build_candidate_cache_context(payload: RecommendationRequest, k_per_source: int) -> dict:
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
    }


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
        "last_active": int(user_features.last_active) if has_personalization_signal else 0,
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
) -> Dict[str, Any]:
    return {
        "serving_versions": serving_versions,
        "user_feature_token": user_feature_token,
        "user_sequence_token": user_sequence_token or _default_user_sequence_token(),
    }


def _serving_version_context(runtime) -> Dict[str, Any]:
    ranking_path = runtime.config.model_config.ranking_model_path
    vector_path = runtime.config.vector_config.index_path
    sasrec_checkpoint_path = runtime.config.recommendation_config.sasrec_checkpoint_path
    sasrec_vocab_path = runtime.config.recommendation_config.sasrec_vocab_path
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
        "sasrec_model": (
            recommendation_engine.loaded_sasrec_version
            if recommendation_engine
            else None
        ),
        "sasrec_checkpoint_mtime": _safe_file_mtime(sasrec_checkpoint_path),
        "sasrec_vocab_mtime": _safe_file_mtime(sasrec_vocab_path),
        "vector_index_mtime": _safe_file_mtime(vector_path),
        "catalog": _catalog_serving_version_context(),
    }


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


@app.on_event("startup")
async def startup_event():
    global feature_store, vector_search, recommendation_engine, ranking_model, ranking_batcher, kafka_manager
    global system_store
    global ranking_checkpoint_sync_task
    global object_storage, artifact_manager

    runtime = app.state.runtime
    runtime.config = Config()
    configure_service_logging(runtime)
    _configure_torch_runtime(runtime)

    feature_store = FeatureStore(runtime.config.redis_config, runtime.config.cache_config)
    await feature_store.initialize()

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
    if system_store and runtime.config.recommendation_config.publish_catalog_snapshot_on_startup:
        await system_store.store_product_catalog_snapshot_batch(vector_search.product_metadata)

    recommendation_engine = RecommendationEngine(
        feature_store,
        vector_search,
        runtime.config.recommendation_config,
        artifact_manager=artifact_manager,
    )
    await recommendation_engine.load_serving_state()

    ranking_model = RankingModel(runtime.config.ranking_config)
    ranking_checkpoint = None
    if artifact_manager:
        ranking_checkpoint = await artifact_manager.sync_latest_ranking_checkpoint()
    await ranking_model.load_model(runtime.config.model_config.ranking_model_path)
    if ranking_checkpoint:
        ranking_model.model_version = ranking_checkpoint.model_version
    ranking_model.enable_profiling_logs = runtime.config.monitoring_config.enable_profiling_logs
    ranking_model.profiling_log_min_duration_ms = (
        runtime.config.monitoring_config.profiling_log_min_duration_ms
    )
    ranking_batcher = RankingBatcher(ranking_model, runtime.config.ranking_config)
    await ranking_batcher.start()
    if runtime.config.ranking_config.checkpoint_sync_interval_seconds > 0:
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
    global ranking_checkpoint_sync_task

    if ranking_checkpoint_sync_task:
        ranking_checkpoint_sync_task.cancel()
        try:
            await ranking_checkpoint_sync_task
        except asyncio.CancelledError:
            pass
        ranking_checkpoint_sync_task = None
    if ranking_batcher:
        await ranking_batcher.close()
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
            "error": None if producer_health.get("connected") else "Kafka producer unavailable",
        }
    elif runtime.config.kafka_config.enable:
        kafka_health = {"status": "degraded", "error": "Kafka producer unavailable"}

    return build_readiness_response(
        runtime,
        {
            "redis": feature_store_health,
            "database": database_health,
            "vector_search": vector_health,
            "ranking_model": ranking_health,
            "kafka": kafka_health,
        },
    )


@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    http_request: Request,
    response: Response,
    payload: RecommendationRequest = Body(...),
):
    start_time = time.time()
    started_at = time.perf_counter()
    runtime = app.state.runtime
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
            (started_at - getattr(http_request.state, "request_started_at", started_at)) * 1000,
            2,
        ),
        "cache_lookup_ms": 0.0,
        "candidate_cache_lookup_ms": 0.0,
        "content_features_ms": 0.0,
        "user_features_ms": 0.0,
        "user_sequence_token_ms": 0.0,
        "candidate_generation_ms": 0.0,
        "candidate_cache_write_ms": 0.0,
        "metadata_lookup_ms": 0.0,
        "ranking_ms": 0.0,
        "cache_write_ms": 0.0,
        "analytics_log_ms": 0.0,
        "kafka_schedule_ms": 0.0,
        "total_ms": 0.0,
        "candidate_count": 0,
        "ranked_count": 0,
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
        stage_started = time.perf_counter()
        user_features = await _bounded_hot_path_read(
            runtime,
            "user_features",
            feature_store.get_user_features(payload.user_id, cache_default=False),
            None,
        )
        profile["user_features_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
        if user_features is None:
            user_features = _default_user_features(payload.user_id)

        stage_started = time.perf_counter()
        user_sequence_token = await _bounded_hot_path_read(
            runtime,
            "user_sequence_token",
            feature_store.get_user_sequence_token(payload.user_id),
            _default_user_sequence_token(),
        )
        profile["user_sequence_token_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)

        serving_versions = _serving_version_context(runtime)
        user_feature_token = _user_feature_cache_token(user_features)
        freshness_context = _build_cache_freshness_context(
            serving_versions,
            user_feature_token,
            user_sequence_token,
        )
        cache_key = feature_store.generate_context_hash(
            {
                **_build_recommendation_cache_context(
                    payload,
                    current_time=start_time,
                ),
                **freshness_context,
            }
        )
        stage_started = time.perf_counter()
        cached = await _bounded_hot_path_read(
            runtime,
            "cached_recommendations",
            feature_store.get_cached_recommendations(payload.user_id, cache_key),
            None,
        )
        profile["cache_lookup_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
        if cached:
            profile["cache_hit"] = True
            profile["serving_path"] = "recommendation_cache"
            profile["ranked_count"] = len(cached)
            profile["total_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            _attach_profile_headers(response, profile)
            _log_recommendation_profile(runtime, payload, profile)
            return RecommendationResponse(
                user_id=payload.user_id,
                recommendations=cached,
                metadata={
                    "total_candidates": len(cached),
                    "response_time_ms": int((time.time() - start_time) * 1000),
                    "model_version": "v1.0.0",
                    "cache_hit": True,
                    "cache_freshness": "model_user_sequence_and_catalog_versioned",
                    "content_processed": payload.content_id is not None,
                    **(
                        {"profile": profile}
                        if runtime.config.monitoring_config.enable_profiling_logs
                        else {}
                    ),
                },
            )

        singleflight_key = f"{payload.user_id}:{cache_key}"
        singleflight_future, owns_singleflight = await _join_or_create_recommendation_singleflight(
            singleflight_key
        )
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
            _attach_profile_headers(response, profile)
            _log_recommendation_profile(runtime, payload, profile)
            return RecommendationResponse(
                user_id=payload.user_id,
                recommendations=shared_result.get("recommendations", []),
                metadata={
                    "total_candidates": shared_result.get("total_candidates", 0),
                    "response_time_ms": int((time.time() - start_time) * 1000),
                    "model_version": "v1.0.0",
                    "cache_hit": False,
                    "cache_freshness": "model_user_sequence_and_catalog_versioned",
                    "singleflight_joined": True,
                    "content_processed": shared_result.get(
                        "content_processed",
                        payload.content_id is not None,
                    ),
                    **(
                        {"fallback": True}
                        if shared_result.get("fallback")
                        else {}
                    ),
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
        content_features_task = None
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

        candidates, profile["candidate_cache_lookup_ms"] = await candidate_cache_task
        content_features = None
        if content_features_task:
            content_features, profile["content_features_ms"] = await content_features_task

        candidate_profile = {"path": "candidate_cache", "candidate_count": len(candidates or [])}
        if candidates is not None:
            profile["candidate_cache_hit"] = True
            profile["serving_path"] = "candidate_cache_then_rank"
        else:
            stage_started = time.perf_counter()
            candidates, candidate_profile = await recommendation_engine.generate_candidates(
                user_id=payload.user_id,
                content_features=content_features,
                context=payload.context,
                k_per_source=k_per_source,
                include_profile=True,
                user_features=user_features,
            )
            profile["candidate_generation_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
            stage_started = time.perf_counter()
            _schedule_best_effort_task(
                "cache_candidate_products",
                feature_store.cache_candidate_products(
                    payload.user_id,
                    candidate_cache_key,
                    candidates,
                    user_features=user_features,
                ),
                timeout_seconds=runtime.config.cache_config.background_write_timeout_ms / 1000.0,
            )
            profile["candidate_cache_write_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
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
            return RecommendationResponse(
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
            )

        stage_started = time.perf_counter()
        product_ids = [candidate.product_id for candidate in candidates]
        product_metadata_map = await _bounded_hot_path_read(
            runtime,
            "product_metadata_batch",
            feature_store.get_product_metadata_batch(product_ids),
            {},
        )
        missing_product_ids = [product_id for product_id in product_ids if product_id not in product_metadata_map]
        if missing_product_ids:
            fetched_metadata = await vector_search.get_product_metadata_batch(missing_product_ids)
            if fetched_metadata:
                feature_store.prime_product_metadata_memory_cache(fetched_metadata)
                _schedule_best_effort_task(
                    "store_product_metadata_batch",
                    feature_store.store_product_metadata_batch(fetched_metadata),
                    timeout_seconds=runtime.config.cache_config.background_write_timeout_ms / 1000.0,
                )
                product_metadata_map.update(fetched_metadata)
            profile["metadata_cache_miss_count"] = len(missing_product_ids) - len(fetched_metadata)
        profile["metadata_lookup_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
        ranked_candidate_count = len(candidates)
        candidates = _filter_recommendable_candidates(candidates, product_metadata_map)
        profile["candidate_count_before_eligibility_filter"] = ranked_candidate_count
        profile["candidate_count"] = len(candidates)
        profile["filtered_unavailable_candidates"] = ranked_candidate_count - len(candidates)

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
            return RecommendationResponse(
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
            )

        stage_started = time.perf_counter()
        ranked_recommendations, ranking_profile = await ranking_batcher.rank_candidates(
            candidates=candidates,
            user_features=user_features,
            context=payload.context,
            product_metadata_map=product_metadata_map,
            k=payload.k,
            include_profile=True,
        )
        profile["ranking_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
        profile["ranked_count"] = len(ranked_recommendations)
        profile["ranking_profile"] = ranking_profile
        response_time = time.time() - start_time

        stage_started = time.perf_counter()
        _schedule_best_effort_task(
            "cache_recommendations",
            feature_store.cache_recommendations(
                payload.user_id,
                cache_key,
                [recommendation.dict() for recommendation in ranked_recommendations],
                user_features=user_features,
            ),
            timeout_seconds=runtime.config.cache_config.background_write_timeout_ms / 1000.0,
        )
        profile["cache_write_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)

        stage_started = time.perf_counter()
        _schedule_best_effort_task(
            "log_recommendation_request",
            feature_store.log_recommendation_request(
                payload.user_id,
                len(ranked_recommendations),
                response_time,
            ),
            timeout_seconds=runtime.config.cache_config.background_write_timeout_ms / 1000.0,
        )
        profile["analytics_log_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)

        if kafka_manager:
            stage_started = time.perf_counter()
            _schedule_best_effort_task(
                "send_recommendation_event",
                kafka_manager.send_recommendation_event(
                    user_id=payload.user_id,
                    recommendations=[item.product_id for item in ranked_recommendations],
                    response_time_ms=int(response_time * 1000),
                    request_id=getattr(http_request.state, "request_id", None),
                    metadata={
                        "content_id": payload.content_id,
                        "candidate_count": len(candidates),
                    },
                ),
                timeout_seconds=runtime.config.cache_config.background_write_timeout_ms / 1000.0,
            )
            profile["kafka_schedule_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)

        profile["total_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        _attach_profile_headers(response, profile)
        _log_recommendation_profile(runtime, payload, profile)
        await _resolve_recommendation_singleflight(
            singleflight_key,
            singleflight_future if owns_singleflight else None,
            result=_build_singleflight_result(
                ranked_recommendations,
                total_candidates=len(candidates),
                content_processed=payload.content_id is not None,
            ),
        )

        return RecommendationResponse(
            user_id=payload.user_id,
            recommendations=ranked_recommendations,
            metadata={
                "total_candidates": len(candidates),
                "response_time_ms": int(response_time * 1000),
                "model_version": "v1.0.0",
                "cache_freshness": "model_user_sequence_and_catalog_versioned",
                "cache_hit": False,
                "content_processed": payload.content_id is not None,
                **(
                    {"profile": profile}
                    if runtime.config.monitoring_config.enable_profiling_logs
                    else {}
                ),
            },
        )

    except asyncio.CancelledError as exc:
        await _resolve_recommendation_singleflight(
            singleflight_key,
            singleflight_future if owns_singleflight else None,
            exception=exc,
        )
        raise
    except Exception as exc:
        logger.error(f"Recommendation request failed: {exc}")
        logger.error(traceback.format_exc())
        profile["error"] = str(exc)
        profile["total_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        _attach_profile_headers(response, profile)
        _log_recommendation_profile(runtime, payload, profile, level="error")
        try:
            trending_recommendations = await recommendation_engine.get_trending_recommendations(payload.k)
            await _resolve_recommendation_singleflight(
                singleflight_key,
                singleflight_future if owns_singleflight else None,
                result=_build_singleflight_result(
                    trending_recommendations,
                    total_candidates=len(trending_recommendations),
                    content_processed=payload.content_id is not None,
                    fallback=True,
                    fallback_reason="serving_error",
                ),
            )
            return RecommendationResponse(
                user_id=payload.user_id,
                recommendations=trending_recommendations,
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


def _attach_profile_headers(response: Response, profile: dict) -> None:
    response.headers["X-Service-Process-Pid"] = str(profile["process_id"])
    response.headers["X-Worker-Active-Requests"] = str(profile["worker_active_requests_at_entry"])
    response.headers["X-Worker-Handled-Requests"] = str(profile["worker_handled_requests"])
    response.headers["X-Handler-Queue-Ms"] = str(profile["handler_queue_ms"])
    response.headers["X-Recommendation-Total-Ms"] = str(profile["total_ms"])
    response.headers["X-Torch-Num-Threads"] = str(profile["torch_num_threads"])
    response.headers["X-Torch-Num-Interop-Threads"] = str(profile["torch_num_interop_threads"])


async def _timed_awaitable(awaitable):
    started_at = time.perf_counter()
    result = await awaitable
    return result, round((time.perf_counter() - started_at) * 1000, 2)


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
    asyncio.create_task(
        _run_best_effort_task(task_name, awaitable, timeout_seconds),
        name=f"best-effort-{task_name}",
    )


async def _run_best_effort_task(task_name: str, awaitable, timeout_seconds: float) -> None:
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
    interval_seconds = max(1, int(runtime.config.ranking_config.checkpoint_sync_interval_seconds))
    model_path = runtime.config.model_config.ranking_model_path
    last_ranking_version: Optional[str] = ranking_model.model_version if ranking_model else None
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            if ranking_model and model_path and artifact_manager:
                latest_ranking = await artifact_manager.get_latest_model_checkpoint(
                    ModelArtifactManager.RANKING_MODEL_NAME
                )
                if latest_ranking and latest_ranking.model_version != last_ranking_version:
                    await artifact_manager.sync_latest_ranking_checkpoint()
                    if await ranking_model.reload_model_if_updated(model_path):
                        ranking_model.model_version = latest_ranking.model_version
                        last_ranking_version = latest_ranking.model_version
            if recommendation_engine:
                await recommendation_engine.sync_serving_artifacts_if_updated()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(f"Ranking checkpoint sync failed: {exc}")
            await asyncio.sleep(min(interval_seconds, 60))


def _detect_cgroup_cpu_limit() -> int:
    cpu_max_path = "/sys/fs/cgroup/cpu.max"
    if os.path.exists(cpu_max_path):
        try:
            quota, period = open(cpu_max_path, "r", encoding="utf-8").read().strip().split()
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


def _log_recommendation_profile(runtime, request, profile: dict, level: str = "info") -> None:
    runtime.observability.record_recommendation(
        result="error" if profile.get("error") else "success",
        cache_hit=bool(profile.get("cache_hit")),
        serving_path=str(profile.get("serving_path", "unknown")),
        candidate_count=int(profile.get("candidate_count") or 0),
        ranked_count=int(profile.get("ranked_count") or 0),
    )
    if not (
        runtime.config.monitoring_config.enable_profiling_logs
        or profile["total_ms"] >= runtime.config.monitoring_config.profiling_log_min_duration_ms
    ):
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
            "error": None if producer_health.get("connected") else "Kafka producer unavailable",
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
