"""
Dedicated internal ranking inference service.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, Response
import torch

from video_commerce.common.cache_codec import json_dumps, json_loads
from video_commerce.common.config import Config
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.ml.ranking import RankingModel
from video_commerce.ranking_runtime.ranking_batcher import (
    RankingBatcher,
    RankingQueueFullError,
    RankingQueueTimeoutError,
)
from video_commerce.ranking_runtime.ranking_coordinator_client import (
    RankingCoordinatorClientPool,
    RankingCoordinatorError,
)
from video_commerce.ranking_runtime.ranking_payloads import (
    RankRequest,
    coerce_rank_payload,
    model_payload,
)
from video_commerce.common.service_common import (
    build_health_response,
    build_liveness_payload,
    build_metrics_response,
    build_readiness_response,
    component_health,
    configure_service_logging,
    create_service_app,
    require_internal_service_auth,
)
from video_commerce.data_plane.system_store import SystemStore

logger = logging.getLogger(__name__)

app = create_service_app(
    title="Ranking Service",
    description="Internal batched ranking inference service",
    service_name="ranking-service",
)

ranking_model: Optional[RankingModel] = None
ranking_batcher: Optional[RankingBatcher] = None
system_store: Optional[SystemStore] = None
object_storage: Optional[ObjectStorage] = None
artifact_manager: Optional[ModelArtifactManager] = None
ranking_checkpoint_sync_task: Optional[asyncio.Task] = None
ranking_coordinator_client: Optional[RankingCoordinatorClientPool] = None


@app.middleware("http")
async def internal_auth(request: Request, call_next):
    require_internal_service_auth(request, app.state.runtime)
    return await call_next(request)


@app.on_event("startup")
async def startup_event():
    global ranking_model, ranking_batcher, system_store, object_storage
    global artifact_manager, ranking_checkpoint_sync_task
    global ranking_coordinator_client

    runtime = app.state.runtime
    runtime.config = Config()
    configure_service_logging(runtime)
    _configure_torch_runtime(runtime)

    topology = runtime.config.service_topology_config
    if topology.ranking_coordinator_host:
        ranking_coordinator_client = RankingCoordinatorClientPool(
            topology.ranking_coordinator_host,
            topology.ranking_coordinator_port,
            pool_size=topology.ranking_coordinator_client_pool_size,
            connect_timeout_seconds=topology.ranking_coordinator_connect_timeout_seconds,
            request_timeout_seconds=topology.ranking_coordinator_request_timeout_seconds,
        )
        logger.info(
            "ranking_service_proxy_worker_started",
            extra={
                "service": runtime.service_name,
                "process_id": os.getpid(),
                "coordinator_host": topology.ranking_coordinator_host,
                "coordinator_port": topology.ranking_coordinator_port,
                "coordinator_pool_size": topology.ranking_coordinator_client_pool_size,
                "configured_workers": topology.ranking_workers,
            },
        )
        return

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

    ranking_batcher = RankingBatcher(
        ranking_model,
        runtime.config.ranking_config,
        observability=runtime.observability,
    )
    await ranking_batcher.start()
    if runtime.config.ranking_config.checkpoint_sync_interval_seconds > 0:
        ranking_checkpoint_sync_task = asyncio.create_task(
            _periodic_ranking_checkpoint_sync(runtime),
            name="ranking-service-checkpoint-sync",
        )
    compile_status = ranking_model.get_stats()

    logger.info(
        "ranking_service_worker_started",
        extra={
            "service": runtime.service_name,
            "process_id": os.getpid(),
            "configured_workers": runtime.config.service_topology_config.ranking_workers,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "batch_max_requests": runtime.config.ranking_config.batch_max_requests,
            "batch_wait_ms": runtime.config.ranking_config.batch_wait_ms,
            "batch_runner_count": runtime.config.ranking_config.batch_runner_count,
            "torch_compile_enabled": compile_status["torch_compile_enabled"],
            "torch_compile_active": compile_status["torch_compile_active"],
            "torch_compile_backend": compile_status["torch_compile_backend"],
            "torch_compile_mode": compile_status["torch_compile_mode"],
            "torch_compile_dynamic": compile_status["torch_compile_dynamic"],
            "torch_compile_error": compile_status["torch_compile_error"],
            "torch_compile_warmup_ms": compile_status["torch_compile_warmup_ms"],
            "torch_compile_fallback_count": compile_status[
                "torch_compile_fallback_count"
            ],
            "torch_compile_last_fallback_error": compile_status[
                "torch_compile_last_fallback_error"
            ],
            "torch_compile_last_inference_path": compile_status[
                "torch_compile_last_inference_path"
            ],
        },
    )


@app.on_event("shutdown")
async def shutdown_event():
    global ranking_checkpoint_sync_task
    global ranking_coordinator_client

    if ranking_checkpoint_sync_task:
        ranking_checkpoint_sync_task.cancel()
        await asyncio.gather(ranking_checkpoint_sync_task, return_exceptions=True)
        ranking_checkpoint_sync_task = None
    if ranking_batcher:
        await ranking_batcher.close()
    if ranking_coordinator_client:
        await ranking_coordinator_client.aclose()
        ranking_coordinator_client = None
    if system_store:
        await system_store.close()


@app.get("/")
async def root():
    return {
        "service": "ranking-service",
        "version": "1.0.0",
        "health": "/health",
        "readyz": "/readyz",
    }


@app.get("/livez")
async def livez():
    return build_liveness_payload(app.state.runtime)


@app.get("/readyz")
async def readyz():
    if ranking_coordinator_client:
        try:
            upstream = await ranking_coordinator_client.health()
            return Response(
                content=upstream.body,
                media_type=upstream.content_type,
                status_code=upstream.status_code,
            )
        except RankingCoordinatorError as exc:
            return build_readiness_response(
                app.state.runtime,
                {
                    "ranking_coordinator": {
                        "status": "unhealthy",
                        "response_time_ms": 0.0,
                        "error": str(exc),
                    }
                },
            )
    ranking_health = (
        ranking_model.health_check() if ranking_model else {"status": "unhealthy"}
    )
    database_health = {"status": "healthy", "response_time_ms": 0.0}
    if system_store:
        database_status = await system_store.health_check()
        database_health = {
            "status": database_status.status,
            "response_time_ms": database_status.response_time_ms,
            "error": database_status.error,
        }
    return build_readiness_response(
        app.state.runtime,
        {
            "ranking_model": ranking_health,
            "database": database_health,
        },
    )


@app.get("/health")
async def health_check():
    if ranking_coordinator_client:
        try:
            upstream = await ranking_coordinator_client.health()
            return Response(
                content=upstream.body,
                media_type=upstream.content_type,
                status_code=upstream.status_code,
            )
        except RankingCoordinatorError as exc:
            return build_health_response(
                {
                    "ranking_coordinator": component_health(
                        "unhealthy",
                        error_message=str(exc),
                    )
                },
                app.state.runtime.started_at,
            )
    ranking_health = (
        ranking_model.health_check() if ranking_model else {"status": "unhealthy"}
    )
    database_health = {"status": "healthy", "response_time_ms": 0.0}
    if system_store:
        database_status = await system_store.health_check()
        database_health = {
            "status": database_status.status,
            "response_time_ms": database_status.response_time_ms,
            "error": database_status.error,
        }
    return build_health_response(
        {
            "ranking_model": component_health(
                ranking_health.get("status", "unhealthy"),
                error_message=ranking_health.get("error"),
            ),
            "database": component_health(
                database_health.get("status", "unhealthy"),
                database_health.get("response_time_ms"),
                database_health.get("error"),
            ),
        },
        app.state.runtime.started_at,
    )


@app.get("/metrics")
async def metrics():
    if ranking_coordinator_client:
        try:
            upstream = await ranking_coordinator_client.metrics()
            return Response(
                content=upstream.body,
                media_type=upstream.content_type,
                status_code=upstream.status_code,
            )
        except RankingCoordinatorError as exc:
            logger.warning("ranking_coordinator_metrics_unavailable: %s", exc)
    return await build_metrics_response(
        app.state.runtime,
        system_store=system_store,
    )


@app.post("/internal/rank")
async def rank(request: Request):
    if ranking_coordinator_client:
        try:
            upstream = await ranking_coordinator_client.rank(
                _body_with_deadline(await request.body(), app.state.runtime)
            )
            return Response(
                content=upstream.body,
                media_type=upstream.content_type,
                status_code=upstream.status_code,
            )
        except RankingCoordinatorError as exc:
            raise HTTPException(
                status_code=503,
                detail="Ranking coordinator unavailable",
            ) from exc

    if ranking_batcher is None:
        raise HTTPException(status_code=503, detail="Ranking batcher unavailable")
    payload = await _parse_rank_request(request)
    try:
        recommendations, profile = await ranking_batcher.rank_candidates(
            candidates=payload.candidates,
            user_features=payload.user_features,
            context={
                **payload.context,
                **(
                    {"temporal_multimodal": payload.multimodal_context}
                    if payload.multimodal_context
                    else {}
                ),
            },
            product_metadata_map=payload.product_metadata_map,
            k=payload.k,
            include_profile=True,
            deadline_unix_seconds=payload.deadline_unix_seconds,
        )
    except (RankingQueueFullError, RankingQueueTimeoutError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    profile = {
        **profile,
        "ranking_service_process_id": os.getpid(),
        "ranking_service_request_id": getattr(request.state, "request_id", None),
    }
    return Response(
        content=json_dumps(
            {
                "recommendations": [model_payload(item) for item in recommendations],
                "profile": profile,
            }
        ),
        media_type="application/json",
    )


async def _parse_rank_request(request: Request) -> RankRequest:
    """Parse the internal rank payload with low hot-path overhead.

    The recommendation service is the normal caller and already validates the
    public request. This keeps the internal JSON shape unchanged while avoiding
    FastAPI's recursive Pydantic conversion for every candidate on the ranking
    hot path.
    """
    try:
        raw_payload = json_loads(await request.body())
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    return coerce_rank_payload(raw_payload)


def _body_with_deadline(body: bytes, runtime) -> bytes:
    if not getattr(runtime, "config", None):
        return body
    try:
        payload = json_loads(body)
    except Exception:
        return body
    if not isinstance(payload, dict):
        return body
    timeout_seconds = (
        runtime.config.service_topology_config.ranking_coordinator_request_timeout_seconds
    )
    local_deadline = time.time() + max(
        0.05,
        float(timeout_seconds) - 0.1,
    )
    payload["deadline_unix_seconds"] = _conservative_deadline(
        payload.get("deadline_unix_seconds"),
        local_deadline,
    )
    return json_dumps(payload)


def _conservative_deadline(existing_deadline: Any, local_deadline: float) -> float:
    try:
        parsed_deadline = float(existing_deadline)
    except (TypeError, ValueError, OverflowError):
        return local_deadline
    if not math.isfinite(parsed_deadline):
        return local_deadline
    return min(parsed_deadline, local_deadline)


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
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(f"Ranking service checkpoint sync failed: {exc}")
            await asyncio.sleep(min(interval_seconds, 60))


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
