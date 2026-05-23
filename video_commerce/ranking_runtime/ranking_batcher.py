"""
Micro-batching queue for recommendation ranking inference.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from video_commerce.common.cache_codec import json_dumps, json_loads
from video_commerce.common.models import CandidateProduct, ProductRecommendation, UserFeatures
from video_commerce.ml.ranking import RankingModel
from video_commerce.ranking_runtime.ranking_coordinator_client import MAX_FRAME_BYTES
from video_commerce.ranking_runtime.ranking_payloads import coerce_candidate, coerce_user_features, model_payload
from video_commerce.ranking_runtime.ranking_runner_client import (
    RankingRunnerTimeout,
    RankingRunnerUnavailable,
    RankingRunnerClientPool,
)

logger = logging.getLogger(__name__)

_PROCESS_RANKING_MODEL: Optional[RankingModel] = None
_RANKING_PRODUCT_METADATA_FIELDS = (
    "title",
    "price",
    "category",
    "brand",
    "rating",
    "num_reviews",
    "in_stock",
    "created_at",
    "tags",
)


class RankingQueueFullError(RuntimeError):
    """Raised when ranking admission control rejects a request."""


class RankingQueueTimeoutError(TimeoutError):
    """Raised when a ranking request waits too long before inference."""


@dataclass
class RankingBatchRequest:
    candidates: List[CandidateProduct]
    user_features: UserFeatures
    context: Dict[str, Any]
    product_metadata_map: Dict[str, Dict[str, Any]]
    k: int
    include_profile: bool
    future: asyncio.Future
    enqueued_at: float
    deadline_unix_seconds: Optional[float] = None
    started_at: Optional[float] = None
    queue_timeout_handle: Optional[asyncio.TimerHandle] = None


def _configure_process_torch_runtime(config) -> None:
    """Torch thread settings are inherited from the coordinator before fork."""
    return None


def _initialize_ranking_process_worker() -> None:
    global _PROCESS_RANKING_MODEL
    from video_commerce.common.config import Config

    config = Config()
    _configure_process_torch_runtime(config)
    ranking_model = RankingModel(config.ranking_config)
    asyncio.run(ranking_model.load_model(config.model_config.ranking_model_path))
    _PROCESS_RANKING_MODEL = ranking_model


def _ranking_process_warmup() -> int:
    if _PROCESS_RANKING_MODEL is None:
        _initialize_ranking_process_worker()
    return os.getpid()


def _trim_product_metadata(metadata: Any) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {
        field: metadata[field]
        for field in _RANKING_PRODUCT_METADATA_FIELDS
        if field in metadata
    }


def normalize_ranking_batch_payloads(raw_payload: Any) -> List[Dict[str, Any]]:
    """Decode v2 batch-level metadata payloads while preserving v1 request lists."""
    if isinstance(raw_payload, dict):
        requests = raw_payload.get("requests")
        if not isinstance(requests, list):
            return []
        if raw_payload.get("payload_version") != 2:
            return requests

        batch_metadata_map = raw_payload.get("product_metadata_map") or {}
        if not isinstance(batch_metadata_map, dict):
            batch_metadata_map = {}
        normalized_requests: List[Dict[str, Any]] = []
        for request in requests:
            if not isinstance(request, dict):
                continue
            request_metadata_map = request.get("product_metadata_map")
            if not isinstance(request_metadata_map, dict):
                request_metadata_map = {}
            if not request_metadata_map:
                product_ids = request.get("product_ids")
                if not isinstance(product_ids, list):
                    product_ids = [
                        candidate.get("product_id")
                        for candidate in (request.get("candidates") or [])
                        if isinstance(candidate, dict)
                    ]
                request_metadata_map = {
                    str(product_id): batch_metadata_map[str(product_id)]
                    for product_id in product_ids
                    if product_id is not None
                    and str(product_id) in batch_metadata_map
                }
            normalized_requests.append(
                {
                    **request,
                    "product_metadata_map": request_metadata_map,
                }
            )
        return normalized_requests
    if isinstance(raw_payload, list):
        return raw_payload
    return []


def run_ranking_batch_payloads(
    ranking_model: RankingModel,
    batch_payloads: Any,
    *,
    profile_path: str = "torch_microbatch_process",
) -> Dict[str, Any]:
    """Execute a full ranking micro-batch and return JSON-serializable results."""
    batch_execution_started = time.perf_counter()
    stages: Dict[str, float] = {}
    normalized_requests: List[Dict[str, Any]] = []
    batch_payloads = normalize_ranking_batch_payloads(batch_payloads)

    feature_prep_started = time.perf_counter()
    for payload in batch_payloads:
        candidates = [
            candidate
            if isinstance(candidate, CandidateProduct)
            else coerce_candidate(candidate)
            for candidate in (payload.get("candidates") or [])
        ]
        user_features = payload["user_features"]
        if not isinstance(user_features, UserFeatures):
            user_features = coerce_user_features(user_features)
        normalized_requests.append(
            {
                "index": payload["index"],
                "candidates": candidates,
                "user_features": user_features,
                "context": payload.get("context") or {},
                "product_metadata_map": payload.get("product_metadata_map") or {},
                "k": int(payload["k"]),
                "batch_wait_ms": payload["batch_wait_ms"],
            }
        )

    feature_matrix, prepared, _ = ranking_model.prepare_batch_matrix(
        normalized_requests
    )
    stages["feature_prep"] = time.perf_counter() - feature_prep_started

    if not prepared:
        stages["total_execution"] = time.perf_counter() - batch_execution_started
        return {"results": [], "stages": stages}

    total_batch_candidates = 0 if feature_matrix is None else feature_matrix.shape[0]
    if feature_matrix is None:
        stages["total_execution"] = time.perf_counter() - batch_execution_started
        results = [
            (
                item["index"],
                [],
                {
                    "path": f"{profile_path}_empty",
                    "batch_request_count": len(prepared),
                    "batch_candidate_count": 0,
                    "batch_wait_ms": item["batch_wait_ms"],
                    "feature_extraction_ms": item["feature_extraction_ms"],
                    "tensor_prep_ms": 0.0,
                    "model_forward_ms": 0.0,
                    "response_build_ms": 0.0,
                    "total_ms": round(stages["total_execution"] * 1000, 2),
                    "candidate_count": item["candidate_count"],
                    "ranked_count": 0,
                },
            )
            for item in prepared
        ]
        return {"results": results, "stages": stages}

    inference_started = time.perf_counter()
    predictions, inference_profile = ranking_model.run_inference_batch(feature_matrix)
    stages["inference_total"] = time.perf_counter() - inference_started
    stages["tensor_prep"] = float(inference_profile.get("tensor_prep_ms", 0.0)) / 1000.0
    stages["model_forward"] = (
        float(inference_profile.get("model_forward_ms", 0.0)) / 1000.0
    )

    results = []
    response_build_started = time.perf_counter()
    for item in prepared:
        row_start = int(item["row_start"])
        row_end = int(item["row_end"])
        if row_end <= row_start:
            results.append(
                (
                    item["index"],
                    [],
                    {
                        "path": f"{profile_path}_empty",
                        "batch_request_count": len(prepared),
                        "batch_candidate_count": total_batch_candidates,
                        "batch_wait_ms": item["batch_wait_ms"],
                        "feature_extraction_ms": item["feature_extraction_ms"],
                        "tensor_prep_ms": inference_profile["tensor_prep_ms"],
                        "model_forward_ms": inference_profile["model_forward_ms"],
                        "response_build_ms": 0.0,
                        "total_ms": round(
                            (time.perf_counter() - batch_execution_started) * 1000, 2
                        ),
                        "candidate_count": item["candidate_count"],
                        "ranked_count": 0,
                    },
                )
            )
            continue

        request_predictions = {
            key: values[row_start:row_end] for key, values in predictions.items()
        }
        (
            recommendations,
            response_build_ms,
        ) = ranking_model.build_recommendations_from_predictions(
            item["valid_candidates"],
            request_predictions,
            int(item["k"]),
        )
        results.append(
            (
                item["index"],
                [model_payload(recommendation) for recommendation in recommendations],
                {
                    "path": profile_path,
                    "batch_request_count": len(prepared),
                    "batch_candidate_count": total_batch_candidates,
                    "batch_wait_ms": item["batch_wait_ms"],
                    "feature_extraction_ms": item["feature_extraction_ms"],
                    "tensor_prep_ms": inference_profile["tensor_prep_ms"],
                    "model_forward_ms": inference_profile["model_forward_ms"],
                    "response_build_ms": response_build_ms,
                    "total_ms": round(
                        (time.perf_counter() - batch_execution_started) * 1000, 2
                    ),
                    "candidate_count": item["candidate_count"],
                    "ranked_count": len(recommendations),
                },
            )
        )
    stages["response_build"] = time.perf_counter() - response_build_started
    stages["total_execution"] = time.perf_counter() - batch_execution_started
    return {"results": results, "stages": stages}


def _run_ranking_process_batch(batch_payloads: Any) -> Dict[str, Any]:
    if _PROCESS_RANKING_MODEL is None:
        _initialize_ranking_process_worker()
    ranking_model = _PROCESS_RANKING_MODEL
    assert ranking_model is not None
    return run_ranking_batch_payloads(
        ranking_model,
        batch_payloads,
        profile_path="torch_microbatch_process",
    )


class RankingBatcher:
    """Collect short-lived ranking requests into global batches.

    A single dispatcher owns batch formation so multiple workers do not split
    the same hot queue into undersized micro-batches. ``runner_count`` controls
    concurrent batch execution after a batch has already been formed.
    """

    def __init__(
        self,
        ranking_model: Optional[RankingModel],
        config,
        observability=None,
        runner_pool: Optional[RankingRunnerClientPool] = None,
    ) -> None:
        self.ranking_model = ranking_model
        self.observability = observability
        self.runner_pool = runner_pool
        self.enabled = config.enable_async_batching
        self.max_batch_requests = max(1, config.batch_max_requests)
        self.target_batch_requests = max(
            1,
            min(
                self.max_batch_requests,
                int(
                    getattr(config, "batch_target_requests", 0)
                    or self.max_batch_requests
                ),
            ),
        )
        self.batch_wait_seconds = max(config.batch_wait_ms, 0.0) / 1000.0
        self.max_queue_wait_seconds = max(
            0.0,
            float(getattr(config, "max_queue_wait_ms", 0.0) or 0.0) / 1000.0,
        )
        self.runner_payload_v2_enabled = bool(
            getattr(config, "runner_payload_v2_enabled", True)
        )
        self._runner_payload_v2_capability_warning_logged = False
        self.runner_payload_max_bytes = max(
            1024,
            min(
                int(getattr(config, "runner_payload_max_bytes", MAX_FRAME_BYTES - 1)),
                MAX_FRAME_BYTES - 1,
            ),
        )
        self.offload_inference_to_thread = getattr(
            config, "offload_inference_to_thread", True
        )
        self.runner_count = max(
            1,
            getattr(
                config,
                "coordinator_dispatch_concurrency",
                getattr(config, "batch_runner_count", 1),
            )
            if self.runner_pool is not None
            else getattr(config, "batch_runner_count", 1),
        )
        self.process_workers = max(
            0,
            0
            if self.runner_pool is not None
            else int(getattr(config, "inference_process_workers", 0) or 0),
        )
        self._runner_capacity = (
            self.runner_pool.capacity
            if self.runner_pool is not None
            else self.process_workers or self.runner_count
        )
        self.executor_workers = max(0, getattr(config, "inference_executor_workers", 0))
        self.queue: asyncio.Queue[Optional[RankingBatchRequest]] = asyncio.Queue(
            maxsize=max(1, config.batch_queue_size)
        )
        self._batch_queue: asyncio.Queue[
            Optional[List[RankingBatchRequest]]
        ] = asyncio.Queue(maxsize=max(1, self.runner_count * 2))
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._worker_tasks: List[asyncio.Task] = []
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(
                max_workers=self.executor_workers,
                thread_name_prefix="ranking-inference",
            )
            if self.runner_pool is None
            and self.offload_inference_to_thread
            and self.executor_workers > 0
            and self.process_workers <= 0
            else None
        )
        self._process_executor: Optional[ProcessPoolExecutor] = (
            ProcessPoolExecutor(
                max_workers=self.process_workers,
                initializer=_initialize_ranking_process_worker,
            )
            if self.process_workers > 0
            else None
        )
        self._closing = False
        self._active_batch_count = 0
        self._batch_execution_ewma_seconds = max(self.batch_wait_seconds, 0.001)
        self._runner_service_ewma_seconds = max(self.batch_wait_seconds, 0.001)
        self._dispatch_wait_ewma_seconds = 0.0

    async def start(self) -> None:
        if not self.enabled or self._dispatcher_task:
            return
        if self._process_executor is not None:
            await self._warm_process_executor()
        self._dispatcher_task = asyncio.create_task(
            self._dispatch(), name="ranking-batcher-dispatcher"
        )
        self._worker_tasks = [
            asyncio.create_task(
                self._run_worker(worker_id), name=f"ranking-batcher-worker-{worker_id}"
            )
            for worker_id in range(self.runner_count)
        ]

    async def _warm_process_executor(self) -> None:
        assert self._process_executor is not None
        loop = asyncio.get_running_loop()
        warmups = [
            loop.run_in_executor(self._process_executor, _ranking_process_warmup)
            for _ in range(max(1, self.process_workers))
        ]
        await asyncio.gather(*warmups)

    async def close(self) -> None:
        self._closing = True
        if self._dispatcher_task:
            await self.queue.put(None)
            await self._dispatcher_task
            self._dispatcher_task = None
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            self._worker_tasks = []
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        if self._process_executor:
            self._process_executor.shutdown(wait=True, cancel_futures=False)
            self._process_executor = None

    async def rank_candidates(
        self,
        candidates: List[CandidateProduct],
        user_features: UserFeatures,
        context: Dict[str, Any],
        k: int,
        product_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
        include_profile: bool = False,
        deadline_unix_seconds: Optional[float] = None,
    ):
        product_metadata_map = product_metadata_map or {}
        if self._deadline_expired(deadline_unix_seconds):
            self._record_direct("deadline_expired_before_enqueue")
            raise RankingQueueTimeoutError("ranking_queue_wait_exceeded")

        if not self.enabled:
            self._record_direct("batching_disabled")
            return await self._rank_direct(
                candidates=candidates,
                user_features=user_features,
                context=context,
                product_metadata_map=product_metadata_map,
                k=k,
                include_profile=include_profile,
            )

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request = RankingBatchRequest(
            candidates=candidates,
            user_features=user_features,
            context=context,
            product_metadata_map=product_metadata_map,
            k=k,
            include_profile=include_profile,
            future=future,
            enqueued_at=time.perf_counter(),
            deadline_unix_seconds=deadline_unix_seconds,
        )

        if self._estimated_queue_wait_exceeded(deadline_unix_seconds):
            self._record_direct("estimated_queue_wait_exceeded")
            raise RankingQueueTimeoutError("ranking_queue_wait_exceeded")

        if self.max_queue_wait_seconds > 0:
            queue_timeout_seconds = self.max_queue_wait_seconds
            if deadline_unix_seconds is not None:
                queue_timeout_seconds = min(
                    queue_timeout_seconds,
                    max(0.0, float(deadline_unix_seconds) - time.time()),
                )
            request.queue_timeout_handle = loop.call_later(
                queue_timeout_seconds,
                self._expire_queued_request,
                request,
            )

        try:
            self.queue.put_nowait(request)
        except asyncio.QueueFull as exc:
            self._cancel_queue_timeout(request)
            self._record_direct("queue_full")
            raise RankingQueueFullError("ranking_batch_queue_full") from exc

        self._record_queue_depth()
        return await future

    def _estimated_queue_wait_exceeded(
        self,
        deadline_unix_seconds: Optional[float] = None,
    ) -> bool:
        if self.max_queue_wait_seconds <= 0:
            return False
        if (
            self.runner_pool is not None
            and not self.runner_pool.has_available_endpoint()
        ):
            return True

        pending_requests = self.queue.qsize()
        pending_batches = self._batch_queue.qsize()
        queued_batch_waves = math.ceil(
            pending_requests / max(1, self.target_batch_requests)
        )
        runner_capacity = max(
            1,
            self.runner_pool.capacity
            if self.runner_pool is not None
            else self._runner_capacity,
        )
        service_seconds = max(
            self.batch_wait_seconds,
            self._runner_service_ewma_seconds,
            0.001,
        )
        batches_before_start = max(
            0,
            self._active_batch_count + pending_batches + queued_batch_waves + 1,
        )
        batches_ahead = max(0, batches_before_start - runner_capacity)
        estimated_wait_seconds = (batches_ahead / runner_capacity) * service_seconds
        allowed_wait_seconds = self.max_queue_wait_seconds
        if deadline_unix_seconds is not None:
            allowed_wait_seconds = min(
                allowed_wait_seconds,
                max(0.0, float(deadline_unix_seconds) - time.time()),
            )
        return estimated_wait_seconds > allowed_wait_seconds

    def should_reject_new_request(
        self,
        deadline_unix_seconds: Optional[float] = None,
    ) -> bool:
        """Return whether admission control would reject another request now."""
        if self._closing:
            return True
        if self.queue.full():
            return True
        return self._estimated_queue_wait_exceeded(deadline_unix_seconds)

    @staticmethod
    def _deadline_expired(deadline_unix_seconds: Optional[float]) -> bool:
        return (
            deadline_unix_seconds is not None
            and float(deadline_unix_seconds) <= time.time()
        )

    def _expire_queued_request(self, request: RankingBatchRequest) -> None:
        if request.started_at is not None:
            return
        if request.future.done():
            return
        request.future.set_exception(
            RankingQueueTimeoutError("ranking_queue_wait_exceeded")
        )
        self._record_cancelled("queue_wait_exceeded")

    def _cancel_queue_timeout(self, request: RankingBatchRequest) -> None:
        handle = getattr(request, "queue_timeout_handle", None)
        if handle is not None and not handle.cancelled():
            handle.cancel()
        request.queue_timeout_handle = None

    def _runner_dispatch_slot_available(self) -> bool:
        if self.runner_pool is None:
            return True
        if hasattr(self.runner_pool, "has_dispatch_capacity"):
            return bool(self.runner_pool.has_dispatch_capacity())
        return bool(self.runner_pool.has_available_endpoint())

    def _mark_started(
        self, requests: List[RankingBatchRequest], started_at: float
    ) -> None:
        for request in requests:
            request.started_at = started_at
            self._cancel_queue_timeout(request)

    async def _dispatch(self) -> None:
        try:
            while True:
                first = await self.queue.get()
                if first is None:
                    break

                batch = [first]
                deadline = first.enqueued_at + self.batch_wait_seconds
                while len(batch) < self.max_batch_requests:
                    if (
                        len(batch) >= self.target_batch_requests
                        and self._runner_dispatch_slot_available()
                    ):
                        break
                    try:
                        next_item = self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        timeout = deadline - time.perf_counter()
                        if timeout <= 0:
                            break
                        try:
                            next_item = await asyncio.wait_for(
                                self.queue.get(), timeout=timeout
                            )
                        except asyncio.TimeoutError:
                            break

                    if next_item is None:
                        self._closing = True
                        break
                    batch.append(next_item)

                await self._batch_queue.put(batch)
                self._record_queue_depth()
                if self._closing:
                    break
        finally:
            for _ in self._worker_tasks:
                await self._batch_queue.put(None)

    async def _run_worker(self, worker_id: int) -> None:
        while True:
            batch = await self._batch_queue.get()
            if batch is None:
                break
            await self._fulfill_batch(batch)

    async def _rank_direct(
        self,
        candidates: List[CandidateProduct],
        user_features: UserFeatures,
        context: Dict[str, Any],
        product_metadata_map: Dict[str, Dict[str, Any]],
        k: int,
        include_profile: bool,
    ):
        if self.ranking_model is None:
            raise RankingQueueTimeoutError("ranking_model_unavailable")
        return await self._run_blocking(
            self.ranking_model._rank_candidates_sync,
            candidates,
            user_features,
            context,
            k,
            include_profile,
            product_metadata_map,
        )

    async def _run_blocking(self, func, *args):
        if self._executor is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, func, *args)
        if self.offload_inference_to_thread:
            return await asyncio.to_thread(func, *args)
        return func(*args)

    async def _fulfill_single(self, request: RankingBatchRequest) -> None:
        started_at = time.perf_counter()
        if not self._request_is_executable(request, started_at):
            return
        self._mark_started([request], started_at)
        self._record_batch(
            request_count=1,
            candidate_count=len(request.candidates),
            queue_wait_seconds=max(0.0, started_at - request.enqueued_at),
            path="single_request",
        )
        try:
            result = await self._rank_direct(
                candidates=request.candidates,
                user_features=request.user_features,
                context=request.context,
                product_metadata_map=request.product_metadata_map,
                k=request.k,
                include_profile=request.include_profile,
            )
        except Exception as exc:
            if not request.future.done():
                request.future.set_exception(exc)
            return

        if not request.future.done():
            request.future.set_result(result)

    async def _fulfill_batch(self, batch: List[RankingBatchRequest]) -> None:
        batch_started = time.perf_counter()
        executable_batch = [
            request
            for request in batch
            if self._request_is_executable(request, batch_started)
        ]
        if not executable_batch:
            return
        self._mark_started(executable_batch, batch_started)

        self._record_batch(
            request_count=len(executable_batch),
            candidate_count=sum(
                len(request.candidates) for request in executable_batch
            ),
            queue_wait_seconds=max(
                0.0,
                max(
                    batch_started - request.enqueued_at for request in executable_batch
                ),
            ),
            path="microbatch",
        )
        self._inc_active_batches()
        try:
            if self.runner_pool is not None:
                results = await self._run_remote_batch(
                    executable_batch,
                    batch_started,
                )
            elif self._process_executor is not None:
                results = await self._run_process_batch(
                    executable_batch,
                    batch_started,
                )
            else:
                results = await self._run_blocking(
                    self._build_batch_results,
                    executable_batch,
                    batch_started,
                )
            for request, recommendations, profile in results:
                self._set_result(request, recommendations, profile)

        except Exception as exc:
            logger.error(f"ranking_microbatch_failed: {exc}")
            if self.runner_pool is not None:
                self._record_batch_stage(
                    path="microbatch_runner",
                    stage=self._remote_failure_stage(exc),
                    duration_seconds=max(0.0, time.perf_counter() - batch_started),
                )
                error_detail = (
                    str(exc)
                    if isinstance(exc, RankingQueueTimeoutError) and str(exc)
                    else "ranking_runner_unavailable"
                )
                error = RankingQueueTimeoutError(error_detail)
                for request in executable_batch:
                    self._record_direct("remote_microbatch_failed")
                    if not request.future.done():
                        request.future.set_exception(error)
                return
            for request in executable_batch:
                self._record_direct("microbatch_failed")
                await self._fulfill_single(request)
        finally:
            batch_duration = max(0.001, time.perf_counter() - batch_started)
            if self.runner_pool is None:
                self._update_runner_service_ewma(batch_duration)
            self._update_batch_execution_ewma(batch_duration)
            self._dec_active_batches()

    def _build_batch_payloads(
        self,
        batch: List[RankingBatchRequest],
        batch_started: float,
        *,
        serialize_models: bool,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "index": index,
                "candidates": (
                    [model_payload(candidate) for candidate in request.candidates]
                    if serialize_models
                    else request.candidates
                ),
                "user_features": (
                    model_payload(request.user_features)
                    if serialize_models
                    else request.user_features
                ),
                "context": request.context,
                "product_metadata_map": request.product_metadata_map,
                "k": request.k,
                "batch_wait_ms": round(
                    max(0.0, batch_started - request.enqueued_at) * 1000,
                    2,
                ),
                "deadline_unix_seconds": request.deadline_unix_seconds,
            }
            for index, request in enumerate(batch)
        ]

    def _build_remote_batch_payload(
        self,
        batch: List[RankingBatchRequest],
        batch_started: float,
    ) -> Dict[str, Any]:
        if not self._remote_payload_v2_supported():
            return {
                "payload_version": 1,
                "requests": self._build_batch_payloads(
                    batch,
                    batch_started,
                    serialize_models=True,
                ),
            }

        requests: List[Dict[str, Any]] = []
        product_metadata_map: Dict[str, Dict[str, Any]] = {}
        for index, request in enumerate(batch):
            candidates = [model_payload(candidate) for candidate in request.candidates]
            product_ids: List[str] = []
            for candidate_payload in candidates:
                product_id = str(candidate_payload.get("product_id") or "")
                if not product_id:
                    continue
                product_ids.append(product_id)
                metadata = request.product_metadata_map.get(product_id)
                if metadata is not None and product_id not in product_metadata_map:
                    product_metadata_map[product_id] = _trim_product_metadata(metadata)
            requests.append(
                {
                    "index": index,
                    "candidates": candidates,
                    "product_ids": product_ids,
                    "user_features": model_payload(request.user_features),
                    "context": request.context,
                    "k": request.k,
                    "batch_wait_ms": round(
                        max(0.0, batch_started - request.enqueued_at) * 1000,
                        2,
                    ),
                    "deadline_unix_seconds": request.deadline_unix_seconds,
                }
            )
        return {
            "payload_version": 2,
            "product_metadata_map": product_metadata_map,
            "requests": requests,
        }

    def _remote_payload_v2_supported(self) -> bool:
        if not self.runner_payload_v2_enabled:
            return False
        if self.runner_pool is None:
            return True

        supports_version = getattr(
            self.runner_pool,
            "supports_batch_payload_version",
            None,
        )
        supported = bool(supports_version and supports_version(2))
        if supported:
            return True

        self._record_direct("runner_payload_v2_capability_mismatch")
        if not self._runner_payload_v2_capability_warning_logged:
            logger.warning(
                "ranking_runner_payload_v2_disabled_capability_mismatch"
            )
            self._runner_payload_v2_capability_warning_logged = True
        return False

    def _build_remote_batch_body(
        self,
        batch: List[RankingBatchRequest],
        batch_started: float,
    ) -> bytes:
        payload = self._build_remote_batch_payload(batch, batch_started)
        body = json_dumps(payload)
        payload_size = len(body)
        self._record_runner_payload_size(
            payload_version=str(payload.get("payload_version", "unknown")),
            size_bytes=payload_size,
        )
        if payload_size > self.runner_payload_max_bytes:
            self._record_direct("remote_payload_too_large")
            raise RankingQueueTimeoutError("ranking_runner_payload_too_large")
        return body

    def _decode_batch_payload_result(
        self,
        batch: List[RankingBatchRequest],
        process_result: Dict[str, Any],
    ) -> List[Tuple[RankingBatchRequest, List[ProductRecommendation], Dict[str, Any]]]:
        runner_process_id = process_result.get("runner_process_id")
        raw_results = process_result.get("results")
        if not isinstance(raw_results, list):
            raise RankingQueueTimeoutError("ranking_runner_invalid_response")

        results: List[
            Tuple[RankingBatchRequest, List[ProductRecommendation], Dict[str, Any]]
        ] = []
        seen_indexes = set()
        for raw_result in raw_results:
            if not isinstance(raw_result, (list, tuple)) or len(raw_result) != 3:
                raise RankingQueueTimeoutError("ranking_runner_invalid_response")

            index, recommendation_payloads, profile = raw_result
            try:
                result_index = int(index)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RankingQueueTimeoutError(
                    "ranking_runner_invalid_response"
                ) from exc
            if result_index < 0 or result_index >= len(batch):
                raise RankingQueueTimeoutError("ranking_runner_invalid_response")
            if result_index in seen_indexes:
                raise RankingQueueTimeoutError(
                    "ranking_runner_incomplete_response"
                )
            seen_indexes.add(result_index)

            if not isinstance(recommendation_payloads, list):
                raise RankingQueueTimeoutError("ranking_runner_invalid_response")
            if not isinstance(profile, dict):
                raise RankingQueueTimeoutError("ranking_runner_invalid_response")

            request = batch[result_index]
            try:
                recommendations = [
                    ProductRecommendation.construct(**payload)
                    for payload in recommendation_payloads
                ]
            except Exception as exc:
                raise RankingQueueTimeoutError(
                    "ranking_runner_invalid_response"
                ) from exc
            if runner_process_id is not None:
                profile = {**profile, "ranking_runner_process_id": runner_process_id}
            results.append((request, recommendations, profile))

        if seen_indexes != set(range(len(batch))):
            raise RankingQueueTimeoutError("ranking_runner_incomplete_response")
        return results

    async def _run_remote_batch(
        self,
        batch: List[RankingBatchRequest],
        batch_started: float,
    ) -> List[Tuple[RankingBatchRequest, List[ProductRecommendation], Dict[str, Any]]]:
        if self.runner_pool is None:
            return self._build_batch_results(batch, batch_started)

        body = self._build_remote_batch_body(batch, batch_started)
        timeout_seconds = self._remaining_batch_deadline_seconds(batch)
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise RankingQueueTimeoutError("ranking_queue_wait_exceeded")
        response = await self.runner_pool.rank_batch(
            body,
            timeout_seconds=timeout_seconds,
        )
        dispatch_wait_seconds = getattr(response, "runner_slot_wait_seconds", None)
        if dispatch_wait_seconds is not None:
            self._update_dispatch_wait_ewma(float(dispatch_wait_seconds))
        if response.status_code != 200:
            detail = "ranking_runner_unavailable"
            try:
                error_payload = json_loads(response.body)
                if isinstance(error_payload, dict) and error_payload.get("detail"):
                    detail = str(error_payload["detail"])
            except Exception:
                pass
            raise RankingQueueTimeoutError(detail)

        process_result = json_loads(response.body)
        if not isinstance(process_result, dict):
            raise RankingQueueTimeoutError("ranking_runner_invalid_response")
        stages = process_result.get("stages") or {}
        total_execution = stages.get("total_execution")
        if total_execution is not None:
            self._update_runner_service_ewma(float(total_execution))
        for stage, duration_seconds in stages.items():
            self._record_batch_stage(
                path="microbatch_runner",
                stage=stage,
                duration_seconds=float(duration_seconds),
            )
        return self._decode_batch_payload_result(batch, process_result)

    @staticmethod
    def _remote_failure_stage(exc: BaseException) -> str:
        message = str(exc)
        if isinstance(exc, RankingRunnerTimeout) or "request timeout" in message:
            return "remote_timeout"
        if "payload_too_large" in message or "frame too large" in message:
            return "remote_payload_too_large"
        if "deadline" in message:
            return "remote_deadline_exceeded"
        if "overloaded" in message or "capacity unavailable" in message:
            return "remote_overloaded"
        if "protocol" in message or "invalid_response" in message:
            return "remote_protocol_error"
        if isinstance(exc, RankingRunnerUnavailable) or "unavailable" in message:
            return "remote_unhealthy"
        return "remote_failure"

    async def _run_process_batch(
        self,
        batch: List[RankingBatchRequest],
        batch_started: float,
    ) -> List[Tuple[RankingBatchRequest, List[ProductRecommendation], Dict[str, Any]]]:
        if self._process_executor is None:
            return self._build_batch_results(batch, batch_started)

        payloads = self._build_batch_payloads(
            batch,
            batch_started,
            serialize_models=True,
        )
        loop = asyncio.get_running_loop()
        process_result = await loop.run_in_executor(
            self._process_executor,
            _run_ranking_process_batch,
            payloads,
        )
        for stage, duration_seconds in (process_result.get("stages") or {}).items():
            self._record_batch_stage(
                path="microbatch_process",
                stage=stage,
                duration_seconds=float(duration_seconds),
            )

        return self._decode_batch_payload_result(batch, process_result)

    def _request_is_executable(self, request: RankingBatchRequest, now: float) -> bool:
        if request.future.cancelled() or request.future.done():
            self._cancel_queue_timeout(request)
            self._record_cancelled("caller_cancelled")
            return False
        if self._deadline_expired(request.deadline_unix_seconds):
            if not request.future.done():
                request.future.set_exception(
                    RankingQueueTimeoutError("ranking_queue_wait_exceeded")
                )
            self._cancel_queue_timeout(request)
            self._record_cancelled("deadline_exceeded")
            return False
        if (
            self.max_queue_wait_seconds > 0
            and now - request.enqueued_at > self.max_queue_wait_seconds
        ):
            if not request.future.done():
                request.future.set_exception(
                    RankingQueueTimeoutError("ranking_queue_wait_exceeded")
                )
            self._cancel_queue_timeout(request)
            self._record_cancelled("queue_wait_exceeded")
            return False
        return True

    @staticmethod
    def _remaining_batch_deadline_seconds(
        batch: List[RankingBatchRequest],
    ) -> Optional[float]:
        deadlines = [
            float(request.deadline_unix_seconds)
            for request in batch
            if request.deadline_unix_seconds is not None
        ]
        if not deadlines:
            return None
        return min(deadlines) - time.time()

    def _build_batch_results(
        self,
        batch: List[RankingBatchRequest],
        batch_started: float,
    ) -> List[Tuple[RankingBatchRequest, List[ProductRecommendation], Dict[str, Any]]]:
        if self.ranking_model is None:
            raise RankingQueueTimeoutError("ranking_model_unavailable")
        payloads = self._build_batch_payloads(
            batch,
            batch_started,
            serialize_models=False,
        )
        process_result = run_ranking_batch_payloads(
            self.ranking_model,
            payloads,
            profile_path="torch_microbatch",
        )
        for stage, duration_seconds in (process_result.get("stages") or {}).items():
            self._record_batch_stage(
                path="microbatch",
                stage=stage,
                duration_seconds=float(duration_seconds),
            )
        return self._decode_batch_payload_result(batch, process_result)

    def _set_result(
        self,
        request: RankingBatchRequest,
        recommendations: List[ProductRecommendation],
        profile: Dict[str, Any],
    ) -> None:
        if request.future.done():
            return
        if request.include_profile:
            request.future.set_result((recommendations, profile))
        else:
            request.future.set_result(recommendations)

    def _record_queue_depth(self) -> None:
        if self.observability and hasattr(
            self.observability, "set_ranking_queue_depth"
        ):
            self.observability.set_ranking_queue_depth(self.queue.qsize())

    def _update_batch_execution_ewma(self, duration_seconds: float) -> None:
        previous = self._batch_execution_ewma_seconds
        self._batch_execution_ewma_seconds = (previous * 0.8) + (duration_seconds * 0.2)

    def _update_runner_service_ewma(self, duration_seconds: float) -> None:
        previous = self._runner_service_ewma_seconds
        self._runner_service_ewma_seconds = (previous * 0.8) + (
            max(0.001, float(duration_seconds)) * 0.2
        )

    def _update_dispatch_wait_ewma(self, duration_seconds: float) -> None:
        previous = self._dispatch_wait_ewma_seconds
        self._dispatch_wait_ewma_seconds = (previous * 0.8) + (
            max(0.0, float(duration_seconds)) * 0.2
        )

    def _record_batch(
        self,
        *,
        request_count: int,
        candidate_count: int,
        queue_wait_seconds: float,
        path: str,
    ) -> None:
        if self.observability and hasattr(self.observability, "record_ranking_batch"):
            self.observability.record_ranking_batch(
                request_count=request_count,
                candidate_count=candidate_count,
                queue_wait_seconds=queue_wait_seconds,
                path=path,
                max_batch_requests=self.max_batch_requests,
                target_batch_requests=self.target_batch_requests,
            )

    def _record_direct(self, reason: str) -> None:
        if self.observability and hasattr(self.observability, "record_ranking_direct"):
            self.observability.record_ranking_direct(reason)

    def _record_cancelled(self, reason: str) -> None:
        if self.observability and hasattr(
            self.observability, "record_ranking_cancelled"
        ):
            self.observability.record_ranking_cancelled(reason)

    def _record_batch_stage(
        self,
        *,
        path: str,
        stage: str,
        duration_seconds: float,
    ) -> None:
        if self.observability and hasattr(
            self.observability, "record_ranking_batch_stage"
        ):
            self.observability.record_ranking_batch_stage(
                path=path,
                stage=stage,
                duration_seconds=duration_seconds,
            )

    def _record_runner_payload_size(
        self,
        *,
        payload_version: str,
        size_bytes: int,
    ) -> None:
        if self.observability and hasattr(
            self.observability,
            "observe_ranking_runner_payload_bytes",
        ):
            self.observability.observe_ranking_runner_payload_bytes(
                payload_version=payload_version,
                size_bytes=size_bytes,
            )

    def _inc_active_batches(self) -> None:
        self._active_batch_count += 1
        if self.observability and hasattr(
            self.observability, "inc_ranking_active_batches"
        ):
            self.observability.inc_ranking_active_batches()

    def _dec_active_batches(self) -> None:
        self._active_batch_count = max(0, self._active_batch_count - 1)
        if self.observability and hasattr(
            self.observability, "dec_ranking_active_batches"
        ):
            self.observability.dec_ranking_active_batches()
