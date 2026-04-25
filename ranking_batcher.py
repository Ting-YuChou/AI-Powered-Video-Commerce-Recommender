"""
Micro-batching queue for recommendation ranking inference.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models import CandidateProduct, ProductRecommendation, UserFeatures
from ranking import RankingModel

logger = logging.getLogger(__name__)


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


class RankingBatcher:
    """Collect short-lived ranking requests into one model forward pass."""

    def __init__(self, ranking_model: RankingModel, config) -> None:
        self.ranking_model = ranking_model
        self.enabled = config.enable_async_batching
        self.max_batch_requests = max(1, config.batch_max_requests)
        self.batch_wait_seconds = max(config.batch_wait_ms, 0.0) / 1000.0
        self.offload_inference_to_thread = getattr(config, "offload_inference_to_thread", True)
        self.runner_count = max(1, getattr(config, "batch_runner_count", 1))
        self.executor_workers = max(0, getattr(config, "inference_executor_workers", 0))
        self.queue: asyncio.Queue[Optional[RankingBatchRequest]] = asyncio.Queue(
            maxsize=max(1, config.batch_queue_size)
        )
        self._runner_tasks: List[asyncio.Task] = []
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(
                max_workers=self.executor_workers,
                thread_name_prefix="ranking-inference",
            )
            if self.offload_inference_to_thread and self.executor_workers > 0
            else None
        )
        self._closing = False

    async def start(self) -> None:
        if not self.enabled or self._runner_tasks:
            return
        self._runner_tasks = [
            asyncio.create_task(self._run(runner_id), name=f"ranking-batcher-{runner_id}")
            for runner_id in range(self.runner_count)
        ]

    async def close(self) -> None:
        if self._runner_tasks:
            self._closing = True
            for _ in self._runner_tasks:
                await self.queue.put(None)
            await asyncio.gather(*self._runner_tasks, return_exceptions=True)
            self._runner_tasks = []
        self._closing = True
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

    async def rank_candidates(
        self,
        candidates: List[CandidateProduct],
        user_features: UserFeatures,
        context: Dict[str, Any],
        k: int,
        product_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
        include_profile: bool = False,
    ):
        product_metadata_map = product_metadata_map or {}
        if not self.enabled:
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
        )

        try:
            self.queue.put_nowait(request)
        except asyncio.QueueFull:
            logger.warning("ranking_batch_queue_full_fallback_direct")
            return await self._rank_direct(
                candidates=candidates,
                user_features=user_features,
                context=context,
                product_metadata_map=product_metadata_map,
                k=k,
                include_profile=include_profile,
            )

        return await future

    async def _run(self, runner_id: int) -> None:
        while True:
            first = await self.queue.get()
            if first is None:
                break

            batch = [first]
            deadline = time.perf_counter() + self.batch_wait_seconds

            while len(batch) < self.max_batch_requests:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(self.queue.get(), timeout)
                except asyncio.TimeoutError:
                    break
                if next_item is None:
                    self._closing = True
                    break
                batch.append(next_item)

            if len(batch) == 1:
                await self._fulfill_single(batch[0])
            else:
                await self._fulfill_batch(batch)

            if self._closing and self.queue.empty():
                break

    async def _fulfill_single(self, request: RankingBatchRequest) -> None:
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

    async def _rank_direct(
        self,
        candidates: List[CandidateProduct],
        user_features: UserFeatures,
        context: Dict[str, Any],
        product_metadata_map: Dict[str, Dict[str, Any]],
        k: int,
        include_profile: bool,
    ):
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

    async def _fulfill_batch(self, batch: List[RankingBatchRequest]) -> None:
        batch_started = time.perf_counter()
        try:
            results = await self._run_blocking(
                self._build_batch_results,
                batch,
                batch_started,
            )
            for request, recommendations, profile in results:
                self._set_result(request, recommendations, profile)

        except Exception as exc:
            logger.error(f"ranking_microbatch_failed: {exc}")
            for request in batch:
                await self._fulfill_single(request)

    def _build_batch_results(
        self,
        batch: List[RankingBatchRequest],
        batch_started: float,
    ) -> List[Tuple[RankingBatchRequest, List[ProductRecommendation], Dict[str, Any]]]:
        prepared = []
        feature_matrices = []
        total_batch_candidates = 0

        for request in batch:
            feature_matrix, valid_candidates, feature_extraction_ms = (
                self.ranking_model.prepare_request_matrix(
                    request.candidates,
                    request.user_features,
                    request.context,
                    product_metadata_map=request.product_metadata_map,
                )
            )
            prepared.append(
                {
                    "request": request,
                    "feature_matrix": feature_matrix,
                    "valid_candidates": valid_candidates,
                    "feature_extraction_ms": feature_extraction_ms,
                }
            )
            if feature_matrix is not None:
                feature_matrices.append(feature_matrix)
                total_batch_candidates += feature_matrix.shape[0]

        if not feature_matrices:
            return [
                (
                    item["request"],
                    [],
                    {
                        "path": "torch_microbatch_empty",
                        "batch_request_count": len(batch),
                        "batch_candidate_count": 0,
                        "batch_wait_ms": round(
                            (batch_started - item["request"].enqueued_at) * 1000, 2
                        ),
                        "feature_extraction_ms": item["feature_extraction_ms"],
                        "tensor_prep_ms": 0.0,
                        "model_forward_ms": 0.0,
                        "response_build_ms": 0.0,
                        "total_ms": round((time.perf_counter() - batch_started) * 1000, 2),
                        "candidate_count": len(item["request"].candidates),
                        "ranked_count": 0,
                    },
                )
                for item in prepared
            ]

        predictions, inference_profile = self.ranking_model.run_inference_batch(
            np.vstack(feature_matrices)
        )

        results: List[Tuple[RankingBatchRequest, List[ProductRecommendation], Dict[str, Any]]] = []
        offset = 0
        for item in prepared:
            request = item["request"]
            feature_matrix = item["feature_matrix"]
            if feature_matrix is None:
                results.append(
                    (
                        request,
                        [],
                        {
                            "path": "torch_microbatch_empty",
                            "batch_request_count": len(batch),
                            "batch_candidate_count": total_batch_candidates,
                            "batch_wait_ms": round(
                                (batch_started - request.enqueued_at) * 1000, 2
                            ),
                            "feature_extraction_ms": item["feature_extraction_ms"],
                            "tensor_prep_ms": inference_profile["tensor_prep_ms"],
                            "model_forward_ms": inference_profile["model_forward_ms"],
                            "response_build_ms": 0.0,
                            "total_ms": round((time.perf_counter() - batch_started) * 1000, 2),
                            "candidate_count": len(request.candidates),
                            "ranked_count": 0,
                        },
                    )
                )
                continue

            size = feature_matrix.shape[0]
            request_predictions = {
                key: values[offset : offset + size]
                for key, values in predictions.items()
            }
            offset += size
            recommendations, response_build_ms = (
                self.ranking_model.build_recommendations_from_predictions(
                    item["valid_candidates"],
                    request_predictions,
                    request.k,
                )
            )
            results.append(
                (
                    request,
                    recommendations,
                    {
                        "path": "torch_microbatch",
                        "batch_request_count": len(batch),
                        "batch_candidate_count": total_batch_candidates,
                        "batch_wait_ms": round(
                            (batch_started - request.enqueued_at) * 1000,
                            2,
                        ),
                        "feature_extraction_ms": item["feature_extraction_ms"],
                        "tensor_prep_ms": inference_profile["tensor_prep_ms"],
                        "model_forward_ms": inference_profile["model_forward_ms"],
                        "response_build_ms": response_build_ms,
                        "total_ms": round((time.perf_counter() - batch_started) * 1000, 2),
                        "candidate_count": len(request.candidates),
                        "ranked_count": len(recommendations),
                    },
                )
            )
        return results

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
