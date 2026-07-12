"""Internal model runner for ranking micro-batches."""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import torch
from fastapi import HTTPException

from video_commerce.common.cache_codec import json_dumps, json_loads
from video_commerce.common.config import Config
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.ml.ranking import RankingModel
from video_commerce.ranking_runtime.ranking_batcher import (
    normalize_ranking_batch_payloads,
    run_ranking_batch_payloads,
)
from video_commerce.ranking_runtime.ranking_coordinator_client import (
    HEALTH_OPERATION,
    METRICS_OPERATION,
    RankingCoordinatorProtocolError,
    encode_response,
    read_frame,
)
from video_commerce.ranking_runtime.ranking_runner_client import BATCH_RANK_OPERATION
from video_commerce.common.service_common import (
    ServiceRuntime,
    configure_service_logging,
)
from video_commerce.data_plane.system_store import SystemStore


logger = logging.getLogger(__name__)


class RankingRunner:
    def __init__(self) -> None:
        self.runtime = ServiceRuntime("ranking-runner")
        self.config: Optional[Config] = None
        self.system_store: Optional[SystemStore] = None
        self.object_storage: Optional[ObjectStorage] = None
        self.artifact_manager: Optional[ModelArtifactManager] = None
        self.ranking_model: Optional[RankingModel] = None
        self.checkpoint_sync_task: Optional[asyncio.Task] = None
        self.server: Optional[asyncio.base_events.Server] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._batch_queue: Optional[
            asyncio.Queue[
                Optional[
                    Tuple[
                        List[dict],
                        asyncio.Future[Tuple[int, bytes]],
                        Optional[float],
                        float,
                    ]
                ]
            ]
        ] = None
        self._batch_workers: List[asyncio.Task] = []
        self._active_batch_count = 0
        self._runner_batch_concurrency = 1
        self._runner_queue_size = 4

    async def start(self) -> None:
        self.config = Config()
        self.runtime.config = self.config
        configure_service_logging(self.runtime)
        _configure_torch_runtime(self.runtime)

        if self.config.database_config.enable:
            self.system_store = SystemStore(
                self.config.database_config,
                observability=self.runtime.observability,
            )
            await self.system_store.initialize()

        self.object_storage = ObjectStorage(self.config.object_storage_config)
        await self.object_storage.initialize()
        self.artifact_manager = ModelArtifactManager(
            system_store=self.system_store,
            object_storage=self.object_storage,
            model_config=self.config.model_config,
            recommendation_config=self.config.recommendation_config,
        )

        self.ranking_model = RankingModel(
            self.config.ranking_config,
            observability=self.runtime.observability,
        )
        ranking_checkpoint = (
            await self.artifact_manager.sync_latest_ranking_checkpoint()
        )
        await self.ranking_model.load_model(self.config.model_config.ranking_model_path)
        if ranking_checkpoint:
            self.ranking_model.model_version = ranking_checkpoint.model_version
        self.ranking_model.enable_profiling_logs = (
            self.config.monitoring_config.enable_profiling_logs
        )
        self.ranking_model.profiling_log_min_duration_ms = (
            self.config.monitoring_config.profiling_log_min_duration_ms
        )
        runner_concurrency = max(
            1, int(getattr(self.config.ranking_config, "runner_batch_concurrency", 1))
        )
        runner_queue_size = max(
            0, int(getattr(self.config.ranking_config, "runner_queue_size", 4))
        )
        self._runner_batch_concurrency = runner_concurrency
        self._runner_queue_size = runner_queue_size
        self._executor = ThreadPoolExecutor(
            max_workers=runner_concurrency,
            thread_name_prefix="ranking-runner",
        )
        self._batch_queue = asyncio.Queue(
            maxsize=max(1, runner_concurrency + runner_queue_size)
        )
        self._batch_workers = [
            asyncio.create_task(
                self._run_batch_worker(worker_id),
                name=f"ranking-runner-worker-{worker_id}",
            )
            for worker_id in range(runner_concurrency)
        ]

        if self.config.ranking_config.checkpoint_sync_interval_seconds > 0:
            self.checkpoint_sync_task = asyncio.create_task(
                self._periodic_ranking_checkpoint_sync(),
                name="ranking-runner-checkpoint-sync",
            )

        topology = self.config.service_topology_config
        host = topology.ranking_runner_bind_host
        port = topology.ranking_runner_port
        self.server = await asyncio.start_server(
            self._handle_client,
            host=host,
            port=port,
            backlog=topology.ranking_runner_backlog,
            limit=topology.ranking_runner_stream_limit,
        )
        compile_status = self.ranking_model.get_stats()
        logger.info(
            "ranking_runner_started",
            extra={
                "service": self.runtime.service_name,
                "process_id": os.getpid(),
                "host": host,
                "port": port,
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "runner_batch_concurrency": runner_concurrency,
                "runner_queue_size": runner_queue_size,
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

    async def close(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        if self.checkpoint_sync_task:
            self.checkpoint_sync_task.cancel()
            await asyncio.gather(self.checkpoint_sync_task, return_exceptions=True)
            self.checkpoint_sync_task = None
        if self._batch_queue is not None and self._batch_workers:
            for _ in self._batch_workers:
                await self._batch_queue.put(None)
            await asyncio.gather(*self._batch_workers, return_exceptions=True)
            self._batch_workers = []
            self._batch_queue = None
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        if self.system_store:
            await self.system_store.close()
            self.system_store = None

    async def serve_forever(self) -> None:
        if not self.server:
            raise RuntimeError("ranking runner not started")
        async with self.server:
            await self.server.serve_forever()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while True:
                try:
                    frame = await read_frame(reader)
                except Exception:
                    break
                if not frame:
                    break
                response = await self._handle_frame(frame)
                writer.write(response)
                await writer.drain()
        except Exception as exc:
            if self.runtime.observability and hasattr(
                self.runtime.observability, "record_ranking_runner_late_write"
            ):
                self.runtime.observability.record_ranking_runner_late_write(
                    type(exc).__name__
                )
            logger.warning(
                "ranking_runner_client_failed",
                extra={
                    "service": self.runtime.service_name,
                    "process_id": os.getpid(),
                    "exception_type": type(exc).__name__,
                    "exception_repr": repr(exc),
                },
            )
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_frame(self, frame: bytes) -> bytes:
        operation = frame[:1]
        body = frame[1:]
        if operation == BATCH_RANK_OPERATION:
            return await self._handle_batch_rank(body)
        if operation == HEALTH_OPERATION:
            return await self._handle_health()
        if operation == METRICS_OPERATION:
            return await self._handle_metrics()
        return _json_response(400, {"detail": "invalid ranking runner operation"})

    async def _handle_batch_rank(self, body: bytes) -> bytes:
        started_at = time.perf_counter()
        status_code = 200
        try:
            if (
                self.ranking_model is None
                or self._executor is None
                or self._batch_queue is None
            ):
                status_code = 503
                return _json_response(503, {"detail": "ranking_runner_unavailable"})
            payload = json_loads(body)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("requests"),
                list,
            ):
                status_code = 400
                return _json_response(400, {"detail": "requests must be a list"})
            requests = normalize_ranking_batch_payloads(payload)
            deadline_unix_seconds = _batch_deadline_unix_seconds(requests)
            if _deadline_expired(deadline_unix_seconds):
                status_code = 503
                self._record_runner_batch("deadline_exceeded")
                return _json_response(
                    503, {"detail": "ranking_runner_deadline_exceeded"}
                )
            if self._runner_queue_capacity_full():
                status_code = 429
                self._record_runner_batch("overloaded")
                self._record_runner_queue_depth()
                return _json_response(429, {"detail": "ranking_runner_overloaded"})
            loop = asyncio.get_running_loop()
            future: asyncio.Future[Tuple[int, bytes]] = loop.create_future()
            try:
                self._batch_queue.put_nowait(
                    (requests, future, deadline_unix_seconds, time.perf_counter())
                )
            except asyncio.QueueFull:
                status_code = 429
                self._record_runner_batch("overloaded")
                self._record_runner_queue_depth()
                return _json_response(429, {"detail": "ranking_runner_overloaded"})
            self._record_runner_batch("queued")
            self._record_runner_queue_depth()
            status_code, response = await future
            return response
        except HTTPException as exc:
            status_code = exc.status_code
            return _json_response(exc.status_code, {"detail": exc.detail})
        except RankingCoordinatorProtocolError as exc:
            status_code = 400
            return _json_response(400, {"detail": str(exc)})
        except Exception:
            status_code = 500
            logger.exception("ranking_runner_batch_failed")
            return _json_response(500, {"detail": "ranking runner failed"})
        finally:
            self.runtime.observability.record_request(
                "POST",
                "/internal/rank-batch",
                status_code,
                time.perf_counter() - started_at,
            )

    async def _run_batch_worker(self, worker_id: int) -> None:
        assert self._batch_queue is not None
        while True:
            item = await self._batch_queue.get()
            if item is None:
                self._batch_queue.task_done()
                break
            requests, future, deadline_unix_seconds, queued_at = item
            execution_started = time.perf_counter()
            active_incremented = False
            try:
                if future.cancelled():
                    self._record_runner_batch("cancelled")
                    continue
                if _deadline_expired(deadline_unix_seconds):
                    self._complete_batch_future(
                        future,
                        503,
                        {"detail": "ranking_runner_deadline_exceeded"},
                    )
                    self._record_runner_batch("deadline_exceeded")
                    continue
                if self.ranking_model is None or self._executor is None:
                    self._complete_batch_future(
                        future,
                        503,
                        {"detail": "ranking_runner_unavailable"},
                    )
                    self._record_runner_batch("unavailable")
                    continue
                self._active_batch_count += 1
                active_incremented = True
                self._record_runner_queue_depth()
                self._record_runner_stage(
                    "runner_queue_wait",
                    max(0.0, execution_started - queued_at),
                )
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    functools.partial(
                        run_ranking_batch_payloads,
                        self.ranking_model,
                        requests,
                        profile_path="torch_microbatch_runner",
                    ),
                )
                result["runner_process_id"] = os.getpid()
                result["runner_worker_id"] = worker_id
                for stage, duration_seconds in (result.get("stages") or {}).items():
                    self._record_runner_stage(stage, float(duration_seconds))
                if not future.cancelled():
                    future.set_result((200, _json_response(200, result)))
                self._record_runner_batch("success")
            except HTTPException as exc:
                self._complete_batch_future(
                    future, exc.status_code, {"detail": exc.detail}
                )
                self._record_runner_batch("error")
            except RankingCoordinatorProtocolError as exc:
                self._complete_batch_future(future, 400, {"detail": str(exc)})
                self._record_runner_batch("bad_request")
            except Exception:
                logger.exception("ranking_runner_batch_failed")
                self._complete_batch_future(
                    future,
                    500,
                    {"detail": "ranking runner failed"},
                )
                self._record_runner_batch("error")
            finally:
                if active_incremented and self._active_batch_count > 0:
                    self._active_batch_count -= 1
                self._record_runner_stage(
                    "worker_total",
                    max(0.0, time.perf_counter() - execution_started),
                )
                self._batch_queue.task_done()
                self._record_runner_queue_depth()

    def _complete_batch_future(
        self,
        future: asyncio.Future[Tuple[int, bytes]],
        status_code: int,
        payload: dict,
    ) -> None:
        if not future.cancelled() and not future.done():
            future.set_result((status_code, _json_response(status_code, payload)))

    def _record_runner_queue_depth(self) -> None:
        if (
            self._batch_queue is not None
            and self.runtime.observability
            and hasattr(self.runtime.observability, "set_ranking_runner_queue_depth")
        ):
            self.runtime.observability.set_ranking_runner_queue_depth(
                self._batch_queue.qsize()
            )

    def _runner_queue_capacity_full(self) -> bool:
        if self._batch_queue is None:
            return True
        capacity = self._runner_batch_concurrency + self._runner_queue_size
        return self._active_batch_count + self._batch_queue.qsize() >= capacity

    def _record_runner_batch(self, status: str) -> None:
        if self.runtime.observability and hasattr(
            self.runtime.observability, "record_ranking_runner_batch"
        ):
            self.runtime.observability.record_ranking_runner_batch(status)

    def _record_runner_stage(self, stage: str, duration_seconds: float) -> None:
        if self.runtime.observability and hasattr(
            self.runtime.observability, "record_ranking_batch_stage"
        ):
            self.runtime.observability.record_ranking_batch_stage(
                path="microbatch_runner_local",
                stage=stage,
                duration_seconds=duration_seconds,
            )

    async def _handle_health(self) -> bytes:
        started_at = time.perf_counter()
        ranking_health = (
            self.ranking_model.health_check()
            if self.ranking_model
            else {"status": "unhealthy", "error": "ranking model unavailable"}
        )
        ready = ranking_health.get("status") == "healthy"
        payload = {
            "status": "ready" if ready else "not_ready",
            "service": self.runtime.service_name,
            "checks": {"ranking_model": ranking_health},
            "process_id": os.getpid(),
            "batch_payload_versions": [1, 2, 3],
            "capabilities": {"batch_payload_versions": [1, 2, 3]},
        }
        self.runtime.observability.record_request(
            "GET",
            "/readyz",
            200 if ready else 503,
            time.perf_counter() - started_at,
        )
        return _json_response(200 if ready else 503, payload)

    async def _handle_metrics(self) -> bytes:
        if self.config and self.config.monitoring_config.enable_prometheus_metrics:
            await self.runtime.observability.collect_runtime_metrics(
                system_store=self.system_store,
            )
        return encode_response(
            200,
            self.runtime.observability.prometheus_content_type,
            self.runtime.observability.prometheus_payload(),
        )

    async def _periodic_ranking_checkpoint_sync(self) -> None:
        assert self.config is not None
        interval_seconds = max(
            1,
            int(self.config.ranking_config.checkpoint_sync_interval_seconds),
        )
        model_path = self.config.model_config.ranking_model_path
        last_ranking_version: Optional[str] = (
            self.ranking_model.model_version if self.ranking_model else None
        )
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                if self.ranking_model and model_path and self.artifact_manager:
                    latest_ranking = (
                        await self.artifact_manager.get_latest_model_checkpoint(
                            ModelArtifactManager.RANKING_MODEL_NAME
                        )
                    )
                    if (
                        latest_ranking
                        and latest_ranking.model_version != last_ranking_version
                    ):
                        await self.artifact_manager.sync_latest_ranking_checkpoint()
                        if await self.ranking_model.reload_model_if_updated(model_path):
                            self.ranking_model.model_version = (
                                latest_ranking.model_version
                            )
                            last_ranking_version = latest_ranking.model_version
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Ranking runner checkpoint sync failed: {exc}")
                await asyncio.sleep(min(interval_seconds, 60))


def _json_response(status_code: int, payload: dict) -> bytes:
    return encode_response(status_code, "application/json", json_dumps(payload))


def _batch_deadline_unix_seconds(requests: List[dict]) -> Optional[float]:
    deadlines = []
    for request in requests:
        raw_deadline = request.get("deadline_unix_seconds")
        if raw_deadline is None:
            continue
        try:
            deadlines.append(float(raw_deadline))
        except (TypeError, ValueError):
            continue
    return min(deadlines) if deadlines else None


def _deadline_expired(deadline_unix_seconds: Optional[float]) -> bool:
    return (
        deadline_unix_seconds is not None
        and float(deadline_unix_seconds) <= time.time()
    )


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
    return max(1, os.cpu_count() or 1)


async def _main() -> None:
    runner = RankingRunner()
    try:
        await runner.start()
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass
        serve_task = asyncio.create_task(runner.serve_forever())
        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            {serve_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if task is serve_task:
                task.result()
    finally:
        await runner.close()


if __name__ == "__main__":
    asyncio.run(_main())
