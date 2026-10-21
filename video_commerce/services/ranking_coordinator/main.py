"""Single-process global ranking batch coordinator.

The public internal HTTP contract remains on the ranking-service proxy. That
service can run multiple HTTP workers and forward raw rank payloads here over a
small length-prefixed TCP protocol. The coordinator owns the only batching
queue, so HTTP worker/process count no longer fragments ranking batches.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import signal
import time
from typing import Optional

import torch
from fastapi import HTTPException

from video_commerce.common.cache_codec import json_dumps, json_loads
from video_commerce.common.config import Config
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.ml.ranking import RankingModel, RankingModelNotReadyError
from video_commerce.ranking_runtime.ranking_batcher import (
    RankingBatcher,
    RankingOverloadedError,
    RankingQueueFullError,
    RankingQueueTimeoutError,
)
from video_commerce.ranking_runtime.ranking_auth import (
    ConnectionAuthState,
    RankingRuntimeAuthenticator,
    RankingRuntimeAuthSettings,
)
from video_commerce.ranking_runtime.ranking_coordinator_client import (
    HEALTH_OPERATION,
    METRICS_OPERATION,
    RANK_OPERATION,
    RankingCoordinatorProtocolError,
    encode_response,
    read_frame,
)
from video_commerce.ranking_runtime.ranking_payloads import (
    coerce_rank_payload,
    model_payload,
)
from video_commerce.ranking_runtime.ranking_runner_client import RankingRunnerClientPool
from video_commerce.common.service_common import (
    ServiceRuntime,
    configure_service_logging,
)
from video_commerce.data_plane.system_store import SystemStore


logger = logging.getLogger(__name__)


class RankingCoordinator:
    def __init__(self) -> None:
        self.runtime = ServiceRuntime("ranking-coordinator")
        self.config: Optional[Config] = None
        self.system_store: Optional[SystemStore] = None
        self.object_storage: Optional[ObjectStorage] = None
        self.artifact_manager: Optional[ModelArtifactManager] = None
        self.ranking_model: Optional[RankingModel] = None
        self.ranking_batcher: Optional[RankingBatcher] = None
        self.runner_pool: Optional[RankingRunnerClientPool] = None
        self.checkpoint_sync_task: Optional[asyncio.Task] = None
        self.server: Optional[asyncio.base_events.Server] = None
        self.authenticator = RankingRuntimeAuthenticator(
            RankingRuntimeAuthSettings(),
            role="coordinator",
            observability=self.runtime.observability,
        )

    async def start(self) -> None:
        self.config = Config()
        self.runtime.config = self.config
        configure_service_logging(self.runtime)
        _configure_torch_runtime(self.runtime)
        auth_settings = RankingRuntimeAuthSettings.from_config(self.config)
        self.authenticator = RankingRuntimeAuthenticator(
            auth_settings,
            role="coordinator",
            observability=self.runtime.observability,
        )

        if self.config.database_config.enable:
            self.system_store = SystemStore(
                self.config.database_config,
                observability=self.runtime.observability,
            )
            await self.system_store.initialize()

        topology = self.config.service_topology_config
        runner_urls = (topology.ranking_runner_urls or "").strip()
        if runner_urls:
            self.runner_pool = await RankingRunnerClientPool.create(
                runner_urls,
                default_port=topology.ranking_runner_port,
                dispatch_concurrency=self.config.ranking_config.coordinator_dispatch_concurrency,
                runner_max_inflight_batches=(
                    max(1, self.config.ranking_config.runner_batch_concurrency)
                ),
                connect_timeout_seconds=topology.ranking_runner_connect_timeout_seconds,
                request_timeout_seconds=topology.ranking_runner_request_timeout_seconds,
                unhealthy_cooldown_seconds=topology.ranking_runner_unhealthy_cooldown_seconds,
                dns_refresh_seconds=topology.ranking_runner_dns_refresh_seconds,
                endpoint_missing_refreshes=topology.ranking_runner_endpoint_missing_refreshes,
                endpoint_missing_grace_seconds=topology.ranking_runner_endpoint_missing_grace_seconds,
                observability=self.runtime.observability,
                auth_settings=auth_settings,
            )
        else:
            self.object_storage = ObjectStorage(self.config.object_storage_config)
            await self.object_storage.initialize()
            self.artifact_manager = ModelArtifactManager(
                system_store=self.system_store,
                object_storage=self.object_storage,
                model_config=self.config.model_config,
                recommendation_config=self.config.recommendation_config,
                observability=self.runtime.observability,
            )

            self.ranking_model = RankingModel(
                self.config.ranking_config,
                observability=self.runtime.observability,
            )
            ranking_checkpoint = await self.artifact_manager.sync_selected_ranking_checkpoint(
                gate_mode=self.config.model_release_config.gate_mode,
                environment=self.config.model_release_config.environment,
                expected_feature_schema_version=self.ranking_model.feature_schema_version,
            )
            if ranking_checkpoint:
                self.ranking_model = (
                    await self.artifact_manager.prepare_ranking_replacement(
                        self.ranking_model,
                        ranking_checkpoint,
                        environment=self.config.model_release_config.environment,
                    )
                )
                await self.artifact_manager.record_generation_activation(
                    ranking_checkpoint,
                    environment=self.config.model_release_config.environment,
                    state="ready",
                )
            else:
                await self.ranking_model.load_model(
                    self.config.model_config.ranking_model_path
                )
            self.ranking_model.enable_profiling_logs = (
                self.config.monitoring_config.enable_profiling_logs
            )
            self.ranking_model.profiling_log_min_duration_ms = (
                self.config.monitoring_config.profiling_log_min_duration_ms
            )

        self.ranking_batcher = RankingBatcher(
            self.ranking_model,
            self.config.ranking_config,
            observability=self.runtime.observability,
            runner_pool=self.runner_pool,
        )
        await self.ranking_batcher.start()
        if (
            self.runner_pool is None
            and self.config.ranking_config.checkpoint_sync_interval_seconds > 0
        ):
            self.checkpoint_sync_task = asyncio.create_task(
                self._periodic_ranking_checkpoint_sync(),
                name="ranking-coordinator-checkpoint-sync",
            )

        host = self.config.service_topology_config.ranking_coordinator_bind_host
        port = self.config.service_topology_config.ranking_coordinator_port
        self.server = await asyncio.start_server(
            self._handle_client,
            host=host,
            port=port,
            backlog=self.config.service_topology_config.ranking_coordinator_backlog,
            limit=self.config.service_topology_config.ranking_coordinator_stream_limit,
        )
        compile_status = (
            self.ranking_model.get_stats()
            if self.ranking_model
            else {
                "torch_compile_enabled": (
                    self.config.ranking_config.torch_compile_enabled
                ),
                "torch_compile_active": False,
                "torch_compile_backend": (
                    self.config.ranking_config.torch_compile_backend
                ),
                "torch_compile_mode": self.config.ranking_config.torch_compile_mode,
                "torch_compile_dynamic": (
                    self.config.ranking_config.torch_compile_dynamic
                ),
                "torch_compile_error": None,
                "torch_compile_warmup_ms": None,
                "torch_compile_fallback_count": 0,
                "torch_compile_last_fallback_error": None,
                "torch_compile_last_inference_path": "eager",
            }
        )
        logger.info(
            "ranking_coordinator_started",
            extra={
                "service": self.runtime.service_name,
                "process_id": os.getpid(),
                "host": host,
                "port": port,
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "batch_max_requests": self.config.ranking_config.batch_max_requests,
                "batch_target_requests": self.config.ranking_config.batch_target_requests,
                "batch_wait_ms": self.config.ranking_config.batch_wait_ms,
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
                "batch_runner_count": self.config.ranking_config.batch_runner_count,
                "ranking_runner_urls": runner_urls,
                "coordinator_dispatch_concurrency": self.config.ranking_config.coordinator_dispatch_concurrency,
                "runner_queue_size": self.config.ranking_config.runner_queue_size,
                "runner_batch_concurrency": self.config.ranking_config.runner_batch_concurrency,
                "runtime_auth_mode": auth_settings.mode,
                "runtime_auth_key_configured": self.authenticator.verifies_handshakes,
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
        if self.ranking_batcher:
            await self.ranking_batcher.close()
            self.ranking_batcher = None
        if self.runner_pool:
            await self.runner_pool.aclose()
            self.runner_pool = None
        if self.system_store:
            await self.system_store.close()
            self.system_store = None

    async def serve_forever(self) -> None:
        if not self.server:
            raise RuntimeError("ranking coordinator not started")
        async with self.server:
            await self.server.serve_forever()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        auth_state = ConnectionAuthState()
        try:
            while True:
                pre_auth_limits = self.authenticator.pre_auth_limits(auth_state)
                try:
                    if pre_auth_limits is None:
                        frame = await read_frame(reader)
                    else:
                        max_frame_bytes, timeout_seconds = pre_auth_limits
                        frame = await asyncio.wait_for(
                            read_frame(reader, max_frame_bytes),
                            timeout=timeout_seconds,
                        )
                except (RankingCoordinatorProtocolError, asyncio.TimeoutError):
                    if pre_auth_limits is not None:
                        self.authenticator.record_pre_auth_violation()
                    break
                except Exception:
                    break
                if not frame:
                    break
                decision = self.authenticator.screen_frame(
                    auth_state, frame[:1], frame[1:]
                )
                if decision is not None:
                    writer.write(
                        _json_response(decision.status_code, decision.payload)
                    )
                    await writer.drain()
                    if decision.close_connection:
                        break
                    continue
                response = await self._handle_frame(frame)
                writer.write(response)
                await writer.drain()
        except Exception as exc:
            logger.warning(
                "ranking_coordinator_client_failed",
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
        if operation == RANK_OPERATION:
            return await self._handle_rank(body)
        if operation == HEALTH_OPERATION:
            return await self._handle_health()
        if operation == METRICS_OPERATION:
            return await self._handle_metrics()
        return _json_response(400, {"detail": "invalid coordinator operation"})

    async def _handle_rank(self, body: bytes) -> bytes:
        started_at = time.perf_counter()
        status_code = 200
        try:
            if self.ranking_batcher is None:
                raise RankingQueueTimeoutError("ranking_batcher_unavailable")
            payload = coerce_rank_payload(json_loads(body))
            if self._deadline_expired(payload.deadline_unix_seconds):
                raise RankingQueueTimeoutError("ranking_queue_wait_exceeded")
            if self.ranking_batcher.should_reject_new_request(
                payload.deadline_unix_seconds
            ):
                raise RankingOverloadedError("ranking_queue_wait_exceeded")
            recommendations, profile = await self.ranking_batcher.rank_candidates(
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
            profile = {
                **profile,
                "ranking_coordinator_process_id": os.getpid(),
                "ranking_service_request_id": payload.request_id,
            }
            response_body = json_dumps(
                {
                    "recommendations": [
                        model_payload(item) for item in recommendations
                    ],
                    "profile": profile,
                }
            )
            return encode_response(200, "application/json", response_body)
        except HTTPException as exc:
            status_code = exc.status_code
            return _json_response(exc.status_code, {"detail": exc.detail})
        except (RankingOverloadedError, RankingQueueFullError) as exc:
            status_code = 429
            return _json_response(
                429,
                {"detail": "ranking_overloaded", "retry_after_seconds": 1},
            )
        except RankingQueueTimeoutError as exc:
            status_code = 503
            return _json_response(503, {"detail": str(exc)})
        except RankingModelNotReadyError:
            status_code = 503
            return _json_response(503, {"detail": "ranking_model_not_ready"})
        except RankingCoordinatorProtocolError as exc:
            status_code = 400
            return _json_response(400, {"detail": str(exc)})
        except Exception as exc:
            status_code = 500
            logger.exception("ranking_coordinator_rank_failed")
            return _json_response(500, {"detail": "ranking coordinator failed"})
        finally:
            duration = time.perf_counter() - started_at
            self.runtime.observability.record_request(
                "POST",
                "/internal/rank",
                status_code,
                duration,
            )

    @staticmethod
    def _deadline_expired(deadline_unix_seconds: Optional[float]) -> bool:
        return (
            deadline_unix_seconds is not None
            and float(deadline_unix_seconds) <= time.time()
        )

    async def _handle_health(self) -> bytes:
        started_at = time.perf_counter()
        allow_degraded = bool(
            getattr(
                getattr(self.config, "ranking_config", None),
                "allow_untrained_fallback",
                False,
            )
        )
        if self.runner_pool is not None:
            ranking_health = await self.runner_pool.health_check()
            required_healthy_count = max(
                1,
                int(
                    getattr(
                        self.config.service_topology_config,
                        "ranking_min_healthy_runners",
                        2,
                    )
                ),
            )
            healthy_count = int(ranking_health.get("healthy_count", 0) or 0)
            degraded_count = int(ranking_health.get("degraded_count", 0) or 0)
            ranking_health = {
                **ranking_health,
                "status": (
                    "healthy"
                    if healthy_count >= required_healthy_count
                    else "degraded"
                    if allow_degraded
                    and healthy_count + degraded_count >= required_healthy_count
                    else "degraded"
                    if healthy_count > 0
                    else "unhealthy"
                ),
                "required_healthy_count": required_healthy_count,
            }
            if hasattr(self.runtime.observability, "set_ranking_healthy_runners"):
                self.runtime.observability.set_ranking_healthy_runners(healthy_count)
            ranking_check_name = "ranking_runners"
        else:
            ranking_health = (
                self.ranking_model.health_check()
                if self.ranking_model
                else {"status": "unhealthy", "error": "ranking model unavailable"}
            )
            ranking_check_name = "ranking_model"
        database_health = {"status": "healthy", "response_time_ms": 0.0, "error": None}
        if self.system_store:
            database_status = await self.system_store.health_check()
            database_health = {
                "status": database_status.status,
                "response_time_ms": database_status.response_time_ms,
                "error": database_status.error,
            }
        checks = {
            ranking_check_name: ranking_health,
            "database": database_health,
        }
        ready = all(
            check.get("status") == "healthy"
            or (allow_degraded and check.get("status") == "degraded")
            for check in checks.values()
        )
        response_status = (
            "ready"
            if ready
            and all(check.get("status") == "healthy" for check in checks.values())
            else "degraded"
            if ready
            else "not_ready"
        )
        payload = {
            "status": response_status,
            "service": self.runtime.service_name,
            "checks": checks,
            "process_id": os.getpid(),
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
            1, int(self.config.ranking_config.checkpoint_sync_interval_seconds)
        )
        last_ranking_identity = (
            self.ranking_model.artifact_metadata.get("model_release_id")
            if self.ranking_model
            else None,
            self.ranking_model.artifact_metadata.get("active_generation")
            if self.ranking_model
            else None,
            self.ranking_model.artifact_metadata.get("bundle_digest")
            if self.ranking_model
            else None,
        )
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                if self.ranking_model and self.artifact_manager:
                    latest_ranking = await self.artifact_manager.sync_selected_ranking_checkpoint(
                        gate_mode=self.config.model_release_config.gate_mode,
                        environment=self.config.model_release_config.environment,
                        expected_feature_schema_version=self.ranking_model.feature_schema_version,
                    )
                    selected_identity = (
                        latest_ranking.payload.get("model_release_id")
                        if latest_ranking
                        else None,
                        latest_ranking.payload.get("active_generation")
                        if latest_ranking
                        else None,
                        latest_ranking.payload.get("bundle_digest")
                        if latest_ranking
                        else None,
                    )
                    if latest_ranking and selected_identity != last_ranking_identity:
                        replacement = await self.artifact_manager.prepare_ranking_replacement(
                            self.ranking_model,
                            latest_ranking,
                            environment=self.config.model_release_config.environment,
                        )
                        if self.ranking_batcher is not None:
                            self.ranking_batcher.replace_ranking_model(replacement)
                        self.ranking_model = replacement
                        await self.artifact_manager.record_generation_activation(
                            latest_ranking,
                            environment=self.config.model_release_config.environment,
                            state="ready",
                        )
                        await self.artifact_manager.cleanup_local_generations(
                            latest_ranking,
                            environment=self.config.model_release_config.environment,
                            inflight_identities=(
                                self.ranking_batcher.inflight_generation_identities()
                                if self.ranking_batcher is not None
                                else set()
                            ),
                        )
                        last_ranking_identity = selected_identity
                    if latest_ranking and selected_identity == last_ranking_identity:
                        await self.artifact_manager.record_generation_activation(
                            latest_ranking,
                            environment=self.config.model_release_config.environment,
                            state="ready",
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Ranking coordinator checkpoint sync failed: {exc}")
                await asyncio.sleep(min(interval_seconds, 60))


def _json_response(status_code: int, payload: dict) -> bytes:
    return encode_response(status_code, "application/json", json_dumps(payload))


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


async def _main() -> None:
    coordinator = RankingCoordinator()
    try:
        await coordinator.start()
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass
        serve_task = asyncio.create_task(coordinator.serve_forever())
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
        await coordinator.close()


if __name__ == "__main__":
    asyncio.run(_main())
