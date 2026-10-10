"""Thin ASGI proxy for the internal ranking-service HTTP surface."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import HTTPException

from video_commerce.common.cache_codec import json_dumps, json_loads
from video_commerce.common.config import Config
from video_commerce.common.observability import configure_logging
from video_commerce.ranking_runtime.ranking_coordinator_client import (
    RankingCoordinatorClientPool,
    RankingCoordinatorError,
    RankingCoordinatorTimeout,
)
from video_commerce.ranking_runtime.ranking_payloads import model_payload
from video_commerce.ranking_runtime.ranking_triton import (
    RankingTritonAdapter,
    TritonGrpcClient,
    RankingTritonOverloaded,
    RankingTritonUnavailable,
    triton_numeric_model_version,
)
from video_commerce.common.service_common import ServiceRuntime
from video_commerce.common.telemetry import configure_tracing


logger = logging.getLogger(__name__)


def _validate_required_release_version(
    *,
    gate_mode: str,
    required_version: str,
    active_checkpoint: Optional[Dict[str, Any]],
) -> None:
    if str(gate_mode).lower() == "legacy":
        return
    if active_checkpoint is None:
        raise RuntimeError("Triton ranking requires an active release pointer")
    active_version = str(active_checkpoint.get("model_version") or "")
    if active_version != str(required_version):
        raise RuntimeError(
            "RANKING_REQUIRED_MODEL_VERSION must match the active release pointer"
        )


class ClientDisconnected(RuntimeError):
    """Raised when the HTTP caller disconnects before the body is available."""


class RankingProxyApp:
    def __init__(self) -> None:
        self.runtime = ServiceRuntime("ranking-service")
        self.client: Optional[RankingCoordinatorClientPool] = None
        self.adapter: Optional[Any] = None
        self.inference_backend = "legacy"
        self.system_store: Optional[Any] = None
        self.object_storage: Optional[Any] = None

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            await self._handle_lifespan(receive, send)
            return
        if scope["type"] != "http":
            raise RuntimeError(f"Unsupported ASGI scope: {scope['type']}")
        await self._handle_http(scope, receive, send)

    async def _handle_lifespan(self, receive, send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                try:
                    await self.startup()
                except Exception as exc:
                    await send(
                        {
                            "type": "lifespan.startup.failed",
                            "message": str(exc),
                        }
                    )
                else:
                    await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await self.shutdown()
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def startup(self) -> None:
        self.runtime.config = Config()
        configure_logging(self.runtime.config.monitoring_config)
        configure_tracing(
            self.runtime.service_name,
            self.runtime.config.monitoring_config,
            app=None,
        )
        topology = self.runtime.config.service_topology_config
        self.inference_backend = self.runtime.config.ranking_config.inference_backend
        if self.inference_backend == "triton":
            await self._startup_triton()
            return
        if not topology.ranking_coordinator_host:
            raise RuntimeError(
                "SERVICE_RANKING_COORDINATOR_HOST is required for ranking_proxy_asgi"
            )
        self.client = RankingCoordinatorClientPool(
            topology.ranking_coordinator_host,
            topology.ranking_coordinator_port,
            pool_size=topology.ranking_coordinator_client_pool_size,
            connect_timeout_seconds=topology.ranking_coordinator_connect_timeout_seconds,
            request_timeout_seconds=topology.ranking_coordinator_request_timeout_seconds,
        )
        logger.info(
            "ranking_proxy_worker_started",
            extra={
                "service": self.runtime.service_name,
                "process_id": os.getpid(),
                "coordinator_host": topology.ranking_coordinator_host,
                "coordinator_port": topology.ranking_coordinator_port,
                "coordinator_pool_size": topology.ranking_coordinator_client_pool_size,
                "configured_workers": topology.ranking_workers,
            },
        )

    async def _startup_triton(self) -> None:
        from video_commerce.data_plane.object_storage import ObjectStorage
        from video_commerce.data_plane.system_store import SystemStore
        from video_commerce.ml.model_artifacts import ModelArtifactManager
        from video_commerce.ml.ranking import RankingModel

        config = self.runtime.config
        assert config is not None
        ranking_config = config.ranking_config
        topology = config.service_topology_config
        if topology.ranking_coordinator_direct_enabled:
            raise RuntimeError(
                "Triton ranking requires SERVICE_RANKING_COORDINATOR_DIRECT_ENABLED=false"
            )
        if not ranking_config.required_model_version:
            raise RuntimeError("RANKING_REQUIRED_MODEL_VERSION is required")
        if ranking_config.trimodal_enabled:
            raise RuntimeError("Triton ranking does not support trimodal models")
        if not config.database_config.enable:
            raise RuntimeError("Triton ranking requires the artifact metadata store")
        triton_client = None
        try:
            self.system_store = SystemStore(
                config.database_config,
                observability=self.runtime.observability,
            )
            await self.system_store.initialize()
            self.object_storage = ObjectStorage(config.object_storage_config)
            await self.object_storage.initialize()
            manager = ModelArtifactManager(
                system_store=self.system_store,
                object_storage=self.object_storage,
                model_config=config.model_config,
                recommendation_config=config.recommendation_config,
            )
            ranking_model = RankingModel(
                ranking_config,
                observability=self.runtime.observability,
            )
            active_checkpoint = await self.system_store.get_active_model_checkpoint(
                ModelArtifactManager.RANKING_MODEL_NAME,
                environment=config.model_release_config.environment,
            )
            _validate_required_release_version(
                gate_mode=config.model_release_config.gate_mode,
                required_version=ranking_config.required_model_version,
                active_checkpoint=active_checkpoint,
            )
            record = await manager.sync_ranking_checkpoint_version(
                ranking_config.required_model_version,
                expected_feature_schema_version=ranking_model.feature_schema_version,
                require_onnx=True,
            )
            if record is None:
                raise RuntimeError("required ranking ONNX artifact is unavailable")
            await ranking_model.load_model(config.model_config.ranking_model_path)
            if not ranking_model.mark_artifact_record_verified(record):
                raise RuntimeError("required ranking artifact verification failed")
            onnx_manifest = dict(
                ((record.payload or {}).get("artifact_manifest") or {}).get(
                    "onnx_model"
                )
                or {}
            )
            onnx_sha256 = str(onnx_manifest.get("sha256") or "")
            numeric_version = str(triton_numeric_model_version(onnx_sha256))
            triton_client = TritonGrpcClient(
                ranking_config.triton_grpc_url,
                ranking_config.triton_model_name,
                ranking_config.triton_request_timeout_ms / 1000.0,
            )
            if not await triton_client.is_ready(model_version=numeric_version):
                raise RuntimeError("required Triton model version is not ready")
            self.adapter = RankingTritonAdapter(
                ranking_model=ranking_model,
                triton_client=triton_client,
                required_model_version=ranking_config.required_model_version,
                numeric_model_version=numeric_version,
                max_inflight=ranking_config.triton_max_inflight,
                max_batch_size=ranking_config.triton_max_batch_size,
                request_timeout_seconds=(
                    ranking_config.triton_request_timeout_ms / 1000.0
                ),
                onnx_sha256=onnx_sha256,
                observability=self.runtime.observability,
            )
        except Exception:
            if triton_client is not None:
                await triton_client.close()
            if self.system_store is not None:
                await self.system_store.close()
                self.system_store = None
            self.object_storage = None
            raise
        logger.info(
            "ranking_triton_adapter_started",
            extra={
                "service": self.runtime.service_name,
                "process_id": os.getpid(),
                "model_version": ranking_config.required_model_version,
                "triton_numeric_model_version": numeric_version,
                "feature_schema_version": ranking_model.feature_schema_version,
                "din_enabled": ranking_config.din_enabled,
            },
        )

    async def shutdown(self) -> None:
        if self.adapter is not None:
            await self.adapter.close()
            self.adapter = None
        if self.client:
            await self.client.aclose()
            self.client = None
        if self.system_store is not None:
            await self.system_store.close()
            self.system_store = None

    async def _handle_http(self, scope, receive, send) -> None:
        started_at = time.perf_counter()
        method = scope.get("method", "GET")
        path = scope.get("path", "/")
        request_id = self._request_id(scope)
        status_code = 500
        self.runtime.active_requests += 1
        self.runtime.max_active_requests = max(
            self.runtime.max_active_requests,
            self.runtime.active_requests,
        )
        self.runtime.observability.http_requests_in_progress.inc()
        try:
            if path in {"/", ""} and method == "GET":
                status_code = 200
                await self._send_json(
                    send,
                    status_code,
                    {
                        "service": self.runtime.service_name,
                        "version": "1.0.0",
                        "health": "/health",
                        "readyz": "/readyz",
                    },
                    request_id,
                )
                return
            if path == "/livez" and method == "GET":
                status_code = 200
                await self._send_json(
                    send,
                    status_code,
                    {
                        "status": "ok",
                        "service": self.runtime.service_name,
                        "process_id": os.getpid(),
                        "uptime_seconds": round(
                            time.time() - self.runtime.started_at,
                            2,
                        ),
                    },
                    request_id,
                )
                return
            if path in {"/readyz", "/health"} and method == "GET":
                if self.inference_backend == "triton":
                    if self.adapter is None:
                        raise RankingTritonUnavailable("ranking adapter unavailable")
                    health = await self.adapter.health()
                    status_code = 200 if health.get("status") == "healthy" else 503
                    await self._send_json(send, status_code, health, request_id)
                    return
                response = await self._client().health()
                status_code = response.status_code
                await self._send_response(
                    send,
                    status_code,
                    response.content_type,
                    response.body,
                    request_id,
                )
                return
            if path == "/metrics" and method == "GET":
                if self.inference_backend == "triton":
                    status_code = 200
                    await self._send_response(
                        send,
                        status_code,
                        "text/plain; version=0.0.4",
                        self.runtime.observability.prometheus_payload(),
                        request_id,
                    )
                    return
                response = await self._client().metrics()
                status_code = response.status_code
                await self._send_response(
                    send,
                    status_code,
                    response.content_type,
                    response.body,
                    request_id,
                )
                return
            if path == "/internal/rank" and method == "POST":
                if not self._is_internal_request(scope):
                    status_code = 401
                    await self._send_json(
                        send,
                        status_code,
                        {"detail": "Invalid internal service key"},
                        request_id,
                    )
                    return
                try:
                    body = await self._read_body(receive)
                except ClientDisconnected:
                    status_code = 499
                    return
                body = self._body_with_deadline(body)
                if self.inference_backend == "triton":
                    if self.adapter is None:
                        raise RankingTritonUnavailable("ranking adapter unavailable")
                    try:
                        payload = json_loads(body)
                    except Exception:
                        status_code = 400
                        await self._send_json(
                            send,
                            status_code,
                            {"detail": "Invalid JSON body"},
                            request_id,
                        )
                        return
                    recommendations, profile = await self.adapter.rank_payload(
                        payload,
                        request_id=request_id,
                    )
                    status_code = 200
                    await self._send_json(
                        send,
                        status_code,
                        {
                            "recommendations": [
                                model_payload(item) for item in recommendations
                            ],
                            "profile": profile,
                        },
                        request_id,
                    )
                    return
                response = await self._client().rank(body)
                status_code = response.status_code
                await self._send_response(
                    send,
                    status_code,
                    response.content_type,
                    response.body,
                    request_id,
                )
                return

            status_code = 404
            await self._send_json(
                send, status_code, {"detail": "Not Found"}, request_id
            )
        except HTTPException as exc:
            status_code = int(exc.status_code)
            await self._send_json(
                send,
                status_code,
                {"detail": exc.detail},
                request_id,
            )
        except RankingTritonOverloaded as exc:
            status_code = 429
            await self._send_json(
                send,
                status_code,
                {"detail": "ranking_overloaded", "retry_after_seconds": 1},
                request_id,
            )
            logger.warning("ranking_triton_overloaded: %s", exc)
        except RankingTritonUnavailable as exc:
            status_code = 503
            await self._send_json(
                send,
                status_code,
                {"detail": "ranking_model_not_ready"},
                request_id,
            )
            logger.warning("ranking_triton_unavailable: %s", exc)
        except RankingCoordinatorTimeout as exc:
            status_code = 503
            if hasattr(
                self.runtime.observability, "record_ranking_coordinator_client_error"
            ):
                self.runtime.observability.record_ranking_coordinator_client_error(
                    "timeout"
                )
            await self._send_json(
                send,
                status_code,
                {"detail": "ranking_coordinator_timeout"},
                request_id,
            )
            logger.warning("ranking_coordinator_timeout: %s", exc)
        except asyncio.TimeoutError as exc:
            status_code = 503
            if hasattr(
                self.runtime.observability, "record_ranking_coordinator_client_error"
            ):
                self.runtime.observability.record_ranking_coordinator_client_error(
                    "timeout"
                )
            await self._send_json(
                send,
                status_code,
                {"detail": "ranking_coordinator_timeout"},
                request_id,
            )
            logger.warning("ranking_coordinator_timeout: %s", exc)
        except RankingCoordinatorError as exc:
            status_code = 503
            if hasattr(
                self.runtime.observability, "record_ranking_coordinator_client_error"
            ):
                self.runtime.observability.record_ranking_coordinator_client_error(
                    "unavailable"
                )
            await self._send_json(
                send,
                status_code,
                {"detail": "Ranking coordinator unavailable"},
                request_id,
            )
            logger.warning("ranking_coordinator_unavailable: %s", exc)
        finally:
            duration = time.perf_counter() - started_at
            self.runtime.observability.record_request(
                method,
                path,
                status_code,
                duration,
            )
            self.runtime.handled_requests += 1
            self.runtime.active_requests = max(0, self.runtime.active_requests - 1)
            self.runtime.observability.http_requests_in_progress.dec()

    def _client(self) -> RankingCoordinatorClientPool:
        if self.client is None:
            raise RankingCoordinatorError("ranking coordinator client unavailable")
        return self.client

    def _body_with_deadline(self, body: bytes) -> bytes:
        if not self.runtime.config:
            return body
        try:
            payload = json_loads(body)
        except Exception:
            return body
        if not isinstance(payload, dict):
            return body
        timeout_seconds = (
            self.runtime.config.service_topology_config.ranking_coordinator_request_timeout_seconds
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

    def _is_internal_request(self, scope) -> bool:
        if self.runtime.config is None:
            return True
        expected_key = self.runtime.config.security_config.internal_service_key
        if not expected_key:
            return True
        headers = _headers(scope)
        header_name = (
            self.runtime.config.security_config.internal_service_header.lower()
        )
        return headers.get(header_name) == expected_key

    def _request_id(self, scope) -> str:
        header_name = "x-request-id"
        if self.runtime.config:
            header_name = (
                self.runtime.config.monitoring_config.request_id_header.lower()
            )
        return _headers(scope).get(header_name) or str(uuid.uuid4())

    async def _read_body(self, receive) -> bytes:
        chunks = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                raise ClientDisconnected()
            if message["type"] != "http.request":
                continue
            body = message.get("body", b"")
            if body:
                chunks.append(body)
            if not message.get("more_body", False):
                return b"".join(chunks)

    async def _send_json(
        self,
        send,
        status_code: int,
        payload: Dict[str, Any],
        request_id: str,
    ) -> None:
        await self._send_response(
            send,
            status_code,
            "application/json",
            json_dumps(payload),
            request_id,
        )

    async def _send_response(
        self,
        send,
        status_code: int,
        content_type: str,
        body: bytes,
        request_id: str,
    ) -> None:
        header_name = b"x-request-id"
        if self.runtime.config:
            header_name = (
                self.runtime.config.monitoring_config.request_id_header.lower().encode()
            )
        headers = [
            (b"content-type", content_type.encode("ascii", errors="ignore")),
            (header_name, request_id.encode("utf-8")),
        ]
        if int(status_code) == 429:
            headers.append((b"retry-after", b"1"))
        await send(
            {
                "type": "http.response.start",
                "status": int(status_code),
                "headers": headers,
            }
        )
        await send({"type": "http.response.body", "body": body})


def _headers(scope) -> Dict[str, str]:
    return {
        key.decode("latin1").lower(): value.decode("latin1")
        for key, value in scope.get("headers") or []
    }


def _conservative_deadline(existing_deadline: Any, local_deadline: float) -> float:
    try:
        parsed_deadline = float(existing_deadline)
    except (TypeError, ValueError, OverflowError):
        return local_deadline
    if not math.isfinite(parsed_deadline):
        return local_deadline
    return min(parsed_deadline, local_deadline)


app = RankingProxyApp()
