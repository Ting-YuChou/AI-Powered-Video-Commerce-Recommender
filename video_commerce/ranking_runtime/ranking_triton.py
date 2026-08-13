"""ONNX Runtime/Triton serving helpers for the ranking adapter.

The adapter owns feature preparation and response construction. Triton owns only
the stateless tensor forward pass and its dynamic batching queue.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np
from fastapi import HTTPException

from video_commerce.ml.din import DIN_SEQUENCE_CONTEXT_KEY
from video_commerce.ml.model_artifacts import ModelArtifactRecord
from video_commerce.ranking_runtime.ranking_payloads import coerce_rank_payload


TRITON_OUTPUT_NAMES = ("ctr", "cvr", "ctcvr", "gmv", "ranking_score")


class RankingTritonOverloaded(RuntimeError):
    """The adapter or Triton scheduler has no admission capacity."""


class RankingTritonUnavailable(RuntimeError):
    """Triton or the required immutable model version is unavailable."""


@dataclass(frozen=True)
class TritonMaterialization:
    model_name: str
    business_model_version: str
    numeric_version: str
    onnx_sha256: str
    feature_schema_version: str
    din_enabled: bool


def triton_numeric_model_version(onnx_sha256: str) -> int:
    """Derive a stable positive Triton version from an ONNX checksum."""
    digest = str(onnx_sha256 or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("ONNX sha256 must contain 64 hexadecimal characters")
    return max(1, int(digest[:15], 16))


def build_triton_model_config(
    *,
    din_enabled: bool,
    model_name: str = "ranking_onnx",
    input_dim: int = 28,
    max_batch_size: int = 256,
    max_queue_delay_microseconds: int = 2000,
    max_queue_size: int = 128,
    default_timeout_microseconds: int = 75000,
    instance_count: int = 1,
    intra_op_threads: int = 1,
    inter_op_threads: int = 1,
) -> str:
    """Render the explicit, bounded ONNX Runtime model configuration."""
    inputs = [
        ("base_features", "TYPE_FP32", f"[ {int(input_dim)} ]"),
    ]
    if din_enabled:
        inputs.extend(
            [
                ("candidate_indices", "TYPE_INT64", "[ 1 ]"),
                ("history_indices", "TYPE_INT64", "[ 3, 60 ]"),
                ("history_recency", "TYPE_FP32", "[ 3, 60 ]"),
                ("history_mask", "TYPE_BOOL", "[ 3, 60 ]"),
                ("summary_features", "TYPE_FP32", "[ 12 ]"),
            ]
        )
    input_blocks = ",\n".join(
        "  {\n"
        f'    name: "{name}"\n'
        f"    data_type: {data_type}\n"
        f"    dims: {dims}\n"
        "  }"
        for name, data_type, dims in inputs
    )
    output_blocks = ",\n".join(
        "  {\n"
        f'    name: "{name}"\n'
        "    data_type: TYPE_FP32\n"
        "    dims: [ 1 ]\n"
        "  }"
        for name in TRITON_OUTPUT_NAMES
    )
    return f"""name: "{model_name}"
backend: "onnxruntime"
max_batch_size: {int(max_batch_size)}
input [
{input_blocks}
]
output [
{output_blocks}
]
dynamic_batching {{
  max_queue_delay_microseconds: {int(max_queue_delay_microseconds)}
  default_queue_policy {{
    max_queue_size: {int(max_queue_size)}
    default_timeout_microseconds: {int(default_timeout_microseconds)}
    timeout_action: REJECT
  }}
}}
instance_group [ {{ kind: KIND_CPU count: {int(instance_count)} }} ]
parameters {{ key: "intra_op_thread_count" value: {{ string_value: "{int(intra_op_threads)}" }} }}
parameters {{ key: "inter_op_thread_count" value: {{ string_value: "{int(inter_op_threads)}" }} }}
parameters {{ key: "execution_mode" value: {{ string_value: "0" }} }}
optimization {{ graph {{ level: 1 }} }}
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_local_artifact(entry: Mapping[str, Any], label: str) -> Path:
    path = Path(str(entry.get("path") or ""))
    expected = str(entry.get("sha256") or "")
    if not path.is_file() or len(expected) != 64:
        raise ValueError(f"{label} artifact is incomplete")
    if _sha256(path) != expected:
        raise ValueError(f"{label} checksum mismatch")
    return path


def materialize_triton_repository(
    record: ModelArtifactRecord,
    repository: Path | str,
    *,
    required_model_version: str,
    model_name: str = "ranking_onnx",
    max_batch_size: int = 256,
    max_queue_delay_microseconds: int = 2000,
    max_queue_size: int = 128,
    default_timeout_microseconds: int = 75000,
    instance_count: int = 1,
    intra_op_threads: int = 1,
    inter_op_threads: int = 1,
) -> TritonMaterialization:
    """Verify lineage and atomically materialize one immutable Triton model."""
    if record.model_version != required_model_version:
        raise ValueError("ranking artifact model version mismatch")
    manifest = dict((record.payload or {}).get("artifact_manifest") or {})
    checkpoint = dict(manifest.get("checkpoint") or {})
    onnx_model = dict(manifest.get("onnx_model") or {})
    checkpoint_path = _verified_local_artifact(checkpoint, "source checkpoint")
    onnx_path = _verified_local_artifact(onnx_model, "ONNX model")
    if onnx_model.get("source_checkpoint_sha256") != checkpoint.get("sha256"):
        raise ValueError("ONNX source checkpoint lineage mismatch")
    if int(onnx_model.get("opset") or 0) != 17:
        raise ValueError("unsupported ONNX opset")
    if onnx_model.get("parity_verified") is not True:
        raise ValueError("ONNX parity verification is missing")
    input_dim = int(onnx_model.get("input_dim") or 0)
    if input_dim <= 0:
        raise ValueError("ONNX input dimension is missing or invalid")
    if str(onnx_model.get("model_architecture") or "dcn") != "dcn":
        raise ValueError("unsupported ONNX model architecture")
    expected_inputs = {"base_features": [None, input_dim]}
    if bool(onnx_model.get("din_enabled")):
        expected_inputs.update(
            {
                "candidate_indices": [None, 1],
                "history_indices": [None, 3, 60],
                "history_recency": [None, 3, 60],
                "history_mask": [None, 3, 60],
                "summary_features": [None, 12],
            }
        )
    if onnx_model.get("input_contract") != expected_inputs:
        raise ValueError("ONNX input contract does not match the ranking adapter")
    expected_outputs = {name: [None, 1] for name in TRITON_OUTPUT_NAMES}
    if onnx_model.get("output_contract") != expected_outputs:
        raise ValueError("ONNX output contract does not match the ranking adapter")
    if (
        str(onnx_model.get("target_triton_version") or "") != "26.07"
        or str(onnx_model.get("target_onnxruntime_version") or "") != "1.27.0"
    ):
        raise ValueError("ONNX runtime contract does not match Triton 26.07")
    feature_schema = str(onnx_model.get("feature_schema_version") or "")
    if feature_schema != str(
        (record.payload or {}).get("feature_schema_version") or ""
    ):
        raise ValueError("ONNX feature schema lineage mismatch")
    din_enabled = bool(onnx_model.get("din_enabled"))
    if din_enabled:
        din_sidecar = dict(manifest.get("din_embedding_sidecar") or {})
        _verified_local_artifact(din_sidecar, "DIN sidecar")
        if onnx_model.get("din_sidecar_sha256") != din_sidecar.get("sha256"):
            raise ValueError("ONNX DIN sidecar lineage mismatch")

    onnx_sha = str(onnx_model["sha256"])
    numeric_version = str(triton_numeric_model_version(onnx_sha))
    materialization = TritonMaterialization(
        model_name=model_name,
        business_model_version=record.model_version,
        numeric_version=numeric_version,
        onnx_sha256=onnx_sha,
        feature_schema_version=feature_schema,
        din_enabled=din_enabled,
    )
    config = build_triton_model_config(
        din_enabled=din_enabled,
        model_name=model_name,
        input_dim=input_dim,
        max_batch_size=max_batch_size,
        max_queue_delay_microseconds=max_queue_delay_microseconds,
        max_queue_size=max_queue_size,
        default_timeout_microseconds=default_timeout_microseconds,
        instance_count=instance_count,
        intra_op_threads=intra_op_threads,
        inter_op_threads=inter_op_threads,
    )
    repository_path = Path(repository)
    repository_path.mkdir(parents=True, exist_ok=True)
    target_model_root = repository_path / model_name
    if target_model_root.exists():
        existing_versions = sorted(
            path.name for path in target_model_root.iterdir() if path.is_dir()
        )
        existing_model = target_model_root / numeric_version / "model.onnx"
        existing_config = target_model_root / "config.pbtxt"
        if (
            existing_versions != [numeric_version]
            or not existing_model.is_file()
            or _sha256(existing_model) != onnx_sha
            or not existing_config.is_file()
            or existing_config.read_text(encoding="utf-8") != config
        ):
            raise FileExistsError(
                f"Triton repository contains a different model: {target_model_root}"
            )
        return materialization

    temporary = Path(tempfile.mkdtemp(prefix=f".{model_name}.", dir=repository_path))
    try:
        model_root = temporary / model_name
        version_root = model_root / numeric_version
        version_root.mkdir(parents=True)
        staged_model = version_root / "model.onnx"
        shutil.copy2(onnx_path, staged_model)
        if _sha256(staged_model) != onnx_sha:
            raise RuntimeError("staged ONNX checksum mismatch")
        (model_root / "config.pbtxt").write_text(config, encoding="utf-8")
        os.replace(model_root, target_model_root)
        temporary.rmdir()
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    # Keep this read to make corruption between verification and copy observable.
    if (
        _sha256(repository_path / model_name / numeric_version / "model.onnx")
        != onnx_sha
    ):
        raise RuntimeError("materialized ONNX checksum mismatch")
    _ = checkpoint_path
    return materialization


class TritonGrpcClient:
    """Small async client with no inference retry."""

    def __init__(self, url: str, model_name: str, timeout_seconds: float) -> None:
        try:
            import tritonclient.grpc as grpcclient
            import tritonclient.grpc.aio as grpc_aio
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RankingTritonUnavailable("tritonclient is not installed") from exc
        self._grpc = grpcclient
        self._client = grpc_aio.InferenceServerClient(url=url, verbose=False)
        self.model_name = model_name
        self.timeout_seconds = float(timeout_seconds)

    async def close(self) -> None:
        await self._client.close()

    async def is_ready(self, *, model_version: str) -> bool:
        readiness = await self.readiness(model_version=model_version)
        return bool(readiness["server_ready"] and readiness["model_ready"])

    async def readiness(self, *, model_version: str) -> Dict[str, bool]:
        try:
            server_ready = bool(await self._client.is_server_ready())
            model_ready = bool(
                server_ready
                and await self._client.is_model_ready(
                    self.model_name, model_version=model_version
                )
            )
        except Exception:
            return {"server_ready": False, "model_ready": False}
        return {"server_ready": server_ready, "model_ready": model_ready}

    async def infer(
        self,
        *,
        inputs: Mapping[str, np.ndarray],
        model_version: str,
        request_id: Optional[str],
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        infer_inputs = []
        for name, values in inputs.items():
            contiguous = np.ascontiguousarray(values)
            item = self._grpc.InferInput(
                name,
                list(contiguous.shape),
                self._grpc.np_to_triton_dtype(contiguous.dtype),
            )
            item.set_data_from_numpy(contiguous)
            infer_inputs.append(item)
        outputs = [
            self._grpc.InferRequestedOutput(name) for name in TRITON_OUTPUT_NAMES
        ]
        try:
            response = await self._client.infer(
                self.model_name,
                infer_inputs,
                model_version=model_version,
                outputs=outputs,
                request_id=str(request_id or ""),
                client_timeout=float(timeout_seconds or self.timeout_seconds),
            )
        except Exception as exc:
            message = str(exc).lower()
            if any(
                token in message
                for token in ("resource_exhausted", "queue", "timeout", "deadline")
            ):
                raise RankingTritonOverloaded(str(exc)) from exc
            raise RankingTritonUnavailable(str(exc)) from exc
        result = {}
        for name in TRITON_OUTPUT_NAMES:
            values = response.as_numpy(name)
            if values is None:
                raise RankingTritonUnavailable(f"Triton response is missing {name}")
            result[name] = np.asarray(values, dtype=np.float32)
        return result


class RankingTritonAdapter:
    """Feature/post-processing adapter around a stateless Triton forward pass."""

    def __init__(
        self,
        *,
        ranking_model: Any,
        triton_client: Any,
        required_model_version: str,
        numeric_model_version: str,
        max_inflight: int,
        max_batch_size: int = 256,
        request_timeout_seconds: float = 0.5,
        onnx_sha256: str = "",
        observability: Optional[Any] = None,
    ) -> None:
        if not required_model_version:
            raise ValueError("RANKING_REQUIRED_MODEL_VERSION is required")
        if getattr(ranking_model, "model_version", None) != required_model_version:
            raise ValueError(
                "loaded ranking model version does not match required version"
            )
        if bool(
            getattr(
                ranking_model.config if hasattr(ranking_model, "config") else None,
                "trimodal_enabled",
                False,
            )
        ):
            raise ValueError("Triton ranking does not support trimodal models")
        self.ranking_model = ranking_model
        self.triton_client = triton_client
        self.required_model_version = required_model_version
        self.numeric_model_version = str(numeric_model_version)
        self.max_batch_size = max(1, int(max_batch_size))
        self.request_timeout_seconds = max(0.001, float(request_timeout_seconds))
        self.onnx_sha256 = str(onnx_sha256)
        self.observability = observability
        self._active_inflight = 0
        self._inflight = asyncio.Semaphore(max(1, int(max_inflight)))
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ranking-adapter"
        )

    async def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        close = getattr(self.triton_client, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result

    async def health(self) -> Dict[str, Any]:
        failure_reason = self.ranking_model.readiness_failure_reason()
        readiness_method = getattr(self.triton_client, "readiness", None)
        if readiness_method is not None:
            triton_health = await readiness_method(
                model_version=self.numeric_model_version
            )
            server_ready = bool(triton_health.get("server_ready"))
            model_ready = bool(triton_health.get("model_ready"))
        else:
            model_ready = bool(
                await self.triton_client.is_ready(
                    model_version=self.numeric_model_version
                )
            )
            server_ready = model_ready
        ready = failure_reason is None and server_ready and model_ready
        if self.observability is not None:
            self.observability.set_ranking_triton_model_ready(ready)
        return {
            "status": "healthy" if ready else "unhealthy",
            "inference_backend": "triton",
            "triton_server_ready": server_ready,
            "triton_model_ready": model_ready,
            "model_version": self.required_model_version,
            "triton_numeric_model_version": self.numeric_model_version,
            "onnx_sha256": self.onnx_sha256,
            "feature_schema_version": self.ranking_model.feature_schema_version,
            "din_enabled": bool(
                getattr(self.ranking_model.config, "din_enabled", False)
            ),
            "failure_reason": failure_reason if failure_reason else None,
        }

    async def rank_payload(
        self,
        raw_payload: Mapping[str, Any],
        *,
        request_id: Optional[str] = None,
    ):
        if self._inflight.locked():
            if self.observability is not None:
                self.observability.record_ranking_adapter_request("rejected_inflight")
            raise RankingTritonOverloaded("ranking adapter inflight limit reached")
        async with self._inflight:
            self._active_inflight += 1
            if self.observability is not None:
                self.observability.set_ranking_adapter_inflight(self._active_inflight)
            try:
                result = await self._rank_payload_admitted(
                    raw_payload,
                    request_id=request_id,
                )
            except RankingTritonOverloaded:
                if self.observability is not None:
                    self.observability.record_ranking_adapter_request("overloaded")
                raise
            except HTTPException:
                if self.observability is not None:
                    self.observability.record_ranking_adapter_request("invalid_request")
                raise
            except RankingTritonUnavailable:
                if self.observability is not None:
                    self.observability.record_ranking_adapter_request("unavailable")
                raise
            except Exception:
                if self.observability is not None:
                    self.observability.record_ranking_adapter_request("unavailable")
                raise RankingTritonUnavailable("ranking adapter execution failed")
            else:
                if self.observability is not None:
                    self.observability.record_ranking_adapter_request("success")
                return result
            finally:
                self._active_inflight = max(0, self._active_inflight - 1)
                if self.observability is not None:
                    self.observability.set_ranking_adapter_inflight(
                        self._active_inflight
                    )

    async def _rank_payload_admitted(
        self,
        raw_payload: Mapping[str, Any],
        *,
        request_id: Optional[str],
    ):
        self.ranking_model.ensure_ready_for_inference()
        payload = coerce_rank_payload(raw_payload)
        context = dict(payload.context)
        behavior_sequences = raw_payload.get("behavior_sequences")
        if behavior_sequences is not None:
            context[DIN_SEQUENCE_CONTEXT_KEY] = behavior_sequences
        if payload.multimodal_context:
            raise RankingTritonUnavailable("trimodal payload is unsupported by Triton")
        loop = asyncio.get_running_loop()
        feature_started = time.perf_counter()
        feature_matrix, prepared, inputs = await loop.run_in_executor(
            self._executor,
            self._prepare_inputs,
            {
                "index": 0,
                "candidates": payload.candidates,
                "user_features": payload.user_features,
                "context": context,
                "product_metadata_map": payload.product_metadata_map,
                "k": payload.k,
            },
        )
        feature_seconds = time.perf_counter() - feature_started
        if self.observability is not None:
            self.observability.record_ranking_adapter_stage(
                "feature_prep", feature_seconds
            )
        feature_ms = feature_seconds * 1000.0
        if feature_matrix is None or not prepared:
            return [], self._profile(feature_ms, 0.0, 0.0, 0)
        infer_started = time.perf_counter()
        predictions = await self._infer_chunks(
            inputs,
            row_count=int(feature_matrix.shape[0]),
            deadline_unix_seconds=payload.deadline_unix_seconds,
            request_id=request_id or payload.request_id,
        )
        self._validate_predictions(predictions, int(feature_matrix.shape[0]))
        inference_seconds = time.perf_counter() - infer_started
        if self.observability is not None:
            self.observability.record_ranking_adapter_stage(
                "triton_inference", inference_seconds
            )
        inference_ms = inference_seconds * 1000.0
        item = prepared[0]
        valid_candidates = item["valid_candidates"]
        post_started = time.perf_counter()
        recommendations = await loop.run_in_executor(
            self._executor,
            self._postprocess,
            valid_candidates,
            predictions,
            int(payload.k),
        )
        post_seconds = time.perf_counter() - post_started
        if self.observability is not None:
            self.observability.record_ranking_adapter_stage("postprocess", post_seconds)
        post_ms = post_seconds * 1000.0
        return recommendations, self._profile(
            feature_ms,
            inference_ms,
            post_ms,
            len(recommendations),
        )

    async def _infer_chunks(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        row_count: int,
        deadline_unix_seconds: Optional[float],
        request_id: Optional[str],
    ) -> Dict[str, np.ndarray]:
        request_deadline = time.monotonic() + self.request_timeout_seconds
        chunks: Dict[str, list[np.ndarray]] = {name: [] for name in TRITON_OUTPUT_NAMES}
        for start in range(0, row_count, self.max_batch_size):
            end = min(row_count, start + self.max_batch_size)
            chunk_inputs = {name: values[start:end] for name, values in inputs.items()}
            timeout = self._remaining_timeout(
                deadline_unix_seconds,
                monotonic_deadline=request_deadline,
            )
            try:
                chunk_predictions = await self.triton_client.infer(
                    inputs=chunk_inputs,
                    model_version=self.numeric_model_version,
                    request_id=request_id,
                    timeout_seconds=timeout,
                )
            except (RankingTritonOverloaded, RankingTritonUnavailable):
                raise
            except Exception as exc:
                raise RankingTritonUnavailable(str(exc)) from exc
            self._validate_predictions(chunk_predictions, end - start)
            for name in TRITON_OUTPUT_NAMES:
                chunks[name].append(np.asarray(chunk_predictions[name]))
        return {name: np.concatenate(values, axis=0) for name, values in chunks.items()}

    def _prepare_inputs(self, request: Dict[str, Any]):
        feature_matrix, prepared, _ = self.ranking_model.prepare_batch_matrix([request])
        inputs = (
            self._tensor_inputs(feature_matrix) if feature_matrix is not None else {}
        )
        return feature_matrix, prepared, inputs

    def _postprocess(
        self,
        valid_candidates: Any,
        predictions: Dict[str, np.ndarray],
        k: int,
    ):
        value_bucket_ids = self.ranking_model._value_bucket_ids_for_candidates(
            valid_candidates
        )
        self.ranking_model._add_business_predictions(predictions, value_bucket_ids)
        recommendations, _ = self.ranking_model.build_recommendations_from_predictions(
            valid_candidates,
            predictions,
            k,
        )
        return recommendations

    @staticmethod
    def _validate_predictions(
        predictions: Mapping[str, np.ndarray], row_count: int
    ) -> None:
        for name in TRITON_OUTPUT_NAMES:
            value = predictions.get(name)
            if value is None:
                raise RankingTritonUnavailable(
                    f"Triton tensor contract is missing {name}"
                )
            array = np.asarray(value)
            if array.shape != (row_count, 1) or not np.isfinite(array).all():
                raise RankingTritonUnavailable(
                    f"Triton tensor contract is invalid for {name}"
                )

    def _remaining_timeout(
        self,
        deadline: Optional[float],
        *,
        monotonic_deadline: Optional[float] = None,
    ) -> float:
        remaining = self.request_timeout_seconds
        if monotonic_deadline is not None:
            remaining = min(remaining, monotonic_deadline - time.monotonic())
        if deadline is not None:
            remaining = min(remaining, float(deadline) - time.time())
        if remaining <= 0:
            raise RankingTritonOverloaded("ranking deadline expired")
        return remaining

    @staticmethod
    def _tensor_inputs(feature_matrix: Any) -> Dict[str, np.ndarray]:
        result = {"base_features": np.asarray(feature_matrix, dtype=np.float32)}
        din_inputs = getattr(feature_matrix, "din_inputs", None)
        if din_inputs is None:
            return result
        history_indices, history_recency, history_mask = din_inputs.expanded_histories()
        result.update(
            {
                "candidate_indices": din_inputs.candidate_indices.detach()
                .cpu()
                .numpy()
                .astype(np.int64)
                .reshape(-1, 1),
                "history_indices": history_indices.detach()
                .cpu()
                .numpy()
                .astype(np.int64),
                "history_recency": history_recency.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                "history_mask": history_mask.detach().cpu().numpy().astype(bool),
                "summary_features": din_inputs.summary_features.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
            }
        )
        return result

    def _profile(
        self,
        feature_ms: float,
        inference_ms: float,
        post_ms: float,
        ranked_count: int,
    ) -> Dict[str, Any]:
        return {
            "path": "triton_onnx",
            "inference_backend": "triton",
            "model_version": self.required_model_version,
            "triton_numeric_model_version": self.numeric_model_version,
            "feature_schema_version": self.ranking_model.feature_schema_version,
            "feature_extraction_ms": round(feature_ms, 2),
            "model_forward_ms": round(inference_ms, 2),
            "response_build_ms": round(post_ms, 2),
            "ranked_count": int(ranked_count),
        }
