import asyncio
import hashlib
from pathlib import Path

import numpy as np
import pytest

from video_commerce.common.config import RankingConfig
from video_commerce.ml.model_artifacts import ModelArtifactRecord
from video_commerce.ranking_runtime.ranking_triton import (
    RankingTritonAdapter,
    RankingTritonOverloaded,
    RankingTritonUnavailable,
    TRITON_OUTPUT_NAMES,
    build_triton_model_config,
    materialize_triton_repository,
    triton_numeric_model_version,
)


def _record(tmp_path: Path, *, din: bool = False) -> ModelArtifactRecord:
    checkpoint = tmp_path / "ranking.pt"
    onnx_model = tmp_path / "ranking.onnx"
    checkpoint.write_bytes(b"checkpoint")
    onnx_model.write_bytes(b"onnx")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    onnx_sha = hashlib.sha256(onnx_model.read_bytes()).hexdigest()
    manifest = {
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "onnx_model": {
            "path": str(onnx_model),
            "sha256": onnx_sha,
            "source_checkpoint_sha256": checkpoint_sha,
            "opset": 17,
            "feature_schema_version": "ranking_v3_din"
            if din
            else "ranking_v3_00_temporal_multimodal",
            "input_dim": 28,
            "din_enabled": din,
            "parity_verified": True,
            "model_architecture": "dcn",
            "target_triton_version": "26.07",
            "target_onnxruntime_version": "1.27.0",
            "input_contract": {
                "base_features": [None, 28],
                **(
                    {
                        "candidate_indices": [None, 1],
                        "history_indices": [None, 3, 60],
                        "history_recency": [None, 3, 60],
                        "history_mask": [None, 3, 60],
                        "summary_features": [None, 12],
                    }
                    if din
                    else {}
                ),
            },
            "output_contract": {
                name: [None, 1]
                for name in ("ctr", "cvr", "ctcvr", "gmv", "ranking_score")
            },
        },
    }
    if din:
        sidecar = tmp_path / "din.npz"
        sidecar.write_bytes(b"din")
        manifest["din_embedding_sidecar"] = {
            "path": str(sidecar),
            "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        }
        manifest["onnx_model"]["din_sidecar_sha256"] = manifest[
            "din_embedding_sidecar"
        ]["sha256"]
    return ModelArtifactRecord(
        model_name="ranking_model",
        model_version="ranking-v1",
        checkpoint_path=str(checkpoint),
        payload={
            "feature_schema_version": manifest["onnx_model"]["feature_schema_version"],
            "artifact_manifest": manifest,
        },
    )


def test_numeric_model_version_is_positive_and_deterministic():
    digest = "0" * 15 + "f" * 49
    first = triton_numeric_model_version(digest)
    second = triton_numeric_model_version(digest)

    assert first == second == 1


def test_triton_config_defaults_are_opt_in_and_bounded():
    config = RankingConfig()

    assert config.inference_backend == "legacy"
    assert config.triton_grpc_url == "ranking-triton:8001"
    assert config.triton_request_timeout_ms == 500
    assert config.triton_max_inflight == 64
    assert config.triton_max_batch_size == 256
    assert config.triton_queue_delay_microseconds == 2000
    assert config.triton_queue_size == 128
    assert config.triton_queue_timeout_microseconds == 75000
    assert config.triton_instance_count == 1
    assert config.triton_ort_intra_op_threads == 1
    assert config.triton_ort_inter_op_threads == 1
    assert config.onnx_export_enabled is False


def test_ranking_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="legacy or triton"):
        RankingConfig(inference_backend="other")


def test_model_config_has_bounded_dynamic_batching_and_din_contract():
    config = build_triton_model_config(din_enabled=True, model_name="ranker_v2")

    assert 'name: "ranker_v2"' in config
    assert 'backend: "onnxruntime"' in config
    assert "max_batch_size: 256" in config
    assert "max_queue_delay_microseconds: 2000" in config
    assert "max_queue_size: 128" in config
    assert "default_timeout_microseconds: 75000" in config
    assert 'name: "history_indices"' in config
    assert "dims: [ 3, 60 ]" in config
    assert 'key: "intra_op_thread_count"' in config
    assert "\n  },\n  {\n" in config
    assert "\n  }\n  {\n" not in config


def test_materializer_rejects_incomplete_tensor_contract(tmp_path):
    record = _record(tmp_path)
    del record.payload["artifact_manifest"]["onnx_model"]["input_dim"]

    with pytest.raises(ValueError, match="input dimension"):
        materialize_triton_repository(
            record,
            tmp_path / "repository",
            required_model_version="ranking-v1",
        )


def test_materializer_rejects_mismatched_tensor_contract(tmp_path):
    record = _record(tmp_path, din=True)
    record.payload["artifact_manifest"]["onnx_model"]["input_contract"][
        "history_indices"
    ] = [None, 2, 60]

    with pytest.raises(ValueError, match="input contract"):
        materialize_triton_repository(
            record,
            tmp_path / "repository",
            required_model_version="ranking-v1",
        )


def test_materializer_rejects_wrong_target_runtime(tmp_path):
    record = _record(tmp_path)
    record.payload["artifact_manifest"]["onnx_model"][
        "target_onnxruntime_version"
    ] = "1.26.0"

    with pytest.raises(ValueError, match="runtime contract"):
        materialize_triton_repository(
            record,
            tmp_path / "repository",
            required_model_version="ranking-v1",
        )


def test_materializer_builds_checksum_versioned_repository_atomically(tmp_path):
    record = _record(tmp_path, din=True)
    repository = tmp_path / "repository"

    result = materialize_triton_repository(
        record,
        repository,
        required_model_version="ranking-v1",
        model_name="ranking_onnx",
    )

    onnx_sha = record.payload["artifact_manifest"]["onnx_model"]["sha256"]
    numeric_version = str(triton_numeric_model_version(onnx_sha))
    assert result.numeric_version == numeric_version
    assert (
        repository / "ranking_onnx" / numeric_version / "model.onnx"
    ).read_bytes() == b"onnx"
    assert (repository / "ranking_onnx" / "config.pbtxt").is_file()
    assert not list(repository.glob(".*.tmp"))


def test_materializer_is_idempotent_for_the_same_verified_artifact(tmp_path):
    record = _record(tmp_path)
    repository = tmp_path / "repository"

    first = materialize_triton_repository(
        record,
        repository,
        required_model_version="ranking-v1",
    )
    second = materialize_triton_repository(
        record,
        repository,
        required_model_version="ranking-v1",
    )

    assert second == first
    assert not list(repository.glob(".*ranking_onnx.*"))


def test_materializer_rejects_source_lineage_mismatch(tmp_path):
    record = _record(tmp_path)
    record.payload["artifact_manifest"]["onnx_model"]["source_checkpoint_sha256"] = (
        "f" * 64
    )

    with pytest.raises(ValueError, match="source checkpoint lineage"):
        materialize_triton_repository(
            record,
            tmp_path / "repository",
            required_model_version="ranking-v1",
        )


class _FakeRankingModel:
    is_trained = True
    artifact_verified = True
    model_version = "ranking-v1"
    feature_schema_version = "ranking_v3_00_temporal_multimodal"
    config = type("Config", (), {"din_enabled": False, "trimodal_enabled": False})()

    def readiness_failure_reason(self):
        return None

    def ensure_ready_for_inference(self):
        return None

    def prepare_batch_matrix(self, requests):
        return (
            np.ones((2, 28), dtype=np.float32),
            [
                {
                    "valid_candidates": [(object(), {}), (object(), {})],
                    "row_start": 0,
                    "row_end": 2,
                    "candidate_count": 2,
                    "feature_extraction_ms": 1.0,
                    "k": 1,
                }
            ],
            1.0,
        )

    def _value_bucket_ids_for_candidates(self, candidates):
        return [None] * len(candidates)

    def _add_business_predictions(self, predictions, value_bucket_ids):
        predictions["business_score"] = predictions["ranking_score"]

    def build_recommendations_from_predictions(self, candidates, predictions, k):
        return ["ranked"], 0.5


class _FakeTritonClient:
    def __init__(self):
        self.timeout_seconds = None

    async def infer(self, **kwargs):
        self.timeout_seconds = kwargs["timeout_seconds"]
        return {
            "ctr": np.array([[0.2], [0.3]], dtype=np.float32),
            "cvr": np.array([[0.1], [0.1]], dtype=np.float32),
            "ctcvr": np.array([[0.02], [0.03]], dtype=np.float32),
            "gmv": np.array([[1.0], [2.0]], dtype=np.float32),
            "ranking_score": np.array([[0.4], [0.9]], dtype=np.float32),
        }


def test_adapter_executes_pre_and_post_processing_around_triton():
    client = _FakeTritonClient()
    adapter = RankingTritonAdapter(
        ranking_model=_FakeRankingModel(),
        triton_client=client,
        required_model_version="ranking-v1",
        numeric_model_version="7",
        max_inflight=1,
        request_timeout_seconds=0.5,
    )

    recommendations, profile = asyncio.run(
        adapter.rank_payload(
            {
                "candidates": [{"product_id": "p1"}, {"product_id": "p2"}],
                "user_features": {"user_id": "u1"},
                "context": {},
                "product_metadata_map": {},
                "k": 1,
            },
            request_id="r1",
        )
    )

    assert recommendations == ["ranked"]
    assert profile["inference_backend"] == "triton"
    assert profile["model_version"] == "ranking-v1"
    assert profile["triton_numeric_model_version"] == "7"
    assert 0.49 < client.timeout_seconds <= 0.5


def test_adapter_chunks_rows_above_triton_max_batch_and_preserves_order():
    row_count = 300

    class LargeRankingModel(_FakeRankingModel):
        def __init__(self):
            self.ranking_scores = None

        def prepare_batch_matrix(self, requests):
            return (
                np.arange(row_count * 28, dtype=np.float32).reshape(row_count, 28),
                [{"valid_candidates": [(object(), {})] * row_count}],
                1.0,
            )

        def build_recommendations_from_predictions(self, candidates, predictions, k):
            self.ranking_scores = predictions["ranking_score"].copy()
            return ["ranked"], 0.5

    class ChunkClient:
        def __init__(self):
            self.row_counts = []

        async def infer(self, **kwargs):
            rows = kwargs["inputs"]["base_features"]
            self.row_counts.append(rows.shape[0])
            values = rows[:, :1] / 28.0
            return {name: values.copy() for name in TRITON_OUTPUT_NAMES}

    model = LargeRankingModel()
    client = ChunkClient()
    adapter = RankingTritonAdapter(
        ranking_model=model,
        triton_client=client,
        required_model_version="ranking-v1",
        numeric_model_version="7",
        max_inflight=1,
        max_batch_size=256,
    )

    asyncio.run(
        adapter.rank_payload(
            {
                "candidates": [
                    {"product_id": f"p{index}"} for index in range(row_count)
                ],
                "user_features": {"user_id": "u1"},
                "context": {},
                "product_metadata_map": {},
                "k": 1,
            },
            request_id="large-request",
        )
    )

    assert client.row_counts == [256, 44]
    np.testing.assert_array_equal(
        model.ranking_scores[:, 0], np.arange(row_count, dtype=np.float32)
    )


def test_adapter_health_reports_server_and_exact_model_readiness_separately():
    class ReadinessClient:
        async def readiness(self, *, model_version):
            assert model_version == "7"
            return {"server_ready": True, "model_ready": False}

    adapter = RankingTritonAdapter(
        ranking_model=_FakeRankingModel(),
        triton_client=ReadinessClient(),
        required_model_version="ranking-v1",
        numeric_model_version="7",
        max_inflight=1,
    )

    health = asyncio.run(adapter.health())

    assert health["status"] == "unhealthy"
    assert health["triton_server_ready"] is True
    assert health["triton_model_ready"] is False


def test_adapter_uses_the_smaller_of_request_deadline_and_triton_timeout():
    client = _FakeTritonClient()
    adapter = RankingTritonAdapter(
        ranking_model=_FakeRankingModel(),
        triton_client=client,
        required_model_version="ranking-v1",
        numeric_model_version="7",
        max_inflight=1,
        request_timeout_seconds=0.5,
    )

    asyncio.run(
        adapter.rank_payload(
            {
                "candidates": [{"product_id": "p1"}],
                "user_features": {"user_id": "u1"},
                "context": {},
                "product_metadata_map": {},
                "k": 1,
                "deadline_unix_seconds": __import__("time").time() + 0.1,
            }
        )
    )

    assert 0 < client.timeout_seconds <= 0.1


def test_adapter_maps_client_capacity_and_transport_errors():
    class BusyClient:
        async def infer(self, **kwargs):
            raise RankingTritonOverloaded("busy")

    class BrokenClient:
        async def infer(self, **kwargs):
            raise RankingTritonUnavailable("broken")

    for client, error in (
        (BusyClient(), RankingTritonOverloaded),
        (BrokenClient(), RankingTritonUnavailable),
    ):
        adapter = RankingTritonAdapter(
            ranking_model=_FakeRankingModel(),
            triton_client=client,
            required_model_version="ranking-v1",
            numeric_model_version="7",
            max_inflight=1,
        )
        with pytest.raises(error):
            asyncio.run(
                adapter.rank_payload(
                    {
                        "candidates": [{"product_id": "p1"}],
                        "user_features": {"user_id": "u1"},
                        "context": {},
                        "product_metadata_map": {},
                        "k": 1,
                    }
                )
            )


def test_adapter_fails_closed_on_malformed_triton_output():
    class MalformedClient:
        async def infer(self, **kwargs):
            return {
                name: np.zeros((1, 1), dtype=np.float32)
                for name in ("ctr", "cvr", "ctcvr", "gmv", "ranking_score")
            }

    adapter = RankingTritonAdapter(
        ranking_model=_FakeRankingModel(),
        triton_client=MalformedClient(),
        required_model_version="ranking-v1",
        numeric_model_version="7",
        max_inflight=1,
    )

    with pytest.raises(RankingTritonUnavailable, match="tensor contract"):
        asyncio.run(
            adapter.rank_payload(
                {
                    "candidates": [{"product_id": "p1"}],
                    "user_features": {"user_id": "u1"},
                    "context": {},
                    "product_metadata_map": {},
                    "k": 1,
                }
            )
        )


def test_adapter_wraps_unclassified_inference_failure_as_unavailable():
    class RawBrokenClient:
        async def infer(self, **kwargs):
            raise RuntimeError("unexpected protocol failure")

    adapter = RankingTritonAdapter(
        ranking_model=_FakeRankingModel(),
        triton_client=RawBrokenClient(),
        required_model_version="ranking-v1",
        numeric_model_version="7",
        max_inflight=1,
    )

    with pytest.raises(RankingTritonUnavailable, match="unexpected protocol"):
        asyncio.run(
            adapter.rank_payload(
                {
                    "candidates": [{"product_id": "p1"}],
                    "user_features": {"user_id": "u1"},
                    "context": {},
                    "product_metadata_map": {},
                    "k": 1,
                }
            )
        )
