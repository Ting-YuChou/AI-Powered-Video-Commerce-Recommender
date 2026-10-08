from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import json

import numpy as np
import pytest
import torch

from video_commerce.common.config import VectorConfig
from video_commerce.common.models import InteractionType, UserInteractionRequest
from video_commerce.data_plane.kafka_client import KafkaManager
from video_commerce.data_plane.system_store import (
    build_ltr_training_samples_from_impression_records,
)
from video_commerce.ml.ranking_score import (
    SCORE_POLICY_VERSION,
    canonical_score_numpy,
    canonical_score_torch,
)
from video_commerce.ml.vector_search import VectorSearchEngine
from video_commerce.services.content_worker.video_processor import VideoProcessorWorker
from video_commerce.services.recommendation import api as recommendation_api


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_startup_uses_flink_profile_and_checks_official_job():
    startup = (REPO_ROOT / "startup.sh").read_text(encoding="utf-8")

    assert "run_compose --profile flink" in startup
    assert "video-commerce-interaction-features" in startup
    assert "flink list" in startup


@pytest.mark.asyncio
async def test_required_vector_index_does_not_create_sample_data(tmp_path):
    engine = VectorSearchEngine(
        VectorConfig(
            index_path=str(tmp_path / "missing.faiss"),
            bootstrap_mode="required",
        )
    )

    with pytest.raises(FileNotFoundError):
        await engine.load_index()

    assert engine.product_embeddings == {}


def test_vector_sample_bootstrap_is_rejected_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")

    with pytest.raises(ValueError, match="sample"):
        VectorConfig(bootstrap_mode="sample")


@pytest.mark.asyncio
async def test_sample_mode_does_not_implicitly_generate_products(tmp_path, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    engine = VectorSearchEngine(
        VectorConfig(
            index_path=str(tmp_path / "missing.faiss"),
            bootstrap_mode="sample",
        )
    )

    with pytest.raises(FileNotFoundError, match="bootstrap_sample_vector_index"):
        await engine.load_index()

    assert not (tmp_path / "missing.faiss").exists()


@pytest.mark.asyncio
async def test_required_vector_index_rejects_checksum_mismatch(tmp_path):
    path = tmp_path / "vector.faiss"
    writer = VectorSearchEngine(VectorConfig(index_path=str(path), bootstrap_mode="empty"))
    await writer._create_empty_index()
    await writer.save_index()
    active_index_path, _ = writer._resolve_active_artifact_paths(path)
    active_index_path.write_bytes(active_index_path.read_bytes() + b"corrupt")

    loader = VectorSearchEngine(VectorConfig(index_path=str(path), bootstrap_mode="required"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        await loader.load_index()


@pytest.mark.asyncio
async def test_required_vector_index_rejects_manifest_dimension_mismatch(tmp_path):
    path = tmp_path / "vector.faiss"
    writer = VectorSearchEngine(VectorConfig(index_path=str(path), bootstrap_mode="empty"))
    await writer._create_empty_index()
    await writer.save_index()
    active_index_path, _ = writer._resolve_active_artifact_paths(path)
    manifest_path = writer._manifest_path(active_index_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["embedding_dim"] = 999
    manifest_path.write_text(json.dumps(manifest))

    loader = VectorSearchEngine(VectorConfig(index_path=str(path), bootstrap_mode="required"))
    with pytest.raises(ValueError, match="dimension mismatch"):
        await loader.load_index()


@pytest.mark.asyncio
async def test_vector_index_switches_one_active_generation_pointer(tmp_path):
    path = tmp_path / "vector.faiss"
    writer = VectorSearchEngine(VectorConfig(index_path=str(path), bootstrap_mode="empty"))
    await writer._create_empty_index()
    await writer.save_index()
    pointer_path = writer._active_generation_pointer(path)
    first = json.loads(pointer_path.read_text())["generation"]

    await writer.save_index()
    second = json.loads(pointer_path.read_text())["generation"]

    assert first != second
    assert (writer._generation_root(path) / first).is_dir()
    assert (writer._generation_root(path) / second).is_dir()


def test_interaction_request_accepts_stable_event_id():
    event_id = uuid4()

    request = UserInteractionRequest(
        event_id=event_id,
        user_id="u1",
        product_id="p1",
        action=InteractionType.CLICK,
    )

    assert request.event_id == event_id


@pytest.mark.asyncio
async def test_kafka_interaction_preserves_caller_event_id():
    sent = {}

    class Producer:
        async def send(self, **kwargs):
            sent.update(kwargs)
            return True

    manager = object.__new__(KafkaManager)
    manager.producer = Producer()
    manager.config = SimpleNamespace(user_interactions_topic="user-interactions")
    event_id = str(uuid4())

    assert await manager.send_user_interaction(
        event_id=event_id,
        user_id="u1",
        product_id="p1",
        action="click",
        context={},
        event_time=100.0,
        server_received_at=101.0,
    )

    assert sent["value"]["event_id"] == event_id
    assert sent["value"]["source_event_id"] == event_id
    assert dict(sent["headers"])["x-event-id"] == event_id.encode()


def test_canonical_score_matches_numpy_and_torch():
    ctcvr = np.array([1.2, 0.25, -0.1], dtype=np.float32)
    value = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    raw = np.array([3.0, 2.0, 1.0], dtype=np.float32)

    numpy_score = canonical_score_numpy(
        ctcvr=ctcvr,
        predicted_value=value,
        raw_ranking_score=raw,
        business_score_enabled=True,
    )
    torch_score = canonical_score_torch(
        ctcvr=torch.from_numpy(ctcvr),
        predicted_value=torch.from_numpy(value),
        raw_ranking_score=torch.from_numpy(raw),
        business_score_enabled=True,
    )

    assert SCORE_POLICY_VERSION == "business-value-v1"
    np.testing.assert_allclose(numpy_score, [10.0, 5.0, 0.0])
    np.testing.assert_allclose(torch_score.numpy(), numpy_score)


def test_served_only_items_are_not_training_negatives():
    impressions = [
        {
            "impression_id": "imp-1",
            "request_id": "req-1",
            "user_id": "u1",
            "product_id": "p-served",
            "position": 1,
            "context": {},
            "feature_snapshot": {},
            "scores": {"ranking_score": 0.9},
            "created_at": 100.0,
        },
        {
            "impression_id": "imp-1",
            "request_id": "req-1",
            "user_id": "u1",
            "product_id": "p-viewed",
            "position": 2,
            "context": {},
            "feature_snapshot": {},
            "scores": {"ranking_score": 0.8},
            "created_at": 100.0,
        },
    ]
    views = [
        {
            "event_id": "view-1",
            "impression_id": "imp-1",
            "product_id": "p-viewed",
            "viewed_at": 101.0,
        }
    ]

    samples = build_ltr_training_samples_from_impression_records(
        impressions,
        interactions=[],
        impression_views=views,
    )

    assert [sample["product_id"] for sample in samples] == ["p-viewed"]
    assert samples[0]["context"]["impression_viewed"] is True


def test_late_interaction_does_not_relabel_viewed_negative():
    impressions = [
        {
            "impression_id": "imp-1",
            "user_id": "u1",
            "product_id": "p1",
            "position": 1,
            "context": {},
            "feature_snapshot": {},
            "scores": {},
            "created_at": 100.0,
        }
    ]
    samples = build_ltr_training_samples_from_impression_records(
        impressions,
        interactions=[
            {
                "event_id": "late-click",
                "user_id": "u1",
                "product_id": "p1",
                "action": "click",
                "context": {"impression_id": "imp-1"},
                "occurred_at": 100.0 + 8 * 24 * 3600,
            }
        ],
        impression_views=[{"impression_id": "imp-1", "product_id": "p1"}],
        attribution_window_hours=168,
    )

    assert samples[0]["action"] == "view"
    assert samples[0]["context"]["attributed_click"] is False


@pytest.mark.asyncio
async def test_existing_matching_served_event_remains_durable(monkeypatch):
    class Kafka:
        def build_recommendation_event(self, **kwargs):
            return {
                "event_id": kwargs["event_id"],
                "metadata": kwargs["metadata"],
            }

    class Store:
        async def enqueue_recommendation_event(self, event):
            return "duplicate"

    monkeypatch.setattr(recommendation_api, "kafka_manager", Kafka())
    monkeypatch.setattr(recommendation_api, "system_store", Store())
    monkeypatch.setattr(
        recommendation_api.app.state.runtime.observability,
        "record_recommendation_impression",
        lambda *args: None,
    )

    durable = await recommendation_api._stage_served_impression_event(
        user_id="u1",
        recommendations=["p1"],
        response_time_ms=10,
        request_id="req-1",
        metadata={"impression_id": "imp-1"},
    )

    assert durable is True


def test_reliability_migration_tracks_interaction_lease_owner():
    migration = (
        REPO_ROOT / "migrations/postgres/008_recommendation_event_reliability.sql"
    ).read_text(encoding="utf-8")

    assert "lease_owner VARCHAR(64)" in migration


@pytest.mark.asyncio
async def test_content_worker_marks_missing_file_failed_and_reraises(tmp_path):
    statuses = []

    class FeatureStore:
        async def update_content_status(self, content_id, status):
            statuses.append(status)

    class Observability:
        def record_worker_message(self, *args):
            return None

    worker = object.__new__(VideoProcessorWorker)
    worker.feature_store = FeatureStore()
    worker.system_store = None
    worker.object_storage = None
    worker.observability = Observability()
    worker.config = SimpleNamespace(
        data_config=SimpleNamespace(cleanup_temp_files=True)
    )

    with pytest.raises(FileNotFoundError):
        await worker._handle_video_task(
            "video-processing",
            "content-1",
            {
                "content_id": "content-1",
                "file_path": str(tmp_path / "missing.mp4"),
                "filename": "missing.mp4",
            },
            None,
        )

    assert statuses == ["processing", "failed"]


@pytest.mark.asyncio
async def test_content_worker_preserves_processing_error_when_bookkeeping_fails(tmp_path):
    class FeatureStore:
        async def update_content_status(self, content_id, status):
            if status == "failed":
                raise RuntimeError("status store unavailable")

    class Observability:
        def record_worker_message(self, *args):
            return None

    worker = object.__new__(VideoProcessorWorker)
    worker.feature_store = FeatureStore()
    worker.system_store = None
    worker.object_storage = None
    worker.observability = Observability()
    worker.config = SimpleNamespace(
        data_config=SimpleNamespace(cleanup_temp_files=True)
    )

    with pytest.raises(FileNotFoundError, match="missing.mp4"):
        await worker._handle_video_task(
            "video-processing",
            "content-1",
            {
                "content_id": "content-1",
                "file_path": str(tmp_path / "missing.mp4"),
                "filename": "missing.mp4",
            },
            None,
        )
