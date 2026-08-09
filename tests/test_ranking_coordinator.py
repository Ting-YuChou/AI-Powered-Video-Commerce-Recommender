from types import SimpleNamespace

import pytest

from video_commerce.common.cache_codec import json_dumps
from video_commerce.services.ranking_coordinator import (
    main as ranking_coordinator_module,
)
from video_commerce.services.ranking_coordinator.main import RankingCoordinator
from video_commerce.ranking_runtime.ranking_coordinator_client import decode_response
from video_commerce.ranking_runtime.ranking_batcher import (
    RankingOverloadedError,
    RankingQueueTimeoutError,
)
from video_commerce.ml.ranking import RankingModelNotReadyError


@pytest.mark.asyncio
async def test_coordinator_admission_uses_request_deadline(monkeypatch):
    class CapturingBatcher:
        def __init__(self):
            self.deadline_unix_seconds = None

        def should_reject_new_request(self, deadline_unix_seconds=None):
            self.deadline_unix_seconds = deadline_unix_seconds
            return True

        async def rank_candidates(self, **kwargs):
            raise RankingQueueTimeoutError("should not rank")

    batcher = CapturingBatcher()
    coordinator = RankingCoordinator()
    coordinator.ranking_batcher = batcher
    coordinator.runtime.observability = SimpleNamespace(
        record_request=lambda *args, **kwargs: None
    )
    monkeypatch.setattr(ranking_coordinator_module.time, "time", lambda: 100.0)
    body = json_dumps(
        {
            "request_id": "req-1",
            "deadline_unix_seconds": 123.4,
            "candidates": [],
            "user_features": {"user_id": "u1"},
            "context": {},
            "product_metadata_map": {},
            "k": 1,
        }
    )

    response = decode_response((await coordinator._handle_rank(body))[4:])

    assert response.status_code == 429
    assert batcher.deadline_unix_seconds == pytest.approx(123.4)


@pytest.mark.asyncio
async def test_coordinator_overload_has_stable_429_shape():
    class OverloadedBatcher:
        def should_reject_new_request(self, deadline_unix_seconds=None):
            return False

        async def rank_candidates(self, **kwargs):
            raise RankingOverloadedError("no_dispatch_capacity")

    coordinator = RankingCoordinator()
    coordinator.ranking_batcher = OverloadedBatcher()
    coordinator.runtime.observability = SimpleNamespace(
        record_request=lambda *args, **kwargs: None
    )
    body = json_dumps(
        {
            "request_id": "req-overload",
            "candidates": [],
            "user_features": {"user_id": "u1"},
            "context": {},
            "product_metadata_map": {},
            "k": 1,
        }
    )

    response = decode_response((await coordinator._handle_rank(body))[4:])

    assert response.status_code == 429
    assert response.body == json_dumps(
        {"detail": "ranking_overloaded", "retry_after_seconds": 1}
    )


@pytest.mark.asyncio
async def test_coordinator_requires_minimum_healthy_runners():
    class OneHealthyRunnerPool:
        async def health_check(self):
            return {
                "status": "healthy",
                "healthy_count": 1,
                "total_count": 4,
                "endpoints": [
                    {
                        "endpoint": "runner-1",
                        "status": "healthy",
                        "model_version": "ranking-v1",
                        "feature_schema_version": "ranking_v3_00_temporal_multimodal",
                    }
                ],
            }

    coordinator = RankingCoordinator()
    coordinator.runner_pool = OneHealthyRunnerPool()
    coordinator.config = SimpleNamespace(
        service_topology_config=SimpleNamespace(ranking_min_healthy_runners=2)
    )
    coordinator.runtime.observability = SimpleNamespace(
        record_request=lambda *args, **kwargs: None
    )

    response = decode_response((await coordinator._handle_health())[4:])
    payload = ranking_coordinator_module.json_loads(response.body)

    assert response.status_code == 503
    assert payload["checks"]["ranking_runners"]["status"] == "degraded"
    assert payload["checks"]["ranking_runners"]["required_healthy_count"] == 2


@pytest.mark.asyncio
async def test_coordinator_maps_model_not_ready_to_503():
    class UnreadyBatcher:
        def should_reject_new_request(self, deadline_unix_seconds=None):
            return False

        async def rank_candidates(self, **kwargs):
            raise RankingModelNotReadyError("ranking_model_not_ready")

    coordinator = RankingCoordinator()
    coordinator.ranking_batcher = UnreadyBatcher()
    coordinator.runtime.observability = SimpleNamespace(
        record_request=lambda *args, **kwargs: None
    )
    response = decode_response(
        (
            await coordinator._handle_rank(
                json_dumps(
                    {
                        "request_id": "req-unready",
                        "candidates": [],
                        "user_features": {"user_id": "u1"},
                        "context": {},
                        "product_metadata_map": {},
                        "k": 1,
                    }
                )
            )
        )[4:]
    )

    assert response.status_code == 503
    assert response.body == json_dumps({"detail": "ranking_model_not_ready"})


@pytest.mark.asyncio
async def test_coordinator_local_fallback_can_report_degraded_without_blocking_startup():
    class DegradedRunnerPool:
        async def health_check(self):
            return {
                "status": "degraded",
                "healthy_count": 0,
                "degraded_count": 4,
                "total_count": 4,
                "endpoints": [],
            }

    coordinator = RankingCoordinator()
    coordinator.runner_pool = DegradedRunnerPool()
    coordinator.config = SimpleNamespace(
        ranking_config=SimpleNamespace(allow_untrained_fallback=True),
        service_topology_config=SimpleNamespace(ranking_min_healthy_runners=2),
    )
    coordinator.runtime.observability = SimpleNamespace(
        record_request=lambda *args, **kwargs: None
    )

    response = decode_response((await coordinator._handle_health())[4:])
    payload = ranking_coordinator_module.json_loads(response.body)

    assert response.status_code == 200
    assert payload["status"] == "degraded"
    assert payload["checks"]["ranking_runners"]["status"] == "degraded"
