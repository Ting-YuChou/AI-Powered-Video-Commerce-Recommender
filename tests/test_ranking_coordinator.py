from types import SimpleNamespace

import pytest

from video_commerce.common.cache_codec import json_dumps
from video_commerce.services.ranking_coordinator import main as ranking_coordinator_module
from video_commerce.services.ranking_coordinator.main import RankingCoordinator
from video_commerce.ranking_runtime.ranking_coordinator_client import decode_response
from video_commerce.ranking_runtime.ranking_batcher import RankingQueueTimeoutError


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

    assert response.status_code == 503
    assert batcher.deadline_unix_seconds == pytest.approx(123.4)
