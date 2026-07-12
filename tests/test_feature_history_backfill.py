from datetime import datetime, timezone

import pytest

from video_commerce.services.feature_history_backfill.main import (
    FeatureHistoryBackfillRunner,
    build_interaction_backfill_event,
)
from video_commerce.ml.feature_history_reconciliation import (
    BackfillReconciliationEvidence,
)


def test_interaction_backfill_uses_original_event_and_availability_times():
    event = build_interaction_backfill_event(
        {
            "event_id": "e1",
            "request_id": "r1",
            "user_id": "u1",
            "product_id": "p1",
            "action": "click",
            "context": {"surface": "home"},
            "occurred_at": datetime.fromtimestamp(100, tz=timezone.utc),
            "created_at": datetime.fromtimestamp(105, tz=timezone.utc),
        },
        run_id="run-1",
    )

    assert event["event_time"] == 100.0
    assert event["available_at"] == 105.0
    assert event["source_event_id"] == "e1"
    assert event["source_version"] == "interaction-v1"
    assert event["payload_hash"]
    assert event["backfill_run_id"] == "run-1"


class FakeStore:
    def __init__(self):
        self.checkpoints = []
        self.pages = [
            [
                {
                    "event_id": "e1",
                    "user_id": "u1",
                    "product_id": "p1",
                    "action": "view",
                    "context": {},
                    "occurred_at": datetime.fromtimestamp(100, tz=timezone.utc),
                    "created_at": datetime.fromtimestamp(101, tz=timezone.utc),
                }
            ],
            [],
        ]

    async def get_feature_history_backfill_run(self, run_id):
        return {
            "run_id": run_id,
            "phase": "interactions",
            "cursor_time": None,
            "cursor_id": None,
            "counts": {},
        }

    async def get_backfill_interactions_page(self, **kwargs):
        return self.pages.pop(0)

    async def checkpoint_feature_history_backfill(self, run_id, **kwargs):
        self.checkpoints.append((run_id, kwargs))


class FakeKafka:
    def __init__(self, succeeds=True):
        self.succeeds = succeeds
        self.events = []

    async def send_feature_history_backfill_event(self, *, topic, event, key):
        self.events.append((topic, event, key))
        return self.succeeds


@pytest.mark.asyncio
async def test_backfill_advances_cursor_only_after_broker_ack():
    store = FakeStore()
    kafka = FakeKafka(succeeds=True)
    runner = FeatureHistoryBackfillRunner(
        system_store=store,
        kafka_manager=kafka,
        topics={"interactions": "user-interactions-backfill"},
        page_size=10,
    )

    await runner.run_interactions("run-1", range_end=200.0)

    assert len(kafka.events) == 1
    assert store.checkpoints[0][1]["cursor_id"] == "e1"
    assert store.checkpoints[-1][1]["phase"] == "observations"


@pytest.mark.asyncio
async def test_backfill_does_not_checkpoint_unacknowledged_page():
    store = FakeStore()
    runner = FeatureHistoryBackfillRunner(
        system_store=store,
        kafka_manager=FakeKafka(succeeds=False),
        topics={"interactions": "user-interactions-backfill"},
        page_size=10,
    )

    with pytest.raises(RuntimeError, match="acknowledge"):
        await runner.run_interactions("run-1", range_end=200.0)

    assert store.checkpoints == []


class FakeReconciliationStore:
    def __init__(self):
        self.checkpoints = []

    async def get_feature_history_backfill_run(self, run_id):
        return {
            "run_id": run_id,
            "status": "active",
            "phase": "reconcile",
            "counts": {"interactions_published": 8, "catalog_published": 2},
        }

    async def checkpoint_feature_history_backfill(self, run_id, **kwargs):
        self.checkpoints.append((run_id, kwargs))


@pytest.mark.asyncio
async def test_backfill_reconciliation_fails_closed_on_unexplained_rows():
    store = FakeReconciliationStore()
    runner = FeatureHistoryBackfillRunner(
        system_store=store,
        kafka_manager=FakeKafka(),
        topics={},
    )

    with pytest.raises(RuntimeError, match="reconciliation mismatch"):
        await runner.reconcile(
            "run-1",
            evidence=BackfillReconciliationEvidence(
                kafka_source_event_ids=frozenset({f"e{i}" for i in range(10)}),
                iceberg_source_event_ids=frozenset({f"e{i}" for i in range(8)}),
                dlq_source_event_ids=frozenset({"e8"}),
                iceberg_row_count=8,
                kafka_end_offsets={},
                iceberg_snapshot_ids={},
            ),
        )

    assert store.checkpoints == []


@pytest.mark.asyncio
async def test_backfill_reconciliation_marks_run_complete_only_when_exact():
    store = FakeReconciliationStore()
    runner = FeatureHistoryBackfillRunner(
        system_store=store,
        kafka_manager=FakeKafka(),
        topics={},
    )

    await runner.reconcile(
        "run-1",
        evidence=BackfillReconciliationEvidence(
            kafka_source_event_ids=frozenset({f"e{i}" for i in range(10)}),
            iceberg_source_event_ids=frozenset({f"e{i}" for i in range(9)}),
            dlq_source_event_ids=frozenset({"e9"}),
            iceberg_row_count=9,
            kafka_end_offsets={"topic:0": 10},
            iceberg_snapshot_ids={"interaction_history": 42},
        ),
    )

    checkpoint = store.checkpoints[-1][1]
    assert checkpoint["phase"] == "complete"
    assert checkpoint["status"] == "complete"
    assert checkpoint["reconciliation"] == {
        "kafka_accepted": 10,
        "iceberg_accepted": 9,
        "dlq_count": 1,
        "duplicate_count": 0,
        "kafka_end_offsets": {"topic:0": 10},
        "iceberg_snapshot_ids": {"interaction_history": 42},
    }
