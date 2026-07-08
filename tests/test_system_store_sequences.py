import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy.dialects import postgresql

from video_commerce.common.config import DatabaseConfig
from video_commerce.data_plane.system_store import (
    SystemStore,
    TWO_TOWER_POSITIVE_ACTIONS,
    _impression_context_snapshot,
    build_two_tower_training_negatives_from_impression_records,
    build_chronological_user_sequences,
    build_ltr_training_samples_from_impression_records,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _event(user_id, product_id, action, event_id, timestamp):
    return {
        "event_id": event_id,
        "schema_version": 1,
        "request_id": f"req-{event_id}",
        "user_id": user_id,
        "product_id": product_id,
        "action": action,
        "context": {"page": "test"},
        "occurred_at": datetime.fromtimestamp(timestamp, tz=timezone.utc),
    }


def test_build_chronological_user_sequences_groups_and_orders_per_user():
    sequences = build_chronological_user_sequences(
        [
            _event("u2", "p4", "purchase", "e4", 4),
            _event("u1", "p2", "click", "e2", 2),
            _event("u1", "p1", "view", "e1", 1),
            _event("u2", "p3", "view", "e3", 3),
        ],
        max_events_per_user=10,
        min_sequence_length=2,
    )

    assert [event["product_id"] for event in sequences["u1"]] == ["p1", "p2"]
    assert [event["product_id"] for event in sequences["u2"]] == ["p3", "p4"]


def test_build_chronological_user_sequences_uses_event_id_for_timestamp_ties():
    sequences = build_chronological_user_sequences(
        [
            _event("u1", "p2", "click", "e2", 10),
            _event("u1", "p1", "view", "e1", 10),
        ],
        max_events_per_user=10,
        min_sequence_length=2,
    )

    assert [event["event_id"] for event in sequences["u1"]] == ["e1", "e2"]


def test_build_chronological_user_sequences_filters_noisy_actions():
    sequences = build_chronological_user_sequences(
        [
            _event("u1", "p1", "view", "e1", 1),
            _event("u1", "p2", "remove_from_cart", "e2", 2),
            _event("u1", "p3", "share", "e3", 3),
            _event("u1", "p4", "purchase", "e4", 4),
        ],
        max_events_per_user=10,
        min_sequence_length=2,
    )

    assert [event["product_id"] for event in sequences["u1"]] == ["p1", "p4"]


def test_build_chronological_user_sequences_keeps_recent_window_in_order():
    sequences = build_chronological_user_sequences(
        [
            _event("u1", "p1", "view", "e1", 1),
            _event("u1", "p2", "click", "e2", 2),
            _event("u1", "p3", "add_to_cart", "e3", 3),
            _event("u1", "p4", "purchase", "e4", 4),
        ],
        max_events_per_user=2,
        min_sequence_length=2,
    )

    assert [event["product_id"] for event in sequences["u1"]] == ["p3", "p4"]


def test_impression_context_snapshot_drops_large_realtime_features():
    snapshot = _impression_context_snapshot(
        {
            "session_id": "session-1",
            "surface": "recommendations",
            "_realtime_window_features": {"p1": {"clicks": 100}},
            "untrusted_blob": {"large": "value"},
            "location": "x" * 600,
        }
    )

    assert snapshot["session_id"] == "session-1"
    assert snapshot["surface"] == "recommendations"
    assert snapshot["location"] == "x" * 512
    assert "_realtime_window_features" not in snapshot
    assert "untrusted_blob" not in snapshot


class RecordingBeginSession:
    def __init__(self):
        self.executed = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, statement):
        self.executed.append(statement)

    async def get(self, *args, **kwargs):
        raise AssertionError("update_content_job_status should not issue a SELECT")


class RecordingBeginFactory:
    def __init__(self):
        self.session = RecordingBeginSession()

    def begin(self):
        return self.session


def _compiled_postgres(statement):
    return str(statement.compile(dialect=postgresql.dialect()))


def _update_clause(sql):
    return sql.split("DO UPDATE SET", 1)[1]


def test_update_content_job_status_uses_single_upsert_without_select():
    session_factory = RecordingBeginFactory()
    store = SystemStore(DatabaseConfig(enable=True))
    store.session_factory = session_factory

    asyncio.run(
        store.update_content_job_status(
            "content-1",
            "completed",
            storage_path="s3://bucket/content-1.mp4",
            payload={"request_id": "req-1"},
        )
    )

    assert len(session_factory.session.executed) == 1
    sql = _compiled_postgres(session_factory.session.executed[0])
    update_clause = _update_clause(sql)
    assert "INSERT INTO content_jobs" in sql
    assert "ON CONFLICT (content_id) DO UPDATE SET" in sql
    assert "status" in update_clause
    assert "error_message" in update_clause
    assert "updated_at" in update_clause
    assert "storage_path" in update_clause
    assert "payload" in update_clause


def test_update_content_job_status_does_not_overwrite_optional_fields_with_none():
    session_factory = RecordingBeginFactory()
    store = SystemStore(DatabaseConfig(enable=True))
    store.session_factory = session_factory

    asyncio.run(store.update_content_job_status("content-1", "failed"))

    sql = _compiled_postgres(session_factory.session.executed[0])
    update_clause = _update_clause(sql)
    assert "status" in update_clause
    assert "error_message" in update_clause
    assert "updated_at" in update_clause
    assert "storage_path" not in update_clause
    assert "payload" not in update_clause


class FakeResult:
    def __init__(self, *, one_value=None, all_rows=None):
        self.one_value = one_value
        self.all_rows = all_rows or []

    def one(self):
        return self.one_value

    def all(self):
        return self.all_rows


class AnalyticsSession:
    def __init__(self, factory):
        self.factory = factory

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, statement):
        self.factory.execute_count += 1
        return self.factory.results.pop(0)


class AnalyticsSessionFactory:
    def __init__(self, *summary_rows):
        self.results = []
        self.execute_count = 0
        for totals, action_rows in summary_rows:
            self.results.append(FakeResult(one_value=totals))
            self.results.append(FakeResult(all_rows=action_rows))

    def __call__(self):
        return AnalyticsSession(self)


def test_analytics_summary_uses_ttl_cache_for_same_window():
    session_factory = AnalyticsSessionFactory(
        ((10, 2, 3), [("view", 8), ("click", 2)]),
    )
    store = SystemStore(
        DatabaseConfig(
            enable=True,
            analytics_window_hours=24,
            analytics_summary_cache_ttl_seconds=15,
        )
    )
    store.session_factory = session_factory

    first = asyncio.run(store.get_analytics_summary())
    first["action_counts"]["view"] = 999
    second = asyncio.run(store.get_analytics_summary())

    assert session_factory.execute_count == 2
    assert second["total_interactions"] == 10
    assert second["action_counts"] == {"view": 8, "click": 2}
    assert second["source"] == "postgres"
    assert second["window_hours"] == 24


def test_analytics_summary_recomputes_after_ttl_expires():
    session_factory = AnalyticsSessionFactory(
        ((10, 2, 3), [("view", 8), ("click", 2)]),
        ((20, 4, 6), [("view", 10), ("purchase", 1)]),
    )
    store = SystemStore(
        DatabaseConfig(
            enable=True,
            analytics_window_hours=24,
            analytics_summary_cache_ttl_seconds=15,
        )
    )
    store.session_factory = session_factory

    first = asyncio.run(store.get_analytics_summary())
    store._analytics_summary_cache[24] = (
        time.time() - 16,
        store._analytics_summary_cache[24][1],
    )
    second = asyncio.run(store.get_analytics_summary())

    assert session_factory.execute_count == 4
    assert first["total_interactions"] == 10
    assert second["total_interactions"] == 20
    assert second["action_counts"] == {"view": 10, "purchase": 1}


def test_analytics_summary_cache_can_be_disabled():
    session_factory = AnalyticsSessionFactory(
        ((10, 2, 3), [("view", 8), ("click", 2)]),
        ((20, 4, 6), [("view", 10), ("purchase", 1)]),
    )
    store = SystemStore(
        DatabaseConfig(
            enable=True,
            analytics_window_hours=24,
            analytics_summary_cache_ttl_seconds=0,
        )
    )
    store.session_factory = session_factory

    first = asyncio.run(store.get_analytics_summary())
    second = asyncio.run(store.get_analytics_summary())

    assert session_factory.execute_count == 4
    assert first["total_interactions"] == 10
    assert second["total_interactions"] == 20


def test_positive_sequence_index_is_declared_in_migrations():
    migration_001 = (
        REPO_ROOT / "migrations/postgres/001_partition_interaction_events.sql"
    ).read_text(encoding="utf-8")
    migration_002 = (
        REPO_ROOT / "migrations/postgres/002_positive_sequence_index.sql"
    ).read_text(encoding="utf-8")

    for sql in (migration_001, migration_002):
        assert "ix_interaction_events_positive_user_sequence" in sql
        assert "ON interaction_events (user_id, occurred_at DESC, event_id DESC)" in sql
        assert "WHERE action IN ('view', 'click', 'add_to_cart', 'purchase')" in sql


def test_recommendation_impression_tables_are_declared_in_migration():
    migration_003 = (
        REPO_ROOT / "migrations/postgres/003_recommendation_impressions.sql"
    ).read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS recommendation_impressions" in migration_003
    assert "CREATE TABLE IF NOT EXISTS recommendation_impression_items" in migration_003
    assert "ix_recommendation_impressions_user_created" in migration_003
    assert "ix_recommendation_impressions_created_at" in migration_003
    assert "ix_recommendation_impression_items_product_created" in migration_003
    assert "ux_recommendation_impression_items_impression_product" in migration_003


class RecordingConnection:
    def __init__(self):
        self.statements = []

    async def execute(self, statement):
        self.statements.append(str(statement))


class RecordingSession:
    def __init__(self):
        self.statements = []

    async def execute(self, statement):
        self.statements.append(str(statement.compile(dialect=postgresql.dialect())))


class RecordingSessionFactory:
    def __init__(self):
        self.session = RecordingSession()

    def begin(self):
        return self

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_positive_sequence_index_is_ensured_on_startup():
    conn = RecordingConnection()
    store = SystemStore(DatabaseConfig(enable=True))

    asyncio.run(store._ensure_operational_indexes(conn))

    combined_sql = "\n".join(conn.statements)
    assert "ix_interaction_events_positive_user_sequence" in combined_sql
    assert "ON interaction_events (user_id, occurred_at DESC, event_id DESC)" in combined_sql
    assert "WHERE action IN ('view', 'click', 'add_to_cart', 'purchase')" in combined_sql


def test_recommendation_impression_indexes_are_ensured_on_startup():
    conn = RecordingConnection()
    store = SystemStore(DatabaseConfig(enable=True))

    asyncio.run(store._ensure_operational_indexes(conn))

    combined_sql = "\n".join(conn.statements)
    assert "ix_recommendation_impressions_user_created" in combined_sql
    assert "ON recommendation_impressions (user_id, created_at DESC)" in combined_sql
    assert "ix_recommendation_impressions_created_at" in combined_sql
    assert "ix_recommendation_impression_items_impression" in combined_sql
    assert "ix_recommendation_impression_items_product_created" in combined_sql
    assert "ux_recommendation_impression_items_impression_product" in combined_sql


def test_record_recommendation_impression_uses_idempotent_upserts():
    session_factory = RecordingSessionFactory()
    store = SystemStore(DatabaseConfig(enable=True))
    store.session_factory = session_factory

    asyncio.run(
        store.record_recommendation_impression(
            {
                "request_id": "req-1",
                "user_id": "u1",
                "timestamp": "2026-06-11T00:00:00Z",
                "metadata": {
                    "impression_id": "imp-1",
                    "session_id": "session-1",
                    "content_id": "content-1",
                    "model_version": "v1",
                    "ranking_model_version": "rank-v1",
                    "context": {"surface": "recommendations"},
                    "displayed_items": [
                        {
                            "product_id": "p1",
                            "position": 1,
                            "candidate_source": "two_tower",
                            "price": 12.0,
                            "ranking_score": 0.9,
                        }
                    ],
                },
            }
        )
    )

    combined_sql = "\n".join(session_factory.session.statements)
    assert "INSERT INTO recommendation_impressions" in combined_sql
    assert "INSERT INTO recommendation_impression_items" in combined_sql
    assert "ON CONFLICT" in combined_sql
    assert "impression_id, product_id" in combined_sql


def test_build_ltr_training_samples_from_impressions_adds_no_click_negatives():
    created_at = datetime.fromtimestamp(1_000, tz=timezone.utc)
    impression_items = [
        {
            "impression_id": "imp-1",
            "request_id": "req-1",
            "user_id": "u1",
            "session_id": "session-1",
            "content_id": "content-1",
            "product_id": "p-click",
            "position": 1,
            "source": "two_tower",
            "context": {"surface": "recommendations"},
            "feature_snapshot": {"price": 12.0, "category": "Shoes", "brand": "A"},
            "scores": {"ranking_score": 0.9, "combined_score": 0.7},
            "created_at": created_at,
        },
        {
            "impression_id": "imp-1",
            "request_id": "req-1",
            "user_id": "u1",
            "session_id": "session-1",
            "content_id": "content-1",
            "product_id": "p-skip",
            "position": 2,
            "source": "popular",
            "context": {"surface": "recommendations"},
            "feature_snapshot": {"price": 20.0, "category": "Bags", "brand": "B"},
            "scores": {"ranking_score": 0.4, "combined_score": 0.3},
            "created_at": created_at,
        },
    ]
    interactions = [
        {
            "event_id": "evt-click",
            "schema_version": 1,
            "user_id": "u1",
            "product_id": "p-click",
            "action": "click",
            "context": {"impression_id": "imp-1", "recommendation_position": 1},
            "occurred_at": datetime.fromtimestamp(1_010, tz=timezone.utc),
        }
    ]

    samples = build_ltr_training_samples_from_impression_records(
        impression_items,
        interactions,
    )

    assert [sample["product_id"] for sample in samples] == ["p-click", "p-skip"]
    assert [sample["action"] for sample in samples] == ["click", "view"]
    clicked = samples[0]
    skipped = samples[1]
    assert clicked["context"]["impression_id"] == "imp-1"
    assert clicked["context"]["attributed_click"] is True
    assert clicked["context"]["attributed_purchase"] is False
    assert clicked["context"]["recommendation_source"] == "two_tower"
    assert clicked["context"]["candidate_scores"]["ranking_score"] == 0.9
    assert clicked["product_metadata"]["price"] == 12.0
    assert skipped["event_id"] == "imp-1:p-skip:impression"
    assert skipped["context"]["recommendation_position"] == 2
    assert skipped["context"]["attributed_click"] is False
    assert skipped["context"]["attributed_purchase"] is False


def test_build_two_tower_training_negatives_uses_only_weak_no_click_items():
    created_at = datetime.fromtimestamp(1_000, tz=timezone.utc)
    impression_items = [
        {
            "impression_id": "imp-1",
            "user_id": "u1",
            "product_id": "p-click",
            "position": 1,
            "source": "two_tower",
            "feature_snapshot": {"candidate_source": "two_tower"},
            "scores": {"ranking_score": 0.9},
            "created_at": created_at,
        },
        {
            "impression_id": "imp-1",
            "user_id": "u1",
            "product_id": "p-view-only",
            "position": 2,
            "source": "popular",
            "feature_snapshot": {"candidate_source": "popular"},
            "scores": {"ranking_score": 0.4},
            "created_at": created_at,
        },
        {
            "impression_id": "imp-1",
            "user_id": "u1",
            "product_id": "p-skip",
            "position": 3,
            "source": "content",
            "feature_snapshot": {"candidate_source": "content"},
            "scores": {"ranking_score": 0.3},
            "created_at": created_at,
        },
    ]
    interactions = [
        {
            "event_id": "evt-click",
            "user_id": "u1",
            "product_id": "p-click",
            "action": "click",
            "context": {"impression_id": "imp-1"},
            "occurred_at": datetime.fromtimestamp(1_010, tz=timezone.utc),
        },
        {
            "event_id": "evt-view",
            "user_id": "u1",
            "product_id": "p-view-only",
            "action": "view",
            "context": {"impression_id": "imp-1"},
            "occurred_at": datetime.fromtimestamp(1_011, tz=timezone.utc),
        },
    ]

    negatives = build_two_tower_training_negatives_from_impression_records(
        impression_items,
        interactions,
    )

    assert [negative["product_id"] for negative in negatives] == [
        "p-view-only",
        "p-skip",
    ]
    assert {negative["source"] for negative in negatives} == {"impression_no_click"}
    assert all(negative["exposed"] is True for negative in negatives)
    assert all(negative["weight"] == 0.25 for negative in negatives)
    assert negatives[0]["rank_position"] == 2


@pytest.mark.parametrize("positive_action", sorted(TWO_TOWER_POSITIVE_ACTIONS))
def test_build_two_tower_training_negatives_excludes_all_positive_actions(
    positive_action,
):
    created_at = datetime.fromtimestamp(1_000, tz=timezone.utc)
    impression_items = [
        {
            "impression_id": "imp-1",
            "user_id": "u1",
            "product_id": "p-positive",
            "position": 1,
            "source": "two_tower",
            "created_at": created_at,
        },
        {
            "impression_id": "imp-1",
            "user_id": "u1",
            "product_id": "p-skip",
            "position": 2,
            "source": "popular",
            "created_at": created_at,
        },
    ]
    interactions = [
        {
            "event_id": f"evt-{positive_action}",
            "user_id": "u1",
            "product_id": "p-positive",
            "action": positive_action,
            "context": {"impression_id": "imp-1"},
            "occurred_at": datetime.fromtimestamp(1_010, tz=timezone.utc),
        }
    ]

    negatives = build_two_tower_training_negatives_from_impression_records(
        impression_items,
        interactions,
    )

    assert [negative["product_id"] for negative in negatives] == ["p-skip"]


def test_build_two_tower_training_negatives_includes_ranker_rejected_context_items():
    created_at = datetime.fromtimestamp(1_000, tz=timezone.utc)
    impression_items = [
        {
            "impression_id": "imp-1",
            "user_id": "u1",
            "product_id": "p-click",
            "position": 1,
            "source": "two_tower",
            "feature_snapshot": {"candidate_source": "two_tower"},
            "scores": {"ranking_score": 0.9},
            "created_at": created_at,
            "context": {
                "rejected_candidate_items": [
                    {
                        "product_id": "p-rejected",
                        "candidate_source": "two_tower",
                        "position": 12,
                        "scores": {"ranking_score": 0.2},
                    },
                    {
                        "product_id": "p-click",
                        "candidate_source": "two_tower",
                        "position": 13,
                    },
                ]
            },
        }
    ]
    interactions = [
        {
            "event_id": "evt-click",
            "user_id": "u1",
            "product_id": "p-click",
            "action": "purchase",
            "context": {"impression_id": "imp-1"},
            "occurred_at": datetime.fromtimestamp(1_010, tz=timezone.utc),
        }
    ]

    negatives = build_two_tower_training_negatives_from_impression_records(
        impression_items,
        interactions,
    )

    assert [negative["product_id"] for negative in negatives] == ["p-rejected"]
    assert negatives[0]["source"] == "ranker_rejected"
    assert negatives[0]["exposed"] is False
    assert negatives[0]["weight"] == 0.15
    assert negatives[0]["rank_position"] == 12


def test_ltr_impression_samples_mark_purchase_as_implicit_click_with_business_value():
    created_at = datetime.fromtimestamp(1_000, tz=timezone.utc)
    impression_items = [
        {
            "impression_id": "imp-1",
            "request_id": "req-1",
            "user_id": "u1",
            "product_id": "p1",
            "position": 1,
            "source": "two_tower",
            "context": {},
            "feature_snapshot": {"price": 120.0, "category": "Shoes", "brand": "A"},
            "scores": {"ranking_score": 0.9},
            "created_at": created_at,
        }
    ]
    interactions = [
        {
            "event_id": "evt-purchase",
            "schema_version": 1,
            "user_id": "u1",
            "product_id": "p1",
            "action": "purchase",
            "context": {"impression_id": "imp-1", "profit": 42.0, "purchase_value": 120.0},
            "occurred_at": datetime.fromtimestamp(1_020, tz=timezone.utc),
        }
    ]

    samples = build_ltr_training_samples_from_impression_records(
        impression_items,
        interactions,
    )

    assert len(samples) == 1
    assert samples[0]["action"] == "purchase"
    assert samples[0]["context"]["attributed_click"] is True
    assert samples[0]["context"]["attributed_purchase"] is True
    assert samples[0]["business_value"] == 42.0
    assert samples[0]["value"] == 42.0


def test_ltr_impression_samples_use_purchase_value_before_price_fallback():
    created_at = datetime.fromtimestamp(1_000, tz=timezone.utc)
    impression_items = [
        {
            "impression_id": "imp-1",
            "request_id": "req-1",
            "user_id": "u1",
            "product_id": "p1",
            "position": 1,
            "source": "two_tower",
            "context": {},
            "feature_snapshot": {"price": 25.0, "category": "Shoes", "brand": "A"},
            "scores": {"ranking_score": 0.9},
            "created_at": created_at,
        }
    ]
    interactions = [
        {
            "event_id": "evt-purchase",
            "schema_version": 1,
            "user_id": "u1",
            "product_id": "p1",
            "action": "purchase",
            "context": {"impression_id": "imp-1", "purchase_value": 120.0},
            "occurred_at": datetime.fromtimestamp(1_020, tz=timezone.utc),
        }
    ]

    samples = build_ltr_training_samples_from_impression_records(
        impression_items,
        interactions,
    )

    assert len(samples) == 1
    assert samples[0]["business_value"] == 120.0
    assert samples[0]["value"] == 120.0
    assert samples[0]["purchase_value"] == 120.0


def test_ltr_impression_matching_requires_same_user_id():
    created_at = datetime.fromtimestamp(1_000, tz=timezone.utc)
    impression_items = [
        {
            "impression_id": "imp-1",
            "request_id": "req-1",
            "user_id": "u1",
            "product_id": "p1",
            "position": 1,
            "source": "two_tower",
            "context": {},
            "feature_snapshot": {"price": 12.0},
            "scores": {"ranking_score": 0.9},
            "created_at": created_at,
        }
    ]
    interactions = [
        {
            "event_id": "evt-spoof",
            "schema_version": 1,
            "user_id": "attacker",
            "product_id": "p1",
            "action": "purchase",
            "context": {"impression_id": "imp-1"},
            "occurred_at": datetime.fromtimestamp(1_010, tz=timezone.utc),
        }
    ]

    samples = build_ltr_training_samples_from_impression_records(
        impression_items,
        interactions,
    )

    assert len(samples) == 1
    assert samples[0]["action"] == "view"
    assert samples[0]["event_id"] == "imp-1:p1:impression"
