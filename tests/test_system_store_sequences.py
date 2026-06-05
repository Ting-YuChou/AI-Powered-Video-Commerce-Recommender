import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.dialects import postgresql

from video_commerce.common.config import DatabaseConfig
from video_commerce.data_plane.system_store import SystemStore, build_chronological_user_sequences


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


class RecordingConnection:
    def __init__(self):
        self.statements = []

    async def execute(self, statement):
        self.statements.append(str(statement))


def test_positive_sequence_index_is_ensured_on_startup():
    conn = RecordingConnection()
    store = SystemStore(DatabaseConfig(enable=True))

    asyncio.run(store._ensure_operational_indexes(conn))

    combined_sql = "\n".join(conn.statements)
    assert "ix_interaction_events_positive_user_sequence" in combined_sql
    assert "ON interaction_events (user_id, occurred_at DESC, event_id DESC)" in combined_sql
    assert "WHERE action IN ('view', 'click', 'add_to_cart', 'purchase')" in combined_sql
