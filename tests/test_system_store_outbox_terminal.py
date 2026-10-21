"""Postgres-backed checks for WS-2: DDL lock, outbox terminal state, timeouts.

These need a real Postgres (``DATABASE_URL``, as in the Compose backend-tests
service) and skip otherwise.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from video_commerce.common.config import DatabaseConfig
from video_commerce.data_plane.system_store import (
    SystemStore,
    _engine_options,
    outbox_retry_delay_seconds,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _database_url() -> str:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is required for Postgres outbox integration tests")
    return database_url


def _store(url: str, **overrides) -> SystemStore:
    return SystemStore(
        DatabaseConfig(
            enable=True,
            url=url,
            auto_create_schema=True,
            enable_retention_cleanup=False,
            **overrides,
        )
    )


def test_migration_012_adds_terminal_columns_to_both_outboxes():
    migration = (
        REPO_ROOT / "migrations/postgres/012_recommendation_outbox_terminal.sql"
    ).read_text(encoding="utf-8")

    assert "ALTER TABLE recommendation_event_outbox" in migration
    assert "ALTER TABLE catalog_feature_outbox" in migration
    assert migration.count("ADD COLUMN IF NOT EXISTS terminal_at") == 2
    assert migration.count("ADD COLUMN IF NOT EXISTS terminal_reason") == 2
    assert "ADD COLUMN IF NOT EXISTS next_attempt_at" in migration


def test_engine_options_bound_pool_and_connection_without_global_statement_timeout():
    options = _engine_options(
        DatabaseConfig(
            pool_timeout_seconds=2,
            idle_in_transaction_session_timeout_ms=60000,
            statement_cache_size=0,
        )
    )

    assert options["pool_timeout"] == 2.0
    assert options["pool_recycle"] == 1800
    assert options["connect_args"]["timeout"] == 5.0
    assert options["connect_args"]["statement_cache_size"] == 0
    assert options["connect_args"]["server_settings"] == {
        "idle_in_transaction_session_timeout": "60000"
    }
    assert "statement_timeout" not in options["connect_args"]["server_settings"]
    assert "statement_cache_size" not in _engine_options(DatabaseConfig())[
        "connect_args"
    ]


def test_outbox_backoff_is_exponential_and_capped():
    # Same curve as before terminal state existed: doubling, capped at 2**8.
    assert [outbox_retry_delay_seconds(n) for n in (1, 2, 3, 8, 20)] == [
        2,
        4,
        8,
        256,
        256,
    ]


@pytest.mark.asyncio
async def test_concurrent_initialize_on_fresh_database_is_serialized():
    base_url = make_url(_database_url())
    database_name = f"vc_init_race_{uuid.uuid4().hex[:12]}"
    admin = create_async_engine(base_url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f'CREATE DATABASE "{database_name}"'))
    fresh_url = base_url.set(database=database_name).render_as_string(
        hide_password=False
    )
    stores = [_store(fresh_url) for _ in range(4)]
    try:
        # Without the advisory lock concurrent create_all races on pg_type /
        # pg_class and some replicas crash with duplicate-key errors.
        await asyncio.gather(*(store.initialize() for store in stores))
        async with stores[0].session_factory() as session:
            columns = {
                row[0]
                for row in await session.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'recommendation_event_outbox'"
                    )
                )
            }
        assert {"terminal_at", "terminal_reason"} <= columns
        # A restart against an up-to-date schema stays a no-op.
        restart = _store(fresh_url)
        await restart.initialize()
        await restart.close()
    finally:
        for store in stores:
            await store.close()
        async with admin.connect() as conn:
            await conn.execute(
                text(f'DROP DATABASE IF EXISTS "{database_name}" WITH (FORCE)')
            )
        await admin.dispose()


@pytest.mark.asyncio
async def test_hot_path_session_sets_local_statement_timeout():
    store = _store(_database_url(), hot_path_statement_timeout_ms=750)
    await store.initialize()
    try:
        async with store._hot_path_session() as session:
            inside = (await session.execute(text("SHOW statement_timeout"))).scalar()
        async with store.session_factory() as session:
            outside = (await session.execute(text("SHOW statement_timeout"))).scalar()
        assert inside == "750ms"
        assert outside == "0"
        with pytest.raises(Exception, match="statement timeout"):
            async with store._hot_path_session() as session:
                await session.execute(text("SELECT pg_sleep(2)"))
    finally:
        await store.close()


def _event(impression_id: str) -> dict:
    return {
        "event_id": uuid.uuid5(uuid.NAMESPACE_URL, impression_id).hex,
        "user_id": "u-terminal",
        "metadata": {"impression_id": impression_id},
    }


async def _force_due(store: SystemStore, event_id: str) -> None:
    async with store.session_factory.begin() as session:
        await session.execute(
            text(
                "UPDATE recommendation_event_outbox SET next_attempt_at = now() "
                "WHERE event_id = :event_id"
            ),
            {"event_id": event_id},
        )


async def _claim_ids(store: SystemStore, worker_id: str) -> set:
    events = await store.claim_recommendation_outbox(
        worker_id=worker_id, batch_size=1000, lease_seconds=60
    )
    return {event["event_id"] for event in events}


@pytest.mark.asyncio
async def test_recommendation_outbox_terminal_lifecycle():
    store = _store(_database_url())
    await store.initialize()
    prefix = f"terminal-{uuid.uuid4().hex[:8]}"
    oversized = _event(f"{prefix}-oversized")
    flaky = _event(f"{prefix}-flaky")
    late = _event(f"{prefix}-late")
    try:
        for event in (oversized, flaky, late):
            assert await store.enqueue_recommendation_event(event) == "inserted"

        # Insert committed after the serving budget: never published, never a slate.
        assert await store.mark_recommendation_outbox_not_exposed(late["event_id"])
        assert not await store.is_known_recommendation_impression(
            late["metadata"]["impression_id"]
        )
        assert (
            await store.get_recommendation_impression_slate(
                late["metadata"]["impression_id"]
            )
            is None
        )

        claimed = await _claim_ids(store, "worker-1")
        assert oversized["event_id"] in claimed and flaky["event_id"] in claimed
        assert late["event_id"] not in claimed

        terminal = await store.mark_recommendation_outbox_failed(
            oversized["event_id"],
            "record too large",
            worker_id="worker-1",
            retryable=False,
            terminal_reason="message_too_large",
            max_attempts=3,
        )
        assert terminal == "message_too_large"

        reasons = []
        for attempt in range(3):
            if attempt:
                await _force_due(store, flaky["event_id"])
                assert flaky["event_id"] in await _claim_ids(store, "worker-1")
            reasons.append(
                await store.mark_recommendation_outbox_failed(
                    flaky["event_id"],
                    "broker down",
                    worker_id="worker-1",
                    max_attempts=3,
                )
            )
        assert reasons == [None, None, "max_attempts"]

        await _force_due(store, flaky["event_id"])
        assert not (await _claim_ids(store, "worker-2")) & {
            oversized["event_id"],
            flaky["event_id"],
            late["event_id"],
        }
        async with store.session_factory() as session:
            rows = dict(
                (
                    await session.execute(
                        text(
                            "SELECT event_id, terminal_reason "
                            "FROM recommendation_event_outbox "
                            "WHERE event_id IN (:a, :b, :c)"
                        ),
                        {
                            "a": oversized["event_id"],
                            "b": flaky["event_id"],
                            "c": late["event_id"],
                        },
                    )
                ).all()
            )
            pending_terminal = (
                await session.execute(
                    text(
                        "SELECT count(*) FROM recommendation_event_outbox "
                        "WHERE published_at IS NULL AND terminal_at IS NOT NULL "
                        "AND event_id IN (:a, :b, :c)"
                    ),
                    {
                        "a": oversized["event_id"],
                        "b": flaky["event_id"],
                        "c": late["event_id"],
                    },
                )
            ).scalar_one()
        assert rows == {
            oversized["event_id"]: "message_too_large",
            flaky["event_id"]: "max_attempts",
            late["event_id"]: "not_exposed",
        }
        assert pending_terminal == 3
        stats_before = await store.get_recommendation_outbox_stats()
        extra = _event(f"{prefix}-pending")
        await store.enqueue_recommendation_event(extra)
        stats_after = await store.get_recommendation_outbox_stats()
        # Terminal rows are excluded; only the new pending row is counted.
        assert stats_after["pending"] == stats_before["pending"] + 1
        await store.mark_recommendation_outbox_not_exposed(extra["event_id"])
    finally:
        async with store.session_factory.begin() as session:
            await session.execute(
                text(
                    "DELETE FROM recommendation_event_outbox "
                    "WHERE impression_id LIKE :prefix"
                ),
                {"prefix": f"{prefix}%"},
            )
        await store.close()
