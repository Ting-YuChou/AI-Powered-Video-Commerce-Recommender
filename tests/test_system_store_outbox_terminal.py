"""Postgres-backed checks for WS-2: DDL lock, outbox terminal state, timeouts.

These need a real Postgres (``DATABASE_URL``, as in the Compose backend-tests
service) and skip otherwise.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from video_commerce.common.config import DatabaseConfig
from video_commerce.data_plane.system_store import (
    OutboxCommitDeadlineExceeded,
    OutboxWriteRolledBack,
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
        assert (
            await store.mark_recommendation_outbox_not_exposed(late["event_id"])
            == "retired"
        )
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

        # A broker outage backs off but never uses up the cap.
        for _ in range(4):
            await _force_due(store, flaky["event_id"])
            await _claim_ids(store, "worker-1")
            assert (
                await store.mark_recommendation_outbox_failed(
                    flaky["event_id"],
                    "broker unreachable",
                    worker_id="worker-1",
                    max_attempts=3,
                    counts_toward_max_attempts=False,
                )
                is None
            )
        await _force_due(store, flaky["event_id"])
        assert flaky["event_id"] in await _claim_ids(store, "worker-1")

        # attempts is already past the cap; the first counted failure stops it.
        assert (
            await store.mark_recommendation_outbox_failed(
                flaky["event_id"],
                "broker rejected record",
                worker_id="worker-1",
                max_attempts=3,
            )
            == "max_attempts"
        )

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
        expired = _event(f"{prefix}-expired")
        with pytest.raises(OutboxCommitDeadlineExceeded):
            await store.enqueue_recommendation_event(expired, commit_deadline=0.0)
        assert not await store.is_known_recommendation_impression(
            expired["metadata"]["impression_id"]
        )
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


@pytest.mark.asyncio
async def test_late_commit_hold_lets_retire_win_and_lost_race_is_reported():
    store = _store(_database_url())
    await store.initialize()
    prefix = f"hold-{uuid.uuid4().hex[:8]}"
    held = _event(f"{prefix}-held")
    raced = _event(f"{prefix}-raced")
    try:
        async with store.session_factory() as session:
            assert await session.scalar(
                text("SELECT to_regclass('ix_recommendation_event_outbox_dispatchable')")
            )
        # Staged with a hold-off: no dispatcher can claim it before the retire.
        await store.enqueue_recommendation_event(held, publish_hold_seconds=30)
        assert held["event_id"] not in await _claim_ids(store, "worker-1")
        assert (
            await store.mark_recommendation_outbox_not_exposed(held["event_id"])
            == "retired"
        )

        # Without the hold a dispatcher can claim between COMMIT and retire;
        # the retire must not "succeed" on a live claim.
        await store.enqueue_recommendation_event(raced)
        assert raced["event_id"] in await _claim_ids(store, "worker-1")
        assert (
            await store.mark_recommendation_outbox_not_exposed(raced["event_id"])
            == "already_dispatched"
        )
        async with store.session_factory() as session:
            claimed_by = await session.scalar(
                text(
                    "SELECT claimed_by FROM recommendation_event_outbox "
                    "WHERE event_id = :event_id"
                ),
                {"event_id": raced["event_id"]},
            )
        assert claimed_by == "worker-1"
        assert (
            await store.mark_recommendation_outbox_not_exposed(uuid.uuid4().hex)
            == "missing"
        )
        assert (
            await store.mark_recommendation_outbox_not_exposed(held["event_id"])
            == "already_terminal"
        )

        # Terminal rows are pruned after retention; live rows are kept.
        async with store.session_factory.begin() as session:
            await session.execute(
                text(
                    "UPDATE recommendation_event_outbox "
                    "SET terminal_at = now() - interval '30 days' "
                    "WHERE event_id = :event_id"
                ),
                {"event_id": held["event_id"]},
            )
        assert await store.prune_terminal_recommendation_outbox(retention_days=7) >= 1
        async with store.session_factory() as session:
            remaining = set(
                (
                    await session.execute(
                        text(
                            "SELECT event_id FROM recommendation_event_outbox "
                            "WHERE event_id IN (:held, :raced)"
                        ),
                        {"held": held["event_id"], "raced": raced["event_id"]},
                    )
                ).scalars()
            )
        assert remaining == {raced["event_id"]}
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


async def _outbox_rows(store: SystemStore, prefix: str) -> set:
    async with store.session_factory() as session:
        return set(
            (
                await session.execute(
                    text(
                        "SELECT impression_id FROM recommendation_event_outbox "
                        "WHERE impression_id LIKE :prefix"
                    ),
                    {"prefix": f"{prefix}%"},
                )
            ).scalars()
        )


async def _delete_outbox_rows(store: SystemStore, prefix: str) -> None:
    async with store.session_factory.begin() as session:
        await session.execute(
            text(
                "DELETE FROM recommendation_event_outbox "
                "WHERE impression_id LIKE :prefix"
            ),
            {"prefix": f"{prefix}%"},
        )


async def _enqueue_while_key_is_held(
    store: SystemStore, held: dict, entries: list, hold_seconds: float = 0.6
) -> list:
    """Run a batch while another transaction holds ``held``'s key uncommitted."""
    blocker = create_async_engine(store.config.url)
    try:
        async with blocker.connect() as connection:
            transaction = await connection.begin()
            await connection.execute(
                text(
                    "INSERT INTO recommendation_event_outbox "
                    "(event_id, impression_id, payload_hash, event_payload, attempts) "
                    "VALUES (:event_id, :impression_id, 'blocker', '{}', 0)"
                ),
                {
                    "event_id": held["event_id"],
                    "impression_id": held["metadata"]["impression_id"],
                },
            )
            write = asyncio.create_task(store.enqueue_recommendation_events(entries))
            await asyncio.sleep(hold_seconds)
            assert not write.done()
            await transaction.rollback()
            return await write
    finally:
        await blocker.dispose()


@pytest.mark.asyncio
async def test_batched_enqueue_resolves_each_row_and_drops_late_rows_before_commit():
    store = _store(_database_url())
    await store.initialize()
    prefix = f"batch-{uuid.uuid4().hex[:8]}"
    existing = _event(f"{prefix}-existing")
    collided = _event(f"{prefix}-collided")
    try:
        await store.enqueue_recommendation_event(existing)
        await store.enqueue_recommendation_event(collided)
        fresh = [_event(f"{prefix}-fresh-{index}") for index in range(5)]
        changed = {**collided, "user_id": "someone-else"}
        far = time.monotonic() + 30
        outcomes = await store.enqueue_recommendation_events(
            [(event, far) for event in fresh]
            + [
                (existing, far),
                (changed, far),
                ({"event_id": "", "metadata": {}}, far),
                (_event(f"{prefix}-expired"), 0.0),
            ],
            publish_hold_seconds=30,
        )

        assert outcomes[:6] == ["inserted"] * 5 + ["duplicate"]
        async with store.session_factory() as session:
            stored = await session.scalar(
                text(
                    "SELECT event_payload FROM recommendation_event_outbox "
                    "WHERE event_id = :event_id"
                ),
                {"event_id": fresh[0]["event_id"]},
            )
        # Pre-serialized once and cast to JSON in SQL: the row reads back as
        # the original event.
        assert stored == fresh[0]
        assert isinstance(outcomes[6], ValueError)
        assert isinstance(outcomes[7], ValueError)
        assert isinstance(outcomes[8], OutboxCommitDeadlineExceeded)
        assert await _outbox_rows(store, prefix) == {
            f"{prefix}-existing",
            f"{prefix}-collided",
            *(f"{prefix}-fresh-{index}" for index in range(5)),
        }
        # Held for the publish hold like single-row inserts.
        assert not await _claim_ids(store, "worker-batch") & {
            event["event_id"] for event in fresh
        }

        # A deadline that passes during the INSERT drops only that row; the
        # transaction still commits the rows that are on time. The INSERT is
        # held up by another transaction's uncommitted row with the same key.
        on_time = _event(f"{prefix}-on-time")
        missed = _event(f"{prefix}-missed")
        contended = _event(f"{prefix}-contended")
        outcomes = await _enqueue_while_key_is_held(
            store,
            contended,
            [
                (on_time, time.monotonic() + 30),
                (missed, time.monotonic() + 0.2),
                (contended, time.monotonic() + 30),
            ],
        )
        assert outcomes[0] == "inserted"
        assert isinstance(outcomes[1], OutboxCommitDeadlineExceeded)
        assert outcomes[2] == "inserted"
        rows = await _outbox_rows(store, prefix)
        assert {f"{prefix}-on-time", f"{prefix}-contended"} <= rows
        assert f"{prefix}-missed" not in rows

        # Every inserted row late: the whole transaction rolls back.
        all_late = _event(f"{prefix}-all-late")
        outcomes = await _enqueue_while_key_is_held(
            store, all_late, [(all_late, time.monotonic() + 0.2)]
        )
        assert isinstance(outcomes[0], OutboxCommitDeadlineExceeded)
        assert f"{prefix}-all-late" not in await _outbox_rows(store, prefix)
    finally:
        await _delete_outbox_rows(store, prefix)
        await store.close()


@pytest.mark.asyncio
async def test_outbox_write_blocked_by_table_lock_rolls_back_without_commit():
    """The lock drill: the insert hits the hot-path statement timeout.

    It fails before COMMIT, so the caller must not schedule a retire (the
    retire would block on the same lock and end as retire_failed).
    """
    database_url = _database_url()
    store = _store(database_url, hot_path_statement_timeout_ms=200)
    await store.initialize()
    prefix = f"locked-{uuid.uuid4().hex[:8]}"
    locker = create_async_engine(database_url)
    try:
        async with locker.connect() as connection:
            transaction = await connection.begin()
            await connection.execute(
                text("LOCK TABLE recommendation_event_outbox IN ACCESS EXCLUSIVE MODE")
            )
            with pytest.raises(OutboxWriteRolledBack) as raised:
                await store.enqueue_recommendation_events(
                    [(_event(f"{prefix}-blocked"), time.monotonic() + 30)]
                )
            await transaction.rollback()
        assert raised.value.reason == "db_error"
        assert "statement timeout" in str(raised.value)
        assert await _outbox_rows(store, prefix) == set()
    finally:
        await locker.dispose()
        await _delete_outbox_rows(store, prefix)
        await store.close()


@pytest.mark.asyncio
async def test_outbox_write_without_a_free_connection_reports_pool_timeout():
    store = _store(
        _database_url(), pool_size=1, max_overflow=0, pool_timeout_seconds=0.2
    )
    await store.initialize()
    prefix = f"pool-{uuid.uuid4().hex[:8]}"
    try:
        async with store.engine.connect():
            with pytest.raises(OutboxWriteRolledBack) as raised:
                await store.enqueue_recommendation_events(
                    [(_event(f"{prefix}-starved"), time.monotonic() + 30)]
                )
        assert raised.value.reason == "pool_timeout"
        assert await _outbox_rows(store, prefix) == set()
    finally:
        await _delete_outbox_rows(store, prefix)
        await store.close()


@pytest.mark.asyncio
async def test_batched_retire_reports_each_row_with_one_update():
    store = _store(_database_url())
    await store.initialize()
    prefix = f"retire-many-{uuid.uuid4().hex[:8]}"
    late = [_event(f"{prefix}-late-{index}") for index in range(3)]
    claimed = _event(f"{prefix}-claimed")
    try:
        for event in late:
            await store.enqueue_recommendation_event(event, publish_hold_seconds=30)
        await store.enqueue_recommendation_event(claimed)
        assert claimed["event_id"] in await _claim_ids(store, "worker-retire")
        await store.mark_recommendation_outbox_not_exposed(late[0]["event_id"])
        missing = uuid.uuid4().hex

        statuses = await store.mark_recommendation_outbox_not_exposed_many(
            [event["event_id"] for event in (*late, claimed)] + [missing]
        )

        assert statuses == {
            late[0]["event_id"]: "already_terminal",
            late[1]["event_id"]: "retired",
            late[2]["event_id"]: "retired",
            claimed["event_id"]: "already_dispatched",
            missing: "missing",
        }
        assert late[1]["event_id"] not in await _claim_ids(store, "worker-other")
    finally:
        await _delete_outbox_rows(store, prefix)
        await store.close()
