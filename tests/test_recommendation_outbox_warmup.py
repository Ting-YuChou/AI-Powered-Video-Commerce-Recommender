"""Startup warmup of the served-impression outbox write path.

The first impression after a deploy used to open its DB connection and
prepare its INSERT inside the 100 ms serving budget, and could come back
impression_tracking "unavailable". The Postgres-backed checks need
``DATABASE_URL`` (as in the Compose backend-tests service) and skip otherwise.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import logging
import os
import textwrap
import time
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from video_commerce.common.config import DatabaseConfig
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.services.recommendation import api as recommendation_api


class _WarmupStore:
    def __init__(self, *, error=None, delay_seconds=0.0):
        self.error = error
        self.delay_seconds = delay_seconds
        self.calls = []
        self.cancelled = False

    async def warm_hot_path_pool(self, connections):
        self.calls.append(connections)
        try:
            await asyncio.sleep(self.delay_seconds)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.error is not None:
            raise self.error
        return connections


def _runtime(pool_size=2):
    return SimpleNamespace(
        config=SimpleNamespace(database_config=SimpleNamespace(pool_size=pool_size))
    )


def test_startup_warms_outbox_pool_right_after_store_initialize():
    tree = ast.parse(textwrap.dedent(inspect.getsource(recommendation_api.startup_event)))
    awaited = [
        ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Await)
    ]
    initialize = awaited.index("system_store.initialize()")
    # Before readiness can pass and before any other startup DB work.
    assert awaited[initialize + 1] == "_warm_impression_outbox_pool(runtime)"


@pytest.mark.asyncio
async def test_warmup_primes_pool_size_connections(monkeypatch, caplog):
    store = _WarmupStore()
    monkeypatch.setattr(recommendation_api, "system_store", store)

    with caplog.at_level(logging.INFO, logger=recommendation_api.logger.name):
        await recommendation_api._warm_impression_outbox_pool(_runtime(pool_size=3))

    assert store.calls == [3]
    assert "recommendation_impression_outbox_warmup_complete" in caplog.text


@pytest.mark.asyncio
async def test_warmup_failure_is_logged_and_does_not_fail_startup(monkeypatch, caplog):
    store = _WarmupStore(error=ConnectionRefusedError("postgres down"))
    monkeypatch.setattr(recommendation_api, "system_store", store)

    with caplog.at_level(logging.WARNING, logger=recommendation_api.logger.name):
        await recommendation_api._warm_impression_outbox_pool(_runtime())

    assert store.calls == [2]
    assert "recommendation_impression_outbox_warmup_failed" in caplog.text


@pytest.mark.asyncio
async def test_slow_warmup_is_bounded_and_cancelled(monkeypatch, caplog):
    store = _WarmupStore(delay_seconds=30.0)
    monkeypatch.setattr(recommendation_api, "system_store", store)
    monkeypatch.setattr(
        recommendation_api, "_IMPRESSION_OUTBOX_WARMUP_TIMEOUT_SECONDS", 0.05
    )

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=recommendation_api.logger.name):
        await recommendation_api._warm_impression_outbox_pool(_runtime())

    assert time.monotonic() - started < 1.0
    assert store.cancelled
    assert "recommendation_impression_outbox_warmup_failed" in caplog.text


@pytest.mark.asyncio
async def test_warmup_is_a_noop_without_a_store(monkeypatch):
    monkeypatch.setattr(recommendation_api, "system_store", None)

    await recommendation_api._warm_impression_outbox_pool(_runtime())


@pytest.mark.asyncio
async def test_disabled_store_warmup_primes_nothing():
    assert await SystemStore(DatabaseConfig(enable=False)).warm_hot_path_pool(2) == 0


# ------------------------------------------------------------------ Postgres


def _database_url() -> str:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is required for Postgres outbox warmup tests")
    return database_url


async def _cold_store(pool_size=2) -> SystemStore:
    """A store whose pool has opened no connections yet."""
    schema = SystemStore(
        DatabaseConfig(
            enable=True,
            url=_database_url(),
            auto_create_schema=True,
            enable_retention_cleanup=False,
        )
    )
    await schema.initialize()
    await schema.close()
    store = SystemStore(
        DatabaseConfig(
            enable=True,
            url=_database_url(),
            auto_create_schema=False,
            enable_retention_cleanup=False,
            pool_size=pool_size,
            max_overflow=1,
        )
    )
    await store.initialize()
    assert store.engine.pool.checkedin() == 0
    return store


def _served_event(user_id):
    impression_id = uuid.uuid4().hex
    return {
        "event_id": uuid.uuid4().hex,
        "user_id": user_id,
        "recommendations": ["p1", "p2", "p3"],
        "metadata": {"impression_id": impression_id},
    }


@pytest.mark.asyncio
async def test_warmup_opens_distinct_connections_and_commits_nothing():
    store = await _cold_store(pool_size=2)
    try:
        assert await store.warm_hot_path_pool(2) == 2
        assert store.engine.pool.checkedin() == 2
        async with store.session_factory() as session:
            leaked = await session.scalar(
                text(
                    "SELECT count(*) FROM recommendation_event_outbox "
                    "WHERE event_id LIKE 'pool-warmup-%'"
                )
            )
        assert leaked == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_first_outbox_writes_after_warmup_fit_the_serving_budget():
    budget_seconds = 0.1  # RECOMMENDATION_IMPRESSION_OUTBOX_TIMEOUT_MS default
    user_id = f"warmup-budget-{uuid.uuid4().hex[:8]}"
    store = await _cold_store(pool_size=2)
    try:
        await store.warm_hot_path_pool(2)

        async def timed_write():
            started = time.monotonic()
            outcome = await store.enqueue_recommendation_event(
                _served_event(user_id),
                commit_deadline=started + budget_seconds,
            )
            return outcome, time.monotonic() - started

        # Concurrent, so both warmed connections take a first write.
        results = await asyncio.gather(timed_write(), timed_write())
        assert store.engine.pool.checkedin() == 2
        for outcome, elapsed in results:
            assert outcome == "inserted"
            assert elapsed < budget_seconds, results
    finally:
        async with store.session_factory.begin() as session:
            await session.execute(
                text(
                    "DELETE FROM recommendation_event_outbox "
                    "WHERE event_payload->>'user_id' = :user_id"
                ),
                {"user_id": user_id},
            )
        await store.close()
