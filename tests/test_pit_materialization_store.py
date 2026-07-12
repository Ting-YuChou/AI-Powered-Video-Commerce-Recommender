from pathlib import Path
import os
import uuid

import pytest
from sqlalchemy import text

from video_commerce.common.config import DatabaseConfig
from video_commerce.data_plane.system_store import SystemStore


def test_pit_materialization_migration_declares_lease_and_lineage_columns():
    migration = (
        Path(__file__).resolve().parents[1]
        / "migrations/postgres/006_pit_materialization_runs.sql"
    ).read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS pit_materialization_runs" in migration
    assert "lease_expires_at" in migration
    assert "snapshot_id" in migration
    assert "manifest_uri" in migration
    assert "quarantine_count" in migration
    assert "export_attempt" in migration


@pytest.mark.asyncio
async def test_pit_materialization_run_lease_takeover_and_completion():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is required for PIT run-store integration test")
    run_id = f"pit-test-{uuid.uuid4().hex[:16]}"
    store = SystemStore(
        DatabaseConfig(
            enable=True,
            url=database_url,
            auto_create_schema=True,
            enable_retention_cleanup=False,
        )
    )
    await store.initialize()
    try:
        first = await store.claim_pit_materialization_run(
            run_id=run_id,
            cutoff_ts=1_700_000_000.0,
            worker_id="worker-1",
            lease_seconds=60,
        )
        assert first["claimed"] is True
        assert first["attempts"] == 1
        assert first["phase"] == "export"

        blocked = await store.claim_pit_materialization_run(
            run_id=run_id,
            cutoff_ts=1_700_000_000.0,
            worker_id="worker-2",
            lease_seconds=60,
        )
        assert blocked["claimed"] is False

        async with store.session_factory.begin() as session:
            await session.execute(
                text(
                    "UPDATE pit_materialization_runs "
                    "SET lease_expires_at = now() - interval '1 second' "
                    "WHERE run_id = :run_id"
                ),
                {"run_id": run_id},
            )

        with pytest.raises(RuntimeError, match="lost its lease"):
            await store.mark_pit_materialization_phase(
                run_id, phase="manifest", worker_id="worker-1"
            )

        takeover = await store.claim_pit_materialization_run(
            run_id=run_id,
            cutoff_ts=1_700_000_000.0,
            worker_id="worker-2",
            lease_seconds=60,
        )
        assert takeover["claimed"] is True
        assert takeover["attempts"] == 2

        await store.mark_pit_materialization_phase(
            run_id, phase="manifest", worker_id="worker-2", export_attempt=2
        )
        await store.complete_pit_materialization_run(
            run_id,
            worker_id="worker-2",
            snapshot_id="42",
            manifest_uri="s3://features/runs/pit-test/manifest.json",
            row_count=10,
            quarantine_count=0,
        )
        completed = await store.claim_pit_materialization_run(
            run_id=run_id,
            cutoff_ts=1_700_000_000.0,
            worker_id="worker-3",
            lease_seconds=60,
        )
        assert completed["claimed"] is False
        assert completed["status"] == "completed"

        training = await store.claim_pit_training_run(
            run_id=run_id, worker_id="trainer-1", lease_seconds=300
        )
        assert training["claimed"] is True
        await store.renew_pit_training_lease(
            run_id=run_id,
            worker_id="trainer-1",
            lease_seconds=300,
        )
        async with store.session_factory.begin() as session:
            await session.execute(
                text(
                    "UPDATE pit_materialization_runs "
                    "SET training_lease_expires_at = now() - interval '1 second' "
                    "WHERE run_id = :run_id"
                ),
                {"run_id": run_id},
            )
        with pytest.raises(RuntimeError, match="lost ownership"):
            await store.renew_pit_training_lease(
                run_id=run_id,
                worker_id="trainer-1",
                lease_seconds=300,
            )
    finally:
        async with store.session_factory.begin() as session:
            await session.execute(
                text("DELETE FROM pit_materialization_runs WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
        await store.close()


@pytest.mark.asyncio
async def test_waiting_pit_run_remains_terminal_for_its_daily_cutoff():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is required for PIT run-store integration test")
    run_id = f"pit-waiting-{uuid.uuid4().hex[:16]}"
    store = SystemStore(
        DatabaseConfig(
            enable=True,
            url=database_url,
            auto_create_schema=True,
            enable_retention_cleanup=False,
        )
    )
    await store.initialize()
    try:
        first = await store.claim_pit_materialization_run(
            run_id=run_id,
            cutoff_ts=1_700_000_000.0,
            worker_id="worker-1",
            lease_seconds=60,
        )
        assert first["claimed"] is True
        await store.mark_pit_materialization_phase(
            run_id, phase="manifest", worker_id="worker-1"
        )
        await store.mark_pit_materialization_waiting(run_id, worker_id="worker-1")

        retried = await store.claim_pit_materialization_run(
            run_id=run_id,
            cutoff_ts=1_700_000_000.0,
            worker_id="worker-2",
            lease_seconds=60,
        )

        assert retried["claimed"] is False
        assert retried["status"] == "waiting_for_eligible_rows"
        assert retried["phase"] == "export"
        assert retried["attempts"] == 1
    finally:
        async with store.session_factory.begin() as session:
            await session.execute(
                text("DELETE FROM pit_materialization_runs WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
        await store.close()


@pytest.mark.asyncio
async def test_next_daily_run_takes_over_expired_prior_run_before_starting_new_one():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is required for PIT run-store integration test")
    suffix = uuid.uuid4().hex[:16]
    old_run_id = f"pit-expired-{suffix}"
    new_run_id = f"pit-next-{suffix}"
    store = SystemStore(
        DatabaseConfig(
            enable=True,
            url=database_url,
            auto_create_schema=True,
            enable_retention_cleanup=False,
        )
    )
    await store.initialize()
    try:
        await store.claim_pit_materialization_run(
            run_id=old_run_id,
            cutoff_ts=1_700_000_000.0,
            worker_id="worker-old",
            lease_seconds=60,
        )
        async with store.session_factory.begin() as session:
            await session.execute(
                text(
                    "UPDATE pit_materialization_runs "
                    "SET lease_expires_at = now() - interval '1 second' "
                    "WHERE run_id = :run_id"
                ),
                {"run_id": old_run_id},
            )

        takeover = await store.claim_pit_materialization_run(
            run_id=new_run_id,
            cutoff_ts=1_700_086_400.0,
            worker_id="worker-new",
            lease_seconds=60,
        )

        assert takeover["claimed"] is True
        assert takeover["run_id"] == old_run_id
        assert takeover["cutoff_ts"] == 1_700_000_000.0
        assert takeover["worker_id"] == "worker-new"
        async with store.session_factory() as session:
            count = await session.scalar(
                text(
                    "SELECT count(*) FROM pit_materialization_runs "
                    "WHERE run_id = :run_id"
                ),
                {"run_id": new_run_id},
            )
        assert count == 0
    finally:
        async with store.session_factory.begin() as session:
            await session.execute(
                text(
                    "DELETE FROM pit_materialization_runs "
                    "WHERE run_id IN (:old_run_id, :new_run_id)"
                ),
                {"old_run_id": old_run_id, "new_run_id": new_run_id},
            )
        await store.close()
