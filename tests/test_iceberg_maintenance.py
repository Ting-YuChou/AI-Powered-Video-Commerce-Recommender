from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

from video_commerce.ml.iceberg_maintenance import (
    ICEBERG_MAINTENANCE_TABLES,
    IcebergFile,
    IcebergMaintenancePolicy,
    IcebergMaintenancePlan,
    IcebergSnapshot,
    IcebergTableState,
    build_maintenance_plan,
    validate_execution_plan,
    collect_snapshot_references,
)


def test_maintenance_plan_preserves_current_recent_and_referenced_snapshots():
    now = datetime(2026, 10, 12, tzinfo=timezone.utc)
    snapshots = tuple(
        IcebergSnapshot(snapshot_id=index, committed_at=now - timedelta(days=index))
        for index in range(1, 36)
    )
    state = IcebergTableState(
        table="ranking_training_pit",
        head_snapshot_id=1,
        snapshots=snapshots,
        orphan_files=(
            IcebergFile(
                "s3://warehouse/table/data/old.parquet", now - timedelta(days=8)
            ),
            IcebergFile(
                "s3://warehouse/table/data/new.parquet", now - timedelta(days=2)
            ),
        ),
        table_prefix="s3://warehouse/table/",
    )

    plan = build_maintenance_plan(
        state,
        protected_snapshot_ids={33},
        policy=IcebergMaintenancePolicy(),
        now=now,
    )

    assert 1 not in plan.expire_snapshot_ids
    assert 33 not in plan.expire_snapshot_ids
    assert set(range(1, 21)).isdisjoint(plan.expire_snapshot_ids)
    assert plan.delete_orphan_files == ("s3://warehouse/table/data/old.parquet",)
    assert len(plan.plan_sha256) == 64


def test_maintenance_execution_rejects_changed_head_or_plan_hash():
    now = datetime.now(timezone.utc)
    state = IcebergTableState(
        table="interaction_history",
        head_snapshot_id=7,
        snapshots=(IcebergSnapshot(7, now),),
        orphan_files=(),
        table_prefix="s3://warehouse/history/",
    )
    plan = build_maintenance_plan(
        state,
        protected_snapshot_ids=set(),
        policy=IcebergMaintenancePolicy(),
        now=now,
    )

    validate_execution_plan(plan, expected_sha256=plan.plan_sha256, current_head=7)
    with pytest.raises(ValueError, match="head snapshot changed"):
        validate_execution_plan(plan, expected_sha256=plan.plan_sha256, current_head=8)
    with pytest.raises(ValueError, match="plan hash"):
        validate_execution_plan(plan, expected_sha256="0" * 64, current_head=7)


def test_maintenance_fails_closed_for_unapproved_table_and_prefix_mismatch():
    now = datetime.now(timezone.utc)
    assert "retrieval_training_pit" in ICEBERG_MAINTENANCE_TABLES
    with pytest.raises(ValueError, match="allowlist"):
        build_maintenance_plan(
            IcebergTableState("unknown", 1, (), (), "s3://warehouse/table/"),
            protected_snapshot_ids=set(),
            policy=IcebergMaintenancePolicy(),
            now=now,
        )
    with pytest.raises(ValueError, match="prefix"):
        build_maintenance_plan(
            IcebergTableState(
                "interaction_history",
                1,
                (IcebergSnapshot(1, now),),
                (IcebergFile("s3://other/path/file", now - timedelta(days=8)),),
                "s3://warehouse/table/",
            ),
            protected_snapshot_ids=set(),
            policy=IcebergMaintenancePolicy(),
            now=now,
        )


def test_plan_round_trip_recomputes_hash_and_rejects_tampering():
    now = datetime.now(timezone.utc)
    state = IcebergTableState(
        table="interaction_history",
        head_snapshot_id=7,
        snapshots=(IcebergSnapshot(7, now),),
        orphan_files=(),
        table_prefix="s3://warehouse/history/",
    )
    plan = build_maintenance_plan(
        state,
        protected_snapshot_ids=set(),
        policy=IcebergMaintenancePolicy(),
        now=now,
    )
    restored = IcebergMaintenancePlan.from_json(plan.to_json())
    assert restored == plan
    payload = json.loads(plan.to_json())
    payload["expire_snapshot_ids"].append(999)
    with pytest.raises(ValueError, match="hash"):
        IcebergMaintenancePlan.from_json(json.dumps(payload))


def test_snapshot_reference_collection_keeps_table_lineage():
    references = collect_snapshot_references(
        {
            "iceberg_table_id": "video_commerce.ranking_training_pit",
            "iceberg_snapshot_id": "42",
            "nested": {
                "table_name": "retrieval_training_pit",
                "snapshot_id": 99,
            },
        }
    )
    assert references == {
        "ranking_training_pit": {42},
        "retrieval_training_pit": {99},
    }
