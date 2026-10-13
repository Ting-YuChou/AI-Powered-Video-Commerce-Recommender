"""Fail-closed Iceberg maintenance planning and execution fencing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any, Iterable


ICEBERG_MAINTENANCE_TABLES = frozenset(
    {
        "interaction_history",
        "recommendation_view_history",
        "ranking_observations",
        "user_feature_history",
        "item_feature_history",
        "window_feature_history",
        "ranking_training_pit",
        "ranking_training_pit_quarantine",
        "retrieval_training_pit",
    }
)


def collect_snapshot_references(payload: Any) -> dict[str, set[int]]:
    """Collect table-scoped Iceberg snapshot IDs from nested lineage JSON."""
    collected: dict[str, set[int]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            table_value = value.get("iceberg_table_id") or value.get("table_name")
            snapshot_value = value.get("iceberg_snapshot_id") or value.get(
                "snapshot_id"
            )
            if table_value is not None and snapshot_value is not None:
                table = str(table_value).rsplit(".", 1)[-1]
                if table in ICEBERG_MAINTENANCE_TABLES:
                    try:
                        collected.setdefault(table, set()).add(int(snapshot_value))
                    except (TypeError, ValueError):
                        raise ValueError(
                            "Iceberg lineage contains an invalid snapshot ID"
                        )
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(payload)
    return collected


@dataclass(frozen=True)
class IcebergMaintenancePolicy:
    snapshot_retention_days: int = 30
    retain_last_snapshots: int = 20
    orphan_min_age_days: int = 7
    delete_batch_size: int = 1000


@dataclass(frozen=True)
class IcebergSnapshot:
    snapshot_id: int
    committed_at: datetime


@dataclass(frozen=True)
class IcebergFile:
    uri: str
    modified_at: datetime


@dataclass(frozen=True)
class IcebergTableState:
    table: str
    head_snapshot_id: int
    snapshots: tuple[IcebergSnapshot, ...]
    orphan_files: tuple[IcebergFile, ...]
    table_prefix: str


@dataclass(frozen=True)
class IcebergMaintenancePlan:
    table: str
    head_snapshot_id: int
    protected_snapshot_ids: tuple[int, ...]
    expire_snapshot_ids: tuple[int, ...]
    delete_orphan_files: tuple[str, ...]
    policy: IcebergMaintenancePolicy
    planned_at: datetime
    plan_sha256: str

    def to_json(self) -> str:
        payload = _plan_payload(self, include_hash=True)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "IcebergMaintenancePlan":
        payload = json.loads(value)
        supplied_hash = str(payload.pop("plan_sha256", ""))
        computed_hash = _payload_hash(payload)
        if supplied_hash != computed_hash:
            raise ValueError("Iceberg maintenance plan hash does not match its payload")
        policy = IcebergMaintenancePolicy(**payload["policy"])
        return cls(
            table=str(payload["table"]),
            head_snapshot_id=int(payload["head_snapshot_id"]),
            protected_snapshot_ids=tuple(
                int(item) for item in payload["protected_snapshot_ids"]
            ),
            expire_snapshot_ids=tuple(
                int(item) for item in payload["expire_snapshot_ids"]
            ),
            delete_orphan_files=tuple(
                str(item) for item in payload["delete_orphan_files"]
            ),
            policy=policy,
            planned_at=datetime.fromisoformat(str(payload["planned_at"])),
            plan_sha256=supplied_hash,
        )


def _payload_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _plan_payload(
    plan: IcebergMaintenancePlan, *, include_hash: bool
) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract_version": "iceberg_maintenance_plan_v1",
        "table": plan.table,
        "head_snapshot_id": plan.head_snapshot_id,
        "protected_snapshot_ids": list(plan.protected_snapshot_ids),
        "expire_snapshot_ids": list(plan.expire_snapshot_ids),
        "delete_orphan_files": list(plan.delete_orphan_files),
        "policy": asdict(plan.policy),
        "planned_at": plan.planned_at.isoformat(),
    }
    if include_hash:
        payload["plan_sha256"] = plan.plan_sha256
    return payload


def build_maintenance_plan(
    state: IcebergTableState,
    *,
    protected_snapshot_ids: Iterable[int],
    policy: IcebergMaintenancePolicy,
    now: datetime | None = None,
) -> IcebergMaintenancePlan:
    if state.table not in ICEBERG_MAINTENANCE_TABLES:
        raise ValueError(
            f"Iceberg table {state.table!r} is not in the maintenance allowlist"
        )
    if policy.snapshot_retention_days < 1 or policy.retain_last_snapshots < 1:
        raise ValueError("snapshot retention policy must be positive")
    if policy.orphan_min_age_days < 1 or policy.delete_batch_size < 1:
        raise ValueError("orphan cleanup policy must be positive")
    planned_at = now or datetime.now(timezone.utc)
    protected = {int(value) for value in protected_snapshot_ids}
    protected.add(int(state.head_snapshot_id))
    newest = sorted(
        state.snapshots,
        key=lambda snapshot: (snapshot.committed_at, snapshot.snapshot_id),
        reverse=True,
    )
    protected.update(
        snapshot.snapshot_id for snapshot in newest[: policy.retain_last_snapshots]
    )
    cutoff = planned_at - timedelta(days=policy.snapshot_retention_days)
    expire = tuple(
        sorted(
            snapshot.snapshot_id
            for snapshot in state.snapshots
            if snapshot.committed_at < cutoff and snapshot.snapshot_id not in protected
        )
    )
    orphan_cutoff = planned_at - timedelta(days=policy.orphan_min_age_days)
    orphan_candidates: list[str] = []
    for artifact in sorted(state.orphan_files, key=lambda item: item.uri):
        if not artifact.uri.startswith(state.table_prefix):
            raise ValueError(
                "orphan file prefix does not match the Iceberg table prefix"
            )
        if artifact.modified_at < orphan_cutoff:
            orphan_candidates.append(artifact.uri)
    orphan_candidates = orphan_candidates[: policy.delete_batch_size]

    payload = {
        "contract_version": "iceberg_maintenance_plan_v1",
        "table": state.table,
        "head_snapshot_id": int(state.head_snapshot_id),
        "protected_snapshot_ids": sorted(protected),
        "expire_snapshot_ids": list(expire),
        "delete_orphan_files": orphan_candidates,
        "policy": asdict(policy),
        "planned_at": planned_at.isoformat(),
    }
    plan_hash = _payload_hash(payload)
    return IcebergMaintenancePlan(
        table=state.table,
        head_snapshot_id=int(state.head_snapshot_id),
        protected_snapshot_ids=tuple(payload["protected_snapshot_ids"]),
        expire_snapshot_ids=expire,
        delete_orphan_files=tuple(orphan_candidates),
        policy=policy,
        planned_at=planned_at,
        plan_sha256=plan_hash,
    )


def validate_execution_plan(
    plan: IcebergMaintenancePlan,
    *,
    expected_sha256: str,
    current_head: int,
) -> None:
    if str(expected_sha256) != plan.plan_sha256:
        raise ValueError("Iceberg maintenance plan hash does not match")
    if int(current_head) != int(plan.head_snapshot_id):
        raise ValueError("Iceberg table head snapshot changed; create a new plan")
