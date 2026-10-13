"""Plan, fence, execute, and verify bounded Iceberg maintenance."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid
from typing import Any

import httpx

from video_commerce.common.config import Config
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.ml.iceberg_maintenance import (
    ICEBERG_MAINTENANCE_TABLES,
    IcebergMaintenancePlan,
    IcebergMaintenancePolicy,
    IcebergSnapshot,
    IcebergTableState,
    build_maintenance_plan,
    validate_execution_plan,
)


ENTRY_CLASS = "com.videocommerce.flink.IcebergMaintenanceJob"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iceberg-maintenance")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    status.add_argument("--table", choices=sorted(ICEBERG_MAINTENANCE_TABLES))
    plan = commands.add_parser("plan")
    plan.add_argument("--table", choices=sorted(ICEBERG_MAINTENANCE_TABLES))
    execute = commands.add_parser("execute")
    execute.add_argument("--run-id", required=True)
    execute.add_argument("--plan-sha256", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-id", required=True)
    scheduled = commands.add_parser("scheduled", help=argparse.SUPPRESS)
    scheduled.add_argument("--execute", action="store_true")
    return parser


def _catalog(config: Config):
    from pyiceberg.catalog import load_catalog

    lake = config.feature_lake_config
    storage = config.object_storage_config
    return load_catalog(
        lake.catalog_name,
        type="rest",
        uri=lake.catalog_uri,
        warehouse=lake.warehouse_uri,
        **{
            "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
            "s3.endpoint": storage.endpoint_url,
            "s3.access-key-id": storage.access_key_id,
            "s3.secret-access-key": storage.secret_access_key,
            "s3.region": storage.region,
        },
    )


def _table_state(config: Config, table_name: str) -> IcebergTableState:
    table = _catalog(config).load_table(
        f"{config.feature_lake_config.namespace}.{table_name}"
    )
    current = table.current_snapshot()
    if current is None:
        raise RuntimeError(f"Iceberg table {table_name} has no current snapshot")
    snapshots = tuple(
        IcebergSnapshot(
            snapshot_id=int(snapshot.snapshot_id),
            committed_at=datetime.fromtimestamp(
                float(snapshot.timestamp_ms) / 1000.0, tz=timezone.utc
            ),
        )
        for snapshot in table.snapshots()
    )
    return IcebergTableState(
        table=table_name,
        head_snapshot_id=int(current.snapshot_id),
        snapshots=snapshots,
        # File enumeration and prefix comparison are executed by Iceberg's
        # DeleteOrphanFiles action. The Python plan never lists or deletes S3.
        orphan_files=(),
        table_prefix=str(table.location()).rstrip("/") + "/",
    )


def _policy(config: Config) -> IcebergMaintenancePolicy:
    lake = config.feature_lake_config
    return IcebergMaintenancePolicy(
        snapshot_retention_days=lake.iceberg_maintenance_snapshot_retention_days,
        retain_last_snapshots=lake.iceberg_maintenance_retain_last_snapshots,
        orphan_min_age_days=lake.iceberg_maintenance_orphan_min_age_days,
        delete_batch_size=lake.iceberg_maintenance_delete_batch_size,
    )


async def _submit(config: Config, plan: IcebergMaintenancePlan) -> str:
    lake = config.feature_lake_config
    jar = Path(lake.pit_job_jar_path)
    if not jar.is_file():
        raise RuntimeError(f"Iceberg maintenance jar is missing: {jar}")
    endpoint = str(lake.flink_jobmanager or "").rstrip("/")
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"http://{endpoint}"
    job_id = hashlib.md5(
        f"iceberg-maintenance:{plan.table}:{plan.plan_sha256}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    program_args = [
        "--table",
        plan.table,
        "--namespace",
        lake.namespace,
        "--catalog-uri",
        str(lake.catalog_uri),
        "--warehouse-uri",
        str(lake.warehouse_uri),
        "--expected-head-snapshot",
        str(plan.head_snapshot_id),
        "--snapshot-retention-days",
        str(plan.policy.snapshot_retention_days),
        "--retain-last-snapshots",
        str(plan.policy.retain_last_snapshots),
        "--orphan-min-age-days",
        str(plan.policy.orphan_min_age_days),
        "--delete-batch-size",
        str(plan.policy.delete_batch_size),
        "--run-seconds",
        str(lake.iceberg_maintenance_run_seconds),
    ]
    for snapshot_id in plan.expire_snapshot_ids:
        program_args.extend(("--expire-snapshot", str(snapshot_id)))
    async with httpx.AsyncClient(base_url=endpoint, timeout=30.0) as client:
        with jar.open("rb") as handle:
            upload = await client.post(
                "/jars/upload",
                files={"jarfile": (jar.name, handle, "application/x-java-archive")},
            )
        upload.raise_for_status()
        jar_id = Path(str(upload.json().get("filename") or "")).name
        if not jar_id:
            raise RuntimeError("Flink jar upload did not return a jar ID")
        response = await client.post(
            f"/jars/{jar_id}/run",
            json={
                "jobId": job_id,
                "entryClass": ENTRY_CLASS,
                "programArgsList": program_args,
            },
        )
        response.raise_for_status()
    return job_id


async def _wait_job(config: Config, job_id: str) -> str:
    lake = config.feature_lake_config
    endpoint = str(lake.flink_jobmanager or "").rstrip("/")
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"http://{endpoint}"
    deadline = (
        asyncio.get_running_loop().time() + lake.iceberg_maintenance_run_seconds + 120
    )
    async with httpx.AsyncClient(base_url=endpoint, timeout=30.0) as client:
        while True:
            response = await client.get(f"/jobs/{job_id}/status")
            response.raise_for_status()
            state = str(response.json().get("status") or "").upper()
            if state in {"FINISHED", "CANCELED"}:
                return state
            if state in {"FAILED", "SUSPENDED"}:
                raise RuntimeError(f"Iceberg maintenance Flink job ended as {state}")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    "Iceberg maintenance Flink job did not terminate in time"
                )
            await asyncio.sleep(5)


async def _plan_one(
    config: Config, store: SystemStore, table_name: str
) -> dict[str, Any]:
    protected = await store.list_protected_iceberg_snapshots()
    state = await asyncio.to_thread(_table_state, config, table_name)
    plan = build_maintenance_plan(
        state,
        protected_snapshot_ids=protected.get(table_name, set()),
        policy=_policy(config),
    )
    run_id = f"iceberg-{table_name}-{uuid.uuid4().hex[:12]}"
    await store.save_iceberg_maintenance_plan(
        run_id=run_id,
        table_name=table_name,
        mode="dry_run",
        plan_sha256=plan.plan_sha256,
        head_snapshot_id=str(plan.head_snapshot_id),
        plan=json.loads(plan.to_json()),
        before_counts={"snapshots": len(state.snapshots)},
    )
    return {"run_id": run_id, "plan": json.loads(plan.to_json())}


async def _execute(
    config: Config, store: SystemStore, run_id: str, plan_sha256: str
) -> dict[str, Any]:
    record = await store.get_iceberg_maintenance_run(run_id)
    if record is None:
        raise RuntimeError("Iceberg maintenance run does not exist")
    plan = IcebergMaintenancePlan.from_json(json.dumps(record["plan"]))
    state = await asyncio.to_thread(_table_state, config, plan.table)
    validate_execution_plan(
        plan, expected_sha256=plan_sha256, current_head=state.head_snapshot_id
    )
    protected = await store.list_protected_iceberg_snapshots()
    if set(plan.expire_snapshot_ids) & protected.get(plan.table, set()):
        raise RuntimeError("Iceberg protected references changed; create a new plan")
    owner_token = uuid.uuid4().hex
    if not await store.claim_iceberg_maintenance_run(
        run_id=run_id,
        owner_token=owner_token,
        lease_seconds=config.feature_lake_config.iceberg_maintenance_run_seconds + 180,
    ):
        raise RuntimeError("Iceberg maintenance run is already claimed or completed")
    try:
        job_id = await _submit(config, plan)
    except Exception as exc:
        await store.update_iceberg_maintenance_run(
            run_id=run_id, status="failed", last_error=str(exc)
        )
        raise
    return {"run_id": run_id, "job_id": job_id, "status": "running"}


async def _verify(config: Config, store: SystemStore, run_id: str) -> dict[str, Any]:
    record = await store.get_iceberg_maintenance_run(run_id)
    if record is None:
        raise RuntimeError("Iceberg maintenance run does not exist")
    plan = IcebergMaintenancePlan.from_json(json.dumps(record["plan"]))
    state = await asyncio.to_thread(_table_state, config, plan.table)
    remaining = {snapshot.snapshot_id for snapshot in state.snapshots}
    protected_missing = sorted(set(plan.protected_snapshot_ids) - remaining)
    expired_remaining = sorted(set(plan.expire_snapshot_ids) & remaining)
    if protected_missing or expired_remaining:
        await store.update_iceberg_maintenance_run(
            run_id=run_id,
            status="failed",
            after_counts={"snapshots": len(state.snapshots)},
            last_error=(
                f"protected_missing={protected_missing}; expired_remaining={expired_remaining}"
            ),
        )
        raise RuntimeError("Iceberg maintenance verification failed")
    await store.update_iceberg_maintenance_run(
        run_id=run_id,
        status="completed",
        after_counts={"snapshots": len(state.snapshots)},
        deleted_counts={"snapshots": len(plan.expire_snapshot_ids)},
    )
    return {
        "run_id": run_id,
        "status": "completed",
        "head_snapshot_id": state.head_snapshot_id,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    config = Config()
    lake = config.feature_lake_config
    if not lake.enabled or not lake.catalog_uri or not lake.warehouse_uri:
        raise RuntimeError("Iceberg feature-lake configuration is incomplete")
    store = SystemStore(config.database_config)
    await store.initialize()
    try:
        tables = (
            [args.table]
            if getattr(args, "table", None)
            else sorted(ICEBERG_MAINTENANCE_TABLES)
        )
        if args.command == "status":
            states = [
                await asyncio.to_thread(_table_state, config, table) for table in tables
            ]
            return {
                "tables": [
                    {
                        "table": state.table,
                        "head_snapshot_id": state.head_snapshot_id,
                        "snapshot_count": len(state.snapshots),
                    }
                    for state in states
                ]
            }
        if args.command == "plan":
            return {"runs": [await _plan_one(config, store, table) for table in tables]}
        if args.command == "execute":
            return await _execute(config, store, args.run_id, args.plan_sha256)
        if args.command == "verify":
            return await _verify(config, store, args.run_id)
        results = []
        for table in tables:
            planned = await _plan_one(config, store, table)
            if args.execute:
                planned["execution"] = await _execute(
                    config,
                    store,
                    planned["run_id"],
                    planned["plan"]["plan_sha256"],
                )
                planned["execution"]["flink_status"] = await _wait_job(
                    config, planned["execution"]["job_id"]
                )
                planned["verification"] = await _verify(
                    config, store, planned["run_id"]
                )
            results.append(planned)
        return {"runs": results}
    finally:
        await store.close()


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(asyncio.run(run(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
