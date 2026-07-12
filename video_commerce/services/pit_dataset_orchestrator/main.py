"""Resumable daily PIT export and manifest publication."""

from __future__ import annotations

import asyncio
import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import logging
import os
from pathlib import Path
import socket
from typing import Any

import httpx

from video_commerce.ml.pit_manifest import (
    PitManifestPublisher,
    _is_parquet_shard_uri,
    load_pinned_iceberg_run,
)
from video_commerce.ml.ranking_training import RANKING_LABEL_DEFINITION_VERSION

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PitPublicationResult:
    row_count: int
    quarantine_count: int
    snapshot_id: str
    manifest_uri: str


def deterministic_flink_job_id(run_id: str, attempt: int) -> str:
    return hashlib.sha256(f"{run_id}:{max(1, int(attempt))}".encode()).hexdigest()[:32]


class FlinkRestPitRunner:
    ENTRY_CLASS = "com.videocommerce.flink.PointInTimeFeatureJoinJob"
    ACTIVE_STATUSES = {
        "INITIALIZING",
        "CREATED",
        "RUNNING",
        "FAILING",
        "CANCELLING",
        "RESTARTING",
        "SUSPENDED",
        "RECONCILING",
    }
    FAILED_STATUSES = {"FAILED", "CANCELED"}

    def __init__(
        self,
        *,
        jobmanager: str,
        jar_path: str,
        catalog_name: str,
        catalog_uri: str,
        warehouse_uri: str,
        s3_endpoint: str,
        namespace: str,
        export_uri: str,
        feature_definition_version: str,
        attribution_window_hours: int,
        allowed_lateness_hours: int,
        timeout_seconds: float = 7200,
        poll_seconds: float = 5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.rest_url = (
            jobmanager.rstrip("/")
            if jobmanager.startswith(("http://", "https://"))
            else f"http://{jobmanager.rstrip('/')}"
        )
        self.jar_path = jar_path
        self.catalog_name = catalog_name
        self.catalog_uri = catalog_uri
        self.warehouse_uri = warehouse_uri
        self.s3_endpoint = s3_endpoint
        self.namespace = namespace
        self.export_uri = export_uri
        self.feature_definition_version = feature_definition_version
        self.attribution_window_hours = int(attribution_window_hours)
        self.allowed_lateness_hours = int(allowed_lateness_hours)
        self.timeout_seconds = max(0.001, float(timeout_seconds))
        self.poll_seconds = max(0.01, float(poll_seconds))
        self.client = client

    def build_program_arguments(
        self, *, run_id: str, cutoff_ts: float, attempt: int
    ) -> list[str]:
        return [
            "--catalog-name",
            self.catalog_name,
            "--catalog-uri",
            self.catalog_uri,
            "--warehouse-uri",
            self.warehouse_uri,
            "--s3-endpoint",
            self.s3_endpoint,
            "--feature-definition-version",
            self.feature_definition_version,
            "--attribution-window-hours",
            str(self.attribution_window_hours),
            "--allowed-lateness-hours",
            str(self.allowed_lateness_hours),
            "--materialization-run-id",
            run_id,
            "--materialization-cutoff",
            str(int(cutoff_ts)),
            "--namespace",
            self.namespace,
            "--export-uri",
            self.export_uri,
            "--export-attempt",
            str(max(1, int(attempt))),
        ]

    async def inspect(self, job_id: str) -> str | None:
        async with self._client_scope() as client:
            response = await client.get(f"/jobs/{job_id}/status")
            if response.status_code == 404:
                return None
            if response.status_code == 500 and (
                "FlinkJobNotFoundException" in response.text
                or "Could not find Flink job" in response.text
            ):
                # Flink 1.20 reports an unknown valid JobID as HTTP 500 rather
                # than 404. Normalize only its explicit not-found exception;
                # every other control-plane 500 remains fail-closed.
                return None
            response.raise_for_status()
            return str(response.json().get("status") or "").upper()

    async def run(
        self,
        *,
        run_id: str,
        cutoff_ts: float,
        attempt: int,
        job_id: str,
        heartbeat=None,
    ) -> None:
        started = asyncio.get_running_loop().time()
        status = await self.inspect(job_id)
        if status is None:
            await self._submit(
                run_id=run_id,
                cutoff_ts=cutoff_ts,
                attempt=attempt,
                job_id=job_id,
            )
        try:
            while True:
                status = await self.inspect(job_id)
                if status == "FINISHED":
                    return
                if status in self.FAILED_STATUSES:
                    raise RuntimeError(f"Flink PIT job {job_id} ended as {status}")
                if status is None:
                    raise RuntimeError(
                        f"Flink PIT job {job_id} disappeared after submission"
                    )
                if heartbeat is not None:
                    await heartbeat()
                elapsed = asyncio.get_running_loop().time() - started
                if elapsed >= self.timeout_seconds:
                    raise RuntimeError(
                        f"Flink PIT job {job_id} timed out after "
                        f"{self.timeout_seconds:g} seconds"
                    )
                await asyncio.sleep(self.poll_seconds)
        except asyncio.CancelledError:
            raise

    async def _submit(
        self, *, run_id: str, cutoff_ts: float, attempt: int, job_id: str
    ) -> None:
        jar_path = Path(self.jar_path)
        if not jar_path.is_file():
            raise RuntimeError(f"Flink PIT job jar is missing: {jar_path}")
        async with self._client_scope() as client:
            with jar_path.open("rb") as handle:
                upload = await client.post(
                    "/jars/upload",
                    files={
                        "jarfile": (
                            jar_path.name,
                            handle,
                            "application/x-java-archive",
                        )
                    },
                )
            upload.raise_for_status()
            filename = str(upload.json().get("filename") or "")
            jar_id = Path(filename).name
            if not jar_id:
                raise RuntimeError("Flink jar upload did not return a jar ID")
            submission = await client.post(
                f"/jars/{jar_id}/run",
                json={
                    "jobId": job_id,
                    "entryClass": self.ENTRY_CLASS,
                    "programArgsList": self.build_program_arguments(
                        run_id=run_id,
                        cutoff_ts=cutoff_ts,
                        attempt=attempt,
                    ),
                },
            )
            if submission.is_error:
                existing = await self.inspect(job_id)
                if existing is not None:
                    return
                submission.raise_for_status()
            submitted_id = str(submission.json().get("jobid") or "")
            if submitted_id.lower() != job_id.lower():
                raise RuntimeError(
                    f"Flink returned unexpected job ID {submitted_id or '<missing>'}"
                )

    def _client_scope(self):
        if self.client is not None:
            return _BorrowedAsyncClient(self.client)
        return httpx.AsyncClient(base_url=self.rest_url, timeout=30.0)


class _BorrowedAsyncClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def __aenter__(self) -> httpx.AsyncClient:
        return self.client

    async def __aexit__(self, *_args) -> None:
        return None


class IcebergPitPublisher:
    def __init__(
        self,
        *,
        storage: Any,
        manifest_publisher: PitManifestPublisher,
        feature_lake: Any,
        export_prefix: str,
        run_loader=load_pinned_iceberg_run,
    ) -> None:
        self.storage = storage
        self.manifest_publisher = manifest_publisher
        self.feature_lake = feature_lake
        self.export_prefix = export_prefix.rstrip("/")
        self.run_loader = run_loader

    async def publish(
        self, *, run_id: str, cutoff_ts: float, attempt: int
    ) -> PitPublicationResult | None:
        snapshot_id, row_count, quarantine_count = await asyncio.to_thread(
            self.run_loader,
            catalog_uri=self.feature_lake.catalog_uri,
            warehouse_uri=self.feature_lake.warehouse_uri,
            namespace=self.feature_lake.namespace,
            storage=self.storage,
            run_id=run_id,
        )
        if row_count <= 0:
            return None
        shard_prefix = f"{self.export_prefix}/runs/{run_id}/attempts/{attempt}/shards"
        shard_objects = await self.storage.list_storage_uris(shard_prefix)
        shards = [uri for uri in shard_objects if _is_parquet_shard_uri(uri)]
        await self.manifest_publisher.publish(
            shard_uris=shards,
            output_prefix=self.export_prefix,
            materialization_run_id=run_id,
            iceberg_table_id=(f"{self.feature_lake.namespace}.ranking_training_pit"),
            iceberg_snapshot_id=snapshot_id,
            feature_definition_version=(self.feature_lake.feature_definition_version),
            label_definition_version=RANKING_LABEL_DEFINITION_VERSION,
            attribution_cutoff=cutoff_ts,
            quarantine_row_count=quarantine_count,
            expected_iceberg_row_count=row_count,
        )
        return PitPublicationResult(
            row_count=row_count,
            quarantine_count=quarantine_count,
            snapshot_id=snapshot_id,
            manifest_uri=(f"{self.export_prefix}/runs/{run_id}/manifest.json"),
        )


def daily_run_identity(
    now: datetime | None = None, *, hour_utc: int = 2
) -> tuple[str, float]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("PIT schedule timestamp must be timezone-aware")
    utc_now = current.astimezone(timezone.utc)
    utc_day = utc_now.date()
    if utc_now.time() < time(hour=hour_utc):
        utc_day -= timedelta(days=1)
    cutoff = datetime.combine(utc_day, time(hour=hour_utc), tzinfo=timezone.utc)
    return f"pit-{utc_day:%Y%m%d}", cutoff.timestamp()


def next_scheduled_run(now: datetime, *, hour_utc: int = 2) -> datetime:
    if now.tzinfo is None:
        raise ValueError("PIT schedule timestamp must be timezone-aware")
    current = now.astimezone(timezone.utc)
    candidate = datetime.combine(
        current.date(), time(hour=hour_utc), tzinfo=timezone.utc
    )
    return candidate if current < candidate else candidate + timedelta(days=1)


class PitDatasetOrchestrator:
    def __init__(
        self,
        *,
        system_store: Any,
        flink_runner: Any,
        publisher: Any,
        worker_id: str,
        lease_seconds: int = 7200,
        schedule_hour_utc: int = 2,
        observability: Any = None,
    ) -> None:
        self.system_store = system_store
        self.flink_runner = flink_runner
        self.publisher = publisher
        self.worker_id = worker_id
        self.lease_seconds = max(60, int(lease_seconds))
        self.schedule_hour_utc = int(schedule_hour_utc)
        self.observability = observability

    def _record_status(self, status: str) -> str:
        if self.observability is not None:
            self.observability.record_pit_orchestrator_run(status)
            self.observability.set_pit_orchestrator_waiting_for_rows(
                status == "waiting_for_eligible_rows"
            )
            self.observability.set_pit_orchestrator_run_in_progress(False)
        return status

    async def run_once(self, scheduled_at: datetime | None = None) -> str:
        run_id, cutoff_ts = daily_run_identity(
            scheduled_at, hour_utc=self.schedule_hour_utc
        )
        requested_run_id = run_id
        claim = await self.system_store.claim_pit_materialization_run(
            run_id=run_id,
            cutoff_ts=cutoff_ts,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if not claim.get("claimed", False):
            return self._record_status(str(claim.get("status") or "lease_held"))
        # A new daily CronJob must finish the oldest expired run before it starts
        # its own cutoff. The durable claim is authoritative for both identity
        # and cutoff so an existing Flink job can be reattached safely.
        run_id = str(claim.get("run_id") or run_id)
        cutoff_ts = float(claim.get("cutoff_ts") or cutoff_ts)
        if self.observability is not None:
            self.observability.set_pit_orchestrator_run_in_progress(True)
        phase = str(claim.get("phase") or "export")
        attempt = max(1, int(claim.get("attempts") or 1))
        publication_attempt = max(1, int(claim.get("export_attempt") or attempt))
        try:
            if phase in {"export", "flink"}:
                job_id = str(claim.get("flink_job_id") or "").strip().lower()
                remote_status = (
                    await self.flink_runner.inspect(job_id) if job_id else None
                )
                if (
                    phase == "export"
                    or not job_id
                    or remote_status in self.flink_runner.FAILED_STATUSES
                ):
                    job_id = deterministic_flink_job_id(run_id, attempt)
                    publication_attempt = attempt
                    await self.system_store.mark_pit_materialization_phase(
                        run_id,
                        phase="flink",
                        worker_id=self.worker_id,
                        export_attempt=publication_attempt,
                        flink_job_id=job_id,
                        lease_seconds=self.lease_seconds,
                    )

                async def renew_lease() -> None:
                    await self.system_store.renew_pit_materialization_lease(
                        run_id,
                        worker_id=self.worker_id,
                        lease_seconds=self.lease_seconds,
                    )

                await self.flink_runner.run(
                    run_id=run_id,
                    cutoff_ts=cutoff_ts,
                    attempt=publication_attempt,
                    job_id=job_id,
                    heartbeat=renew_lease,
                )
                await self.system_store.mark_pit_materialization_phase(
                    run_id,
                    phase="manifest",
                    worker_id=self.worker_id,
                    export_attempt=publication_attempt,
                    flink_job_id=job_id,
                    lease_seconds=self.lease_seconds,
                )
            publication = await self.publisher.publish(
                run_id=run_id,
                cutoff_ts=cutoff_ts,
                attempt=publication_attempt,
            )
            if publication is None:
                await self.system_store.mark_pit_materialization_waiting(
                    run_id,
                    worker_id=self.worker_id,
                )
                if run_id != requested_run_id:
                    self._record_status("waiting_for_eligible_rows")
                    return await self.run_once(scheduled_at)
                return self._record_status("waiting_for_eligible_rows")
            await self.system_store.complete_pit_materialization_run(
                run_id,
                worker_id=self.worker_id,
                snapshot_id=publication.snapshot_id,
                manifest_uri=publication.manifest_uri,
                row_count=publication.row_count,
                quarantine_count=publication.quarantine_count,
            )
            if run_id != requested_run_id:
                self._record_status("completed")
                return await self.run_once(scheduled_at)
            return self._record_status("completed")
        except Exception as exc:
            await self.system_store.fail_pit_materialization_run(
                run_id,
                worker_id=self.worker_id,
                error=str(exc),
            )
            self._record_status("failed")
            raise


def _scheduled_datetime(value: str | None, *, hour_utc: int) -> datetime | None:
    if not value:
        return None
    selected = date.fromisoformat(value)
    return datetime.combine(selected, time(hour=hour_utc), tzinfo=timezone.utc)


async def _run_service(*, once: bool, run_date: str | None) -> None:
    from video_commerce.common.config import Config
    from video_commerce.common.observability import (
        ObservabilityManager,
        configure_logging,
        start_worker_metrics_server,
    )
    from video_commerce.data_plane.object_storage import ObjectStorage
    from video_commerce.data_plane.system_store import SystemStore

    config = Config()
    lake = config.feature_lake_config
    configure_logging(config.monitoring_config)
    observability = ObservabilityManager()
    start_worker_metrics_server(
        observability, config.monitoring_config, default_port=9105
    )
    store = SystemStore(config.database_config, observability=observability)
    storage = ObjectStorage(config.object_storage_config)
    await store.initialize()
    await storage.initialize()
    worker_id = f"pit-orchestrator-{socket.gethostname()}-{os.getpid()}"
    orchestrator = PitDatasetOrchestrator(
        system_store=store,
        flink_runner=FlinkRestPitRunner(
            jobmanager=lake.flink_jobmanager,
            jar_path=lake.pit_job_jar_path,
            catalog_name=lake.catalog_name,
            catalog_uri=lake.catalog_uri,
            warehouse_uri=lake.warehouse_uri,
            s3_endpoint=(lake.s3_endpoint or config.object_storage_config.endpoint_url),
            namespace=lake.namespace,
            export_uri=lake.pit_export_uri,
            feature_definition_version=lake.feature_definition_version,
            attribution_window_hours=lake.attribution_window_hours,
            allowed_lateness_hours=lake.allowed_lateness_hours,
            timeout_seconds=max(60, lake.pit_orchestrator_lease_seconds - 300),
        ),
        publisher=IcebergPitPublisher(
            storage=storage,
            manifest_publisher=PitManifestPublisher(storage),
            feature_lake=lake,
            export_prefix=lake.pit_export_uri,
        ),
        worker_id=worker_id,
        lease_seconds=lake.pit_orchestrator_lease_seconds,
        schedule_hour_utc=lake.pit_schedule_hour_utc,
        observability=observability,
    )
    try:
        selected = _scheduled_datetime(run_date, hour_utc=lake.pit_schedule_hour_utc)
        if once:
            status = await orchestrator.run_once(selected)
            logger.info("pit_orchestrator_run", extra={"status": status})
            return
        while True:
            now = datetime.now(timezone.utc)
            today_cutoff = datetime.combine(
                now.date(),
                time(hour=lake.pit_schedule_hour_utc),
                tzinfo=timezone.utc,
            )
            if now < today_cutoff:
                await asyncio.sleep((today_cutoff - now).total_seconds())
                now = datetime.now(timezone.utc)
            status = await orchestrator.run_once(now)
            logger.info("pit_orchestrator_run", extra={"status": status})
            upcoming = next_scheduled_run(
                datetime.now(timezone.utc),
                hour_utc=lake.pit_schedule_hour_utc,
            )
            await asyncio.sleep(
                max(1.0, (upcoming - datetime.now(timezone.utc)).total_seconds())
            )
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily PIT dataset orchestrator")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--date", help="UTC run date in YYYY-MM-DD format")
    arguments = parser.parse_args()
    asyncio.run(_run_service(once=arguments.once, run_date=arguments.date))


if __name__ == "__main__":
    main()
