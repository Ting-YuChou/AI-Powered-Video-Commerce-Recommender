from datetime import datetime, timezone
import json
from types import SimpleNamespace

import httpx
import pytest

from video_commerce.services.pit_dataset_orchestrator.main import (
    FlinkRestPitRunner,
    IcebergPitPublisher,
    PitDatasetOrchestrator,
    PitPublicationResult,
    daily_run_identity,
    deterministic_flink_job_id,
    next_scheduled_run,
)


class FakeRunStore:
    def __init__(
        self,
        *,
        phase="export",
        attempts=1,
        export_attempt=None,
        flink_job_id=None,
        claimed_run_id=None,
        claimed_cutoff_ts=None,
    ):
        self.phase = phase
        self.attempts = attempts
        self.export_attempt = export_attempt
        self.flink_job_id = flink_job_id
        self.claimed_run_id = claimed_run_id
        self.claimed_cutoff_ts = claimed_cutoff_ts
        self.renewals = 0
        self.completed = None
        self.completed_calls = []
        self.waiting = None
        self.failed = None

    async def claim_pit_materialization_run(self, **claim):
        resumed_prior_run = self.claimed_run_id is not None
        claimed_run_id = self.claimed_run_id or claim["run_id"]
        claimed_cutoff_ts = self.claimed_cutoff_ts or claim["cutoff_ts"]
        self.claimed_run_id = None
        self.claimed_cutoff_ts = None
        if not resumed_prior_run and self.completed_calls:
            self.phase = "export"
            self.export_attempt = None
            self.flink_job_id = None
        return {
            **claim,
            "run_id": claimed_run_id,
            "cutoff_ts": claimed_cutoff_ts,
            "phase": self.phase,
            "status": "running",
            "attempts": self.attempts,
            "export_attempt": self.export_attempt,
            "flink_job_id": self.flink_job_id,
            "claimed": True,
        }

    async def mark_pit_materialization_phase(
        self,
        run_id,
        *,
        phase,
        worker_id,
        export_attempt=None,
        flink_job_id=None,
        lease_seconds=None,
    ):
        self.phase = phase
        self.export_attempt = export_attempt
        if flink_job_id is not None:
            self.flink_job_id = flink_job_id

    async def renew_pit_materialization_lease(self, *_args, **_kwargs):
        self.renewals += 1

    async def complete_pit_materialization_run(self, run_id, **result):
        self.completed = (run_id, result)
        self.completed_calls.append((run_id, result))

    async def mark_pit_materialization_waiting(self, run_id, **result):
        self.waiting = (run_id, result)

    async def fail_pit_materialization_run(self, run_id, **result):
        self.failed = (run_id, result)


class FakeFlinkRunner:
    FAILED_STATUSES = {"FAILED", "CANCELED"}

    def __init__(self, status=None):
        self.calls = []
        self.status = status

    async def inspect(self, _job_id):
        return self.status

    async def run(self, *, run_id, cutoff_ts, attempt, job_id, heartbeat):
        self.calls.append((run_id, cutoff_ts, attempt, job_id))
        await heartbeat()


class FakePublisher:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def publish(self, *, run_id, cutoff_ts, attempt):
        self.calls.append((run_id, cutoff_ts, attempt))
        return self.result


class FakeObjectStorage:
    def __init__(self, shard_uris=None):
        self.shard_uris = shard_uris or []
        self.prefixes = []

    async def list_storage_uris(self, prefix):
        self.prefixes.append(prefix)
        return list(self.shard_uris)


class FakeManifestPublisher:
    def __init__(self):
        self.calls = []

    async def publish(self, **payload):
        self.calls.append(payload)
        return "s3://features/training/ranking-pit/latest.json"


def test_daily_run_identity_is_stable_at_0200_utc():
    run_id, cutoff_ts = daily_run_identity(
        datetime(2026, 7, 11, 19, 45, tzinfo=timezone.utc)
    )

    assert run_id == "pit-20260711"
    assert cutoff_ts == datetime(2026, 7, 11, 2, 0, tzinfo=timezone.utc).timestamp()


def test_daily_run_identity_uses_previous_boundary_before_0200_utc():
    run_id, cutoff_ts = daily_run_identity(
        datetime(2026, 7, 11, 1, 45, tzinfo=timezone.utc)
    )

    assert run_id == "pit-20260710"
    assert cutoff_ts == datetime(2026, 7, 10, 2, 0, tzinfo=timezone.utc).timestamp()


def test_next_scheduled_run_uses_today_before_cutoff_and_tomorrow_after():
    assert next_scheduled_run(
        datetime(2026, 7, 11, 1, 0, tzinfo=timezone.utc), hour_utc=2
    ) == datetime(2026, 7, 11, 2, 0, tzinfo=timezone.utc)
    assert next_scheduled_run(
        datetime(2026, 7, 11, 2, 1, tzinfo=timezone.utc), hour_utc=2
    ) == datetime(2026, 7, 12, 2, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_orchestrator_resumes_expired_prior_daily_run_identity():
    prior_cutoff = datetime(2026, 7, 10, 2, 0, tzinfo=timezone.utc).timestamp()
    store = FakeRunStore(
        claimed_run_id="pit-20260710",
        claimed_cutoff_ts=prior_cutoff,
    )
    runner = FakeFlinkRunner()
    publisher = FakePublisher(
        PitPublicationResult(
            row_count=1,
            quarantine_count=0,
            snapshot_id="42",
            manifest_uri="s3://features/runs/pit-20260710/manifest.json",
        )
    )
    orchestrator = PitDatasetOrchestrator(
        system_store=store,
        flink_runner=runner,
        publisher=publisher,
        worker_id="worker-new",
    )

    status = await orchestrator.run_once(
        datetime(2026, 7, 11, 3, 0, tzinfo=timezone.utc)
    )

    assert status == "completed"
    assert [call[0] for call in runner.calls] == ["pit-20260710", "pit-20260711"]
    assert runner.calls[0][1] == prior_cutoff
    assert [call[0] for call in publisher.calls] == [
        "pit-20260710",
        "pit-20260711",
    ]
    assert [call[0] for call in store.completed_calls] == [
        "pit-20260710",
        "pit-20260711",
    ]


def test_flink_runner_builds_explicit_reproducible_job_arguments():
    runner = FlinkRestPitRunner(
        jobmanager="flink-jobmanager:8081",
        jar_path="/opt/flink/usrlib/interaction-features.jar",
        catalog_name="feature_catalog",
        catalog_uri="http://iceberg-rest:8181",
        warehouse_uri="s3://features/warehouse",
        s3_endpoint="http://minio:9000",
        namespace="video_commerce",
        export_uri="s3://features/training/ranking-pit",
        feature_definition_version="ranking_ltr_v1",
        attribution_window_hours=168,
        allowed_lateness_hours=1,
    )

    arguments = runner.build_program_arguments(
        run_id="pit-20260711",
        cutoff_ts=1_752_215_200.0,
        attempt=2,
    )

    assert arguments == [
        "--catalog-name",
        "feature_catalog",
        "--catalog-uri",
        "http://iceberg-rest:8181",
        "--warehouse-uri",
        "s3://features/warehouse",
        "--s3-endpoint",
        "http://minio:9000",
        "--feature-definition-version",
        "ranking_ltr_v1",
        "--attribution-window-hours",
        "168",
        "--allowed-lateness-hours",
        "1",
        "--materialization-run-id",
        "pit-20260711",
        "--materialization-cutoff",
        "1752215200",
        "--namespace",
        "video_commerce",
        "--export-uri",
        "s3://features/training/ranking-pit",
        "--export-attempt",
        "2",
    ]
    assert deterministic_flink_job_id("pit-20260711", 2) != (
        deterministic_flink_job_id("pit-20260711", 1)
    )
    assert len(deterministic_flink_job_id("pit-20260711", 2)) == 32


@pytest.mark.asyncio
async def test_flink_runner_submits_fixed_job_id_and_polls_to_completion(tmp_path):
    job_id = deterministic_flink_job_id("pit-20260711", 1)
    jar = tmp_path / "job.jar"
    jar.write_bytes(b"jar")
    statuses = iter([404, "RUNNING", "FINISHED"])
    submissions = []

    def handler(request):
        if request.method == "GET":
            status = next(statuses)
            if status == 404:
                return httpx.Response(404)
            return httpx.Response(200, json={"status": status})
        if request.url.path == "/jars/upload":
            return httpx.Response(200, json={"filename": "/tmp/uploaded-job.jar"})
        payload = json.loads(request.content)
        submissions.append(payload)
        return httpx.Response(200, json={"jobid": job_id})

    client = httpx.AsyncClient(
        base_url="http://flink:8081", transport=httpx.MockTransport(handler)
    )
    runner = FlinkRestPitRunner(
        jobmanager="flink:8081",
        jar_path=str(jar),
        catalog_name="feature_catalog",
        catalog_uri="http://catalog:8181",
        warehouse_uri="s3://features/warehouse",
        s3_endpoint="http://minio:9000",
        namespace="video_commerce",
        export_uri="s3://features/pit",
        feature_definition_version="ranking_ltr_v1",
        attribution_window_hours=168,
        allowed_lateness_hours=1,
        poll_seconds=0.001,
        client=client,
    )
    heartbeats = 0

    async def heartbeat():
        nonlocal heartbeats
        heartbeats += 1

    await runner.run(
        run_id="pit-20260711",
        cutoff_ts=1.0,
        attempt=1,
        job_id=job_id,
        heartbeat=heartbeat,
    )
    await client.aclose()

    assert submissions[0]["jobId"] == job_id
    assert submissions[0]["programArgsList"][-2:] == ["--export-attempt", "1"]
    assert heartbeats == 1


@pytest.mark.asyncio
async def test_flink_runner_treats_flink_120_not_found_500_as_absent_job(tmp_path):
    def handler(_request):
        return httpx.Response(
            500,
            json={
                "errors": [
                    "Internal server error.",
                    "org.apache.flink.runtime.messages.FlinkJobNotFoundException: "
                    "Could not find Flink job (0123456789abcdef0123456789abcdef)",
                ]
            },
        )

    client = httpx.AsyncClient(
        base_url="http://flink:8081", transport=httpx.MockTransport(handler)
    )
    runner = FlinkRestPitRunner(
        jobmanager="flink:8081",
        jar_path=str(tmp_path / "job.jar"),
        catalog_name="feature_catalog",
        catalog_uri="http://catalog:8181",
        warehouse_uri="s3://features/warehouse",
        s3_endpoint="http://minio:9000",
        namespace="video_commerce",
        export_uri="s3://features/pit",
        feature_definition_version="ranking_ltr_v1",
        attribution_window_hours=168,
        allowed_lateness_hours=1,
        client=client,
    )

    assert await runner.inspect("0123456789abcdef0123456789abcdef") is None
    await client.aclose()


@pytest.mark.asyncio
async def test_iceberg_publisher_returns_waiting_without_creating_empty_manifest():
    manifest_publisher = FakeManifestPublisher()
    publisher = IcebergPitPublisher(
        storage=FakeObjectStorage(),
        manifest_publisher=manifest_publisher,
        feature_lake=SimpleNamespace(
            catalog_uri="http://catalog:8181",
            warehouse_uri="s3://features/warehouse",
            namespace="video_commerce",
            feature_definition_version="ranking_ltr_v1",
        ),
        export_prefix="s3://features/training/ranking-pit",
        run_loader=lambda **_kwargs: ("42", 0, 0),
    )

    result = await publisher.publish(
        run_id="pit-20260711", cutoff_ts=1_752_215_200.0, attempt=1
    )

    assert result is None
    assert manifest_publisher.calls == []


@pytest.mark.asyncio
async def test_iceberg_publisher_returns_completed_lineage():
    manifest_publisher = FakeManifestPublisher()
    shard = "s3://features/training/ranking-pit/runs/pit-20260711/shards/part-1"
    storage = FakeObjectStorage([shard])
    publisher = IcebergPitPublisher(
        storage=storage,
        manifest_publisher=manifest_publisher,
        feature_lake=SimpleNamespace(
            catalog_uri="http://catalog:8181",
            warehouse_uri="s3://features/warehouse",
            namespace="video_commerce",
            feature_definition_version="ranking_ltr_v1",
        ),
        export_prefix="s3://features/training/ranking-pit",
        run_loader=lambda **_kwargs: ("42", 3, 1),
    )

    result = await publisher.publish(
        run_id="pit-20260711", cutoff_ts=1_752_215_200.0, attempt=1
    )

    assert result == PitPublicationResult(
        row_count=3,
        quarantine_count=1,
        snapshot_id="42",
        manifest_uri=(
            "s3://features/training/ranking-pit/runs/" "pit-20260711/manifest.json"
        ),
    )
    assert manifest_publisher.calls[0]["shard_uris"] == [shard]
    assert storage.prefixes == [
        "s3://features/training/ranking-pit/runs/pit-20260711/attempts/1/shards"
    ]


@pytest.mark.asyncio
async def test_orchestrator_completes_export_then_manifest():
    class ObservabilitySpy:
        statuses = []
        waiting = []
        running = []

        def record_pit_orchestrator_run(self, status):
            self.statuses.append(status)

        def set_pit_orchestrator_waiting_for_rows(self, waiting):
            self.waiting.append(waiting)

        def set_pit_orchestrator_run_in_progress(self, running):
            self.running.append(running)

    observability = ObservabilitySpy()
    store = FakeRunStore()
    runner = FakeFlinkRunner()
    publisher = FakePublisher(
        PitPublicationResult(
            row_count=12,
            quarantine_count=0,
            snapshot_id="42",
            manifest_uri="s3://features/runs/pit-20260711/manifest.json",
        )
    )
    orchestrator = PitDatasetOrchestrator(
        system_store=store,
        flink_runner=runner,
        publisher=publisher,
        worker_id="worker-1",
        observability=observability,
    )

    status = await orchestrator.run_once(
        datetime(2026, 7, 11, 2, 5, tzinfo=timezone.utc)
    )

    assert status == "completed"
    assert len(runner.calls) == 1
    assert len(publisher.calls) == 1
    assert store.completed[1]["snapshot_id"] == "42"
    assert observability.statuses == ["completed"]
    assert observability.waiting == [False]
    assert observability.running == [True, False]


@pytest.mark.asyncio
async def test_orchestrator_does_not_publish_empty_run():
    store = FakeRunStore()
    runner = FakeFlinkRunner()
    publisher = FakePublisher(None)
    orchestrator = PitDatasetOrchestrator(
        system_store=store,
        flink_runner=runner,
        publisher=publisher,
        worker_id="worker-1",
    )

    status = await orchestrator.run_once(
        datetime(2026, 7, 11, 2, 5, tzinfo=timezone.utc)
    )

    assert status == "waiting_for_eligible_rows"
    assert store.waiting[0] == "pit-20260711"
    assert store.completed is None


@pytest.mark.asyncio
async def test_orchestrator_reconnects_to_durable_running_flink_job():
    job_id = deterministic_flink_job_id("pit-20260711", 1)
    store = FakeRunStore(
        phase="flink",
        attempts=2,
        export_attempt=1,
        flink_job_id=job_id,
    )
    runner = FakeFlinkRunner(status="RUNNING")
    publisher = FakePublisher(
        PitPublicationResult(
            row_count=2,
            quarantine_count=0,
            snapshot_id="43",
            manifest_uri="s3://features/runs/pit-20260711/manifest.json",
        )
    )
    orchestrator = PitDatasetOrchestrator(
        system_store=store,
        flink_runner=runner,
        publisher=publisher,
        worker_id="worker-2",
    )

    status = await orchestrator.run_once(
        datetime(2026, 7, 11, 4, 0, tzinfo=timezone.utc)
    )

    assert status == "completed"
    assert runner.calls[0][2:] == (1, job_id)
    assert store.renewals == 1


@pytest.mark.asyncio
async def test_orchestrator_resumes_manifest_phase_without_rerunning_flink():
    store = FakeRunStore(phase="manifest", attempts=3, export_attempt=1)
    runner = FakeFlinkRunner()
    publisher = FakePublisher(
        PitPublicationResult(
            row_count=2,
            quarantine_count=0,
            snapshot_id="43",
            manifest_uri="s3://features/runs/pit-20260711/manifest.json",
        )
    )
    orchestrator = PitDatasetOrchestrator(
        system_store=store,
        flink_runner=runner,
        publisher=publisher,
        worker_id="worker-2",
    )

    status = await orchestrator.run_once(
        datetime(2026, 7, 11, 4, 0, tzinfo=timezone.utc)
    )

    assert status == "completed"
    assert runner.calls == []
    assert len(publisher.calls) == 1
    assert publisher.calls[0][2] == 1
