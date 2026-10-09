from argparse import Namespace
import subprocess

from scripts.run_recommendation_outbox_ab import (
    _resource_snapshot,
    evaluate_mode,
    sha256_file,
)


def test_sha256_file_records_artifact_lineage(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b'{"version":"v1"}\n')

    assert (
        sha256_file(manifest)
        == "9a2df226d25ae5f31143e2a1d2f53e8658216683fd44ca53502ea3b0aa535577"
    )


def _run(
    *,
    p95,
    p99=180.0,
    qps=100.0,
    error_rate=0.0,
    five_xx_rate=0.0,
    tracking=1.0,
    durable=3000,
    published=3000,
    pending=0,
):
    return {
        "successful_p95_ms": p95,
        "successful_p99_ms": p99,
        "qps_2xx": qps,
        "error_rate": error_rate,
        "five_xx_rate": five_xx_rate,
        "durable_tracking_rate": tracking,
        "durable_impression_count": durable,
        "outbox_published_count": published,
        "outbox_pending_count": pending,
    }


def test_evaluate_mode_accepts_three_paired_runs_inside_relative_and_absolute_gates():
    controls = [_run(p95=100, qps=100), _run(p95=110, qps=98), _run(p95=90, qps=102)]
    treatments = [_run(p95=108, qps=95), _run(p95=109, qps=94), _run(p95=107, qps=96)]

    result = evaluate_mode("hot", controls, treatments)

    assert result["passed"] is True
    assert result["reasons"] == []
    assert result["control_median_p95_ms"] == 100.0
    assert result["treatment_median_p95_ms"] == 108.0
    assert result["p95_regression"] == 0.08
    assert result["successful_qps_regression"] == 0.05


def test_evaluate_mode_rejects_latency_qps_slo_tracking_and_reconciliation_failures():
    controls = [_run(p95=100, qps=100) for _ in range(3)]
    treatments = [
        _run(
            p95=1200,
            p99=2200,
            qps=80,
            error_rate=0.01,
            five_xx_rate=0.002,
            tracking=0.99,
            durable=2990,
            published=2989,
            pending=1,
        )
        for _ in range(3)
    ]

    result = evaluate_mode("unique", controls, treatments)

    assert result["passed"] is False
    assert set(result["reasons"]) == {
        "p95 regression exceeds 10%",
        "successful QPS regression exceeds 10%",
        "treatment error rate is not below 0.5%",
        "treatment 5xx rate is not below 0.1%",
        "treatment successful p95 is not below 1000ms",
        "treatment successful p99 is not below 2000ms",
        "treatment durable tracking coverage is below 99.5%",
        "treatment outbox did not drain",
        "treatment durable impressions do not match published outbox rows",
    }


def test_evaluate_mode_requires_exactly_three_control_and_treatment_runs():
    result = evaluate_mode("hot", [_run(p95=100)] * 2, [_run(p95=100)] * 3)

    assert result["passed"] is False
    assert result["reasons"] == ["expected exactly three paired runs"]


def test_resource_snapshot_keeps_host_data_when_container_lookup_fails(monkeypatch):
    monkeypatch.setattr(
        "scripts.run_recommendation_outbox_ab._run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["docker"])
        ),
    )

    snapshot = _resource_snapshot(
        Namespace(
            compose_file="docker-compose.yml",
            compose_project_name="missing",
        )
    )

    assert snapshot["host_cpu_count"]
    assert snapshot["host_memory_bytes"]
    assert snapshot["recommendation_container"] is None
