"""prometheus_client multiprocess mode for multi-worker uvicorn services."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from prometheus_client import Gauge
from prometheus_client.parser import text_string_to_metric_families

from video_commerce.common.observability import (
    ObservabilityManager,
    prepare_prometheus_multiprocess_dir,
)


ROOT = Path(__file__).resolve().parents[1]


def _gauge_calls_without_mode(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "Gauge":
            continue
        if not any(keyword.arg == "multiprocess_mode" for keyword in node.keywords):
            yield f"{path.relative_to(ROOT)}:{node.lineno}"


def test_every_gauge_in_the_codebase_declares_multiprocess_mode():
    # The default mode ("all") emits one series per worker pid, which breaks
    # every alert that compares a gauge to a threshold. New Gauges added by any
    # workstream must pick an aggregation explicitly.
    offenders = [
        location
        for path in sorted((ROOT / "video_commerce").rglob("*.py"))
        for location in _gauge_calls_without_mode(path)
    ]
    assert offenders == []


def test_registered_gauges_use_an_aggregating_mode():
    manager = ObservabilityManager()
    gauges = [
        collector
        for collector in manager.registry._collector_to_names
        if isinstance(collector, Gauge)
    ]

    assert len(gauges) >= 69
    assert {
        gauge._name: gauge._multiprocess_mode
        for gauge in gauges
        if gauge._multiprocess_mode in {"all", "liveall"}
    } == {}


_WORKER = textwrap.dedent(
    """
    import json, os, sys
    from video_commerce.common.observability import ObservabilityManager

    requests, in_progress, connected, redis_clients, dead = json.loads(sys.argv[1])
    manager = ObservabilityManager()
    for _ in range(requests):
        manager.record_request("GET", "/health", 200, 0.01)
    for _ in range(in_progress):
        manager.http_requests_in_progress.inc()
    manager.kafka_producer_connected.set(connected)
    if redis_clients is not None:
        manager.redis_connected_clients.labels(role="state").set(redis_clients)
    if dead:
        manager.mark_process_dead()
    """
)

_SCRAPE = textwrap.dedent(
    """
    from video_commerce.common.observability import ObservabilityManager

    print(ObservabilityManager().prometheus_payload().decode("utf-8"))
    """
)


def _run(code: str, multiproc_dir: Path, *args: str) -> str:
    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(multiproc_dir)}
    env.pop("prometheus_multiproc_dir", None)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return result.stdout


def _samples(payload: str):
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(payload)
        for sample in family.samples
    }


def test_two_worker_processes_aggregate_into_one_scrape(tmp_path):
    # Worker A: 3 requests, 2 in flight, producer disconnected, then exits
    # cleanly. Worker B: 5 requests, 1 in flight, producer connected, newer
    # Redis observation, still alive.
    _run(_WORKER, tmp_path, json.dumps([3, 2, 0, 10, True]))
    _run(_WORKER, tmp_path, json.dumps([5, 1, 1, 42, False]))

    samples = _samples(_run(_SCRAPE, tmp_path))

    requests_key = (
        "video_commerce_http_requests_total",
        (("method", "GET"), ("path", "/health"), ("status", "200")),
    )
    # Counters from both processes, including the one that exited, are summed.
    assert samples[requests_key] == 8.0
    # livesum: worker A's in-flight gauge was dropped by mark_process_dead.
    assert samples[("video_commerce_http_requests_in_progress", ())] == 1.0
    # livemax: any live worker with a connected producer reports connected.
    assert samples[("video_commerce_kafka_producer_connected", ())] == 1.0
    # mostrecent: the latest observation of shared Redis state wins.
    assert (
        samples[("video_commerce_redis_connected_clients", (("role", "state"),))]
        == 42.0
    )
    # Aggregated gauges never fan out per pid.
    assert not any(dict(labels).get("pid") for _, labels in samples)


def test_counter_stays_monotonic_across_repeated_scrapes(tmp_path):
    previous = -1.0
    for _ in range(3):
        _run(_WORKER, tmp_path, json.dumps([2, 0, 1, None, True]))
        samples = _samples(_run(_SCRAPE, tmp_path))
        current = samples[
            (
                "video_commerce_http_requests_total",
                (("method", "GET"), ("path", "/health"), ("status", "200")),
            )
        ]
        assert current > previous
        previous = current
    assert previous == 6.0


def test_prepare_removes_stale_files_from_a_previous_container(tmp_path):
    _run(_WORKER, tmp_path, json.dumps([4, 0, 1, None, False]))
    assert list(tmp_path.glob("*.db"))

    removed = prepare_prometheus_multiprocess_dir(str(tmp_path))

    assert removed > 0
    assert list(tmp_path.glob("*.db")) == []
    samples = _samples(_run(_SCRAPE, tmp_path))
    assert not any(
        name == "video_commerce_http_requests_total" for name, _ in samples
    )


def test_prepare_cli_is_a_noop_without_the_env(tmp_path):
    env = {
        key: value
        for key, value in os.environ.items()
        if key.lower() != "prometheus_multiproc_dir"
    }
    result = subprocess.run(
        [sys.executable, "-m", "video_commerce.common.prometheus_multiproc"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0


def test_single_process_export_is_unchanged_without_the_env():
    manager = ObservabilityManager()
    manager.record_request("GET", "/health", 200, 0.01)
    manager.mark_process_dead()

    payload = manager.prometheus_payload().decode("utf-8")

    assert manager.multiprocess_dir is None
    assert (
        'video_commerce_http_requests_total{method="GET",path="/health",status="200"} 1.0'
        in payload
    )


def test_uvicorn_services_clear_the_multiprocess_dir_before_exec():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    values = (ROOT / "charts/video-commerce/values.yaml").read_text(encoding="utf-8")
    prepare = "python -m video_commerce.common.prometheus_multiproc && exec python -m uvicorn"
    for module in (
        "video_commerce.services.gateway.api:app",
        "video_commerce.services.recommendation.api:app",
        "video_commerce.services.ranking_service.proxy_asgi:app",
        "video_commerce.services.interaction_ingest.api:app",
    ):
        assert f"{prepare} {module}" in compose
        assert f"{prepare} {module}" in values


def test_service_apps_drop_live_gauges_on_shutdown():
    from video_commerce.common.service_common import create_service_app

    app = create_service_app("Test", "test", "test-service")

    assert app.state.runtime.observability.mark_process_dead in app.router.on_shutdown
