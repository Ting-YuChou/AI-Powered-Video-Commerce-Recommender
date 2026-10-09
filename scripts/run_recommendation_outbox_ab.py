#!/usr/bin/env python3
"""Run paired recommendation served-impression outbox A/B baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence

import httpx


EXPECTED_RUNS = 3
ORDER = ((False, True), (True, False), (False, True))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median(runs: Sequence[Dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(run[key]) for run in runs))


def _regression(control: float, treatment: float, *, lower_is_better: bool) -> float:
    if control <= 0:
        return 0.0 if treatment <= 0 else 1.0
    delta = (treatment - control) / control
    return round(delta if lower_is_better else -delta, 4)


def evaluate_mode(
    mode: str,
    control_runs: Sequence[Dict[str, Any]],
    treatment_runs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate the fixed three-pair A/B acceptance contract."""
    result: Dict[str, Any] = {"mode": mode, "passed": False, "reasons": []}
    if len(control_runs) != EXPECTED_RUNS or len(treatment_runs) != EXPECTED_RUNS:
        result["reasons"] = ["expected exactly three paired runs"]
        return result

    control_p95 = _median(control_runs, "successful_p95_ms")
    treatment_p95 = _median(treatment_runs, "successful_p95_ms")
    control_qps = _median(control_runs, "qps_2xx")
    treatment_qps = _median(treatment_runs, "qps_2xx")
    p95_regression = _regression(control_p95, treatment_p95, lower_is_better=True)
    qps_regression = _regression(control_qps, treatment_qps, lower_is_better=False)
    result.update(
        {
            "control_median_p95_ms": round(control_p95, 2),
            "treatment_median_p95_ms": round(treatment_p95, 2),
            "control_median_qps_2xx": round(control_qps, 2),
            "treatment_median_qps_2xx": round(treatment_qps, 2),
            "p95_regression": p95_regression,
            "successful_qps_regression": qps_regression,
        }
    )

    reasons: List[str] = []
    if p95_regression > 0.10:
        reasons.append("p95 regression exceeds 10%")
    if qps_regression > 0.10:
        reasons.append("successful QPS regression exceeds 10%")
    if any(float(run["error_rate"]) >= 0.005 for run in treatment_runs):
        reasons.append("treatment error rate is not below 0.5%")
    if any(float(run["five_xx_rate"]) >= 0.001 for run in treatment_runs):
        reasons.append("treatment 5xx rate is not below 0.1%")
    if any(float(run["successful_p95_ms"]) >= 1000 for run in treatment_runs):
        reasons.append("treatment successful p95 is not below 1000ms")
    if any(float(run["successful_p99_ms"]) >= 2000 for run in treatment_runs):
        reasons.append("treatment successful p99 is not below 2000ms")
    if any(float(run["durable_tracking_rate"]) < 0.995 for run in treatment_runs):
        reasons.append("treatment durable tracking coverage is below 99.5%")
    if any(int(run["outbox_pending_count"]) != 0 for run in treatment_runs):
        reasons.append("treatment outbox did not drain")
    if any(
        int(run["durable_impression_count"]) != int(run["outbox_published_count"])
        for run in treatment_runs
    ):
        reasons.append(
            "treatment durable impressions do not match published outbox rows"
        )
    result["reasons"] = reasons
    result["passed"] = not reasons
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare recommendation latency with the durable impression outbox off/on"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--compose-project-name", required=True)
    parser.add_argument("--requests", type=int, default=3000)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--warmup-requests", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--drain-timeout", type=float, default=60.0)
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    parser.add_argument(
        "--output-dir", default="loadtest/results/recommendation-outbox-ab"
    )
    parser.add_argument("--api-key")
    parser.add_argument("--compose-file", default="docker-compose.yml")
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        help="Optional verified artifact manifest recorded with the result",
    )
    parser.add_argument(
        "--environment-class",
        choices=("directional", "fixed-linux"),
        default="directional",
    )
    return parser.parse_args()


def _run(
    command: Sequence[str], *, env: Dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )


def _compose_command(args: argparse.Namespace, *parts: str) -> List[str]:
    return [
        "docker",
        "compose",
        "-f",
        args.compose_file,
        "-p",
        args.compose_project_name,
        *parts,
    ]


def _restart_recommendation_service(
    args: argparse.Namespace, *, outbox_enabled: bool
) -> None:
    env = dict(os.environ)
    env["RECOMMENDATION_IMPRESSION_LOGGING_ENABLED"] = (
        "true" if outbox_enabled else "false"
    )
    _run(
        _compose_command(
            args,
            "up",
            "-d",
            "--force-recreate",
            "recommendation-service",
        ),
        env=env,
    )
    deadline = time.monotonic() + args.ready_timeout
    readiness_url = f"{args.base_url.rstrip('/')}/readyz"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(readiness_url, timeout=5.0, trust_env=False)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"recommendation stack did not become ready: {readiness_url}")


def _outbox_counts(args: argparse.Namespace, run_id: str) -> tuple[int, int]:
    if not re.fullmatch(r"[a-zA-Z0-9-]+", run_id):
        raise ValueError("run_id must contain only letters, digits, and hyphens")
    sql = (
        "SELECT count(*) FILTER (WHERE published_at IS NULL), "
        "count(*) FILTER (WHERE published_at IS NOT NULL) "
        "FROM recommendation_event_outbox "
        f"WHERE event_payload->'metadata'->>'session_id' = '{run_id}'"
    )
    completed = _run(
        _compose_command(
            args,
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            os.environ.get("POSTGRES_USER", "video_commerce"),
            "-d",
            os.environ.get("POSTGRES_DB", "video_commerce"),
            "-At",
            "-F",
            "|",
            "-c",
            sql,
        )
    )
    pending, published = completed.stdout.strip().split("|", 1)
    return int(pending), int(published)


def _wait_for_outbox_drain(args: argparse.Namespace, run_id: str) -> tuple[int, int]:
    deadline = time.monotonic() + args.drain_timeout
    counts = _outbox_counts(args, run_id)
    while counts[0] and time.monotonic() < deadline:
        time.sleep(0.5)
        counts = _outbox_counts(args, run_id)
    return counts


def _resource_snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    """Record enough host/container context to interpret a directional result."""
    host_memory_bytes = None
    if shutil.which("sysctl"):
        completed = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode == 0 and completed.stdout.strip().isdigit():
            host_memory_bytes = int(completed.stdout.strip())
    if host_memory_bytes is None:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_count = os.sysconf("SC_PHYS_PAGES")
            host_memory_bytes = int(page_size * page_count)
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    container = None
    try:
        service_id = _run(
            _compose_command(args, "ps", "-q", "recommendation-service")
        ).stdout.strip()
        if service_id:
            raw = _run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{json .HostConfig}}",
                    service_id,
                ]
            ).stdout.strip()
            host_config = json.loads(raw)
            container = {
                "cpu_quota": host_config.get("CpuQuota"),
                "cpu_period": host_config.get("CpuPeriod"),
                "nano_cpus": host_config.get("NanoCpus"),
                "cpuset_cpus": host_config.get("CpusetCpus"),
                "memory_bytes": host_config.get("Memory"),
                "memory_swap_bytes": host_config.get("MemorySwap"),
            }
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        container = None

    return {
        "host_cpu_count": os.cpu_count(),
        "host_memory_bytes": host_memory_bytes,
        "recommendation_container": container,
    }


def _run_baseline(
    args: argparse.Namespace,
    *,
    mode: str,
    run_id: str,
    output_path: Path,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        "scripts/loadtest_api_baseline.py",
        "--base-url",
        args.base_url,
        "--requests",
        str(args.requests),
        "--concurrency",
        str(args.concurrency),
        "--mode",
        mode,
        "--timeout",
        str(args.timeout),
        "--warmup-requests",
        str(args.warmup_requests),
        "--run-id",
        run_id,
        "--output",
        str(output_path),
    ]
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    _run(command)
    return json.loads(output_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.environment_class == "fixed-linux" and not args.artifact_manifest:
        raise SystemExit("--artifact-manifest is required for fixed-linux results")
    if args.artifact_manifest and not args.artifact_manifest.is_file():
        raise SystemExit(f"artifact manifest does not exist: {args.artifact_manifest}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        mode: {"control": [], "treatment": []} for mode in ("hot", "unique")
    }

    for pair_index, pair_order in enumerate(ORDER, start=1):
        for outbox_enabled in pair_order:
            variant = "treatment" if outbox_enabled else "control"
            _restart_recommendation_service(args, outbox_enabled=outbox_enabled)
            for mode in ("hot", "unique"):
                run_id = f"pair-{pair_index}-{variant}-{mode}"
                output_path = output_dir / f"{run_id}.json"
                summary = _run_baseline(
                    args,
                    mode=mode,
                    run_id=run_id,
                    output_path=output_path,
                )
                pending, published = _wait_for_outbox_drain(args, run_id)
                summary["outbox_pending_count"] = pending
                summary["outbox_published_count"] = published
                output_path.write_text(
                    json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
                )
                runs[mode][variant].append(summary)

    evaluations = {
        mode: evaluate_mode(mode, values["control"], values["treatment"])
        for mode, values in runs.items()
    }
    report = {
        "git_commit": _run(["git", "rev-parse", "HEAD"]).stdout.strip(),
        "environment_class": args.environment_class,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            **_resource_snapshot(args),
        },
        "artifact_manifest": (
            {
                "path": str(args.artifact_manifest),
                "sha256": sha256_file(args.artifact_manifest),
            }
            if args.artifact_manifest
            else None
        ),
        "compose_project_name": args.compose_project_name,
        "requests_per_run": args.requests,
        "concurrency": args.concurrency,
        "warmup_requests": args.warmup_requests,
        "settings": {
            "timeout_seconds": args.timeout,
            "drain_timeout_seconds": args.drain_timeout,
            "ready_timeout_seconds": args.ready_timeout,
            "run_order": [
                ["treatment" if enabled else "control" for enabled in pair]
                for pair in ORDER
            ],
        },
        "runs": runs,
        "evaluations": evaluations,
        "passed": all(result["passed"] for result in evaluations.values()),
    }
    report_path = output_dir / "summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
