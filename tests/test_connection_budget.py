"""Postgres connection budget for the Helm chart at HPA max.

Every uvicorn worker process owns its own SQLAlchemy pool, so the worst case is
``pods * processes * (pool_size + max_overflow)`` summed over every workload
that opens a ``SystemStore``. Pods are counted at HPA ``maxReplicas`` (or
``replicaCount`` without an HPA) plus the rolling-update surge, and optional
workloads are counted as if enabled so turning one on cannot silently break
the budget.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
VALUES_PATH = REPO_ROOT / "charts/video-commerce/values.yaml"

# Every backend workload in the chart opens a SystemStore (DATABASE_ENABLE is
# set chart-wide). Listed explicitly so a new workload must be classified.
DATABASE_WORKLOADS = {
    "gatewayApi",
    "recommendationService",
    "rankingService",
    "rankingCoordinator",
    "rankingRunner",
    "interactionIngestService",
    "contentWorker",
    "contentTaskPublisher",
    "featureWorker",
    "catalogEventPublisher",
    "pitStateExporter",
    "modelTrainer",
}
DATABASE_CRONJOBS = ("pitOrchestrator", "icebergMaintenance")
_WORKERS_ARG = re.compile(r"--workers\s+\$\{(\w+):-(\d+)\}")
_DEFAULT_MAX_SURGE = "25%"


def _values() -> Dict[str, Any]:
    return yaml.safe_load(VALUES_PATH.read_text(encoding="utf-8"))


def _processes(workload: Dict[str, Any]) -> int:
    env = workload.get("env") or {}
    for arg in workload.get("args") or []:
        match = _WORKERS_ARG.search(str(arg))
        if match:
            return int(env.get(match.group(1), match.group(2)))
    return 1


def _surge(replicas: int, workload: Dict[str, Any]) -> int:
    max_surge = (workload.get("strategy") or {}).get("maxSurge", _DEFAULT_MAX_SURGE)
    if isinstance(max_surge, str) and max_surge.endswith("%"):
        return math.ceil(replicas * int(max_surge[:-1]) / 100)
    return int(max_surge)


def _per_process(env: Dict[str, Any], database: Dict[str, Any]) -> int:
    pool_size = int(env.get("DATABASE_POOL_SIZE", database["poolSize"]))
    max_overflow = int(env.get("DATABASE_MAX_OVERFLOW", database["maxOverflow"]))
    return pool_size + max_overflow


def connection_budget_rows(values: Dict[str, Any]) -> List[Tuple[str, int, int, int]]:
    """Return (name, pods, processes_per_pod, connections_per_process) rows."""
    database = values["external"]["database"]
    rows: List[Tuple[str, int, int, int]] = []
    for key, workload in values["backend"]["workloads"].items():
        if key not in DATABASE_WORKLOADS:
            continue
        hpa = workload.get("hpa") or {}
        replicas = int(
            hpa["maxReplicas"] if hpa.get("enabled") else workload["replicaCount"]
        )
        pods = replicas + _surge(replicas, workload)
        rows.append(
            (
                workload["name"],
                pods,
                _processes(workload),
                _per_process(workload.get("env") or {}, database),
            )
        )
    for key in DATABASE_CRONJOBS:
        cronjob = values[key]
        # concurrencyPolicy Forbid keeps one Job pod per schedule.
        assert cronjob["concurrencyPolicy"] == "Forbid"
        rows.append((key, 1, 1, _per_process(cronjob.get("env") or {}, database)))
    return rows


def test_every_backend_workload_is_classified_for_the_budget():
    workloads = set(_values()["backend"]["workloads"])

    assert workloads == DATABASE_WORKLOADS


def test_hpa_max_connections_fit_postgres_budget():
    values = _values()
    rows = connection_budget_rows(values)
    total = sum(pods * processes * per for _, pods, processes, per in rows)
    budget = int(values["postgres"]["maxConnectionsBudget"])
    reserved = int(values["postgres"]["reservedConnections"])
    table = "\n".join(
        f"{name:28s} pods={pods:3d} procs={processes} conns/proc={per} "
        f"total={pods * processes * per}"
        for name, pods, processes, per in rows
    )

    assert total + reserved <= budget, (
        f"HPA-max Postgres connections {total} + reserved {reserved} exceed "
        f"postgres.maxConnectionsBudget {budget}:\n{table}"
    )


def test_hot_path_services_keep_a_small_bounded_pool():
    values = _values()
    database = values["external"]["database"]
    workloads = values["backend"]["workloads"]

    for key in ("gatewayApi", "recommendationService", "interactionIngestService"):
        env = workloads[key].get("env") or {}
        assert 1 <= _per_process(env, database) <= 4, key
    assert float(database["poolTimeoutSeconds"]) <= 5
    assert int(database["hotPathStatementTimeoutMs"]) > 0
    for key in ("rankingRunner", "rankingCoordinator", "rankingService"):
        env = workloads[key]["env"]
        assert _per_process(env, database) == 1, key


def test_offline_database_clients_get_a_longer_pool_timeout():
    values = _values()

    assert values["pitOrchestrator"]["env"]["DATABASE_POOL_TIMEOUT_SECONDS"] == "30"
    assert values["icebergMaintenance"]["env"]["DATABASE_POOL_TIMEOUT_SECONDS"] == "30"
    trainer_env = values["backend"]["workloads"]["modelTrainer"]["env"]
    assert trainer_env["DATABASE_POOL_TIMEOUT_SECONDS"] == "30"


def test_budget_math_counts_surge_and_workers():
    values = {
        "external": {"database": {"poolSize": 2, "maxOverflow": 1}},
        "backend": {
            "workloads": {
                "gatewayApi": {
                    "name": "gateway-api",
                    "replicaCount": 2,
                    "args": ["uvicorn x --workers ${SERVICE_GATEWAY_WORKERS:-4}"],
                    "env": {"SERVICE_GATEWAY_WORKERS": "3"},
                    "hpa": {"enabled": True, "maxReplicas": 8},
                },
                "rankingRunner": {
                    "name": "ranking-runner",
                    "replicaCount": 4,
                    "strategy": {"maxSurge": 1},
                    "env": {"DATABASE_POOL_SIZE": "1", "DATABASE_MAX_OVERFLOW": "0"},
                    "hpa": {"enabled": True, "maxReplicas": 32},
                },
            }
        },
        "pitOrchestrator": {"concurrencyPolicy": "Forbid", "env": {}},
        "icebergMaintenance": {"concurrencyPolicy": "Forbid", "env": {}},
    }

    rows = {name: row for name, *row in connection_budget_rows(values)}

    assert rows["gateway-api"] == [10, 3, 3]
    assert rows["ranking-runner"] == [33, 1, 1]
    assert rows["pitOrchestrator"] == [1, 1, 3]
