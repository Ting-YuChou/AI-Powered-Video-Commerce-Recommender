"""Alert rules must reference metrics that something actually emits."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

from video_commerce.common.observability import ObservabilityManager


ROOT = Path(__file__).resolve().parents[1]

# Emitted by exporters or by Prometheus itself, not by this codebase.
EXTERNAL_METRICS = {"up", "pg_up", "kafka_consumergroup_lag"}
# Flink built-in job/operator metrics; job-specific operator metrics are checked
# against the Flink job sources below.
FLINK_BUILTIN_METRICS = {
    "flink_jobmanager_job_uptime",
    "flink_jobmanager_job_lastCheckpointCompletionTimestamp",
    "flink_taskmanager_job_task_operator_currentInputWatermark",
}
FLINK_OPERATOR_PREFIX = "flink_taskmanager_job_task_operator_"
# Registered by Iceberg's IcebergFilesCommitterMetrics inside the Flink SQL
# Iceberg sink, as addGroup("IcebergFilesCommitter").addGroup("table", name).
# Names were read from iceberg-flink-runtime at ICEBERG_VERSION; re-verify them
# when pom.xml moves.
ICEBERG_VERSION = "1.11.0"
ICEBERG_FLINK_METRICS = {
    FLINK_OPERATOR_PREFIX
    + "IcebergFilesCommitter_table_elapsedSecondsSinceLastSuccessfulCommit",
}

# Alerts on metrics nothing emits yet, kept visible instead of silently dead.
KNOWN_UNEMITTED_METRICS: dict[str, str] = {}

_PROMQL_WORDS = {
    "sum", "min", "max", "avg", "count", "rate", "increase", "irate", "delta",
    "histogram_quantile", "clamp_min", "clamp_max", "absent", "time", "and",
    "or", "unless", "bool", "by", "without", "on", "ignoring", "group_left",
    "group_right", "offset",
}
_HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count")


def _rules():
    parsed = yaml.safe_load((ROOT / "monitoring/prometheus-rules.yml").read_text())
    return [rule for group in parsed["groups"] for rule in group["rules"]]


def _metric_names(expr: str) -> set[str]:
    expr = re.sub(r"\{[^}]*\}", " ", expr)
    expr = re.sub(r"\[[^\]]*\]", " ", expr)
    expr = re.sub(
        r"\b(by|without|on|ignoring|group_left|group_right)\s*\([^)]*\)", " ", expr
    )
    tokens = set(re.findall(r"(?<![\w.])[A-Za-z_:][A-Za-z0-9_:]*", expr))
    return {token for token in tokens if token not in _PROMQL_WORDS}


def _python_metric_names() -> set[str]:
    """First arguments of every Counter/Gauge/Histogram/Summary constructor."""
    names = set()
    for path in (ROOT / "video_commerce").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            kind = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            first = node.args[0]
            if kind in {"Counter", "Gauge", "Histogram", "Summary"} and isinstance(
                first, ast.Constant
            ):
                names.add(first.value)
    return names


def _flink_sources() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "flink-jobs").rglob("*.java")
    )


def _flink_operator_metric_defined(operator_metric: str, sources: str) -> bool:
    # Flink exports getMetricGroup().addGroup("g").counter("c") as g_c.
    groups = re.findall(r'addGroup\("([A-Za-z0-9_]+)"\)', sources)
    counters = re.findall(r'counter\((?:[A-Za-z]+ \+ )?"([A-Za-z0-9_]+)"\)', sources)
    for group in groups:
        if not operator_metric.startswith(group + "_"):
            continue
        rest = operator_metric[len(group) + 1:]
        if any(rest == c or rest.endswith(c if c.startswith("_") else "_" + c) for c in counters):
            return True
    return False


def _defined_in_python(name: str, defined: set[str]) -> bool:
    candidates = [name] + [
        name[: -len(suffix)] for suffix in _HISTOGRAM_SUFFIXES if name.endswith(suffix)
    ]
    return any(candidate in defined for candidate in candidates)


def test_metric_name_extraction():
    assert _metric_names(
        'sum by (job, le) (rate(foo_seconds_bucket{job="x",le!=""}[5m])) > 0.5'
        " and on() max(bar{mode!=\"flink\"}) == 1"
    ) == {"foo_seconds_bucket", "bar"}


def test_every_rule_metric_is_emitted():
    python_sources = _python_metric_names()
    flink_sources = _flink_sources()
    recorded = {rule["record"] for rule in _rules() if "record" in rule}
    unresolved = {}
    referenced = set()
    for rule in _rules():
        for name in _metric_names(rule["expr"]) - recorded:
            referenced.add(name)
            if name in EXTERNAL_METRICS | FLINK_BUILTIN_METRICS | ICEBERG_FLINK_METRICS:
                continue
            if name.startswith(FLINK_OPERATOR_PREFIX):
                operator_metric = name[len(FLINK_OPERATOR_PREFIX):]
                if _flink_operator_metric_defined(operator_metric, flink_sources):
                    continue
                if name in KNOWN_UNEMITTED_METRICS:
                    continue
            elif _defined_in_python(name, python_sources):
                continue
            unresolved[name] = rule.get("alert") or rule.get("record")

    assert unresolved == {}
    for name, alert in KNOWN_UNEMITTED_METRICS.items():
        assert name in referenced, f"{alert} no longer uses {name}; drop the entry"
        assert not _flink_operator_metric_defined(
            name[len(FLINK_OPERATOR_PREFIX):], flink_sources
        ), f"{name} is emitted now; drop it from KNOWN_UNEMITTED_METRICS"


def test_iceberg_metric_names_match_pinned_iceberg_version():
    pom = (ROOT / "flink-jobs/interaction-features/pom.xml").read_text()
    assert f"<iceberg.version>{ICEBERG_VERSION}</iceberg.version>" in pom


def test_flink_job_name_selectors_match_job_sources():
    flink_sources = _flink_sources()
    for rule in _rules():
        for job_name in re.findall(r'job_name="([^"]+)"', rule["expr"]):
            assert f'"{job_name}"' in flink_sources, (rule.get("alert"), job_name)


def test_alerted_label_values_are_precreated():
    manager = ObservabilityManager()
    for attribute in ObservabilityManager.ALERTED_LABEL_VALUES:
        assert hasattr(manager, attribute), attribute

    payload = manager.prometheus_payload().decode("utf-8")
    assert 'video_commerce_rate_limiter_errors_total{reason="cooldown"} 0.0' in payload
    assert (
        'video_commerce_recommendation_impression_events_total{stage="served",status="timeout"} 0.0'
        in payload
    )
    assert 'content_task_publish_retries_total{reason="missing_object"} 0.0' in payload
    for sample in (
        'video_commerce_recommendation_outbox_terminal_total{reason="max_attempts"} 0.0',
        'catalog_outbox_terminal_total{reason="message_too_large"} 0.0',
        'video_commerce_content_processing_redrives_total{outcome="exhausted"} 0.0',
        'video_commerce_ranking_runtime_auth_total{result="unauthenticated",role="coordinator"} 0.0',
    ):
        assert sample in payload, sample
