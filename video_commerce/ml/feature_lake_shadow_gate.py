"""Seven-day feature-lake shadow cutover gate evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Tuple


@dataclass(frozen=True)
class ShadowGateResult:
    allowed: bool
    consecutive_passing_days: int
    failures: Tuple[str, ...]


def _daily_failures(report: Mapping[str, Any]) -> Tuple[str, ...]:
    checks = {
        "pit_leakage_rows": int(report.get("pit_leakage_rows", -1)) == 0,
        "duplicate_source_event_ids": int(report.get("duplicate_source_event_ids", -1))
        == 0,
        "manifest_validation_ratio": float(report.get("manifest_validation_ratio", -1))
        == 1.0,
        "event_reconciliation_ratio": float(
            report.get("event_reconciliation_ratio", -1)
        )
        == 1.0,
        "online_offline_parity_ratio": float(
            report.get("online_offline_parity_ratio", -1)
        )
        >= 0.999,
        "assembler_vector_parity_ratio": float(
            report.get("assembler_vector_parity_ratio", -1)
        )
        == 1.0,
        "label_reconciliation_ratio": float(
            report.get("label_reconciliation_ratio", -1)
        )
        == 1.0,
        "pit_current_state_calls": int(report.get("pit_current_state_calls", -1)) == 0,
        "invalid_feature_or_label_rows": int(
            report.get("invalid_feature_or_label_rows", -1)
        )
        == 0,
        "serving_p95_regression_ratio": float(
            report.get("serving_p95_regression_ratio", float("inf"))
        )
        <= 0.05,
        "serving_throughput_ratio": float(report.get("serving_throughput_ratio", -1))
        >= 0.95,
        "raw_materialization_lag_p99_seconds": float(
            report.get("raw_materialization_lag_p99_seconds", float("inf"))
        )
        < 120.0,
        "window_materialization_lag_p99_seconds": float(
            report.get("window_materialization_lag_p99_seconds", float("inf"))
        )
        < 900.0,
        "catalog_outbox_oldest_pending_seconds": float(
            report.get("catalog_outbox_oldest_pending_seconds", float("inf"))
        )
        < 300.0,
        "catalog_outbox_backlog_growing": not bool(
            report.get("catalog_outbox_backlog_growing", True)
        ),
        "pit_coverage_delta_percentage_points": float(
            report.get("pit_coverage_delta_percentage_points", float("-inf"))
        )
        >= -0.1,
        "shadow_training_success_ratio": float(
            report.get("shadow_training_success_ratio", -1)
        )
        == 1.0,
        "artifact_persist_success_ratio": float(
            report.get("artifact_persist_success_ratio", -1)
        )
        == 1.0,
    }
    return tuple(name for name, passed in checks.items() if not passed)


def evaluate_shadow_gate(
    reports: Iterable[Mapping[str, Any]], *, as_of_date: date | None = None
) -> ShadowGateResult:
    as_of = as_of_date or datetime.now(timezone.utc).date()
    by_date = {}
    for report in reports:
        report_day = date.fromisoformat(str(report["date"]))
        if report_day in by_date:
            return ShadowGateResult(False, 0, ("duplicate_shadow_report_date",))
        by_date[report_day] = report
    ordered = sorted(by_date.items(), key=lambda item: item[0])
    consecutive = 0
    previous_day = None
    latest_failures: Tuple[str, ...] = ("missing_shadow_reports",)
    for report_day, report in ordered:
        failures = _daily_failures(report)
        if previous_day is None or report_day != previous_day + timedelta(days=1):
            consecutive = 0
        if failures:
            consecutive = 0
        else:
            consecutive += 1
        previous_day = report_day
        latest_failures = failures
    if consecutive < 7 and not latest_failures:
        latest_failures = ("seven_consecutive_passing_days",)
    if not ordered or ordered[-1][0] != as_of:
        latest_failures = ("latest_shadow_report_is_stale",)
        consecutive = 0
    return ShadowGateResult(
        allowed=consecutive >= 7 and ordered[-1][0] == as_of,
        consecutive_passing_days=consecutive,
        failures=latest_failures,
    )
