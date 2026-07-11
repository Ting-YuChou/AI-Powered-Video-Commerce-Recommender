from datetime import date

from video_commerce.ml.feature_lake_shadow_gate import evaluate_shadow_gate


def _passing(day):
    return {
        "date": f"2026-07-{day:02d}",
        "pit_leakage_rows": 0,
        "duplicate_source_event_ids": 0,
        "manifest_validation_ratio": 1.0,
        "event_reconciliation_ratio": 1.0,
        "online_offline_parity_ratio": 0.9995,
        "raw_materialization_lag_p99_seconds": 100.0,
        "window_materialization_lag_p99_seconds": 800.0,
        "catalog_outbox_oldest_pending_seconds": 100.0,
        "catalog_outbox_backlog_growing": False,
        "pit_coverage_delta_percentage_points": -0.05,
        "shadow_training_success_ratio": 1.0,
        "artifact_persist_success_ratio": 1.0,
    }


def test_shadow_gate_requires_seven_consecutive_passing_days():
    result = evaluate_shadow_gate(
        [_passing(day) for day in range(1, 8)], as_of_date=date(2026, 7, 7)
    )

    assert result.allowed is True
    assert result.consecutive_passing_days == 7


def test_shadow_gate_fails_on_leakage_or_coverage_regression():
    reports = [_passing(day) for day in range(1, 8)]
    reports[-1]["pit_leakage_rows"] = 1
    reports[-1]["pit_coverage_delta_percentage_points"] = -0.2

    result = evaluate_shadow_gate(reports, as_of_date=date(2026, 7, 7))

    assert result.allowed is False
    assert "pit_leakage_rows" in result.failures
    assert "pit_coverage_delta_percentage_points" in result.failures


def test_shadow_gate_rejects_stale_or_duplicate_daily_reports():
    reports = [_passing(day) for day in range(1, 8)]
    assert evaluate_shadow_gate(reports, as_of_date=date(2026, 7, 8)).failures == (
        "latest_shadow_report_is_stale",
    )
    assert evaluate_shadow_gate(
        [*reports, _passing(7)], as_of_date=date(2026, 7, 7)
    ).failures == ("duplicate_shadow_report_date",)
