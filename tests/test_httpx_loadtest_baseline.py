from argparse import Namespace

from scripts.loadtest_api_baseline import RequestResult, build_payload, summarize


def _args(**overrides):
    values = {
        "base_url": "http://example.test",
        "mode": "hot",
        "requests": 4,
        "concurrency": 2,
        "timeout": 10.0,
        "run_id": "pair-1-on",
        "warmup_requests": 0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_build_payload_uses_run_id_to_isolate_hot_and_unique_users():
    hot = build_payload(7, "hot", run_id="pair-1-on")
    unique = build_payload(7, "unique", run_id="pair-1-on")

    assert hot["user_id"] == "loadtest-pair-1-on-hot-user"
    assert unique["user_id"] == "loadtest-pair-1-on-user-7"
    assert hot["context"]["run_id"] == "pair-1-on"
    assert hot["context"]["session_id"] == "pair-1-on"


def test_build_payload_can_warm_the_hot_user_under_a_separate_correlation_id():
    warmup = build_payload(
        0,
        "hot",
        run_id="pair-1-on-warmup",
        user_run_id="pair-1-on",
    )

    assert warmup["user_id"] == "loadtest-pair-1-on-hot-user"
    assert warmup["context"]["session_id"] == "pair-1-on-warmup"


def test_summarize_reports_success_latency_cache_and_tracking_coverage():
    results = [
        RequestResult(200, 10.0, True, True, "durable", "imp-1"),
        RequestResult(200, 20.0, True, False, "durable", "imp-2"),
        RequestResult(503, 90.0, False, None, None, None),
        RequestResult(0, 100.0, False, None, None, None),
    ]

    summary = summarize(results, _args(), elapsed_seconds=2.0)

    assert summary["success_rate"] == 0.5
    assert summary["error_rate"] == 0.5
    assert summary["server_error_rate"] == 0.5
    assert summary["five_xx_rate"] == 0.25
    assert summary["transport_error_rate"] == 0.25
    assert summary["successful_p50_ms"] == 10.0
    assert summary["successful_p95_ms"] == 20.0
    assert summary["successful_p99_ms"] == 20.0
    assert summary["cache_hit_rate"] == 0.5
    assert summary["durable_tracking_rate"] == 1.0
    assert summary["durable_impression_count"] == 2
    assert summary["unique_durable_impression_count"] == 2


def test_summarize_handles_successes_without_tracking_metadata():
    results = [
        RequestResult(200, 12.0, True, False, None, None),
        RequestResult(200, 18.0, True, False, None, None),
    ]

    summary = summarize(results, _args(requests=2), elapsed_seconds=1.0)

    assert summary["tracking_observation_count"] == 0
    assert summary["durable_tracking_rate"] == 0.0
    assert summary["unavailable_tracking_rate"] == 0.0
