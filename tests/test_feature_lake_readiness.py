from video_commerce.ml.feature_lake_readiness import readiness_reasons


def test_readiness_requires_running_fresh_checkpoint_and_zero_lag():
    assert (
        readiness_reasons(
            job_state="RUNNING",
            completed_checkpoint_at_ms=2_000,
            gate_started_at_ms=1_000,
            committed_offsets={"topic:0": 10},
            end_offsets={"topic:0": 10},
        )
        == []
    )


def test_readiness_reports_each_unmet_boundary():
    reasons = readiness_reasons(
        job_state="FAILED",
        completed_checkpoint_at_ms=999,
        gate_started_at_ms=1_000,
        committed_offsets={"topic:0": 8},
        end_offsets={"topic:0": 10},
    )

    assert "materializer_not_running" in reasons
    assert "no_completed_checkpoint_after_gate_start" in reasons
    assert "consumer_lag:topic:0:2" in reasons
