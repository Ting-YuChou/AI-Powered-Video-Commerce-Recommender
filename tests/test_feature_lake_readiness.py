from video_commerce.ml import feature_lake_readiness
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


def test_readiness_allows_uncommitted_partitions_when_topics_are_empty():
    assert (
        readiness_reasons(
            job_state="RUNNING",
            completed_checkpoint_at_ms=2_000,
            gate_started_at_ms=1_000,
            committed_offsets={},
            end_offsets={"topic:0": 0, "topic:1": 0},
        )
        == []
    )


def test_readiness_rejects_uncommitted_partitions_when_topic_has_data():
    reasons = readiness_reasons(
        job_state="RUNNING",
        completed_checkpoint_at_ms=2_000,
        gate_started_at_ms=1_000,
        committed_offsets={},
        end_offsets={"topic:0": 1},
    )

    assert reasons == ["consumer_lag:topic:0:1"]


def test_flink_status_ignores_terminal_same_name_job_history(monkeypatch):
    def fake_read_json(url):
        if url.endswith("/jobs/overview"):
            return {
                "jobs": [
                    {
                        "jid": "old-job",
                        "name": "materializer",
                        "state": "CANCELED",
                    },
                    {
                        "jid": "live-job",
                        "name": "materializer",
                        "state": "RUNNING",
                    },
                ]
            }
        assert url.endswith("/jobs/live-job/checkpoints")
        return {"latest": {"completed": {"latest_ack_timestamp": 2_000}}}

    monkeypatch.setattr(feature_lake_readiness, "_read_json", fake_read_json)

    assert feature_lake_readiness._flink_status(
        "http://flink:8081", "materializer"
    ) == ("RUNNING", 2_000)
