import pytest

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


def test_materializer_topics_use_history_snapshots_not_legacy_feature_updates(
    monkeypatch,
):
    # Python services set KAFKA_FEATURE_UPDATES_TOPIC for the legacy topic, which
    # the materializer never consumes; it must not leak into the lag check.
    monkeypatch.setenv("KAFKA_FEATURE_UPDATES_TOPIC", "feature-updates")
    monkeypatch.delenv("KAFKA_FEATURE_HISTORY_SNAPSHOTS_TOPIC", raising=False)

    topics = feature_lake_readiness._materializer_topics()

    assert "feature-history-snapshots" in topics
    assert "feature-updates" not in topics


async def _probe_with(statuses, offsets):
    calls = {"status": [], "offsets": []}

    def status_reader(rest_url, job_name):
        calls["status"].append((rest_url, job_name))
        return statuses.pop(0)

    async def offsets_reader(bootstrap, group_id, topics):
        calls["offsets"].append((bootstrap, group_id, tuple(topics)))
        return offsets.pop(0)

    probe = feature_lake_readiness.MaterializerCatchUpProbe(
        flink_rest_url="flink-jobmanager:8081",
        bootstrap_servers="kafka:9092",
        job_name="materializer",
        group_id="group-1",
        topics=["user-interactions"],
        status_reader=status_reader,
        offsets_reader=offsets_reader,
    )
    return probe, calls


@pytest.mark.asyncio
async def test_catchup_probe_compares_committed_offsets_to_first_end_sample():
    cutoff_ts = 1_000.0
    probe, calls = await _probe_with(
        statuses=[("RUNNING", 1_000_500), ("RUNNING", 1_000_900)],
        offsets=[
            ({"user-interactions:0": 5}, {"user-interactions:0": 10}),
            # Traffic keeps arriving (end 15), but the run only needs the first sample.
            ({"user-interactions:0": 10}, {"user-interactions:0": 15}),
        ],
    )

    first = await probe.reasons(run_id="pit-1", cutoff_ts=cutoff_ts)
    second = await probe.reasons(run_id="pit-1", cutoff_ts=cutoff_ts)

    assert first == ["consumer_lag:user-interactions:0:5"]
    assert second == []
    assert calls["status"][0] == ("http://flink-jobmanager:8081", "materializer")
    assert calls["offsets"][0] == ("kafka:9092", "group-1", ("user-interactions",))


@pytest.mark.asyncio
async def test_catchup_probe_requires_checkpoint_after_cutoff():
    probe, _ = await _probe_with(
        statuses=[("RUNNING", 999_999)],
        offsets=[({"user-interactions:0": 10}, {"user-interactions:0": 10})],
    )

    reasons = await probe.reasons(run_id="pit-1", cutoff_ts=1_000.0)

    assert reasons == ["no_completed_checkpoint_after_gate_start"]
