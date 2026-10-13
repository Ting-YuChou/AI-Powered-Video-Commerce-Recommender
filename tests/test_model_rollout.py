from __future__ import annotations

from datetime import datetime, timedelta, timezone

from video_commerce.ml.model_rollout import (
    DesiredGeneration,
    RolloutObservation,
    evaluate_rollout,
)


def _observation(service: str, instance: str, *, generation: int = 4, state="ready"):
    return RolloutObservation(
        model_name="ranking_model",
        environment="production",
        service_name=service,
        instance_id=instance,
        desired_release_id="release-4",
        desired_generation=4,
        observed_release_id="release-4",
        observed_generation=generation,
        observed_model_version="v4",
        bundle_digest="d" * 64,
        state=state,
        heartbeat_at=datetime.now(timezone.utc),
    )


def test_rollout_requires_expected_fresh_instances_and_matching_digest():
    desired = DesiredGeneration(
        release_id="release-4",
        generation=4,
        model_version="v4",
        bundle_digest="d" * 64,
    )
    observations = [
        _observation("ranking-runner", "runner-1"),
        _observation("ranking-runner", "runner-2"),
        _observation("ranking-service", "service-1"),
    ]

    result = evaluate_rollout(
        desired,
        observations,
        expected_service_counts={"ranking-runner": 2, "ranking-service": 1},
        heartbeat_max_age=timedelta(seconds=90),
        now=datetime.now(timezone.utc),
    )

    assert result.ready is True
    assert result.failures == ()


def test_rollout_rejects_stale_failed_and_old_generation_instances():
    now = datetime.now(timezone.utc)
    desired = DesiredGeneration("release-4", 4, "v4", "d" * 64)
    stale = _observation("ranking-runner", "runner-stale")
    stale = RolloutObservation(
        **{
            **stale.__dict__,
            "heartbeat_at": now - timedelta(minutes=10),
        }
    )
    observations = [
        stale,
        _observation("ranking-runner", "runner-old", generation=3),
        _observation("ranking-service", "service-failed", state="failed"),
    ]

    result = evaluate_rollout(
        desired,
        observations,
        expected_service_counts={"ranking-runner": 2, "ranking-service": 1},
        heartbeat_max_age=timedelta(seconds=90),
        now=now,
    )

    assert result.ready is False
    assert {failure.reason for failure in result.failures} >= {
        "stale_heartbeat",
        "generation_mismatch",
        "instance_failed",
        "insufficient_ready_instances",
    }
