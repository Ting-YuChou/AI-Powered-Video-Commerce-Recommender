"""Pure rollout convergence contracts shared by services and the release CLI."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping


@dataclass(frozen=True)
class DesiredGeneration:
    release_id: str
    generation: int
    model_version: str
    bundle_digest: str


@dataclass(frozen=True)
class RolloutObservation:
    model_name: str
    environment: str
    service_name: str
    instance_id: str
    desired_release_id: str
    desired_generation: int
    observed_release_id: str | None
    observed_generation: int | None
    observed_model_version: str | None
    bundle_digest: str | None
    state: str
    heartbeat_at: datetime
    last_error: str | None = None


@dataclass(frozen=True)
class RolloutFailure:
    reason: str
    service_name: str | None = None
    instance_id: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class RolloutEvaluation:
    ready: bool
    ready_counts: Mapping[str, int]
    failures: tuple[RolloutFailure, ...]


def evaluate_rollout(
    desired: DesiredGeneration,
    observations: Iterable[RolloutObservation],
    *,
    expected_service_counts: Mapping[str, int],
    heartbeat_max_age: timedelta,
    now: datetime | None = None,
) -> RolloutEvaluation:
    evaluated_at = now or datetime.now(timezone.utc)
    ready_counts: Counter[str] = Counter()
    failures: list[RolloutFailure] = []
    for observation in observations:
        age = evaluated_at - observation.heartbeat_at
        common = {
            "service_name": observation.service_name,
            "instance_id": observation.instance_id,
        }
        if age > heartbeat_max_age:
            failures.append(RolloutFailure("stale_heartbeat", **common))
            continue
        if observation.state == "failed":
            failures.append(
                RolloutFailure(
                    "instance_failed", detail=observation.last_error, **common
                )
            )
            continue
        if observation.state != "ready":
            failures.append(RolloutFailure("instance_not_ready", **common))
            continue
        if (
            observation.observed_release_id != desired.release_id
            or observation.observed_generation != desired.generation
        ):
            failures.append(RolloutFailure("generation_mismatch", **common))
            continue
        if observation.observed_model_version != desired.model_version:
            failures.append(RolloutFailure("model_version_mismatch", **common))
            continue
        if observation.bundle_digest != desired.bundle_digest:
            failures.append(RolloutFailure("bundle_digest_mismatch", **common))
            continue
        ready_counts[observation.service_name] += 1

    for service_name, expected in sorted(expected_service_counts.items()):
        actual = int(ready_counts.get(service_name, 0))
        if actual < int(expected):
            failures.append(
                RolloutFailure(
                    "insufficient_ready_instances",
                    service_name=service_name,
                    detail=f"expected={int(expected)} actual={actual}",
                )
            )
    return RolloutEvaluation(
        ready=not failures,
        ready_counts=dict(ready_counts),
        failures=tuple(failures),
    )
