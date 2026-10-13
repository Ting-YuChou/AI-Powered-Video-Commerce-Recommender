#!/usr/bin/env python3
"""Operate the durable ranking release lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone
import time
from typing import Any

from video_commerce.common.config import Config
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.ml.model_release import ModelReleaseController
from video_commerce.ml.model_rollout import (
    DesiredGeneration,
    RolloutObservation,
    evaluate_rollout,
)
from video_commerce.ml.ranking import RankingModel


MODEL_NAMES = (
    ModelArtifactManager.RANKING_MODEL_NAME,
    ModelArtifactManager.TWO_TOWER_MODEL_NAME,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-release")
    parser.add_argument(
        "--model-name",
        choices=MODEL_NAMES,
        default=ModelArtifactManager.RANKING_MODEL_NAME,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--json", action="store_true", dest="as_json")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--version", required=True)
    for command in ("promote", "rollback"):
        child = subparsers.add_parser(command)
        child.add_argument("--version", required=True)
        child.add_argument("--expected-generation", type=int, required=True)
        child.add_argument("--actor", required=True)
        child.add_argument("--reason", required=True)
    bootstrap = subparsers.add_parser("bootstrap-active")
    bootstrap.add_argument("--version", required=True)
    bootstrap.add_argument("--actor", required=True)
    bootstrap.add_argument("--reason", required=True)
    rollout = subparsers.add_parser("rollout-status")
    rollout.add_argument("--generation", type=int, required=True)
    rollout.add_argument("--json", action="store_true", dest="as_json")
    wait = subparsers.add_parser("wait-rollout")
    wait.add_argument("--generation", type=int, required=True)
    wait.add_argument(
        "--expected-service-count", action="append", default=[], required=True
    )
    wait.add_argument("--timeout", type=int, default=300)
    return parser


def _expected_service_counts(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        service, separator, count = str(value).partition("=")
        if not separator or not service.strip() or int(count) < 1:
            raise ValueError(
                "expected service count must use service-name=positive-count"
            )
        result[service.strip()] = int(count)
    return result


async def _rollout_status(
    store: SystemStore,
    *,
    model_name: str,
    environment: str,
    generation: int,
    expected_counts: dict[str, int],
) -> dict[str, Any]:
    active = await store.get_active_model_checkpoint(
        model_name, environment=environment
    )
    if active is None or int(active.get("generation") or 0) != int(generation):
        raise RuntimeError("requested generation is not the current active pointer")
    rows = await store.list_model_release_instance_observations(
        model_name=model_name, environment=environment
    )
    desired_digest = str((active.get("payload") or {}).get("bundle_digest") or "")
    if not desired_digest:
        desired_digest = next(
            (
                str(row["desired_bundle_digest"])
                for row in rows
                if row.get("desired_generation") == int(generation)
                and row.get("desired_bundle_digest")
            ),
            "",
        )
    desired = DesiredGeneration(
        release_id=str(active.get("release_id") or ""),
        generation=int(generation),
        model_version=str(active.get("model_version") or ""),
        bundle_digest=desired_digest,
    )
    observations = [
        RolloutObservation(
            model_name=str(row["model_name"]),
            environment=str(row["environment"]),
            service_name=str(row["service_name"]),
            instance_id=str(row["instance_id"]),
            desired_release_id=str(row.get("desired_release_id") or ""),
            desired_generation=int(row.get("desired_generation") or 0),
            observed_release_id=row.get("observed_release_id"),
            observed_generation=row.get("observed_generation"),
            observed_model_version=row.get("observed_model_version"),
            bundle_digest=row.get("bundle_digest"),
            state=str(row["state"]),
            heartbeat_at=datetime.fromtimestamp(
                float(row["heartbeat_at"]), tz=timezone.utc
            ),
            last_error=row.get("last_error"),
        )
        for row in rows
    ]
    evaluation = evaluate_rollout(
        desired,
        observations,
        expected_service_counts=expected_counts,
        heartbeat_max_age=timedelta(seconds=90),
    )
    return {
        "model_name": model_name,
        "environment": environment,
        "desired": desired.__dict__,
        "ready": evaluation.ready,
        "ready_counts": dict(evaluation.ready_counts),
        "failures": [failure.__dict__ for failure in evaluation.failures],
        "observations": rows,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    config = Config()
    store = SystemStore(config.database_config)
    await store.initialize()
    try:
        environment = config.model_release_config.environment
        model_name = str(args.model_name)
        if model_name == ModelArtifactManager.TWO_TOWER_MODEL_NAME:
            environment = config.recommendation_config.retrieval_release_environment
        if args.command == "status":
            return {
                "environment": environment,
                "active": await store.get_model_release_pointer(
                    model_name, environment=environment, slot="active"
                ),
                "staging": await store.get_model_release_pointer(
                    model_name, environment=environment, slot="staging"
                ),
                "releases": await store.list_model_releases(model_name),
            }
        if args.command in {"rollout-status", "wait-rollout"}:
            expected_counts = (
                _expected_service_counts(args.expected_service_count)
                if args.command == "wait-rollout"
                else {}
            )
            deadline = time.monotonic() + int(getattr(args, "timeout", 0))
            while True:
                result = await _rollout_status(
                    store,
                    model_name=model_name,
                    environment=environment,
                    generation=args.generation,
                    expected_counts=expected_counts,
                )
                if args.command == "rollout-status" or result["ready"]:
                    return result
                if time.monotonic() >= deadline:
                    return {**result, "timed_out": True}
                await asyncio.sleep(2)
        release = await store.get_model_release(model_name, args.version)
        if release is None:
            raise RuntimeError(f"{model_name} release {args.version!r} does not exist")
        if args.command == "evaluate":
            return {
                "model_version": args.version,
                "release_id": release["release_id"],
                "lifecycle_state": release["lifecycle_state"],
                "validation_status": release["validation_status"],
                "note": (
                    "PIT evaluation runs automatically in model-trainer; this command "
                    "reports its durable decision"
                ),
            }
        controller = ModelReleaseController(store)
        if args.command == "bootstrap-active":
            object_storage = ObjectStorage(config.object_storage_config)
            await object_storage.initialize()
            manager = ModelArtifactManager(
                system_store=store,
                object_storage=object_storage,
                model_config=config.model_config,
                recommendation_config=config.recommendation_config,
            )
            if model_name == ModelArtifactManager.RANKING_MODEL_NAME:
                model = RankingModel(config.ranking_config)
                verified = await manager.sync_ranking_checkpoint_version(
                    args.version,
                    expected_feature_schema_version=model.feature_schema_version,
                    require_onnx=config.ranking_config.triton_enabled,
                )
            else:
                verified = await manager.sync_two_tower_checkpoint_version(args.version)
            if verified is None:
                raise RuntimeError("bootstrap artifact is missing or incompatible")
            return await controller.bootstrap_active(
                model_name=model_name,
                model_version=args.version,
                environment=environment,
                actor=args.actor,
                reason=args.reason,
            )
        operation = (
            controller.promote if args.command == "promote" else controller.rollback
        )
        return await operation(
            model_name=model_name,
            model_version=args.version,
            environment=environment,
            expected_generation=args.expected_generation,
            actor=args.actor,
            reason=args.reason,
        )
    finally:
        await store.close()


def main() -> int:
    args = build_parser().parse_args()
    result = asyncio.run(_run(args))
    if getattr(args, "as_json", False):
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
