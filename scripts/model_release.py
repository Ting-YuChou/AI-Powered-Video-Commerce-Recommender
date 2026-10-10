#!/usr/bin/env python3
"""Operate the durable ranking release lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from video_commerce.common.config import Config
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.ml.model_release import ModelReleaseController
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
    return parser


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
