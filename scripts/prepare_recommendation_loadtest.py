#!/usr/bin/env python3
"""Prepare a verified synthetic ranker and deterministic serving catalog.

Run this inside a backend Compose container before starting ranking runners.
The checkpoint executes the production ranking architecture with deterministic
random weights.  It is valid for capacity testing only, never quality claims.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_commerce.common.config import Config
from video_commerce.data_plane.feature_store import FeatureStore
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.loadtest_support import (
    build_synthetic_catalog,
    create_synthetic_ranking_checkpoint,
)
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.ml.ranking import RankingModel
from video_commerce.ml.ranking_onnx import export_ranking_onnx


async def _delete_cache_prefixes(feature_store: FeatureStore) -> dict[str, int]:
    client = feature_store.cache_redis_client or feature_store.redis_client
    deleted: dict[str, int] = {}
    for prefix in ("rc:", "cc:"):
        count = 0
        batch = []
        async for key in client.scan_iter(match=f"{prefix}*", count=1000):
            batch.append(key)
            if len(batch) >= 500:
                count += int(await client.unlink(*batch))
                batch = []
        if batch:
            count += int(await client.unlink(*batch))
        deleted[prefix] = count
    return deleted


async def prepare(args: argparse.Namespace) -> dict:
    config = Config()
    checkpoint_path = Path(config.model_config.ranking_model_path)
    version = args.model_version or f"synthetic-capacity-{int(time.time())}"
    checkpoint_manifest = await create_synthetic_ranking_checkpoint(
        checkpoint_path=checkpoint_path,
        ranking_config=config.ranking_config,
        model_version=version,
        seed=args.seed,
    )
    onnx_path = None
    onnx_export = None
    if args.export_onnx:
        ranking_model = RankingModel(config.ranking_config)
        await ranking_model.load_model(str(checkpoint_path))
        ranking_model.mark_artifact_verified(
            model_version=version,
            artifact_sha256=checkpoint_manifest["artifact_sha256"],
            feature_schema_version=checkpoint_manifest["feature_schema_version"],
            metadata=checkpoint_manifest,
        )
        onnx_path = checkpoint_path.with_suffix(".onnx")
        onnx_export = export_ranking_onnx(ranking_model, onnx_path)
        onnx_export["source_checkpoint_sha256"] = checkpoint_manifest[
            "artifact_sha256"
        ]

    system_store = SystemStore(config.database_config)
    object_storage = ObjectStorage(config.object_storage_config)
    feature_store = FeatureStore(config.redis_config, config.cache_config)
    await system_store.initialize()
    await object_storage.initialize()
    await feature_store.initialize()
    try:
        artifact_manager = ModelArtifactManager(
            system_store=system_store,
            object_storage=object_storage,
            model_config=config.model_config,
            recommendation_config=config.recommendation_config,
        )
        record = await artifact_manager.persist_ranking_checkpoint(
            local_path=str(checkpoint_path),
            model_version=version,
            onnx_path=str(onnx_path) if onnx_path else None,
            payload={
                **checkpoint_manifest,
                **({"onnx_export": onnx_export} if onnx_export else {}),
                "trigger": "recommendation_loadtest_prepare",
            },
        )
        if record is None:
            raise RuntimeError("ranking checkpoint metadata was not persisted")

        deleted = await _delete_cache_prefixes(feature_store)
        candidates, metadata = build_synthetic_catalog(
            count=args.candidates,
            seed=args.seed,
        )
        await feature_store.store_product_metadata_batch(metadata)
        await feature_store.store_trending_pool(candidates)
        return {
            **checkpoint_manifest,
            "checkpoint_path": str(checkpoint_path),
            "artifact_record_path": record.checkpoint_path,
            "onnx_path": str(onnx_path) if onnx_path else None,
            "onnx_sha256": (
                record.payload.get("artifact_manifest", {})
                .get("onnx_model", {})
                .get("sha256")
            ),
            "catalog_candidates": len(candidates),
            "cache_keys_deleted": deleted,
        }
    finally:
        await feature_store.close()
        await system_store.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--model-version")
    parser.add_argument("--output")
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export and atomically publish the matching ONNX artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(prepare(args))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
