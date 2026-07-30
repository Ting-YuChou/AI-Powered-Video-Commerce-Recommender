"""Upgrade v2 content artifacts to visual-retrieval v3 without decoding video."""

from __future__ import annotations

import argparse
import asyncio
import json

from video_commerce.common.config import Config
from video_commerce.common.models import ContentFeatures
from video_commerce.data_plane.feature_store import FeatureStore
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.ml.content_artifacts import (
    content_artifact_reference,
    load_content_feature_artifact,
    persist_content_features,
)
from video_commerce.ml.visual_retrieval import (
    load_visual_retrieval_checkpoint,
    upgrade_content_features_to_visual_retrieval_v3,
)


async def run(*, limit: int, apply: bool) -> int:
    config = Config()
    model_config = config.model_config
    model, metadata = load_visual_retrieval_checkpoint(
        model_config.retrieval_visual_checkpoint_path,
        expected_sha256=model_config.retrieval_visual_checkpoint_sha256 or None,
    )
    lineage = metadata["lineage"]
    if (
        lineage["clip_model_id"] != model_config.clip_model
        or lineage["clip_revision"] != model_config.clip_revision
    ):
        raise ValueError("retrieval checkpoint and configured CLIP lineage differ")
    storage = ObjectStorage(config.object_storage_config)
    store = SystemStore(config.database_config)
    feature_store = FeatureStore(config.redis_config, config.cache_config)
    await storage.initialize()
    await store.initialize()
    await feature_store.initialize()
    upgraded = []
    skipped = {}
    try:
        jobs = await store.list_content_jobs_missing_feature_artifact(
            limit=limit,
            expected_schema_version="temporal_multimodal_v3",
        )
        for job in jobs:
            content_id = job["content_id"]
            current_payload = await store.get_content_feature_artifact(content_id)
            if current_payload is None:
                skipped[content_id] = "current artifact pointer missing"
                continue
            current = ContentFeatures(**current_payload)
            reference = content_artifact_reference(current)
            if reference is None:
                skipped[content_id] = "immutable v2 artifact reference missing"
                continue
            current = await load_content_feature_artifact(storage, reference)
            if not current.frame_embeddings:
                skipped[content_id] = "stored frame embeddings missing"
                continue
            updated = upgrade_content_features_to_visual_retrieval_v3(
                current,
                model=model,
                model_version=metadata["model_version"],
                product_index_version=lineage["product_index_version"],
                clip_model_id=lineage["clip_model_id"],
                clip_revision=lineage["clip_revision"],
            )
            if apply:
                await persist_content_features(
                    updated,
                    object_storage=storage,
                    system_store=store,
                    feature_store=feature_store,
                )
            upgraded.append(content_id)
        print(
            json.dumps(
                {
                    "mode": "apply" if apply else "dry-run",
                    "selected": len(jobs),
                    "upgradable": len(upgraded),
                    "content_ids": upgraded,
                    "skipped": skipped,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        await feature_store.close()
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish immutable v3 artifacts; default only reports eligibility",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(limit=max(1, args.limit), apply=args.apply)))


if __name__ == "__main__":
    main()
