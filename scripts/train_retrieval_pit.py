#!/usr/bin/env python3
"""Train, evaluate, and register one immutable Two-Tower retrieval release."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path

from video_commerce.common.config import Config
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.ml.retrieval_evaluation import score_ann_index
from video_commerce.ml.retrieval_pit_dataset import RetrievalPitDatasetReader
from video_commerce.ml.retrieval_release_training import (
    RetrievalReleaseTrainingRunner,
)
from video_commerce.ml.retrieval_training import build_retrieval_training_inputs
from video_commerce.ml.two_tower import TwoTowerTrainer
from video_commerce.ml.vector_search import VectorSearchEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train-retrieval-pit")
    parser.add_argument("--dataset-uri")
    parser.add_argument("--model-version")
    parser.add_argument("--output-dir", default="/tmp/retrieval-pit-training")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


async def _active_champion_scores(config, manager, inputs):
    record = await manager.sync_active_two_tower_artifacts(
        environment=config.recommendation_config.retrieval_release_environment,
        require_compatible=True,
    )
    if record is None:
        return None, None
    trainer = TwoTowerTrainer(
        clip_dim=inputs.product_clip_embeddings[
            next(iter(inputs.product_clip_embeddings))
        ].shape[0],
        output_dim=config.recommendation_config.tt_embedding_dim,
    )
    if not trainer.load_checkpoint(manager.two_tower_local_checkpoint_path):
        raise RuntimeError("active Two-Tower checkpoint could not be loaded")
    loaded_index = VectorSearchEngine.load_cf_index(manager.two_tower_local_index_path)
    if loaded_index is None:
        raise RuntimeError("active Two-Tower ANN index could not be loaded")
    index, metadata = loaded_index
    index_map = {
        int(key): str(value)
        for key, value in dict(metadata.get("index_map") or {}).items()
    }
    scores = score_ann_index(
        inputs.holdout_queries,
        index,
        index_map,
        encode_user=lambda query: trainer.encode_user(
            query.user_id,
            dict(query.user_features),
            current_time=query.as_of_ts,
        ),
    )
    return record.payload.get("model_release_id"), scores


async def _run(args: argparse.Namespace):
    config = Config()
    if config.recommendation_config.retrieval_training_source != "pit":
        raise RuntimeError("RETRIEVAL_TRAINING_SOURCE=pit is required")
    dataset_uri = str(
        args.dataset_uri or config.feature_lake_config.retrieval_pit_dataset_uri or ""
    )
    if not dataset_uri:
        raise RuntimeError("FEATURE_LAKE_RETRIEVAL_PIT_DATASET_URI is required")
    storage = ObjectStorage(config.object_storage_config)
    store = SystemStore(config.database_config)
    await storage.initialize()
    await store.initialize()
    try:
        dataset = await RetrievalPitDatasetReader(storage).read(dataset_uri)
        if (
            dataset.manifest.eligibility_policy_version
            != config.recommendation_config.retrieval_eligibility_policy_version
            or dataset.manifest.label_policy_version
            != config.recommendation_config.retrieval_label_policy_version
        ):
            raise RuntimeError(
                "retrieval PIT policies are incompatible with trainer config"
            )
        inputs = build_retrieval_training_inputs(
            dataset,
            holdout_days=config.recommendation_config.retrieval_holdout_days,
            ranker_rejected_mode=(
                config.recommendation_config.retrieval_ranker_rejected_mode
            ),
        )
        version = str(
            args.model_version
            or f"retrieval-{dataset.manifest.dataset_version}-{dataset.manifest.manifest_sha256[:12]}"
        )
        manager = ModelArtifactManager(
            system_store=store,
            object_storage=storage,
            model_config=config.model_config,
            recommendation_config=config.recommendation_config,
        )
        champion_release_id, champion_scores = await _active_champion_scores(
            config, manager, inputs
        )
        runner = RetrievalReleaseTrainingRunner(
            recommendation_config=config.recommendation_config,
            artifact_manager=manager,
            system_store=store,
        )
        lineage = {
            "manifest_uri": dataset.manifest.manifest_uri,
            "retrieval_pit_manifest_sha256": dataset.manifest.manifest_sha256,
            "catalog_manifest_sha256": dataset.manifest.catalog_sha256,
            "catalog_generation_id": dataset.manifest.catalog_generation_id,
            "eligibility_policy_version": dataset.manifest.eligibility_policy_version,
            "label_policy_version": dataset.manifest.label_policy_version,
            "quality_gate_policy_version": config.recommendation_config.retrieval_quality_gate_policy_version,
            "embedding_dimension": config.recommendation_config.tt_embedding_dim,
            "architecture": config.recommendation_config.tt_architecture,
            "training_code_revision": os.getenv("GIT_COMMIT_SHA", "unknown"),
            "random_seed": config.recommendation_config.retrieval_gate_random_seed,
            "negative_sampling": {
                "logq_correction_enabled": config.recommendation_config.tt_logq_correction_enabled,
                "ranker_rejected_mode": config.recommendation_config.retrieval_ranker_rejected_mode,
                "ann_mining_catalog_generation_id": dataset.manifest.catalog_generation_id,
                "ann_mining_cutoff": inputs.training_as_of_ts,
            },
            "embedding_source_dimension": dataset.catalog.embedding_dimension,
            "embedding_model_revision": dataset.catalog.embedding_model_revision,
        }
        outcome = await runner.run_inputs(
            inputs,
            model_version=version,
            output_dir=str(Path(args.output_dir) / version),
            lineage=lineage,
            champion_scores=champion_scores,
            champion_release_id=champion_release_id,
        )
        return {
            **asdict(outcome),
            "gate": {
                **asdict(outcome.gate),
                "decision": outcome.gate.decision.value,
            },
            "comparison": "active" if champion_scores is not None else "popularity",
        }
    finally:
        await store.close()


def main() -> int:
    args = build_parser().parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
