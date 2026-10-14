"""Create independent Phase C experiment bundles from one immutable PIT source."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from scripts.evaluate_phase_c import _verified_json, _load_ranker
from video_commerce.common.config import Config
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.ml.pit_training_dataset import PitTrainingDatasetReader
from video_commerce.ml.ranking import RankingModel
from video_commerce.ml.ranking_onnx import export_ranking_onnx
from video_commerce.ml.ranking_score import SCORE_POLICY_VERSION


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pit-pointer", required=True)
    parser.add_argument(
        "--experiment",
        choices=("features", "fusion", "calibration", "combined"),
        required=True,
    )
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--champion-bundle")
    parser.add_argument("--champion-bundle-sha256")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--register", action="store_true")
    return parser


async def run(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.model_version):
        raise ValueError("model version is not a safe immutable artifact identifier")
    if args.experiment in {"fusion", "calibration"} and (
        not args.champion_bundle or not args.champion_bundle_sha256
    ):
        raise ValueError(
            "fusion/calibration experiments require a pinned champion bundle"
        )
    config = Config()
    lake = config.feature_lake_config
    storage = ObjectStorage(config.object_storage_config)
    await storage.initialize()
    with tempfile.TemporaryDirectory(prefix="phase-c-training-") as work_dir:
        source = await PitTrainingDatasetReader(
            storage,
            expected_feature_definition_version=lake.feature_definition_version,
        ).validate_and_stage(
            args.pit_pointer,
            staging_dir=Path(work_dir) / "pit",
            batch_rows=lake.pit_batch_rows,
            max_group_rows=lake.pit_max_group_rows,
            max_bytes=lake.pit_staging_max_bytes,
            content_cache_bytes=lake.pit_content_cache_bytes,
        )
        source.assign_splits(
            holdout_days=7,
            attribution_cutoff=source.attribution_cutoff,
            validation_fraction=config.ranking_config.validation_fraction,
            calibration_fraction=0.1,
        )
        if (
            source.row_count_for_split("training")
            < config.ranking_config.training_min_samples
        ):
            return {
                "decision": "insufficient_evidence",
                "reason": "training_split",
                "pit_manifest_sha256": source.manifest_sha256,
            }
        directory = Path(args.output_dir) / args.model_version
        directory.mkdir(parents=True, exist_ok=False)
        checkpoint_path = directory / "ranking.pt"
        if args.experiment in {"features", "combined"}:
            settings = config.ranking_config.copy(deep=True)
            settings.feature_contract_version = "ranking_ltr_v3_categorical_sources"
            settings.din_enabled = settings.trimodal_enabled = False
            settings.training_seed = 42
            settings.value_target_policy = "gross_revenue_v1"
            settings.probability_calibration = (
                "temperature_v1" if args.experiment == "combined" else "none"
            )
            ranker = RankingModel(settings)
            ranker.loaded_model_path = str(checkpoint_path)
            if not await ranker.train_model_source(
                source, batch_rows=lake.pit_batch_rows
            ):
                return {"decision": "insufficient_evidence", "reason": "training_split"}
        else:
            if not args.champion_bundle or not args.champion_bundle_sha256:
                raise ValueError(
                    "fusion/calibration experiments require a pinned champion bundle"
                )
            champion_manifest = _verified_json(
                args.champion_bundle, args.champion_bundle_sha256
            )
            ranker = await _load_ranker(storage, champion_manifest, directory)
            if args.experiment == "calibration":
                ranker.config.probability_calibration = "temperature_v1"
                ranker.model.eval()
                ranker._fit_probability_calibration_from_source(
                    source, batch_rows=lake.pit_batch_rows, seed=42
                )
        ranker.model_version = args.model_version
        await ranker.save_model(str(checkpoint_path))
        checkpoint_sha = ObjectStorage.calculate_sha256(str(checkpoint_path))
        ranker.mark_artifact_verified(
            model_version=args.model_version,
            artifact_sha256=checkpoint_sha,
            feature_schema_version=ranker.feature_schema_version,
            metadata={"training_pit_manifest_sha256": source.manifest_sha256},
        )
        onnx_path = directory / "ranking.onnx"
        onnx_manifest = export_ranking_onnx(ranker, onnx_path)
        onnx_manifest.update(
            path=str(onnx_path),
            sha256=ObjectStorage.calculate_sha256(str(onnx_path)),
            source_checkpoint_sha256=checkpoint_sha,
        )
        payload = {
            "experiment_kind": args.experiment,
            "feature_schema_version": ranker.feature_schema_version,
            "feature_definition_version": ranker.config.feature_contract_version,
            "label_definition_version": ranker.training_tensor_builder.label_builder.version,
            "feature_assembler_version": ranker.feature_assembler.version,
            "score_policy_version": SCORE_POLICY_VERSION,
            "value_transform_stats": ranker.value_transform_stats,
            "value_target_policy": ranker.config.value_target_policy,
            "value_currency": ranker.config.value_currency,
            "categorical_vocabulary": ranker.categorical_vocabulary.to_manifest()
            if ranker.categorical_vocabulary
            else None,
            "dense_normalization": ranker.feature_assembler.dense_normalization.to_manifest()
            if ranker.feature_assembler.dense_normalization
            else None,
            "probability_calibration": ranker.probability_calibration.to_manifest()
            if ranker.config.probability_calibration == "temperature_v1"
            else None,
            "candidate_fusion_policy_version": "candidate_rrf_v1"
            if args.experiment in {"fusion", "combined"}
            else "candidate_legacy_v1",
            "candidate_rrf_k": 60,
            "candidate_source_registry_version": "retrieval_sources_v1",
            "feature_lake_manifest_uri": source.manifest_uri,
            "feature_lake_manifest_sha256": source.manifest_sha256,
            "feature_lake_iceberg_snapshot_id": source.iceberg_snapshot_id,
            "training_pit_materialization_run_id": source.materialization_run_id,
            "training_seed": 42,
            "training_code_revision": os.getenv("GIT_COMMIT_SHA", "unknown"),
            "quality_gate_policy_version": "ranking_quality_gate_v2",
            "holdout_window": {
                "start": source.max_as_of_ts - 7 * 86400,
                "end": source.max_as_of_ts,
            },
            "quality_gate_holdout_start": source.max_as_of_ts - 7 * 86400,
            "quality_gate_holdout_end": source.max_as_of_ts,
            "onnx_export": onnx_manifest,
        }
        if args.register:
            store = SystemStore(config.database_config)
            await store.initialize()
            try:
                model_config = config.model_config.copy(deep=True)
                model_config.ranking_model_path = str(checkpoint_path)
                manager = ModelArtifactManager(
                    system_store=store,
                    object_storage=storage,
                    model_config=model_config,
                    recommendation_config=config.recommendation_config,
                )
                record = await manager.persist_ranking_checkpoint(
                    local_path=str(checkpoint_path),
                    model_version=args.model_version,
                    payload=payload,
                    onnx_path=str(onnx_path),
                )
                release = await store.get_model_release(
                    ModelArtifactManager.RANKING_MODEL_NAME, record.model_version
                )
                manifest = release["bundle_manifest"]
                decision = "registered"
            finally:
                await store.close()
        else:
            manifest = {
                **payload,
                "release_id": None,
                "artifact_manifest": {
                    "checkpoint": {
                        "path": str(checkpoint_path),
                        "sha256": checkpoint_sha,
                    },
                    "onnx_model": onnx_manifest,
                },
            }
            decision = "offline_bundle"
        bundle_path = directory / "bundle.json"
        encoded = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
        bundle_path.write_text(encoded)
        return {
            "decision": decision,
            "bundle": str(bundle_path),
            "bundle_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "pit_manifest_sha256": source.manifest_sha256,
        }


def main():
    result = asyncio.run(run(build_parser().parse_args()))
    print(json.dumps(result, sort_keys=True))
    return 2 if result["decision"] == "insufficient_evidence" else 0


if __name__ == "__main__":
    raise SystemExit(main())
