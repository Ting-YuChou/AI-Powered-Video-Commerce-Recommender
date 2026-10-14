"""Evaluate a pinned Phase C challenger without changing the active pointer."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np

from video_commerce.common.config import Config, RankingConfig
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.ml.phase_c_evidence import bundle_digest
from video_commerce.ml.model_release_compatibility import model_release_compatibility
from video_commerce.ml.pit_training_dataset import PitTrainingDatasetReader
from video_commerce.ml.ranking import RankingModel
from video_commerce.ml.model_release_quality import (
    RankingEvaluationSpool,
    RankingQualityGateConfig,
    evaluate_quality_gate,
    evaluation_rows_from_examples,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pit-pointer", required=True)
    for family in ("champion", "challenger"):
        parser.add_argument(f"--{family}-bundle", required=True)
        parser.add_argument(f"--{family}-bundle-sha256", required=True)
    parser.add_argument(
        "--experiment",
        choices=("features", "fusion", "calibration", "combined"),
        required=True,
    )
    parser.add_argument("--candidate-recall-evidence")
    parser.add_argument(
        "--prerequisite-evaluation-id",
        action="append",
        default=[],
        help="Combined release lineage, e.g. features=UUID (repeat for fusion/calibration)",
    )
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument(
        "--register",
        action="store_true",
        help="Persist the immutable evaluation without changing the active pointer",
    )
    parser.add_argument("--output", required=True)
    return parser


def _verified_json(path, digest):
    value = Path(path).read_bytes()
    if hashlib.sha256(value).hexdigest() != digest:
        raise ValueError("bundle JSON checksum mismatch")
    return json.loads(value)


def candidate_recall_config(
    config, evidence, *, pit_sha, champion_digest, challenger_digest
):
    expected = {
        "pit_manifest_sha256": pit_sha,
        "champion_bundle_digest": champion_digest,
        "challenger_bundle_digest": challenger_digest,
        "candidate_limit": 500,
        "seed": 42,
    }
    catalog_sha = str(evidence.get("catalog_manifest_sha256") or "")
    if (
        any(evidence.get(key) != value for key, value in expected.items())
        or len(catalog_sha) != 64
    ):
        raise ValueError("candidate recall evidence lineage mismatch")
    return replace(
        config,
        candidate_recall_at_500_champion=float(evidence["champion_recall_at_500"]),
        candidate_recall_at_500_challenger=float(evidence["challenger_recall_at_500"]),
    )


async def _load_ranker(storage, manifest, directory):
    compatibility = model_release_compatibility("ranking_model", manifest)
    if not compatibility.compatible:
        raise ValueError(f"ranking bundle is incompatible: {compatibility.reason}")
    checkpoint = dict(manifest["artifact_manifest"]["checkpoint"])
    target = Path(directory) / "ranking.pt"
    staged = await storage.stage_to_local_temp(
        checkpoint["path"], str(target), expected_sha256=checkpoint["sha256"]
    )
    Path(staged).replace(target)
    feature_version = manifest.get("feature_definition_version") or "ranking_ltr_v1"
    config = RankingConfig(
        feature_contract_version=feature_version,
        probability_calibration="temperature_v1"
        if manifest.get("probability_calibration")
        else "none",
        din_enabled=feature_version == "ranking_ltr_v2_din",
        trimodal_enabled=False,
    )
    if config.din_enabled:
        sidecar = dict(manifest["artifact_manifest"]["din_embedding_sidecar"])
        sidecar_target = Path(directory) / "din.npz"
        staged = await storage.stage_to_local_temp(
            sidecar["path"], str(sidecar_target), expected_sha256=sidecar["sha256"]
        )
        Path(staged).replace(sidecar_target)
        config.din_embedding_sidecar_path = str(sidecar_target)
    ranker = RankingModel(config)
    await ranker.load_model(str(target))
    if feature_version == "ranking_ltr_v3_categorical_sources":
        if (
            ranker.categorical_vocabulary.to_manifest()
            != manifest["categorical_vocabulary"]
            or ranker.feature_assembler.dense_normalization.to_manifest()
            != manifest["dense_normalization"]
        ):
            raise ValueError("checkpoint and bundle frozen features differ")
    return ranker


async def run(args):
    config = Config()
    storage = ObjectStorage(config.object_storage_config)
    await storage.initialize()
    champion_manifest = _verified_json(
        args.champion_bundle, args.champion_bundle_sha256
    )
    challenger_manifest = _verified_json(
        args.challenger_bundle, args.challenger_bundle_sha256
    )
    lake = config.feature_lake_config
    with tempfile.TemporaryDirectory(prefix="phase-c-evaluation-") as directory:
        root = Path(directory)
        champion_dir, challenger_dir = root / "champion", root / "challenger"
        champion_dir.mkdir()
        challenger_dir.mkdir()
        champion = await _load_ranker(storage, champion_manifest, champion_dir)
        challenger = await _load_ranker(storage, challenger_manifest, challenger_dir)
        source = await PitTrainingDatasetReader(
            storage,
            expected_feature_definition_version=lake.feature_definition_version,
        ).validate_and_stage(
            args.pit_pointer,
            staging_dir=root / "pit",
            batch_rows=lake.pit_batch_rows,
            max_group_rows=lake.pit_max_group_rows,
            max_bytes=lake.pit_staging_max_bytes,
            content_cache_bytes=lake.pit_content_cache_bytes,
        )
        if (
            challenger_manifest.get("feature_lake_manifest_sha256")
            != source.manifest_sha256
        ):
            raise ValueError("challenger training PIT and evaluation PIT differ")
        source.assign_splits(
            holdout_days=7,
            attribution_cutoff=source.attribution_cutoff,
            validation_fraction=config.ranking_config.validation_fraction,
            calibration_fraction=config.ranking_config.calibration_fraction,
        )
        gate_config = RankingQualityGateConfig(experiment_kind=args.experiment)
        recall_evidence = None
        if args.candidate_recall_evidence:
            recall_evidence = json.loads(
                Path(args.candidate_recall_evidence).read_text()
            )
            gate_config = candidate_recall_config(
                gate_config,
                recall_evidence,
                pit_sha=source.manifest_sha256,
                champion_digest=bundle_digest(champion_manifest),
                challenger_digest=bundle_digest(challenger_manifest),
            )
        stats = source.evaluation_slice_stats()
        onnx_session = None
        onnx_artifact = dict(
            challenger_manifest["artifact_manifest"].get("onnx_model") or {}
        )
        if onnx_artifact:
            import onnxruntime as ort

            onnx_target = root / "challenger.onnx"
            staged = await storage.stage_to_local_temp(
                onnx_artifact["path"],
                str(onnx_target),
                expected_sha256=onnx_artifact["sha256"],
            )
            Path(staged).replace(onnx_target)
            onnx_session = ort.InferenceSession(
                str(onnx_target), providers=["CPUExecutionProvider"]
            )
        spool = RankingEvaluationSpool(root / "evaluation.sqlite3")
        try:
            for examples in source.iter_example_batches(
                split="holdout", batch_rows=lake.pit_batch_rows, seed=42, epoch=0
            ):
                rows = evaluation_rows_from_examples(
                    [],
                    examples,
                    feature_schema_version=challenger.feature_schema_version,
                    value_target_policy=challenger.config.value_target_policy,
                    value_currency=challenger.config.value_currency,
                )
                old = champion.predict_release_examples(examples)
                new = challenger.predict_release_examples(examples)
                if onnx_session is not None:
                    tensors = challenger._prepare_training_tensors(
                        examples, fit_value_transform=False
                    )
                    feed = {
                        "base_features": tensors.base_features.detach().cpu().numpy()
                    }
                    for key, tensor in tensors.din_inputs.items():
                        if key == "candidate_indices":
                            tensor = tensor.unsqueeze(-1)
                        feed[key] = tensor.detach().cpu().numpy()
                    names = ["ctr", "cvr", "ctcvr", "gmv", "ranking_score"]
                    raw = dict(zip(names, onnx_session.run(names, feed)))
                    challenger._add_business_predictions(
                        raw, tensors.labels["value_bucket"].detach().cpu().tolist()
                    )
                    scores = np.asarray(
                        raw["business_score"]
                        if challenger.config.business_score_enabled
                        else raw["ranking_score"]
                    ).reshape(-1)
                    expected = np.asarray(
                        [
                            new[example.observation_id].serving_score
                            for example in examples
                        ]
                    )
                    np.testing.assert_allclose(scores, expected, atol=1e-5, rtol=1e-4)
                    products = [
                        example.bundle.candidate.product_id for example in examples
                    ]
                    if sorted(
                        range(len(products)), key=lambda i: (-scores[i], products[i])
                    ) != sorted(
                        range(len(products)), key=lambda i: (-expected[i], products[i])
                    ):
                        raise ValueError("ONNX holdout canonical top-k parity failed")
                for row in rows:
                    spool.add(
                        replace(
                            row,
                            pre_holdout_exposures=int(
                                stats["exposures"].get(row.product_id, 0)
                            ),
                            is_long_tail=row.product_id not in stats["head_products"],
                            is_high_price=row.price > stats["high_price_threshold"],
                        ),
                        old[row.observation_id],
                        new[row.observation_id],
                    )
            rows, old, new = spool.load()
        finally:
            spool.close()
        result = evaluate_quality_gate(rows, old, new, gate_config)
        prerequisites = {}
        for entry in args.prerequisite_evaluation_id:
            kind, separator, evaluation_id = entry.partition("=")
            if (
                not separator
                or kind not in {"features", "fusion", "calibration"}
                or not evaluation_id
                or kind in prerequisites
            ):
                raise ValueError("invalid or repeated prerequisite evaluation identity")
            prerequisites[kind] = evaluation_id
        gate_snapshot = {
            **asdict(gate_config),
            "candidate_recall_evidence": recall_evidence,
            "prerequisite_evaluations": prerequisites,
        }
        report = {
            **asdict(result),
            "decision": result.decision.value,
            "experiment": args.experiment,
            "gate_config": gate_snapshot,
            "pit_manifest_sha256": source.manifest_sha256,
            "pit_manifest_uri": source.manifest_uri,
            "champion_bundle_digest": bundle_digest(champion_manifest),
            "challenger_bundle_digest": bundle_digest(challenger_manifest),
            "candidate_recall_evidence": recall_evidence,
            "onnx_holdout_parity": onnx_session is not None,
            "holdout_window": {
                "start": source.max_as_of_ts - 7 * 86400,
                "end": source.max_as_of_ts,
            },
        }
        if args.register:
            store = SystemStore(config.database_config)
            await store.initialize()
            try:
                release_id = challenger_manifest["release_id"]
                evaluation = await store.record_model_release_evaluation(
                    release_id=release_id,
                    champion_release_id=champion_manifest["release_id"],
                    dataset_manifest_uri=source.manifest_uri,
                    dataset_manifest_sha256=source.manifest_sha256,
                    holdout_start=report["holdout_window"]["start"],
                    holdout_end=source.max_as_of_ts,
                    policy_version=result.policy_version,
                    decision=result.decision.value,
                    metrics={
                        **dict(result.overall),
                        "reasons": list(result.reasons),
                        "onnx_holdout_parity": onnx_session is not None,
                    },
                    slice_metrics=dict(result.slices),
                    bootstrap=dict(result.bootstrap),
                    gate_config=gate_snapshot,
                )
                if evaluation["decision"] != result.decision.value:
                    raise ValueError(
                        "an immutable evaluation already exists for this release/PIT/policy"
                    )
                report["evaluation_id"] = evaluation["evaluation_id"]
                if result.decision.value == "passed":
                    if onnx_session is None:
                        raise ValueError(
                            "staging requires a verified ONNX artifact and holdout parity"
                        )
                    if not await store.stage_model_release(
                        release_id=release_id,
                        actor="phase-c-offline-evaluator",
                        environment=config.model_release_config.environment,
                    ):
                        raise ValueError("release could not enter staging")
            finally:
                await store.close()
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        target.write_text(encoded)
        return {
            "decision": result.decision.value,
            "report_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "output": str(target),
        }


def main():
    result = asyncio.run(run(build_parser().parse_args()))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["decision"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
