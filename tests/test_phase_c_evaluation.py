from scripts.evaluate_phase_c import build_parser, candidate_recall_config
from video_commerce.ml.model_release_quality import RankingQualityGateConfig
import pytest


@pytest.mark.asyncio
async def test_offline_runner_writes_insufficient_evidence_for_small_verified_pit(
    tmp_path, monkeypatch
):
    import hashlib
    import json
    from test_pit_training_dataset import _write_dataset
    from scripts.evaluate_phase_c import run
    from video_commerce.ml.ranking import RankingModel
    from video_commerce.common.config import RankingConfig
    from video_commerce.common.feature_history_contracts import payload_sha256

    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("OBJECT_STORAGE_BACKEND", "local")
    pointer, manifest, _ = _write_dataset(tmp_path / "dataset")
    ranker = RankingModel(RankingConfig(hidden_dims=[16]))
    ranker._initialize_model()
    ranker.is_trained = True
    checkpoint = tmp_path / "ranking.pt"
    await ranker.save_model(str(checkpoint))
    bundle = {
        "release_id": "fixture-release",
        "feature_schema_version": ranker.feature_schema_version,
        "feature_definition_version": "ranking_ltr_v1",
        "quality_gate_policy_version": "ranking_quality_gate_v1",
        "score_policy_version": "business-value-v1",
        "value_transform_stats": ranker.value_transform_stats,
        "feature_lake_manifest_sha256": payload_sha256(
            json.loads(manifest.read_text())
        ),
        "artifact_manifest": {
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        },
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    args = build_parser().parse_args(
        [
            "--pit-pointer",
            str(pointer),
            "--champion-bundle",
            str(bundle_path),
            "--champion-bundle-sha256",
            bundle_sha,
            "--challenger-bundle",
            str(bundle_path),
            "--challenger-bundle-sha256",
            bundle_sha,
            "--experiment",
            "combined",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    result = await run(args)
    assert result["decision"] == "insufficient_evidence"
    report = json.loads((tmp_path / "report.json").read_text())
    assert "candidate_recall_at_500" in report["reasons"]
    assert report["pit_manifest_sha256"] == bundle["feature_lake_manifest_sha256"]


def test_phase_c_runner_requires_pinned_bundles_and_experiment():
    args = build_parser().parse_args(
        [
            "--pit-pointer",
            "pit/latest.json",
            "--champion-bundle",
            "champion.json",
            "--champion-bundle-sha256",
            "a" * 64,
            "--challenger-bundle",
            "challenger.json",
            "--challenger-bundle-sha256",
            "b" * 64,
            "--experiment",
            "features",
            "--output",
            "evaluation.json",
        ]
    )
    assert args.experiment == "features"
    assert args.seed == 42


def test_phase_c_training_variants_require_independent_model_versions():
    from scripts.train_phase_c import build_parser as training_parser

    args = training_parser().parse_args(
        [
            "--pit-pointer",
            "pit/latest.json",
            "--experiment",
            "combined",
            "--model-version",
            "phase-c-combined-1",
            "--output-dir",
            "models/phase-c",
        ]
    )
    assert args.model_version == "phase-c-combined-1"
    assert not args.register


def test_candidate_recall_evidence_must_match_pit_and_bundle_identity():
    evidence = {
        "pit_manifest_sha256": "a" * 64,
        "catalog_manifest_sha256": "b" * 64,
        "champion_bundle_digest": "c" * 64,
        "challenger_bundle_digest": "d" * 64,
        "champion_recall_at_500": 0.8,
        "challenger_recall_at_500": 0.81,
        "candidate_limit": 500,
        "seed": 42,
    }
    config = candidate_recall_config(
        RankingQualityGateConfig(),
        evidence,
        pit_sha="a" * 64,
        champion_digest="c" * 64,
        challenger_digest="d" * 64,
    )
    assert config.candidate_recall_at_500_challenger == 0.81
    with pytest.raises(ValueError, match="lineage"):
        candidate_recall_config(
            RankingQualityGateConfig(),
            evidence,
            pit_sha="x" * 64,
            champion_digest="c" * 64,
            challenger_digest="d" * 64,
        )


@pytest.mark.asyncio
async def test_training_insufficient_rows_never_starts_optimizer_or_writes_artifact(
    tmp_path, monkeypatch
):
    from test_pit_training_dataset import _write_dataset
    from scripts.train_phase_c import build_parser as training_parser, run as train
    from video_commerce.ml.ranking import RankingModel

    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("OBJECT_STORAGE_BACKEND", "local")
    pointer, _, _ = _write_dataset(tmp_path / "dataset")

    async def forbidden(*args, **kwargs):
        raise AssertionError("optimizer must not start")

    monkeypatch.setattr(RankingModel, "train_model_source", forbidden)
    output = tmp_path / "experiments"
    result = await train(
        training_parser().parse_args(
            [
                "--pit-pointer",
                str(pointer),
                "--experiment",
                "combined",
                "--model-version",
                "small-fixture",
                "--output-dir",
                str(output),
            ]
        )
    )
    assert result["decision"] == "insufficient_evidence"
    assert not output.exists()
