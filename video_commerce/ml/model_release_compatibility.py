"""Model-family specific release bundle compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

from video_commerce.ml.model_release_quality import QUALITY_GATE_POLICY_VERSION
from video_commerce.ml.ranking_score import SCORE_POLICY_VERSION
from video_commerce.ml.retrieval_evaluation import RETRIEVAL_QUALITY_GATE_VERSION
from video_commerce.ml.retrieval_pit_dataset import (
    RETRIEVAL_ELIGIBILITY_POLICY_VERSION,
    RETRIEVAL_LABEL_POLICY_VERSION,
)


@dataclass(frozen=True)
class ModelReleaseCompatibility:
    compatible: bool
    reason: str | None
    policy_version: str | None
    compatibility_keys: tuple[str, ...]


def model_release_compatibility(
    model_name: str, manifest: Mapping[str, Any]
) -> ModelReleaseCompatibility:
    artifact_manifest = dict(manifest.get("artifact_manifest") or {})
    checkpoint = dict(artifact_manifest.get("checkpoint") or {})
    if not checkpoint.get("path"):
        return _failed("missing_checkpoint_path")
    if len(str(checkpoint.get("sha256") or "")) != 64:
        return _failed("invalid_checkpoint_sha256")
    if model_name == "ranking_model":
        requirements = {
            "feature_schema_version": bool(manifest.get("feature_schema_version")),
            "score_policy_version": manifest.get("score_policy_version")
            == SCORE_POLICY_VERSION,
            "value_transform_stats": isinstance(
                manifest.get("value_transform_stats"), dict
            ),
            "quality_gate_policy_version": manifest.get("quality_gate_policy_version")
            == QUALITY_GATE_POLICY_VERSION,
        }
        failed = next((key for key, valid in requirements.items() if not valid), None)
        return (
            _failed(f"missing_or_incompatible_{failed}")
            if failed
            else ModelReleaseCompatibility(
                True,
                None,
                QUALITY_GATE_POLICY_VERSION,
                ("feature_schema_version", "score_policy_version"),
            )
        )
    if model_name == "two_tower_retrieval":
        required_hashes = (
            "retrieval_pit_manifest_sha256",
            "catalog_manifest_sha256",
        )
        for key in required_hashes:
            if len(str(manifest.get(key) or "")) != 64:
                return _failed(f"missing_{key}")
        index_artifact = dict(artifact_manifest.get("cf_index") or {})
        metadata_artifact = dict(artifact_manifest.get("cf_index_metadata") or {})
        sidecar_artifact = dict(artifact_manifest.get("cf_embedding_sidecar") or {})
        requirements = {
            "cf_index_path": bool(manifest.get("cf_index_path")),
            "cf_index_metadata_path": bool(manifest.get("cf_index_metadata_path")),
            "cf_index_checksum": len(str(index_artifact.get("sha256") or "")) == 64,
            "cf_metadata_checksum": len(str(metadata_artifact.get("sha256") or ""))
            == 64,
            "cf_embedding_sidecar_checksum": len(
                str(sidecar_artifact.get("sha256") or "")
            )
            == 64,
            "architecture": bool(manifest.get("architecture")),
            "eligibility_policy_version": manifest.get("eligibility_policy_version")
            == RETRIEVAL_ELIGIBILITY_POLICY_VERSION,
            "label_policy_version": manifest.get("label_policy_version")
            == RETRIEVAL_LABEL_POLICY_VERSION,
            "quality_gate_policy_version": manifest.get("quality_gate_policy_version")
            == RETRIEVAL_QUALITY_GATE_VERSION,
            "embedding_dimension": int(manifest.get("embedding_dimension") or 0) > 0,
        }
        failed = next((key for key, valid in requirements.items() if not valid), None)
        if failed:
            return _failed(f"missing_or_incompatible_{failed}")
        return ModelReleaseCompatibility(
            True,
            None,
            RETRIEVAL_QUALITY_GATE_VERSION,
            (
                "embedding_dimension",
                "eligibility_policy_version",
                "label_policy_version",
                "architecture",
            ),
        )
    return _failed("unsupported_model_family")


def _failed(reason: str) -> ModelReleaseCompatibility:
    return ModelReleaseCompatibility(False, reason, None, ())
