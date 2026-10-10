from video_commerce.ml.model_release_compatibility import (
    model_release_compatibility,
)


def _checkpoint_manifest():
    return {
        "artifact_manifest": {"checkpoint": {"path": "model.pt", "sha256": "a" * 64}}
    }


def test_ranking_compatibility_remains_score_policy_aware():
    manifest = {
        **_checkpoint_manifest(),
        "feature_schema_version": "ranking_v4_01_temporal_trimodal",
        "score_policy_version": "business-value-v1",
        "value_transform_stats": {},
        "quality_gate_policy_version": "ranking_quality_gate_v1",
    }

    result = model_release_compatibility("ranking_model", manifest)

    assert result.compatible is True
    assert result.policy_version == "ranking_quality_gate_v1"


def test_two_tower_compatibility_requires_pit_catalog_and_bundle_lineage():
    manifest = {
        "artifact_manifest": {
            "checkpoint": {"path": "model.pt", "sha256": "a" * 64},
            "cf_index": {"path": "model.faiss", "sha256": "d" * 64},
            "cf_index_metadata": {
                "path": "model.cf_meta.json",
                "sha256": "e" * 64,
            },
            "cf_embedding_sidecar": {
                "path": "model.cf_embeddings.npz",
                "sha256": "f" * 64,
            },
        },
        "retrieval_pit_manifest_sha256": "b" * 64,
        "catalog_manifest_sha256": "c" * 64,
        "eligibility_policy_version": "retrieval_eligibility_v1",
        "label_policy_version": "retrieval_label_v1",
        "quality_gate_policy_version": "retrieval_quality_gate_v1",
        "embedding_dimension": 128,
        "architecture": "dcn",
        "cf_index_path": "model.faiss",
        "cf_index_metadata_path": "model.cf_meta.json",
    }

    result = model_release_compatibility("two_tower_retrieval", manifest)

    assert result.compatible is True
    assert result.compatibility_keys == (
        "embedding_dimension",
        "eligibility_policy_version",
        "label_policy_version",
        "architecture",
    )

    manifest.pop("catalog_manifest_sha256")
    failed = model_release_compatibility("two_tower_retrieval", manifest)
    assert failed.compatible is False
    assert failed.reason == "missing_catalog_manifest_sha256"


def test_unknown_model_family_cannot_be_activated():
    result = model_release_compatibility("unknown", _checkpoint_manifest())

    assert result.compatible is False
    assert result.reason == "unsupported_model_family"
