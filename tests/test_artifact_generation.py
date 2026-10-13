from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

import pytest

from video_commerce.common.config import ObjectStorageConfig
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.ml.artifact_generation import (
    ArtifactGenerationInstaller,
    ArtifactSpec,
    AtomicModelReference,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_generation_installer_activates_complete_bundle_with_one_directory_rename(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "ranking.pt"
    sidecar = source / "ranking.candidates.npz"
    checkpoint.write_bytes(b"checkpoint-v2")
    sidecar.write_bytes(b"sidecar-v2")
    installer = ArtifactGenerationInstaller(
        root=tmp_path / "generations",
        object_storage=ObjectStorage(ObjectStorageConfig(backend="local")),
    )

    resolved = asyncio.run(
        installer.install(
            model_name="ranking_model",
            release_id="release-v2",
            model_version="v2",
            active_generation=2,
            artifacts=(
                ArtifactSpec("checkpoint", str(checkpoint), _sha(b"checkpoint-v2")),
                ArtifactSpec("candidate_sidecar", str(sidecar), _sha(b"sidecar-v2")),
            ),
            metadata={"feature_schema_version": "ranking-v4"},
        )
    )

    assert resolved.root.name == "2-release-v2"
    assert resolved.artifact_paths["checkpoint"].read_bytes() == b"checkpoint-v2"
    assert resolved.artifact_paths["candidate_sidecar"].read_bytes() == b"sidecar-v2"
    assert (resolved.root / "generation.json").exists()
    assert not list((tmp_path / "generations" / "ranking_model").glob(".staging-*"))


def test_generation_installer_failure_never_exposes_partial_generation(tmp_path):
    checkpoint = tmp_path / "ranking.pt"
    checkpoint.write_bytes(b"checkpoint")
    installer = ArtifactGenerationInstaller(
        root=tmp_path / "generations",
        object_storage=ObjectStorage(ObjectStorageConfig(backend="local")),
    )

    with pytest.raises(ValueError, match="checksum"):
        asyncio.run(
            installer.install(
                model_name="ranking_model",
                release_id="release-bad",
                model_version="bad",
                active_generation=3,
                artifacts=(ArtifactSpec("checkpoint", str(checkpoint), "0" * 64),),
                metadata={},
            )
        )

    family_root = tmp_path / "generations" / "ranking_model"
    assert not (family_root / "3-release-bad").exists()
    assert not list(family_root.glob(".staging-*"))


def test_atomic_model_reference_keeps_inflight_snapshot_during_swap():
    reference = AtomicModelReference("v1", identity=("release-v1", 1, "digest-1"))

    with reference.snapshot() as inflight:
        previous = reference.swap("v2", identity=("release-v2", 2, "digest-2"))
        assert inflight == "v1"
        assert previous == "v1"
        assert reference.current() == "v2"
        assert reference.inflight_count(("release-v1", 1, "digest-1")) == 1

    assert reference.inflight_count(("release-v1", 1, "digest-1")) == 0


def test_generation_cleanup_preserves_current_inflight_rollback_and_young(tmp_path):
    family = tmp_path / "generations" / "ranking_model"
    family.mkdir(parents=True)
    now = time.time()
    for generation, release in ((1, "r1"), (2, "r2"), (3, "r3"), (4, "r4"), (5, "r5")):
        directory = family / f"{generation}-{release}"
        directory.mkdir()
        (directory / "generation.json").write_text(
            json.dumps({"release_id": release, "active_generation": generation}),
            encoding="utf-8",
        )
        age_hours = 1 if generation == 5 else 100
        os.utime(directory, (now - age_hours * 3600, now - age_hours * 3600))

    installer = ArtifactGenerationInstaller(
        root=tmp_path / "generations", object_storage=object()
    )
    deleted = installer.cleanup(
        model_name="ranking_model",
        current_identity=("r4", 4, "digest"),
        protected_identities={("r1", 1)},
        inflight_identities={("r2", 2)},
        keep=1,
        min_age_hours=24,
        now_epoch_seconds=now,
    )

    assert deleted == (family / "3-r3",)
    assert sorted(path.name for path in family.iterdir()) == [
        "1-r1",
        "2-r2",
        "4-r4",
        "5-r5",
    ]
