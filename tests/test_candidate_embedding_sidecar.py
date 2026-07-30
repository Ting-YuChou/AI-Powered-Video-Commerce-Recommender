import hashlib

import numpy as np
import pytest

from video_commerce.ml.candidate_embedding_sidecar import (
    CandidateEmbeddingSidecar,
    write_candidate_embedding_sidecar,
)


def test_candidate_sidecar_round_trips_presence_without_random_fallback(tmp_path):
    path = tmp_path / "candidates.npz"
    digest = write_candidate_embedding_sidecar(
        path,
        {
            "p1": {"image": np.ones(512), "text": np.ones(384)},
            "p2": {"two_tower": np.ones(128)},
        },
        model_version="candidate-v1",
    )
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    sidecar = CandidateEmbeddingSidecar.load(
        path, expected_sha256=digest, expected_model_version="candidate-v1"
    )
    p1 = sidecar.get("p1")
    assert p1["presence"].tolist() == [True, True, False]
    assert np.count_nonzero(p1["two_tower"]) == 0
    assert sidecar.get("missing") is None

    with pytest.raises(ValueError, match="checksum"):
        CandidateEmbeddingSidecar.load(path, expected_sha256="0" * 64)


def test_candidate_sidecar_rejects_wrong_embedding_dimensions(tmp_path):
    with pytest.raises(ValueError, match="image embedding dimension"):
        write_candidate_embedding_sidecar(
            tmp_path / "bad.npz", {"p1": {"image": [1.0]}}, model_version="v1"
        )


def test_candidate_sidecar_resolves_latest_point_in_time_version(tmp_path):
    path = tmp_path / "versioned-candidates.npz"
    early = np.zeros(384, dtype=np.float32)
    early[0] = 1.0
    late = np.zeros(384, dtype=np.float32)
    late[1] = 1.0
    digest = write_candidate_embedding_sidecar(
        path,
        {
            "p1": [
                {"available_at": 10.0, "text": early},
                {"available_at": 20.0, "text": late},
            ]
        },
        model_version="candidate-v2",
        training_cutoff=5.0,
    )

    sidecar = CandidateEmbeddingSidecar.load(
        path,
        expected_sha256=digest,
        expected_model_version="candidate-v2",
        expected_training_cutoff=5.0,
    )

    assert sidecar.get("p1", as_of_ts=5.0) is None
    np.testing.assert_allclose(sidecar.get("p1", as_of_ts=15.0)["text"], early)
    np.testing.assert_allclose(sidecar.get("p1", as_of_ts=25.0)["text"], late)
    np.testing.assert_allclose(sidecar.get("p1")["text"], late)
