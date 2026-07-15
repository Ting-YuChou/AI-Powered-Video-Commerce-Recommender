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
