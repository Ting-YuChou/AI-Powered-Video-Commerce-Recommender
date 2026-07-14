"""Checksum-aware candidate embeddings locked to a ranking checkpoint."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


CANDIDATE_SIDECAR_SCHEMA_VERSION = "candidate_embeddings_v1"
_DIMENSIONS = {"image": 512, "text": 384, "two_tower": 128}


def _normalized_embedding(value: Any, *, name: str) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32)
    expected = _DIMENSIONS[name]
    if embedding.shape != (expected,):
        raise ValueError(f"candidate {name} embedding dimension must be {expected}")
    if not np.isfinite(embedding).all():
        raise ValueError(f"candidate {name} embedding must be finite")
    norm = float(np.linalg.norm(embedding))
    return embedding / norm if norm > 0 else embedding


def write_candidate_embedding_sidecar(
    path: str | Path,
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    model_version: str,
) -> str:
    """Atomically publish an NPZ; absent modalities remain explicit zero rows."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    product_ids = sorted(str(product_id) for product_id in candidates)
    arrays = {
        name: np.zeros((len(product_ids), dimension), dtype=np.float32)
        for name, dimension in _DIMENSIONS.items()
    }
    presence = np.zeros((len(product_ids), 3), dtype=np.bool_)
    for row, product_id in enumerate(product_ids):
        values = candidates[product_id]
        for column, name in enumerate(_DIMENSIONS):
            value = values.get(name)
            if value is None:
                continue
            arrays[name][row] = _normalized_embedding(value, name=name)
            presence[row, column] = True

    descriptor, temporary_path = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        with open(temporary_path, "wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.asarray(CANDIDATE_SIDECAR_SCHEMA_VERSION),
                model_version=np.asarray(str(model_version)),
                product_ids=np.asarray(product_ids, dtype=np.str_),
                presence=presence,
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return hashlib.sha256(target.read_bytes()).hexdigest()


class CandidateEmbeddingSidecar:
    def __init__(
        self,
        *,
        product_ids: np.ndarray,
        presence: np.ndarray,
        image: np.ndarray,
        text: np.ndarray,
        two_tower: np.ndarray,
        model_version: str,
    ) -> None:
        self.model_version = model_version
        self._index = {
            str(product_id): row for row, product_id in enumerate(product_ids.tolist())
        }
        self._presence = presence
        self._arrays = {"image": image, "text": text, "two_tower": two_tower}

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        expected_model_version: str | None = None,
    ) -> "CandidateEmbeddingSidecar":
        source = Path(path)
        actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_sha256 != str(expected_sha256):
            raise ValueError("candidate sidecar checksum mismatch")
        with np.load(source, allow_pickle=False) as payload:
            schema = str(payload["schema_version"].item())
            model_version = str(payload["model_version"].item())
            if schema != CANDIDATE_SIDECAR_SCHEMA_VERSION:
                raise ValueError("candidate sidecar schema mismatch")
            if expected_model_version and model_version != expected_model_version:
                raise ValueError("candidate sidecar model version mismatch")
            product_ids = payload["product_ids"].copy()
            presence = payload["presence"].astype(np.bool_, copy=True)
            arrays = {
                name: payload[name].astype(np.float32, copy=True)
                for name in _DIMENSIONS
            }
        row_count = len(product_ids)
        if presence.shape != (row_count, 3) or any(
            arrays[name].shape != (row_count, dimension)
            for name, dimension in _DIMENSIONS.items()
        ):
            raise ValueError("candidate sidecar tensor shape mismatch")
        return cls(
            product_ids=product_ids,
            presence=presence,
            model_version=model_version,
            **arrays,
        )

    def get(self, product_id: str) -> dict[str, np.ndarray] | None:
        row = self._index.get(str(product_id))
        if row is None:
            return None
        return {
            **{name: values[row].copy() for name, values in self._arrays.items()},
            "presence": self._presence[row].copy(),
        }
