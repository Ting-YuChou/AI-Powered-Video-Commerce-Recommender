"""Checksum-aware candidate embeddings locked to a ranking checkpoint."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


CANDIDATE_SIDECAR_SCHEMA_VERSION = "candidate_embeddings_v2"
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
    candidates: Mapping[
        str,
        Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ],
    *,
    model_version: str,
    training_cutoff: float = 0.0,
) -> str:
    """Atomically publish an NPZ; absent modalities remain explicit zero rows."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cutoff = float(training_cutoff)
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("candidate sidecar training cutoff must be finite and >= 0")
    versions: list[tuple[str, float, Mapping[str, Any]]] = []
    for product_id in sorted(str(value) for value in candidates):
        raw_versions = candidates[product_id]
        candidate_versions = (
            list(raw_versions)
            if isinstance(raw_versions, (list, tuple))
            else [raw_versions]
        )
        seen_timestamps: set[float] = set()
        for values in candidate_versions:
            available_at = float(values.get("available_at", 0.0))
            if not np.isfinite(available_at) or available_at < 0.0:
                raise ValueError(
                    "candidate embedding available_at must be finite and >= 0"
                )
            if available_at in seen_timestamps:
                raise ValueError(
                    "candidate embedding versions must have unique available_at "
                    f"timestamps for product {product_id}"
                )
            seen_timestamps.add(available_at)
            versions.append((product_id, available_at, values))
    versions.sort(key=lambda value: (value[0], value[1]))
    product_ids = [product_id for product_id, _, _ in versions]
    available_at = np.asarray(
        [timestamp for _, timestamp, _ in versions],
        dtype=np.float64,
    )
    arrays = {
        name: np.zeros((len(product_ids), dimension), dtype=np.float32)
        for name, dimension in _DIMENSIONS.items()
    }
    presence = np.zeros((len(product_ids), 3), dtype=np.bool_)
    for row, (_, _, values) in enumerate(versions):
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
                training_cutoff=np.asarray(cutoff, dtype=np.float64),
                product_ids=np.asarray(product_ids, dtype=np.str_),
                available_at=available_at,
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
        available_at: np.ndarray,
        training_cutoff: float,
    ) -> None:
        self.model_version = model_version
        self.training_cutoff = float(training_cutoff)
        self._available_at = available_at
        self._index: dict[str, list[int]] = {}
        for row, product_id in enumerate(product_ids.tolist()):
            self._index.setdefault(str(product_id), []).append(row)
        for rows in self._index.values():
            rows.sort(key=lambda row: float(self._available_at[row]))
        self._presence = presence
        self._arrays = {"image": image, "text": text, "two_tower": two_tower}

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        expected_model_version: str | None = None,
        expected_training_cutoff: float | None = None,
    ) -> "CandidateEmbeddingSidecar":
        source = Path(path)
        actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_sha256 != str(expected_sha256):
            raise ValueError("candidate sidecar checksum mismatch")
        with np.load(source, allow_pickle=False) as payload:
            schema = str(payload["schema_version"].item())
            model_version = str(payload["model_version"].item())
            training_cutoff = float(payload["training_cutoff"].item())
            if schema != CANDIDATE_SIDECAR_SCHEMA_VERSION:
                raise ValueError("candidate sidecar schema mismatch")
            if expected_model_version and model_version != expected_model_version:
                raise ValueError("candidate sidecar model version mismatch")
            if expected_training_cutoff is not None and training_cutoff != float(
                expected_training_cutoff
            ):
                raise ValueError("candidate sidecar training cutoff mismatch")
            product_ids = payload["product_ids"].copy()
            available_at = payload["available_at"].astype(np.float64, copy=True)
            presence = payload["presence"].astype(np.bool_, copy=True)
            arrays = {
                name: payload[name].astype(np.float32, copy=True)
                for name in _DIMENSIONS
            }
        row_count = len(product_ids)
        if (
            not np.isfinite(training_cutoff)
            or training_cutoff < 0.0
            or available_at.shape != (row_count,)
            or not np.isfinite(available_at).all()
            or np.any(available_at < 0.0)
            or presence.shape != (row_count, 3)
            or any(
                arrays[name].shape != (row_count, dimension)
                for name, dimension in _DIMENSIONS.items()
            )
        ):
            raise ValueError("candidate sidecar tensor shape mismatch")
        return cls(
            product_ids=product_ids,
            presence=presence,
            model_version=model_version,
            available_at=available_at,
            training_cutoff=training_cutoff,
            **arrays,
        )

    def get(
        self,
        product_id: str,
        *,
        as_of_ts: float | None = None,
    ) -> dict[str, np.ndarray | float] | None:
        rows = self._index.get(str(product_id))
        if not rows:
            return None
        if as_of_ts is None:
            row = rows[-1]
        else:
            cutoff = float(as_of_ts)
            eligible = [row for row in rows if float(self._available_at[row]) <= cutoff]
            if not eligible:
                return None
            row = eligible[-1]
        return {
            **{name: values[row].copy() for name, values in self._arrays.items()},
            "presence": self._presence[row].copy(),
            "available_at": float(self._available_at[row]),
        }
