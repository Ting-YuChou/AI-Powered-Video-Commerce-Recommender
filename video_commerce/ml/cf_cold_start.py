"""
Cold-start bootstrap helpers for Two-Tower collaborative retrieval.

The synthetic embeddings produced here are serving-time priors. They should not
be written into the Two-Tower checkpoint as if the model trained them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from video_commerce.ml.two_tower import ItemFeatureEncoder

SYNTHETIC_SOURCE = "synthetic_clip_neighbor_adapter"
ADAPTER_VERSION = "content-to-cf-ridge-v1"


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm <= 0.0 or not np.isfinite(norm):
        return np.zeros_like(values, dtype=np.float32)
    return (values / norm).astype(np.float32)


def build_content_feature(
    clip_embedding: np.ndarray,
    metadata: Dict[str, Any],
    *,
    clip_dim: int,
) -> np.ndarray:
    clip = np.asarray(clip_embedding, dtype=np.float32).reshape(-1)
    if clip.shape[0] != clip_dim:
        raise ValueError(f"clip embedding dimension {clip.shape[0]} != {clip_dim}")
    return np.concatenate(
        [
            normalize_vector(clip),
            ItemFeatureEncoder.encode(metadata),
        ]
    ).astype(np.float32)


@dataclass
class ContentToCFAdapter:
    weights: np.ndarray
    input_dim: int
    output_dim: int
    ridge_alpha: float
    version: str = ADAPTER_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        ridge_alpha: float = 1e-2,
    ) -> "ContentToCFAdapter":
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("features and targets must be 2D arrays")
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and targets must have the same row count")
        if x.shape[0] < 2:
            raise ValueError("at least two rows are required to fit the adapter")

        x_aug = np.concatenate(
            [x, np.ones((x.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        regularizer = np.eye(x_aug.shape[1], dtype=np.float32) * float(ridge_alpha)
        regularizer[-1, -1] = 0.0
        lhs = x_aug.T @ x_aug + regularizer
        rhs = x_aug.T @ y
        weights = np.linalg.solve(lhs, rhs).astype(np.float32)
        return cls(
            weights=weights,
            input_dim=x.shape[1],
            output_dim=y.shape[1],
            ridge_alpha=float(ridge_alpha),
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.input_dim:
            raise ValueError(f"feature dimension {x.shape[1]} != {self.input_dim}")
        x_aug = np.concatenate(
            [x, np.ones((x.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        predictions = x_aug @ self.weights
        return np.vstack([normalize_vector(row) for row in predictions]).astype(np.float32)

    def save(self, path: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            weights=self.weights.astype(np.float32),
            input_dim=np.array([self.input_dim], dtype=np.int32),
            output_dim=np.array([self.output_dim], dtype=np.int32),
            ridge_alpha=np.array([self.ridge_alpha], dtype=np.float32),
            version=np.array([self.version]),
            metadata_json=np.array([json.dumps(metadata or {}, sort_keys=True)]),
        )

    @classmethod
    def load(cls, path: str) -> Optional["ContentToCFAdapter"]:
        target = Path(path)
        if not target.exists():
            return None
        data = np.load(target, allow_pickle=False)
        metadata: Dict[str, Any] = {}
        if "metadata_json" in data:
            try:
                metadata = json.loads(str(data["metadata_json"][0]))
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
        return cls(
            weights=np.asarray(data["weights"], dtype=np.float32),
            input_dim=int(data["input_dim"][0]),
            output_dim=int(data["output_dim"][0]),
            ridge_alpha=float(data["ridge_alpha"][0]),
            version=str(data["version"][0]),
            metadata=metadata,
        )


def save_item_embedding_sidecar(
    path: str,
    *,
    embedding_map: Dict[str, np.ndarray],
    clip_available: Dict[str, bool],
    item_features: Dict[str, np.ndarray],
    model_version: Optional[str],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    product_ids = sorted(embedding_map.keys())
    embeddings = np.vstack([normalize_vector(embedding_map[pid]) for pid in product_ids]).astype(
        np.float32
    )
    features = np.vstack(
        [
            np.asarray(
                item_features.get(pid, np.zeros(ItemFeatureEncoder.FEATURE_DIM, dtype=np.float32)),
                dtype=np.float32,
            )
            for pid in product_ids
        ]
    )
    np.savez_compressed(
        target,
        product_ids=np.asarray(product_ids),
        embeddings=embeddings,
        clip_available=np.asarray([bool(clip_available.get(pid, False)) for pid in product_ids]),
        item_features=features.astype(np.float32),
        model_version=np.asarray([model_version or ""]),
        created_at=np.asarray([time.time()], dtype=np.float64),
    )


def load_item_embedding_sidecar(
    path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, bool], Dict[str, np.ndarray], Optional[str]]:
    target = Path(path)
    if not target.exists():
        return {}, {}, {}, None
    data = np.load(target, allow_pickle=False)
    product_ids = [str(pid) for pid in data["product_ids"]]
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    clip_available_array = np.asarray(data["clip_available"], dtype=bool)
    features = np.asarray(data["item_features"], dtype=np.float32)
    model_version = str(data["model_version"][0]) if "model_version" in data else None
    return (
        {pid: normalize_vector(embeddings[index]) for index, pid in enumerate(product_ids)},
        {pid: bool(clip_available_array[index]) for index, pid in enumerate(product_ids)},
        {pid: features[index].astype(np.float32) for index, pid in enumerate(product_ids)},
        model_version or None,
    )


def metadata_affinity(query_metadata: Dict[str, Any], neighbor_metadata: Dict[str, Any]) -> float:
    affinity = 1.0
    if query_metadata.get("category") and query_metadata.get("category") == neighbor_metadata.get("category"):
        affinity *= 1.25
    if query_metadata.get("brand") and query_metadata.get("brand") == neighbor_metadata.get("brand"):
        affinity *= 1.1

    query_price = _safe_positive_float(query_metadata.get("price"))
    neighbor_price = _safe_positive_float(neighbor_metadata.get("price"))
    if query_price and neighbor_price:
        ratio = max(query_price, neighbor_price) / max(min(query_price, neighbor_price), 1e-6)
        if ratio >= 3.0:
            affinity *= 0.6
        elif ratio >= 2.0:
            affinity *= 0.8
    return float(affinity)


def build_hybrid_synthetic_embedding(
    *,
    neighbor_embeddings: Sequence[np.ndarray],
    neighbor_similarities: Sequence[float],
    adapter_embedding: np.ndarray,
    query_metadata: Dict[str, Any],
    neighbor_metadatas: Sequence[Dict[str, Any]],
    neighbor_weight: float,
    softmax_temperature: float,
    configured_neighbor_count: int,
) -> Tuple[np.ndarray, float, List[float]]:
    if not neighbor_embeddings:
        raise ValueError("at least one neighbor embedding is required")

    similarities = np.asarray(neighbor_similarities, dtype=np.float32)
    affinities = np.asarray(
        [metadata_affinity(query_metadata, metadata) for metadata in neighbor_metadatas],
        dtype=np.float32,
    )
    weights = softmax_with_affinity(
        similarities,
        affinities,
        temperature=max(float(softmax_temperature), 1e-6),
    )
    normalized_neighbors = np.vstack([normalize_vector(emb) for emb in neighbor_embeddings])
    neighbor_cf = normalize_vector(weights @ normalized_neighbors)
    adapter_cf = normalize_vector(adapter_embedding)
    alpha = min(1.0, max(0.0, float(neighbor_weight)))
    synthetic = normalize_vector(alpha * neighbor_cf + (1.0 - alpha) * adapter_cf)
    confidence = synthetic_confidence(
        similarities,
        configured_neighbor_count=configured_neighbor_count,
        valid_neighbor_count=len(neighbor_embeddings),
    )
    return synthetic, confidence, weights.astype(float).tolist()


def softmax_with_affinity(
    similarities: np.ndarray,
    affinities: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    affinities = np.maximum(np.asarray(affinities, dtype=np.float32), 1e-6)
    logits = np.asarray(similarities, dtype=np.float32) / temperature + np.log(affinities)
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    total = float(np.sum(exp_logits))
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(exp_logits, 1.0 / len(exp_logits), dtype=np.float32)
    return (exp_logits / total).astype(np.float32)


def synthetic_confidence(
    similarities: np.ndarray,
    *,
    configured_neighbor_count: int,
    valid_neighbor_count: int,
) -> float:
    if len(similarities) == 0:
        return 0.0
    average_similarity = float(np.mean(np.asarray(similarities, dtype=np.float32)))
    coverage = min(
        1.0,
        valid_neighbor_count / max(float(configured_neighbor_count), 1.0),
    )
    return float(min(1.0, max(0.0, average_similarity * (0.5 + 0.5 * coverage))))


def is_cold_start_eligible(
    *,
    product_id: str,
    metadata: Dict[str, Any],
    clip_embedding: Optional[np.ndarray],
    trained_item_ids: Iterable[str],
    current_time: float,
    max_age_days: float,
    max_interactions: float,
) -> bool:
    if product_id in set(trained_item_ids):
        return False
    if metadata.get("active") is False or metadata.get("in_stock") is False:
        return False
    if metadata.get("deleted") is True or metadata.get("is_deleted") is True:
        return False
    if clip_embedding is None:
        return False
    if float(np.linalg.norm(np.asarray(clip_embedding, dtype=np.float32))) <= 0.0:
        return False

    interaction_count = _metadata_interaction_count(metadata)
    created_at = float(metadata.get("created_at", current_time))
    age_days = max((current_time - created_at) / 86400.0, 0.0)
    return age_days <= float(max_age_days) and interaction_count <= float(max_interactions)


def _metadata_interaction_count(metadata: Dict[str, Any]) -> float:
    for key in ("interaction_count", "num_interactions", "views", "view_count"):
        if key in metadata:
            return max(0.0, float(metadata.get(key) or 0.0))
    return 0.0


def _safe_positive_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0 or not np.isfinite(parsed):
        return None
    return parsed
