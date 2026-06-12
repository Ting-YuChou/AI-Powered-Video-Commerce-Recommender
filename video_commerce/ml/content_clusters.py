"""Content-embedding product clustering helpers.

The clustering artifact is intentionally independent from online serving. It
can be produced by offline workers and then loaded by recommendation workers as
a cheap candidate-pool hint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass
class ContentClusterArtifact:
    cluster_model_version: str
    num_clusters: int
    centroids: np.ndarray
    product_cluster_map: Dict[str, int]
    product_count: int
    embedding_dim: int
    source_catalog_context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_content_cluster_artifact(
    product_embeddings: Dict[str, Any],
    *,
    num_clusters: int,
    source_catalog_context: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> ContentClusterArtifact:
    """Cluster product content embeddings into stable internal bucket IDs."""
    if num_clusters <= 0:
        raise ValueError("num_clusters must be > 0")

    product_ids, embedding_matrix = _normalized_embedding_matrix(product_embeddings)
    product_count = len(product_ids)
    if product_count == 0:
        raise ValueError("Cannot build content clusters without product embeddings")

    effective_clusters = min(int(num_clusters), product_count)
    if effective_clusters == 1:
        labels = np.zeros(product_count, dtype=np.int64)
        centroids = embedding_matrix[:1].copy()
    else:
        model = MiniBatchKMeans(
            n_clusters=effective_clusters,
            random_state=random_state,
            batch_size=max(1024, effective_clusters * 10),
            n_init=10,
            reassignment_ratio=0.0,
        )
        labels = model.fit_predict(embedding_matrix)
        centroids = _normalize_matrix(model.cluster_centers_)

    product_cluster_map = {
        product_id: int(label)
        for product_id, label in zip(product_ids, labels)
    }
    catalog_context = dict(source_catalog_context or {})
    model_version = _build_cluster_model_version(
        product_ids,
        embedding_matrix,
        effective_clusters,
        catalog_context,
    )
    return ContentClusterArtifact(
        cluster_model_version=model_version,
        num_clusters=effective_clusters,
        centroids=centroids.astype(np.float32),
        product_cluster_map=product_cluster_map,
        product_count=product_count,
        embedding_dim=int(embedding_matrix.shape[1]),
        source_catalog_context=catalog_context,
        created_at=time.time(),
        metadata={
            "requested_num_clusters": int(num_clusters),
            "random_state": int(random_state),
        },
    )


def save_content_cluster_artifact(
    artifact: ContentClusterArtifact,
    *,
    metadata_path: str,
    centroids_path: str,
) -> None:
    """Persist a cluster artifact as JSON metadata plus an NPZ centroid sidecar."""
    metadata_target = Path(metadata_path)
    centroids_target = Path(centroids_path)
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    centroids_target.parent.mkdir(parents=True, exist_ok=True)

    centroids_tmp = centroids_target.with_suffix(centroids_target.suffix + ".tmp")
    with centroids_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            centroids=np.asarray(artifact.centroids, dtype=np.float32),
            created_at=np.asarray([artifact.created_at], dtype=np.float64),
        )
    centroids_tmp.replace(centroids_target)
    centroids_sha256 = _file_sha256(centroids_target)
    metadata = {
        "schema_version": 1,
        "cluster_model_version": artifact.cluster_model_version,
        "num_clusters": int(artifact.num_clusters),
        "product_count": int(artifact.product_count),
        "embedding_dim": int(artifact.embedding_dim),
        "created_at": float(artifact.created_at),
        "centroids_path": centroids_target.name,
        "centroids_sha256": centroids_sha256,
        "source_catalog_context": artifact.source_catalog_context,
        "product_cluster_map": {
            product_id: int(cluster_id)
            for product_id, cluster_id in artifact.product_cluster_map.items()
        },
        "metadata": dict(artifact.metadata or {}),
    }
    tmp_path = metadata_target.with_suffix(metadata_target.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(metadata, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(metadata_target)


def load_content_cluster_artifact(
    *,
    metadata_path: str,
    centroids_path: Optional[str] = None,
) -> ContentClusterArtifact:
    """Load and validate a persisted content cluster artifact."""
    metadata_target = Path(metadata_path)
    with metadata_target.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if int(metadata.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported content cluster artifact schema")

    resolved_centroids_path = _resolve_centroids_path(
        metadata_target,
        centroids_path or metadata.get("centroids_path"),
    )
    expected_centroids_sha256 = metadata.get("centroids_sha256")
    if expected_centroids_sha256:
        actual_centroids_sha256 = _file_sha256(resolved_centroids_path)
        if actual_centroids_sha256 != expected_centroids_sha256:
            raise ValueError("Cluster centroid checksum does not match metadata")
    with np.load(resolved_centroids_path, allow_pickle=False) as data:
        centroids = np.asarray(data["centroids"], dtype=np.float32)

    if centroids.ndim != 2 or centroids.shape[0] <= 0:
        raise ValueError("Cluster centroids must be a non-empty matrix")
    centroids = _normalize_matrix(centroids)

    product_cluster_map = {
        str(product_id): int(cluster_id)
        for product_id, cluster_id in (metadata.get("product_cluster_map") or {}).items()
    }
    num_clusters = int(metadata.get("num_clusters") or centroids.shape[0])
    if num_clusters != int(centroids.shape[0]):
        raise ValueError("Cluster centroid count does not match metadata")

    return ContentClusterArtifact(
        cluster_model_version=str(metadata.get("cluster_model_version") or ""),
        num_clusters=num_clusters,
        centroids=centroids.astype(np.float32),
        product_cluster_map=product_cluster_map,
        product_count=int(metadata.get("product_count") or len(product_cluster_map)),
        embedding_dim=int(metadata.get("embedding_dim") or centroids.shape[1]),
        source_catalog_context=dict(metadata.get("source_catalog_context") or {}),
        created_at=float(metadata.get("created_at") or 0.0),
        metadata=dict(metadata.get("metadata") or {}),
    )


def assign_embedding_to_clusters(
    embedding: Any,
    centroids: np.ndarray,
    *,
    limit: int = 1,
    centroids_normalized: bool = False,
) -> List[int]:
    """Return nearest cluster IDs for a content embedding."""
    if limit <= 0:
        return []
    query = _normalize_embedding(embedding)
    if query is None:
        return []
    centroid_matrix = (
        np.asarray(centroids, dtype=np.float32)
        if centroids_normalized
        else _normalize_matrix(centroids)
    )
    if centroid_matrix.ndim != 2 or centroid_matrix.shape[0] == 0:
        return []
    if query.shape[0] != centroid_matrix.shape[1]:
        return []

    scores = centroid_matrix @ query
    ordered = np.argsort(scores)[::-1]
    return [int(cluster_id) for cluster_id in ordered[:limit]]


def _normalized_embedding_matrix(
    product_embeddings: Dict[str, Any],
) -> Tuple[List[str], np.ndarray]:
    product_ids: List[str] = []
    embeddings: List[np.ndarray] = []
    expected_dim: Optional[int] = None
    for product_id in sorted(product_embeddings.keys()):
        normalized = _normalize_embedding(product_embeddings[product_id])
        if normalized is None:
            continue
        if expected_dim is None:
            expected_dim = int(normalized.shape[0])
        if normalized.shape[0] != expected_dim:
            continue
        product_ids.append(str(product_id))
        embeddings.append(normalized)
    if not embeddings:
        return [], np.empty((0, 0), dtype=np.float32)
    return product_ids, np.vstack(embeddings).astype(np.float32)


def _normalize_embedding(embedding: Any) -> Optional[np.ndarray]:
    try:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        return None
    return (vector / norm).astype(np.float32)


def _normalize_matrix(matrix: Any) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        return values
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where((norms > 0.0) & np.isfinite(norms), norms, 1.0)
    return (values / norms).astype(np.float32)


def _resolve_centroids_path(metadata_path: Path, centroids_path: Optional[str]) -> Path:
    if not centroids_path:
        return metadata_path.with_suffix(".centroids.npz")
    candidate = Path(centroids_path)
    if candidate.is_absolute():
        return candidate
    return metadata_path.parent / candidate


def _build_cluster_model_version(
    product_ids: Iterable[str],
    embedding_matrix: np.ndarray,
    num_clusters: int,
    source_catalog_context: Dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(str(int(num_clusters)).encode("utf-8"))
    digest.update(
        json.dumps(
            source_catalog_context,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    for product_id in product_ids:
        digest.update(str(product_id).encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.asarray(embedding_matrix, dtype=np.float32).tobytes())
    return f"content-cluster-{int(num_clusters)}-{digest.hexdigest()[:16]}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
