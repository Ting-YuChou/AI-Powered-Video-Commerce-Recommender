"""
Post-ranker slate diversity helpers.

These helpers are intentionally pure and synchronous so they can run as a cheap
final-stage rerank after the ranking model has already produced a small pool.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from models import ProductRecommendation

EmbeddingLookup = Callable[[str], Optional[np.ndarray]]


def select_mmr_recommendations(
    recommendations: List[ProductRecommendation],
    *,
    k: int,
    embedding_lookup: EmbeddingLookup,
    lambda_weight: float = 0.8,
) -> List[ProductRecommendation]:
    """Select a final slate using maximal marginal relevance.

    Relevance comes from ProductRecommendation.ranking_score and similarity is
    cosine similarity over product embeddings. Missing embeddings participate
    normally by taking a zero similarity penalty.
    """
    if k <= 0 or not recommendations:
        return []

    k = min(k, len(recommendations))
    lambda_weight = min(1.0, max(0.0, float(lambda_weight)))
    normalized_relevance = _normalize_relevance(
        [float(item.ranking_score) for item in recommendations]
    )
    embeddings = _load_normalized_embeddings(
        (item.product_id for item in recommendations),
        embedding_lookup,
    )

    selected_indices: List[int] = []
    remaining_indices = list(range(len(recommendations)))

    while remaining_indices and len(selected_indices) < k:
        best_index = max(
            remaining_indices,
            key=lambda idx: _mmr_score(
                idx,
                selected_indices,
                normalized_relevance,
                embeddings,
                lambda_weight,
            ),
        )
        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    return [recommendations[idx] for idx in selected_indices]


def _normalize_relevance(scores: List[float]) -> List[float]:
    if not scores:
        return []
    finite_scores = [score if np.isfinite(score) else 0.0 for score in scores]
    min_score = min(finite_scores)
    max_score = max(finite_scores)
    span = max_score - min_score
    if span <= 1e-12:
        return [1.0 for _ in finite_scores]
    return [(score - min_score) / span for score in finite_scores]


def _load_normalized_embeddings(
    product_ids: Iterable[str],
    embedding_lookup: EmbeddingLookup,
) -> Dict[int, Optional[np.ndarray]]:
    embeddings: Dict[int, Optional[np.ndarray]] = {}
    for index, product_id in enumerate(product_ids):
        try:
            embedding = embedding_lookup(product_id)
        except Exception:
            embedding = None
        if embedding is None:
            embeddings[index] = None
            continue
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0 or not np.isfinite(norm):
            embeddings[index] = None
            continue
        embeddings[index] = vector / norm
    return embeddings


def _mmr_score(
    index: int,
    selected_indices: List[int],
    normalized_relevance: List[float],
    embeddings: Dict[int, Optional[np.ndarray]],
    lambda_weight: float,
) -> float:
    relevance_score = normalized_relevance[index]
    diversity_penalty = _max_similarity(index, selected_indices, embeddings)
    return lambda_weight * relevance_score - (1.0 - lambda_weight) * diversity_penalty


def _max_similarity(
    index: int,
    selected_indices: List[int],
    embeddings: Dict[int, Optional[np.ndarray]],
) -> float:
    if not selected_indices:
        return 0.0
    current = embeddings.get(index)
    if current is None:
        return 0.0

    max_similarity = 0.0
    for selected_index in selected_indices:
        selected = embeddings.get(selected_index)
        if selected is None:
            continue
        similarity = float(np.dot(current, selected))
        if np.isfinite(similarity):
            max_similarity = max(max_similarity, similarity)
    return max_similarity
