"""Deterministic source-aware candidate fusion."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from video_commerce.common.models import CandidateProduct, RetrievalSourceSignal


SOURCE_REGISTRY_VERSION = "retrieval_sources_v1"
FUSION_POLICY_VERSION = "candidate_rrf_v1"
CANONICAL_SOURCES = (
    "two_tower",
    "sasrec",
    "swing_itemcf",
    "content",
    "trending",
    "category",
    "cluster",
    "new_item",
    "random",
    "unknown",
)
_SOURCE_ALIASES = {
    "cf": "two_tower",
    "collaborative": "two_tower",
    "content_similarity": "content",
    "trending_pool": "trending",
    "popular_fallback": "trending",
    "category_pool": "category",
    "cluster_pool": "cluster",
}
_SOURCE_ORDER = {source: index for index, source in enumerate(CANONICAL_SOURCES)}


def canonical_source(source: str) -> str:
    normalized = str(source or "").strip().lower()
    canonical = _SOURCE_ALIASES.get(normalized, normalized)
    return canonical if canonical in _SOURCE_ORDER else "unknown"


def _native_score(candidate: CandidateProduct, source: str) -> float:
    if source in {"two_tower", "sasrec", "swing_itemcf"}:
        value = candidate.collaborative_score
    elif source == "content":
        value = candidate.content_similarity_score
    elif source in {"trending", "category", "cluster", "new_item", "random"}:
        value = candidate.popularity_score
    else:
        value = None
    return float(candidate.combined_score if value is None else value)


def _merge_legacy_fields(
    existing: CandidateProduct, incoming: CandidateProduct
) -> None:
    for field_name in (
        "collaborative_score",
        "content_similarity_score",
        "popularity_score",
    ):
        value = getattr(incoming, field_name)
        if value is None:
            continue
        if getattr(existing, field_name) is None:
            setattr(existing, field_name, float(value))


def merge_candidates_rrf(
    source_candidates: Mapping[str, Sequence[CandidateProduct]],
    *,
    limit: int,
    rrf_k: int = 60,
) -> list[CandidateProduct]:
    """Union source pools and rank by source position rather than raw score scale."""
    if limit <= 0:
        return []
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    merged: Dict[str, CandidateProduct] = {}
    for source_name, candidates in source_candidates.items():
        canonical = canonical_source(source_name)
        count = len(candidates)
        if count <= 0:
            continue
        for rank, candidate in enumerate(candidates, start=1):
            product_id = str(candidate.product_id)
            existing = merged.get(product_id)
            if existing is None:
                existing = candidate.copy(deep=True)
                existing.retrieval_signals = {}
                merged[product_id] = existing
            else:
                _merge_legacy_fields(existing, candidate)
            previous_signal = existing.retrieval_signals.get(canonical)
            if previous_signal is not None and previous_signal.source_rank <= rank:
                continue
            normalized_rank = (
                1.0 if count == 1 else float(count - rank) / float(count - 1)
            )
            reciprocal_rank = 1.0 / float(rrf_k + rank)
            existing.retrieval_signals[canonical] = RetrievalSourceSignal(
                raw_source=str(source_name),
                raw_score=_native_score(candidate, canonical),
                source_rank=rank,
                source_count=count,
                normalized_rank=normalized_rank,
                reciprocal_rank=reciprocal_rank,
            )

    for candidate in merged.values():
        sources = sorted(
            candidate.retrieval_signals,
            key=lambda source: (_SOURCE_ORDER.get(source, 999), source),
        )
        candidate.source = "+".join(sources) or "unknown"
        candidate.combined_score = sum(
            signal.reciprocal_rank for signal in candidate.retrieval_signals.values()
        )
        candidate.fusion_policy_version = FUSION_POLICY_VERSION
        candidate.source_registry_version = SOURCE_REGISTRY_VERSION

    ordered = sorted(
        merged.values(),
        key=lambda candidate: (-float(candidate.combined_score), candidate.product_id),
    )[:limit]
    for position, candidate in enumerate(ordered, start=1):
        candidate.candidate_pool_position = position
    return ordered


def retrieval_lineage_snapshot(candidate: CandidateProduct) -> dict:
    """Return bounded JSON-ready fusion lineage for impression and PIT payloads."""
    return {
        "candidate_pool_position": candidate.candidate_pool_position,
        "fusion_policy_version": candidate.fusion_policy_version,
        "source_registry_version": candidate.source_registry_version,
        "retrieval_signals": {
            source: signal.dict()
            for source, signal in sorted(candidate.retrieval_signals.items())
        },
    }
