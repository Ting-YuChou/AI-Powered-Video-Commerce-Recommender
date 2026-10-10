"""Full-catalog metrics and governed quality gate for Two-Tower retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Mapping, Sequence

import numpy as np


RETRIEVAL_QUALITY_GATE_VERSION = "retrieval_quality_gate_v1"


class RetrievalGateDecision(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class RetrievalEvaluationQuery:
    query_id: str
    user_id: str
    relevant_product_ids: frozenset[str]
    eligible_product_ids: tuple[str, ...]
    seen_product_ids: frozenset[str]
    cold_user: bool = False
    cold_item_product_ids: frozenset[str] = frozenset()
    long_tail_product_ids: frozenset[str] = frozenset()
    missing_modality_product_ids: frozenset[str] = frozenset()
    missing_modality_product_ids_by_name: Mapping[str, frozenset[str]] = field(
        default_factory=dict
    )
    as_of_ts: float = 0.0
    user_features: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalEvaluationConfig:
    min_queries: int = 1000
    min_users: int = 200
    min_relevant_labels: int = 100
    min_slice_queries: int = 200
    min_slice_users: int = 50
    min_slice_relevant_labels: int = 20
    bootstrap_samples: int = 2000
    random_seed: int = 42
    recall_at_100_min_delta: float = 0.0
    recall_at_100_ci_lower: float = -0.005
    recall_secondary_max_regression: float = 0.01
    mrr_max_regression: float = 0.005
    slice_recall_at_100_min_delta: float = -0.02
    slice_recall_at_100_ci_lower: float = -0.03
    eligible_encoding_coverage_min: float = 0.995
    long_tail_relative_regression_max: float = 0.10


@dataclass(frozen=True)
class RetrievalQualityGateResult:
    decision: RetrievalGateDecision
    reasons: tuple[str, ...]
    policy_version: str
    overall: Mapping[str, object]
    slices: Mapping[str, Mapping[str, object]]
    bootstrap: Mapping[str, float]


def score_full_catalog_embeddings(
    queries: Sequence[RetrievalEvaluationQuery],
    item_embeddings: Mapping[str, np.ndarray],
    *,
    encode_user,
) -> dict[str, dict[str, float]]:
    """Score every encoded eligible item; no sampled candidate set is accepted."""
    scores: dict[str, dict[str, float]] = {}
    for query in queries:
        user_embedding = np.asarray(encode_user(query), dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(user_embedding)):
            raise ValueError("retrieval user embedding is not finite")
        query_scores = {}
        for product_id in query.eligible_product_ids:
            embedding = item_embeddings.get(product_id)
            if embedding is None:
                continue
            item_embedding = np.asarray(embedding, dtype=np.float64).reshape(-1)
            if item_embedding.shape != user_embedding.shape:
                raise ValueError("retrieval user and item embedding dimensions differ")
            score = float(np.dot(user_embedding, item_embedding))
            if not math.isfinite(score):
                raise ValueError("retrieval score is not finite")
            query_scores[product_id] = score
        scores[query.query_id] = query_scores
    return scores


def score_ann_index(
    queries: Sequence[RetrievalEvaluationQuery],
    index: object,
    index_map: Mapping[int, str],
    *,
    encode_user,
) -> dict[str, dict[str, float]]:
    """Search an ANN index containing the complete pinned eligible catalog."""
    total = int(getattr(index, "ntotal", 0))
    if total <= 0 or len(index_map) != total:
        raise ValueError("retrieval ANN index does not cover its catalog mapping")
    scores: dict[str, dict[str, float]] = {}
    for query in queries:
        vector = np.asarray(encode_user(query), dtype=np.float32).reshape(1, -1)
        if not np.all(np.isfinite(vector)):
            raise ValueError("retrieval ANN query embedding is not finite")
        distances, indices = index.search(vector, total)
        query_scores: dict[str, float] = {}
        for raw_index, raw_score in zip(indices[0], distances[0]):
            faiss_index = int(raw_index)
            if faiss_index < 0:
                continue
            product_id = index_map.get(faiss_index)
            score = float(raw_score)
            if product_id is None or not math.isfinite(score):
                raise ValueError("retrieval ANN result is incompatible or non-finite")
            query_scores[str(product_id)] = score
        scores[query.query_id] = query_scores
    return scores


def evaluate_full_catalog(
    queries: Sequence[RetrievalEvaluationQuery],
    scores: Mapping[str, Mapping[str, float]],
    *,
    cutoffs: Sequence[int] = (50, 100, 200),
) -> dict[str, object]:
    normalized_cutoffs = tuple(sorted({int(k) for k in cutoffs if int(k) > 0}))
    if not normalized_cutoffs:
        raise ValueError("retrieval evaluation requires at least one positive cutoff")
    recalls = {cutoff: [] for cutoff in normalized_cutoffs}
    hit_rates = {cutoff: [] for cutoff in normalized_cutoffs}
    reciprocal_ranks: list[float] = []
    ranked: dict[str, list[str]] = {}
    all_eligible: set[str] = set()
    encoded_eligible: set[str] = set()
    exposed: set[str] = set()
    long_tail_slots = 0
    total_slots = 0
    max_cutoff = max(normalized_cutoffs)

    for query in queries:
        eligible = set(query.eligible_product_ids) - set(query.seen_product_ids)
        relevant = set(query.relevant_product_ids) & eligible
        if not relevant:
            continue
        all_eligible.update(eligible)
        query_scores = scores.get(query.query_id, {})
        encoded_eligible.update(
            product for product in eligible if product in query_scores
        )
        ordered = sorted(
            (
                (product_id, float(query_scores[product_id]))
                for product_id in eligible
                if product_id in query_scores
            ),
            key=lambda item: (-item[1], item[0]),
        )
        ranked_ids = [product_id for product_id, _ in ordered]
        ranked[query.query_id] = ranked_ids
        recommendation_window = ranked_ids[:max_cutoff]
        exposed.update(recommendation_window)
        long_tail_slots += sum(
            product_id in query.long_tail_product_ids
            for product_id in recommendation_window
        )
        total_slots += len(recommendation_window)
        first_rank = next(
            (
                index
                for index, product_id in enumerate(ranked_ids, 1)
                if product_id in relevant
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        for cutoff in normalized_cutoffs:
            hits = len(set(ranked_ids[:cutoff]) & relevant)
            recalls[cutoff].append(hits / len(relevant))
            hit_rates[cutoff].append(float(hits > 0))

    result: dict[str, object] = {
        "queries": len(reciprocal_ranks),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "eligible_item_encoding_coverage": (
            len(encoded_eligible) / len(all_eligible) if all_eligible else 0.0
        ),
        "recommendation_catalog_coverage": (
            len(exposed) / len(all_eligible) if all_eligible else 0.0
        ),
        "long_tail_exposure_share": (
            long_tail_slots / total_slots if total_slots else 0.0
        ),
        "ranked_product_ids": ranked,
    }
    for cutoff in normalized_cutoffs:
        result[f"recall_at_{cutoff}"] = (
            float(np.mean(recalls[cutoff])) if recalls[cutoff] else 0.0
        )
        result[f"hit_rate_at_{cutoff}"] = (
            float(np.mean(hit_rates[cutoff])) if hit_rates[cutoff] else 0.0
        )
    return result


def exact_audit_ann_recall(
    queries: Sequence[RetrievalEvaluationQuery],
    ann_scores: Mapping[str, Mapping[str, float]],
    exact_scores: Mapping[str, Mapping[str, float]],
    *,
    cutoffs: Sequence[int] = (50, 100, 200),
) -> dict[str, float | int]:
    """Measure ANN top-k overlap against exact full-catalog dot products."""
    normalized = tuple(sorted({int(value) for value in cutoffs if int(value) > 0}))
    if not normalized:
        raise ValueError("ANN audit requires at least one positive cutoff")
    overlaps = {cutoff: [] for cutoff in normalized}
    for query in queries:
        eligible = set(query.eligible_product_ids) - set(query.seen_product_ids)
        exact_ranked = _rank_scores(exact_scores.get(query.query_id, {}), eligible)
        ann_ranked = _rank_scores(ann_scores.get(query.query_id, {}), eligible)
        if not exact_ranked:
            continue
        for cutoff in normalized:
            expected = exact_ranked[:cutoff]
            denominator = len(expected)
            overlaps[cutoff].append(
                len(set(expected) & set(ann_ranked[:cutoff])) / denominator
                if denominator
                else 0.0
            )
    audited = max((len(values) for values in overlaps.values()), default=0)
    result: dict[str, float | int] = {"audited_queries": audited}
    for cutoff, values in overlaps.items():
        result[f"ann_recall_at_{cutoff}"] = float(np.mean(values)) if values else 0.0
    return result


def _rank_scores(scores: Mapping[str, float], eligible: set[str]) -> list[str]:
    return [
        product_id
        for product_id, _ in sorted(
            (
                (product_id, float(score))
                for product_id, score in scores.items()
                if product_id in eligible and math.isfinite(float(score))
            ),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def evaluate_retrieval_gate(
    queries: Sequence[RetrievalEvaluationQuery],
    champion_scores: Mapping[str, Mapping[str, float]],
    challenger_scores: Mapping[str, Mapping[str, float]],
    config: RetrievalEvaluationConfig,
) -> RetrievalQualityGateResult:
    if _has_non_finite_scores(champion_scores) or _has_non_finite_scores(
        challenger_scores
    ):
        return RetrievalQualityGateResult(
            RetrievalGateDecision.FAILED,
            ("non_finite_score",),
            RETRIEVAL_QUALITY_GATE_VERSION,
            {},
            {},
            {},
        )
    evidence = _evidence_reasons(queries, config, slice_name=None)
    overall = _compare(queries, champion_scores, challenger_scores)
    bootstrap = _bootstrap_recall_delta(
        queries, champion_scores, challenger_scores, config
    )
    overall = {**overall, **bootstrap}
    slices: dict[str, Mapping[str, object]] = {}
    insufficient: list[str] = []
    failed: list[str] = []
    for name, selected in _slices(queries):
        if not selected:
            slices[name] = {"status": "not_applicable"}
            continue
        reasons = _evidence_reasons(selected, config, slice_name=name)
        if reasons:
            slices[name] = {
                "status": RetrievalGateDecision.INSUFFICIENT_EVIDENCE.value,
                "reasons": reasons,
            }
            insufficient.append(name)
            continue
        metrics = _compare(selected, champion_scores, challenger_scores)
        ci = _bootstrap_recall_delta(
            selected, champion_scores, challenger_scores, config
        )
        metrics = {**metrics, **ci}
        reasons = _gate_failures(metrics, config, slice_name=name)
        slices[name] = {
            "status": (
                RetrievalGateDecision.FAILED.value
                if reasons
                else RetrievalGateDecision.PASSED.value
            ),
            "reasons": reasons,
            "metrics": metrics,
        }
        if reasons:
            failed.append(name)

    if evidence or insufficient:
        decision = RetrievalGateDecision.INSUFFICIENT_EVIDENCE
        reasons = tuple(evidence) + tuple(f"slice_{name}" for name in insufficient)
    else:
        gate_failures = _gate_failures(overall, config, slice_name=None)
        gate_failures.extend(f"slice_{name}" for name in failed)
        decision = (
            RetrievalGateDecision.FAILED
            if gate_failures
            else RetrievalGateDecision.PASSED
        )
        reasons = tuple(gate_failures)
    return RetrievalQualityGateResult(
        decision,
        reasons,
        RETRIEVAL_QUALITY_GATE_VERSION,
        overall,
        slices,
        bootstrap,
    )


def _compare(queries, champion_scores, challenger_scores):
    champion = evaluate_full_catalog(queries, champion_scores)
    challenger = evaluate_full_catalog(queries, challenger_scores)
    return {
        "queries": int(challenger["queries"]),
        "users": len({query.user_id for query in queries}),
        "relevant_labels": sum(len(query.relevant_product_ids) for query in queries),
        "recall_at_50_delta": float(challenger["recall_at_50"])
        - float(champion["recall_at_50"]),
        "recall_at_100_delta": float(challenger["recall_at_100"])
        - float(champion["recall_at_100"]),
        "recall_at_200_delta": float(challenger["recall_at_200"])
        - float(champion["recall_at_200"]),
        "mrr_delta": float(challenger["mrr"]) - float(champion["mrr"]),
        "eligible_item_encoding_coverage": float(
            challenger["eligible_item_encoding_coverage"]
        ),
        "long_tail_exposure_share_champion": float(
            champion["long_tail_exposure_share"]
        ),
        "long_tail_exposure_share_challenger": float(
            challenger["long_tail_exposure_share"]
        ),
    }


def _evidence_reasons(queries, config, *, slice_name):
    prefix = f"slice_{slice_name}" if slice_name else "overall"
    query_count = len(queries)
    user_count = len({query.user_id for query in queries})
    label_count = sum(len(query.relevant_product_ids) for query in queries)
    limits = (
        (
            config.min_slice_queries,
            config.min_slice_users,
            config.min_slice_relevant_labels,
        )
        if slice_name
        else (config.min_queries, config.min_users, config.min_relevant_labels)
    )
    reasons = []
    if query_count < limits[0]:
        reasons.append(f"{prefix}_queries")
    if user_count < limits[1]:
        reasons.append(f"{prefix}_users")
    if label_count < limits[2]:
        reasons.append(f"{prefix}_relevant_labels")
    return reasons


def _slices(queries):
    yield "cold_user", [query for query in queries if query.cold_user]
    yield "cold_item", [
        query
        for query in queries
        if query.relevant_product_ids & query.cold_item_product_ids
    ]
    yield "warm_item", [
        query
        for query in queries
        if query.relevant_product_ids
        and not (query.relevant_product_ids & query.cold_item_product_ids)
    ]
    yield "long_tail", [
        query
        for query in queries
        if query.relevant_product_ids & query.long_tail_product_ids
    ]
    yield "missing_modality", [
        query
        for query in queries
        if query.relevant_product_ids & query.missing_modality_product_ids
    ]
    modality_names = sorted(
        {
            name
            for query in queries
            for name in query.missing_modality_product_ids_by_name
        }
    )
    for name in modality_names:
        yield f"missing_modality_{name}", [
            query
            for query in queries
            if query.relevant_product_ids
            & query.missing_modality_product_ids_by_name.get(name, frozenset())
        ]
    yield "warm_user", [query for query in queries if not query.cold_user]


def _bootstrap_recall_delta(queries, champion_scores, challenger_scores, config):
    by_user: dict[str, list[RetrievalEvaluationQuery]] = {}
    for query in queries:
        by_user.setdefault(query.user_id, []).append(query)
    users = sorted(by_user)
    if not users:
        return {
            "recall_at_100_delta_ci_lower": 0.0,
            "recall_at_100_delta_ci_upper": 0.0,
        }
    generator = np.random.default_rng(config.random_seed)
    deltas = []
    for sample_index in range(max(1, int(config.bootstrap_samples))):
        sampled_users = generator.choice(users, size=len(users), replace=True)
        sampled_queries = []
        sampled_champion = {}
        sampled_challenger = {}
        for cluster_index, user_id in enumerate(sampled_users):
            for query in by_user[str(user_id)]:
                query_id = f"{sample_index}:{cluster_index}:{query.query_id}"
                sampled_queries.append(
                    RetrievalEvaluationQuery(**{**query.__dict__, "query_id": query_id})
                )
                sampled_champion[query_id] = champion_scores[query.query_id]
                sampled_challenger[query_id] = challenger_scores[query.query_id]
        champion = evaluate_full_catalog(sampled_queries, sampled_champion)
        challenger = evaluate_full_catalog(sampled_queries, sampled_challenger)
        deltas.append(
            float(challenger["recall_at_100"]) - float(champion["recall_at_100"])
        )
    return {
        "recall_at_100_delta_ci_lower": float(np.quantile(deltas, 0.025)),
        "recall_at_100_delta_ci_upper": float(np.quantile(deltas, 0.975)),
    }


def _gate_failures(metrics, config, *, slice_name):
    prefix = f"slice_{slice_name}" if slice_name else "overall"
    failures = []
    point_limit = (
        config.slice_recall_at_100_min_delta
        if slice_name
        else config.recall_at_100_min_delta
    )
    ci_limit = (
        config.slice_recall_at_100_ci_lower
        if slice_name
        else config.recall_at_100_ci_lower
    )
    if float(metrics["recall_at_100_delta"]) < point_limit:
        failures.append(f"{prefix}_recall_at_100_delta")
    if float(metrics["recall_at_100_delta_ci_lower"]) < ci_limit:
        failures.append(f"{prefix}_recall_at_100_ci_lower")
    if not slice_name:
        for cutoff in (50, 200):
            if float(metrics[f"recall_at_{cutoff}_delta"]) < -float(
                config.recall_secondary_max_regression
            ):
                failures.append(f"overall_recall_at_{cutoff}_delta")
        if float(metrics["mrr_delta"]) < -float(config.mrr_max_regression):
            failures.append("overall_mrr_delta")
        if float(metrics["eligible_item_encoding_coverage"]) < float(
            config.eligible_encoding_coverage_min
        ):
            failures.append("overall_eligible_item_encoding_coverage")
        champion_tail = float(metrics["long_tail_exposure_share_champion"])
        challenger_tail = float(metrics["long_tail_exposure_share_challenger"])
        relative_regression = (
            (champion_tail - challenger_tail) / champion_tail
            if champion_tail > 0
            else 0.0
        )
        if relative_regression > float(config.long_tail_relative_regression_max):
            failures.append("overall_long_tail_exposure_regression")
    return failures


def _has_non_finite_scores(scores):
    return any(
        not math.isfinite(float(value))
        for query_scores in scores.values()
        for value in query_scores.values()
    )
