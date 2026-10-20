"""Full-catalog metrics and governed quality gate for Two-Tower retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Mapping, Sequence

import numpy as np


RETRIEVAL_QUALITY_GATE_VERSION = "retrieval_quality_gate_v1"

# faiss.METRIC_INNER_PRODUCT / faiss.METRIC_L2; mirrored here so offline
# evaluation does not need to import faiss just to read ``index.metric_type``.
FAISS_METRIC_INNER_PRODUCT = 0
FAISS_METRIC_L2 = 1

# Recall@40 matches the serving max_live_cf_candidates budget. It is reported
# (recall_at_40, recall_at_40_delta) but is not a gate blocker: blocking on it
# would change gate decisions and require a new gate policy version.
DEFAULT_CUTOFFS = (40, 50, 100, 200)
MAX_CUTOFF = max(DEFAULT_CUTOFFS)
# Exact scoring materializes (queries x items) float64 blocks of at most this
# many elements (~32 MiB) at a time.
_SCORING_BLOCK_ELEMENTS = 4_000_000


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
    # Note: builders in retrieval_training store only the cold items among
    # relevant_product_ids; the gate only ever reads relevant & cold.


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
    max_results: int = MAX_CUTOFF,
) -> dict[str, dict[str, float]]:
    """Exact inner-product top-k over every eligible, unseen item.

    Items are scored with chunked matrix products, and the top-k is chosen
    with a partition. Ties at the cut and in the output order break by
    ascending product ID, so results are deterministic. Seen items are
    excluded before selection because evaluation always drops them. Each
    query therefore keeps up to ``max_results`` rankable items.
    """
    limit = int(max_results)
    catalogs = _EligibleCatalogCache(item_embeddings)
    grouped: dict[int, list[RetrievalEvaluationQuery]] = {}
    for query in queries:
        grouped.setdefault(catalogs.key(query.eligible_product_ids), []).append(query)
    scores: dict[str, dict[str, float]] = {}
    for key, group in grouped.items():
        catalog = catalogs.catalog(key)
        item_count = len(catalog.product_ids)
        block_rows = max(1, _SCORING_BLOCK_ELEMENTS // max(item_count, 1))
        for start in range(0, len(group), block_rows):
            block = group[start : start + block_rows]
            users = np.stack(
                [
                    np.asarray(encode_user(query), dtype=np.float64).reshape(-1)
                    for query in block
                ]
            )
            if not np.all(np.isfinite(users)):
                raise ValueError("retrieval user embedding is not finite")
            if item_count == 0 or limit <= 0:
                scores.update({query.query_id: {} for query in block})
                continue
            if users.shape[1] != catalog.matrix.shape[1]:
                raise ValueError("retrieval user and item embedding dimensions differ")
            block_scores = users @ catalog.matrix.T
            if not np.all(np.isfinite(block_scores)):
                raise ValueError("retrieval score is not finite")
            for row, query in zip(block_scores, block):
                scores[query.query_id] = _top_k_row(
                    row, catalog, query.seen_product_ids, limit
                )
    return scores


@dataclass(frozen=True)
class _ItemCatalog:
    product_ids: tuple[str, ...]  # ascending, so column order == tie order
    positions: Mapping[str, int]
    matrix: np.ndarray


class _EligibleCatalogCache:
    """Build one item matrix per distinct eligible tuple (usually exactly one)."""

    def __init__(self, item_embeddings: Mapping[str, np.ndarray]) -> None:
        self.item_embeddings = item_embeddings
        self._key_by_identity: dict[int, tuple[tuple[str, ...], int]] = {}
        self._key_by_value: dict[tuple[str, ...], int] = {}
        self._catalogs: list[_ItemCatalog] = []

    def key(self, eligible: tuple[str, ...]) -> int:
        cached = self._key_by_identity.get(id(eligible))
        if cached is not None and cached[0] is eligible:
            return cached[1]
        key = self._key_by_value.get(eligible)
        if key is None:
            key = len(self._catalogs)
            self._catalogs.append(self._build(eligible))
            self._key_by_value[eligible] = key
        self._key_by_identity[id(eligible)] = (eligible, key)
        return key

    def catalog(self, key: int) -> _ItemCatalog:
        return self._catalogs[key]

    def _build(self, eligible: tuple[str, ...]) -> _ItemCatalog:
        entries: dict[str, np.ndarray] = {}
        for product_id in eligible:
            embedding = self.item_embeddings.get(product_id)
            if embedding is not None:
                entries[str(product_id)] = np.asarray(
                    embedding, dtype=np.float64
                ).reshape(-1)
        product_ids = tuple(sorted(entries))
        if not product_ids:
            return _ItemCatalog((), {}, np.zeros((0, 0), dtype=np.float64))
        if len({entries[product_id].shape for product_id in product_ids}) != 1:
            raise ValueError("retrieval user and item embedding dimensions differ")
        return _ItemCatalog(
            product_ids,
            {product_id: index for index, product_id in enumerate(product_ids)},
            np.stack([entries[product_id] for product_id in product_ids]),
        )


def _top_k_row(row, catalog: _ItemCatalog, seen, limit: int) -> dict[str, float]:
    seen_positions = [
        catalog.positions[str(product_id)]
        for product_id in seen
        if str(product_id) in catalog.positions
    ]
    if seen_positions:
        row = row.copy()
        row[seen_positions] = -np.inf
    k = min(limit, len(row) - len(seen_positions))
    if k <= 0:
        return {}
    if k < len(row):
        threshold = np.partition(row, len(row) - k)[len(row) - k]
        above = np.flatnonzero(row > threshold)
        ties = np.flatnonzero(row == threshold)[: k - len(above)]
        selected = np.concatenate((above, ties))
    else:
        selected = np.arange(len(row))
    ordered = selected[np.lexsort((selected, -row[selected]))]
    return {catalog.product_ids[index]: float(row[index]) for index in ordered}


def score_ann_index(
    queries: Sequence[RetrievalEvaluationQuery],
    index: object,
    index_map: Mapping[int, str],
    *,
    encode_user,
    max_results: int = MAX_CUTOFF,
) -> dict[str, dict[str, float]]:
    """Search an ANN index containing the complete pinned eligible catalog.

    Returned scores are always higher-is-better: inner-product indexes keep
    the raw similarity and L2 indexes (pre-``two_tower_index_v2`` champions)
    report the negated distance. The search over-fetches by the query's seen
    count and drops seen items, so up to ``max_results`` rankable items remain.
    """
    total = int(getattr(index, "ntotal", 0))
    if total <= 0 or len(index_map) != total:
        raise ValueError("retrieval ANN index does not cover its catalog mapping")
    metric_type = getattr(index, "metric_type", None)
    if metric_type == FAISS_METRIC_INNER_PRODUCT:
        score_sign = 1.0
    elif metric_type == FAISS_METRIC_L2:
        score_sign = -1.0
    else:
        raise ValueError(f"retrieval ANN index metric is unsupported: {metric_type!r}")
    scores: dict[str, dict[str, float]] = {}
    for query in queries:
        vector = np.asarray(encode_user(query), dtype=np.float32).reshape(1, -1)
        if not np.all(np.isfinite(vector)):
            raise ValueError("retrieval ANN query embedding is not finite")
        seen = {str(product_id) for product_id in query.seen_product_ids}
        distances, indices = index.search(
            vector, min(total, int(max_results) + len(seen))
        )
        query_scores: dict[str, float] = {}
        for raw_index, raw_score in zip(indices[0], distances[0]):
            faiss_index = int(raw_index)
            if faiss_index < 0:
                continue
            product_id = index_map.get(faiss_index)
            score = score_sign * float(raw_score)
            if product_id is None or not math.isfinite(score):
                raise ValueError("retrieval ANN result is incompatible or non-finite")
            if str(product_id) in seen or len(query_scores) >= int(max_results):
                continue
            query_scores[str(product_id)] = score
        scores[query.query_id] = query_scores
    return scores


def evaluate_full_catalog(
    queries: Sequence[RetrievalEvaluationQuery],
    scores: Mapping[str, Mapping[str, float]],
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    encoding_coverage: float | None = None,
    include_ranked_product_ids: bool = True,
) -> dict[str, object]:
    """Full-catalog recall/hit-rate/MRR over each query's eligible, unseen items.

    Per-query work is O(|scores| log |scores|): eligible sets are built once
    per distinct eligible tuple, never per query. Pass
    ``include_ranked_product_ids=False`` for metrics that are persisted.
    """
    normalized_cutoffs = tuple(sorted({int(k) for k in cutoffs if int(k) > 0}))
    if not normalized_cutoffs:
        raise ValueError("retrieval evaluation requires at least one positive cutoff")
    recalls = {cutoff: [] for cutoff in normalized_cutoffs}
    hit_rates = {cutoff: [] for cutoff in normalized_cutoffs}
    reciprocal_ranks: list[float] = []
    ranked: dict[str, list[str]] = {}
    eligible_sets = _EligibleSetCache()
    # Per eligible set: products seen by every evaluated query. The union of
    # (eligible - seen) over queries is eligible minus that intersection.
    always_seen: dict[int, set[str]] = {}
    encoded_eligible: set[str] = set()
    exposed: set[str] = set()
    long_tail_slots = 0
    total_slots = 0
    max_cutoff = max(normalized_cutoffs)

    for query in queries:
        key, eligible = eligible_sets.get(query.eligible_product_ids)
        outcome = _rank_query(query, scores, eligible)
        if outcome is None:
            continue
        ranked_ids, relevant = outcome
        seen_eligible = {
            product_id for product_id in query.seen_product_ids if product_id in eligible
        }
        always_seen[key] = (
            seen_eligible if key not in always_seen else always_seen[key] & seen_eligible
        )
        encoded_eligible.update(ranked_ids)
        if include_ranked_product_ids:
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

    if len(always_seen) <= 1:
        eligible_count = sum(
            len(eligible_sets.set(key)) - len(seen) for key, seen in always_seen.items()
        )
    else:
        eligible_count = len(
            set().union(
                *(eligible_sets.set(key) - seen for key, seen in always_seen.items())
            )
        )

    result: dict[str, object] = {
        "queries": len(reciprocal_ranks),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "eligible_item_encoding_coverage": (
            float(encoding_coverage)
            if encoding_coverage is not None
            else len(encoded_eligible) / eligible_count if eligible_count else 0.0
        ),
        "recommendation_catalog_coverage": (
            len(exposed) / eligible_count if eligible_count else 0.0
        ),
        "long_tail_exposure_share": (
            long_tail_slots / total_slots if total_slots else 0.0
        ),
    }
    if include_ranked_product_ids:
        result["ranked_product_ids"] = ranked
    for cutoff in normalized_cutoffs:
        result[f"recall_at_{cutoff}"] = (
            float(np.mean(recalls[cutoff])) if recalls[cutoff] else 0.0
        )
        result[f"hit_rate_at_{cutoff}"] = (
            float(np.mean(hit_rates[cutoff])) if hit_rates[cutoff] else 0.0
        )
    return result


class _EligibleSetCache:
    """Share one frozenset per distinct eligible tuple across queries."""

    def __init__(self) -> None:
        self._by_identity: dict[int, tuple[tuple[str, ...], int]] = {}
        self._by_value: dict[tuple[str, ...], int] = {}
        self._sets: list[frozenset[str]] = []

    def get(self, eligible: tuple[str, ...]) -> tuple[int, frozenset[str]]:
        cached = self._by_identity.get(id(eligible))
        if cached is not None and cached[0] is eligible:
            return cached[1], self._sets[cached[1]]
        key = self._by_value.get(eligible)
        if key is None:
            key = len(self._sets)
            self._sets.append(frozenset(eligible))
            self._by_value[eligible] = key
        self._by_identity[id(eligible)] = (eligible, key)
        return key, self._sets[key]

    def set(self, key: int) -> frozenset[str]:
        return self._sets[key]


def _rank_query(query, scores, eligible):
    """Return (ranked unseen eligible IDs, relevant set), or None if unrankable."""
    seen = query.seen_product_ids
    relevant = {
        product_id
        for product_id in query.relevant_product_ids
        if product_id in eligible and product_id not in seen
    }
    if not relevant:
        return None
    query_scores = scores.get(query.query_id, {})
    ordered = sorted(
        (
            (product_id, float(score))
            for product_id, score in query_scores.items()
            if product_id in eligible and product_id not in seen
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return [product_id for product_id, _ in ordered], relevant


def _per_query_recall(queries, scores, cutoff):
    """Recall@cutoff per query and whether the query is evaluable at all."""
    eligible_sets = _EligibleSetCache()
    recall = np.zeros(len(queries), dtype=np.float64)
    evaluated = np.zeros(len(queries), dtype=bool)
    for index, query in enumerate(queries):
        outcome = _rank_query(
            query, scores, eligible_sets.get(query.eligible_product_ids)[1]
        )
        if outcome is None:
            continue
        ranked_ids, relevant = outcome
        evaluated[index] = True
        recall[index] = len(set(ranked_ids[:cutoff]) & relevant) / len(relevant)
    return recall, evaluated


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
    challenger_encoding_coverage: float | None = None,
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
    overall = _compare(
        queries,
        champion_scores,
        challenger_scores,
        challenger_encoding_coverage=challenger_encoding_coverage,
    )
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


def _compare(
    queries,
    champion_scores,
    challenger_scores,
    *,
    challenger_encoding_coverage=None,
):
    champion = evaluate_full_catalog(
        queries, champion_scores, include_ranked_product_ids=False
    )
    challenger = evaluate_full_catalog(
        queries,
        challenger_scores,
        encoding_coverage=challenger_encoding_coverage,
        include_ranked_product_ids=False,
    )
    return {
        "queries": int(challenger["queries"]),
        "users": len({query.user_id for query in queries}),
        "relevant_labels": sum(len(query.relevant_product_ids) for query in queries),
        "recall_at_40_delta": float(challenger["recall_at_40"])
        - float(champion["recall_at_40"]),
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
    """User-cluster bootstrap CI for the Recall@100 delta.

    Per-query recall is computed once per model. Each resample draws user
    indices and sums per-user recall deltas and evaluable-query counts, which
    equals re-running evaluate_full_catalog on the duplicated queries. The
    RNG draws match the previous per-sample ``choice(users)`` calls.
    """
    users = sorted({query.user_id for query in queries})
    if not users:
        return {
            "recall_at_100_delta_ci_lower": 0.0,
            "recall_at_100_delta_ci_upper": 0.0,
        }
    user_index = {user_id: index for index, user_id in enumerate(users)}
    query_users = np.asarray(
        [user_index[query.user_id] for query in queries], dtype=np.int64
    )
    champion, evaluated = _per_query_recall(queries, champion_scores, 100)
    challenger, _ = _per_query_recall(queries, challenger_scores, 100)
    delta_by_user = np.bincount(
        query_users[evaluated],
        weights=(challenger - champion)[evaluated],
        minlength=len(users),
    )
    count_by_user = np.bincount(query_users[evaluated], minlength=len(users))
    generator = np.random.default_rng(config.random_seed)
    samples = max(1, int(config.bootstrap_samples))
    deltas = np.zeros(samples, dtype=np.float64)
    for sample_index in range(samples):
        sampled = generator.choice(len(users), size=len(users), replace=True)
        count = count_by_user[sampled].sum()
        if count:
            deltas[sample_index] = delta_by_user[sampled].sum() / count
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
