"""Adapters that keep Two-Tower training and evaluation pinned to retrieval PIT."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from video_commerce.ml.retrieval_evaluation import RetrievalEvaluationQuery
from video_commerce.ml.retrieval_pit_dataset import (
    RetrievalCatalog,
    RetrievalPitDataset,
    RetrievalPitManifest,
    RetrievalPitRow,
    RetrievalHoldoutWindow,
    split_retrieval_holdout,
)

POSITIVE_LABELS = frozenset({"click", "add_to_cart", "purchase"})
EXPOSURE_NEGATIVE_LABELS = frozenset({"viewed_no_positive", "no_positive"})
NEGATIVE_SOURCE_IMPRESSION_NO_CLICK = "impression_no_click"
NEGATIVE_SOURCE_RANKER_REJECTED = "ranker_rejected"
DEFAULT_TEACHER_SOFT_MIN_RANKER_SCORE_COVERAGE = 0.9
COLD_ITEM_AGE_SECONDS = 30 * 86400.0
COLD_ITEM_MAX_TRAINING_EXPOSURES = 5


def require_teacher_soft_coverage(
    manifest: RetrievalPitManifest | None, *, min_coverage: float
) -> float:
    """Refuse teacher_soft unless enough rejected rows carry a real ranker score.

    Rows without a score fall back to plain weak negatives (no teacher target)
    rather than a fabricated 0.0 logit. Below ``min_coverage`` the experiment
    would mostly be ``weak`` under another name, so it is rejected outright.
    """
    coverage = None if manifest is None else manifest.ranker_score_coverage
    if coverage is None or coverage < float(min_coverage):
        raise ValueError(
            "teacher_soft requires ranker_score coverage >= "
            f"{float(min_coverage):.3f} over ranker_rejected rows; "
            f"dataset coverage is {coverage if coverage is None else round(coverage, 4)}"
        )
    return coverage


def teacher_target_from_ranker_score(score: float | None) -> float | None:
    if score is None:
        return None
    clamped = max(-30.0, min(30.0, float(score)))
    return 1.0 / (1.0 + math.exp(-clamped))


def require_catalog_predates_holdout(
    catalog: RetrievalCatalog, window: RetrievalHoldoutWindow
) -> None:
    """Holdout queries must not be scored against item state from after them."""
    snapshot_at = max(
        float(catalog.available_at), float(catalog.item_features_available_at)
    )
    if snapshot_at > float(window.start_ts):
        raise ValueError(
            "retrieval catalog generation is newer than the holdout start; "
            "pin a generation available at or before the holdout window"
        )


@dataclass(frozen=True)
class RetrievalTrainingInputs:
    interactions: tuple[Mapping[str, Any], ...]
    external_negatives: tuple[Mapping[str, Any], ...]
    product_metadata: Mapping[str, Mapping[str, Any]]
    product_clip_embeddings: Mapping[str, np.ndarray]
    user_features_map: Mapping[str, Mapping[str, Any]]
    training_rows: tuple[RetrievalPitRow, ...]
    holdout_rows: tuple[RetrievalPitRow, ...]
    holdout_queries: tuple[RetrievalEvaluationQuery, ...]
    holdout_window: RetrievalHoldoutWindow
    training_as_of_ts: float


@dataclass(frozen=True)
class RetrievalStreamingInputs:
    source: Any
    product_metadata: Mapping[str, Mapping[str, Any]]
    product_clip_embeddings: Mapping[str, np.ndarray]
    holdout_queries: tuple[RetrievalEvaluationQuery, ...]
    holdout_window: RetrievalHoldoutWindow
    training_as_of_ts: float
    batch_rows: int


def build_retrieval_streaming_inputs(
    source,
    *,
    holdout_days: int,
    batch_rows: int,
) -> RetrievalStreamingInputs:
    """Build O(items + queries) evaluation state while event rows remain on disk."""
    window = source.assign_holdout(holdout_days=holdout_days)
    catalog = source.catalog
    require_catalog_predates_holdout(catalog, window)
    metadata = {}
    embeddings = {}
    eligible_ids = tuple(product.product_id for product in catalog.eligible_products)
    for product in catalog.eligible_products:
        feature = catalog.item_features[product.product_id]
        presence = dict(product.modality_presence)
        presence["clip_embedding"] = feature.clip_embedding is not None
        metadata[product.product_id] = {
            **dict(product.metadata),
            "active": product.active,
            "in_stock": product.in_stock,
            "modality_presence": presence,
        }
        embeddings[product.product_id] = (
            np.asarray(feature.clip_embedding, dtype=np.float32)
            if feature.clip_embedding is not None
            else np.zeros(catalog.embedding_dimension, dtype=np.float32)
        )
    exposures: dict[str, int] = {}
    training_as_of_ts = window.start_ts
    for rows in source.iter_row_batches(
        split="training", batch_rows=batch_rows, seed=42, epoch=0
    ):
        for row in rows:
            exposures[row.product_id] = exposures.get(row.product_id, 0) + 1
            training_as_of_ts = max(training_as_of_ts, float(row.as_of_ts))
    context = _EvaluationContext.build(eligible_ids, metadata, exposures)
    queries = []
    for rows in source.iter_row_batches(
        split="holdout", batch_rows=batch_rows, seed=42, epoch=0
    ):
        grouped: dict[str, list[RetrievalPitRow]] = {}
        for row in rows:
            grouped.setdefault(row.query_id, []).append(row)
        for query_id, query_rows in grouped.items():
            query = context.query(query_id, query_rows)
            if query is not None:
                queries.append(query)
    return RetrievalStreamingInputs(
        source=source,
        product_metadata=metadata,
        product_clip_embeddings=embeddings,
        holdout_queries=tuple(sorted(queries, key=lambda query: query.query_id)),
        holdout_window=window,
        training_as_of_ts=training_as_of_ts,
        batch_rows=int(batch_rows),
    )


def build_retrieval_training_inputs(
    dataset: RetrievalPitDataset,
    *,
    holdout_days: int,
    ranker_rejected_mode: str = "weak",
    teacher_soft_min_ranker_score_coverage: float = (
        DEFAULT_TEACHER_SOFT_MIN_RANKER_SCORE_COVERAGE
    ),
) -> RetrievalTrainingInputs:
    """Build legacy trainer tensors without consulting mutable online state."""
    rejected_mode = str(ranker_rejected_mode or "").strip().lower()
    if rejected_mode not in {"disabled", "weak", "teacher_soft"}:
        raise ValueError("ranker_rejected_mode must be disabled, weak, or teacher_soft")
    if rejected_mode == "teacher_soft":
        require_teacher_soft_coverage(
            dataset.manifest, min_coverage=teacher_soft_min_ranker_score_coverage
        )
    training_rows, holdout_rows, window = split_retrieval_holdout(
        dataset.rows,
        holdout_days=holdout_days,
        attribution_cutoff=dataset.manifest.attribution_cutoff,
    )
    require_catalog_predates_holdout(dataset.catalog, window)
    interactions = []
    negatives = []
    latest_user_features: dict[str, tuple[float, Mapping[str, Any]]] = {}
    for row in sorted(
        training_rows,
        key=lambda value: (
            value.as_of_ts,
            value.query_id,
            value.product_id,
            value.label_source,
        ),
    ):
        current = latest_user_features.get(row.user_id)
        if current is None or row.as_of_ts >= current[0]:
            latest_user_features[row.user_id] = (
                row.as_of_ts,
                dict(row.user_features),
            )
        if row.label_type in POSITIVE_LABELS:
            interactions.append(
                {
                    "user_id": row.user_id,
                    "product_id": row.product_id,
                    "action": row.label_type,
                    "event_id": f"retrieval:{row.query_id}:{row.product_id}:{row.label_type}",
                    "as_of_ts": row.as_of_ts,
                    "user_features": dict(row.user_features),
                }
            )
        elif row.label_type in EXPOSURE_NEGATIVE_LABELS:
            negatives.append(
                {
                    "user_id": row.user_id,
                    "product_id": row.product_id,
                    "source": NEGATIVE_SOURCE_IMPRESSION_NO_CLICK,
                    "weight": float(row.label_weight),
                    "exposed": True,
                    "sample_prob": 0.0,
                    "as_of_ts": row.as_of_ts,
                }
            )
        elif (
            row.label_source == NEGATIVE_SOURCE_RANKER_REJECTED
            or row.label_type == NEGATIVE_SOURCE_RANKER_REJECTED
        ):
            if rejected_mode == "disabled":
                continue
            # Unscored rows stay plain weak negatives in teacher_soft mode.
            teacher_target = (
                teacher_target_from_ranker_score(row.ranker_score)
                if rejected_mode == "teacher_soft"
                else None
            )
            negatives.append(
                {
                    "user_id": row.user_id,
                    "product_id": row.product_id,
                    "source": NEGATIVE_SOURCE_RANKER_REJECTED,
                    "weight": float(row.label_weight),
                    "exposed": False,
                    "sample_prob": 0.0,
                    "ranker_score": row.ranker_score,
                    "as_of_ts": row.as_of_ts,
                    **(
                        {"teacher_target": teacher_target}
                        if teacher_target is not None
                        else {}
                    ),
                }
            )

    catalog = dataset.catalog
    metadata = {}
    for product in catalog.eligible_products:
        feature = catalog.item_features[product.product_id]
        modality_presence = dict(product.modality_presence)
        modality_presence["clip_embedding"] = feature.clip_embedding is not None
        metadata[product.product_id] = {
            **dict(product.metadata),
            "active": product.active,
            "in_stock": product.in_stock,
            "modality_presence": modality_presence,
        }
    embeddings = {}
    for product in catalog.eligible_products:
        feature = catalog.item_features[product.product_id]
        embeddings[product.product_id] = (
            np.asarray(feature.clip_embedding, dtype=np.float32)
            if feature.clip_embedding is not None
            else np.zeros(catalog.embedding_dimension, dtype=np.float32)
        )

    holdout_queries = _evaluation_queries(
        training_rows=training_rows,
        holdout_rows=holdout_rows,
        product_metadata=metadata,
        eligible_product_ids=tuple(
            product.product_id for product in catalog.eligible_products
        ),
    )
    return RetrievalTrainingInputs(
        interactions=tuple(interactions),
        external_negatives=tuple(negatives),
        product_metadata=metadata,
        product_clip_embeddings=embeddings,
        user_features_map={
            user_id: dict(value[1]) for user_id, value in latest_user_features.items()
        },
        training_rows=tuple(training_rows),
        holdout_rows=tuple(holdout_rows),
        holdout_queries=holdout_queries,
        holdout_window=window,
        training_as_of_ts=max(
            (row.as_of_ts for row in training_rows), default=window.start_ts
        ),
    )


def _evaluation_queries(
    *,
    training_rows,
    holdout_rows,
    product_metadata,
    eligible_product_ids,
):
    training_exposures: dict[str, int] = {}
    for row in training_rows:
        training_exposures[row.product_id] = (
            training_exposures.get(row.product_id, 0) + 1
        )
    context = _EvaluationContext.build(
        tuple(eligible_product_ids), product_metadata, training_exposures
    )
    grouped: dict[str, list[RetrievalPitRow]] = {}
    for row in holdout_rows:
        grouped.setdefault(row.query_id, []).append(row)
    queries = []
    for query_id, rows in sorted(grouped.items()):
        query = context.query(query_id, rows)
        if query is not None:
            queries.append(query)
    return tuple(queries)


@dataclass(frozen=True)
class _EvaluationContext:
    """Query-independent evaluation state, computed once per holdout.

    ``long_tail`` and the missing-modality sets are shared by every query.
    Cold items depend on the query's ``as_of_ts``, but the gate only uses them
    as ``relevant & cold``. Each query therefore stores just its cold relevant
    items, an O(|relevant|) lookup, instead of an O(catalog) set per query.
    """

    eligible_ids: tuple[str, ...]
    created_at: Mapping[str, float | None]
    exposures: Mapping[str, int]
    long_tail: frozenset[str]
    missing_by_name: Mapping[str, frozenset[str]]
    missing_modality: frozenset[str]

    @classmethod
    def build(cls, eligible_ids, metadata, exposures) -> "_EvaluationContext":
        eligible_ids = tuple(eligible_ids)
        ordered = sorted(
            eligible_ids,
            key=lambda product_id: (-exposures.get(product_id, 0), product_id),
        )
        head_count = max(1, int(math.ceil(len(ordered) * 0.2))) if ordered else 0
        presence = {
            product_id: dict(metadata[product_id].get("modality_presence") or {})
            for product_id in eligible_ids
        }
        modality_names = sorted(
            {name for values in presence.values() for name in values}
        )
        missing_by_name = {
            name: frozenset(
                product_id
                for product_id in eligible_ids
                if not bool(presence[product_id].get(name, False))
            )
            for name in modality_names
        }
        return cls(
            eligible_ids=eligible_ids,
            created_at={
                product_id: (
                    float(metadata[product_id]["created_at"])
                    if metadata[product_id].get("created_at") is not None
                    else None
                )
                for product_id in eligible_ids
            },
            exposures=dict(exposures),
            long_tail=frozenset(ordered[head_count:]),
            missing_by_name=missing_by_name,
            missing_modality=(
                frozenset().union(*missing_by_name.values())
                if missing_by_name
                else frozenset()
            ),
        )

    def is_cold(self, product_id: str, as_of_ts: float) -> bool:
        if product_id not in self.created_at:
            return False
        if self.exposures.get(product_id, 0) <= COLD_ITEM_MAX_TRAINING_EXPOSURES:
            return True
        created_at = self.created_at[product_id]
        # A missing created_at counts as brand new, as before.
        return created_at is None or as_of_ts - created_at <= COLD_ITEM_AGE_SECONDS

    def query(self, query_id, rows) -> RetrievalEvaluationQuery | None:
        relevant = frozenset(
            row.product_id for row in rows if row.label_type in POSITIVE_LABELS
        )
        if not relevant:
            return None
        representative = min(rows, key=lambda row: (row.as_of_ts, row.product_id))
        as_of_ts = float(representative.as_of_ts)
        return RetrievalEvaluationQuery(
            query_id=query_id,
            user_id=representative.user_id,
            relevant_product_ids=relevant,
            eligible_product_ids=self.eligible_ids,
            seen_product_ids=frozenset(representative.seen_product_ids),
            cold_user=int(
                representative.user_features.get("total_interactions", 0) or 0
            )
            < 5,
            cold_item_product_ids=frozenset(
                product_id
                for product_id in relevant
                if self.is_cold(product_id, as_of_ts)
            ),
            long_tail_product_ids=self.long_tail,
            missing_modality_product_ids=self.missing_modality,
            missing_modality_product_ids_by_name=self.missing_by_name,
            as_of_ts=as_of_ts,
            user_features=dict(representative.user_features),
        )
