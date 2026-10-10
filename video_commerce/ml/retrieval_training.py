"""Adapters that keep Two-Tower training and evaluation pinned to retrieval PIT."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from video_commerce.ml.retrieval_evaluation import RetrievalEvaluationQuery
from video_commerce.ml.retrieval_pit_dataset import (
    RetrievalPitDataset,
    RetrievalPitRow,
    RetrievalHoldoutWindow,
    split_retrieval_holdout,
)

POSITIVE_LABELS = frozenset({"click", "add_to_cart", "purchase"})
EXPOSURE_NEGATIVE_LABELS = frozenset({"viewed_no_positive", "no_positive"})
NEGATIVE_SOURCE_IMPRESSION_NO_CLICK = "impression_no_click"
NEGATIVE_SOURCE_RANKER_REJECTED = "ranker_rejected"


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


def build_retrieval_training_inputs(
    dataset: RetrievalPitDataset,
    *,
    holdout_days: int,
    ranker_rejected_mode: str = "weak",
) -> RetrievalTrainingInputs:
    """Build legacy trainer tensors without consulting mutable online state."""
    rejected_mode = str(ranker_rejected_mode or "").strip().lower()
    if rejected_mode not in {"disabled", "weak", "teacher_soft"}:
        raise ValueError("ranker_rejected_mode must be disabled, weak, or teacher_soft")
    training_rows, holdout_rows, window = split_retrieval_holdout(
        dataset.rows,
        holdout_days=holdout_days,
        attribution_cutoff=dataset.manifest.attribution_cutoff,
    )
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
            teacher_target = None
            if rejected_mode == "teacher_soft":
                score = float(row.ranker_score or 0.0)
                teacher_target = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
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
    ordered = sorted(
        eligible_product_ids,
        key=lambda product_id: (-training_exposures.get(product_id, 0), product_id),
    )
    head_count = max(1, int(math.ceil(len(ordered) * 0.2))) if ordered else 0
    long_tail = frozenset(ordered[head_count:])
    grouped: dict[str, list[RetrievalPitRow]] = {}
    for row in holdout_rows:
        grouped.setdefault(row.query_id, []).append(row)
    queries = []
    for query_id, rows in sorted(grouped.items()):
        relevant = frozenset(
            row.product_id for row in rows if row.label_type in POSITIVE_LABELS
        )
        if not relevant:
            continue
        representative = min(rows, key=lambda row: (row.as_of_ts, row.product_id))
        cold_items = frozenset(
            product_id
            for product_id in eligible_product_ids
            if (
                representative.as_of_ts
                - float(
                    product_metadata[product_id].get(
                        "created_at", representative.as_of_ts
                    )
                )
                <= 30 * 86400.0
                or training_exposures.get(product_id, 0) <= 5
            )
        )
        modality_names = sorted(
            {
                name
                for product_id in eligible_product_ids
                for name in dict(
                    product_metadata[product_id].get("modality_presence") or {}
                )
            }
        )
        missing_by_name = {
            name: frozenset(
                product_id
                for product_id in eligible_product_ids
                if not bool(
                    dict(
                        product_metadata[product_id].get("modality_presence") or {}
                    ).get(name, False)
                )
            )
            for name in modality_names
        }
        missing_modality = (
            frozenset().union(*missing_by_name.values())
            if missing_by_name
            else frozenset()
        )
        queries.append(
            RetrievalEvaluationQuery(
                query_id=query_id,
                user_id=representative.user_id,
                relevant_product_ids=relevant,
                eligible_product_ids=tuple(eligible_product_ids),
                seen_product_ids=frozenset(representative.seen_product_ids),
                cold_user=int(
                    representative.user_features.get("total_interactions", 0) or 0
                )
                < 5,
                cold_item_product_ids=cold_items,
                long_tail_product_ids=long_tail,
                missing_modality_product_ids=missing_modality,
                missing_modality_product_ids_by_name=missing_by_name,
                as_of_ts=representative.as_of_ts,
                user_features=dict(representative.user_features),
            )
        )
    return tuple(queries)
