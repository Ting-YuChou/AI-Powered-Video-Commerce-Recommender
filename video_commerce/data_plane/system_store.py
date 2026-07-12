"""
Durable Postgres-backed system store for operational state and training data.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import time
from collections import Counter
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Set

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    Integer,
    JSON,
    String,
    Text,
    delete,
    desc,
    event,
    func,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from video_commerce.common.config import DatabaseConfig
from video_commerce.common.feature_history_contracts import (
    build_catalog_feature_event,
    payload_sha256,
)

logger = logging.getLogger(__name__)

POSITIVE_SEQUENCE_ACTIONS = ("view", "click", "add_to_cart", "purchase")
MAX_REJECTED_CANDIDATE_ITEMS = 50
IMPRESSION_CONTEXT_KEYS = {
    "session_id",
    "surface",
    "device",
    "time_of_day",
    "location",
    "priority",
    "demo_mode",
    "content_id",
    "category",
    "request_category",
    "recommendation_count",
}


@dataclass(frozen=True)
class PreparedCatalogActivation:
    activation_id: str
    source_version: str
    expected_count: int
    manifest_hash: str
    snapshot_rows: List[Dict[str, Any]]
    outbox_rows: List[Dict[str, Any]]


def prepare_catalog_activation(
    *,
    source_version: str,
    metadata_map: Mapping[str, Mapping[str, Any]],
    event_time: float,
    available_at: float,
) -> PreparedCatalogActivation:
    """Build deterministic snapshot/outbox rows before opening a DB transaction."""
    normalized_version = str(source_version or "").strip()
    if not normalized_version:
        raise ValueError("source_version must not be blank")
    activation_id = hashlib.sha256(
        f"catalog_activation\x00{normalized_version}".encode("utf-8")
    ).hexdigest()
    snapshot_rows: List[Dict[str, Any]] = []
    outbox_rows: List[Dict[str, Any]] = []
    for product_id in sorted(metadata_map):
        payload = dict(metadata_map[product_id])
        event = build_catalog_feature_event(
            product_id=product_id,
            source_version=normalized_version,
            event_time=event_time,
            available_at=available_at,
            payload=payload,
        )
        event["activation_id"] = activation_id
        snapshot_rows.append(
            {
                "product_id": product_id,
                "snapshot": payload,
                "updated_at": _coerce_datetime(available_at),
            }
        )
        outbox_rows.append(
            {
                "event_id": event["event_id"],
                "activation_id": activation_id,
                "product_id": product_id,
                "payload": event["payload"],
                "payload_hash": event["payload_hash"],
                "event_payload": event,
                "event_time": _coerce_datetime(event_time),
                "available_at": _coerce_datetime(available_at),
            }
        )
    return PreparedCatalogActivation(
        activation_id=activation_id,
        source_version=normalized_version,
        expected_count=len(outbox_rows),
        manifest_hash=payload_sha256(
            {
                "source_version": normalized_version,
                "products": {
                    row["product_id"]: row["snapshot"] for row in snapshot_rows
                },
            }
        ),
        snapshot_rows=snapshot_rows,
        outbox_rows=outbox_rows,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _event_value(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(key, default)
    return getattr(event, key, default)


def _event_to_sequence_dict(event: Any) -> Dict[str, Any]:
    occurred_at = (
        _coerce_datetime(_event_value(event, "event_time"))
        or _coerce_datetime(_event_value(event, "occurred_at"))
        or _utc_now()
    )
    return {
        "event_id": _event_value(event, "event_id"),
        "schema_version": int(_event_value(event, "schema_version", 1) or 1),
        "request_id": _event_value(event, "request_id"),
        "user_id": _event_value(event, "user_id"),
        "product_id": _event_value(event, "product_id"),
        "action": _event_value(event, "action"),
        "context": _event_value(event, "context", {}) or {},
        "event_time": occurred_at.timestamp(),
        "timestamp": occurred_at.timestamp(),
        "occurred_at": occurred_at.timestamp(),
    }


def build_chronological_user_sequences(
    events: Iterable[Any],
    *,
    max_events_per_user: int,
    min_sequence_length: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build bounded positive product sequences keyed by user id."""
    if max_events_per_user <= 0:
        return {}

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        user_id = _event_value(event, "user_id")
        product_id = _event_value(event, "product_id")
        action = _event_value(event, "action")
        if not user_id or not product_id or action not in POSITIVE_SEQUENCE_ACTIONS:
            continue
        grouped.setdefault(user_id, []).append(_event_to_sequence_dict(event))

    min_sequence_length = max(1, int(min_sequence_length))
    bounded: Dict[str, List[Dict[str, Any]]] = {}
    for user_id, sequence in grouped.items():
        sequence.sort(
            key=lambda item: (
                float(item.get("occurred_at") or 0.0),
                str(item.get("event_id") or ""),
            )
        )
        sequence = sequence[-max_events_per_user:]
        if len(sequence) >= min_sequence_length:
            bounded[user_id] = sequence
    return bounded


def _interaction_relevance(action: Any) -> float:
    normalized = str(action or "").lower()
    if normalized == "purchase":
        return 4.0
    if normalized == "add_to_cart":
        return 3.0
    if normalized == "click":
        return 2.0
    if normalized == "view":
        return 1.0
    return 0.0


def _first_numeric_value(*values: Any) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _safe_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bounded_json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            return value[:512]
        return value
    if isinstance(value, list):
        bounded = []
        for item in value[:20]:
            if isinstance(item, (str, int, float, bool)) or item is None:
                bounded.append(item[:512] if isinstance(item, str) else item)
        return bounded
    return None


def _impression_context_snapshot(context: Any) -> Dict[str, Any]:
    source = _safe_dict(context)
    snapshot: Dict[str, Any] = {}
    for key in IMPRESSION_CONTEXT_KEYS:
        if key not in source:
            continue
        bounded = _bounded_json_value(source.get(key))
        if bounded is not None:
            snapshot[key] = bounded
    return snapshot


def _timestamp(value: Any, fallback: Optional[datetime] = None) -> float:
    coerced = _coerce_datetime(value) or fallback or _utc_now()
    return coerced.timestamp()


def _score_snapshot_from_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    scores = _safe_dict(item.get("scores"))
    for key in (
        "collaborative_score",
        "content_similarity_score",
        "popularity_score",
        "combined_score",
        "ranking_score",
        "confidence_score",
    ):
        if key in item and key not in scores:
            scores[key] = item.get(key)
    return scores


def build_ltr_training_samples_from_impression_records(
    impression_items: Iterable[Mapping[str, Any]],
    interactions: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Build slate-level ranking samples with no-click negatives."""
    strongest_interactions: Dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for interaction in interactions:
        context = _safe_dict(interaction.get("context"))
        impression_id = str(context.get("impression_id") or "").strip()
        product_id = str(interaction.get("product_id") or "").strip()
        user_id = str(interaction.get("user_id") or "").strip()
        if not impression_id or not product_id or not user_id:
            continue
        key = (impression_id, product_id, user_id)
        previous = strongest_interactions.get(key)
        if previous is None or _interaction_relevance(
            interaction.get("action")
        ) >= _interaction_relevance(previous.get("action")):
            strongest_interactions[key] = interaction

    samples: List[Dict[str, Any]] = []
    for item in impression_items:
        impression_id = str(item.get("impression_id") or "").strip()
        product_id = str(item.get("product_id") or "").strip()
        user_id = str(item.get("user_id") or "").strip()
        if not impression_id or not product_id or not user_id:
            continue

        matched_interaction = strongest_interactions.get(
            (impression_id, product_id, user_id)
        )
        action = str(
            matched_interaction.get("action") if matched_interaction else "view"
        )
        impression_context = _safe_dict(item.get("context"))
        interaction_context = (
            _safe_dict(matched_interaction.get("context"))
            if matched_interaction
            else {}
        )
        feature_snapshot = _safe_dict(item.get("feature_snapshot"))
        scores = _score_snapshot_from_item(item)
        source = (
            item.get("source")
            or feature_snapshot.get("candidate_source")
            or interaction_context.get("recommendation_source")
        )
        position = item.get("position")
        created_at = _coerce_datetime(item.get("created_at")) or _utc_now()
        occurred_at = (
            _coerce_datetime(matched_interaction.get("occurred_at"))
            if matched_interaction
            else created_at
        )
        context = {
            **impression_context,
            **interaction_context,
            "impression_id": impression_id,
            "recommendation_position": position,
            "recommendation_source": source,
            "recommendation_ranking_score": scores.get("ranking_score"),
            "candidate_scores": scores,
        }
        if item.get("content_id") is not None:
            context["content_id"] = item.get("content_id")
        if item.get("session_id") is not None:
            context["session_id"] = item.get("session_id")

        product_metadata = {
            key: feature_snapshot.get(key)
            for key in ("price", "category", "brand")
            if key in feature_snapshot
        }
        value = _first_numeric_value(
            matched_interaction.get("value") if matched_interaction else None,
            interaction_context.get("value"),
            matched_interaction.get("gmv") if matched_interaction else None,
            interaction_context.get("gmv"),
            matched_interaction.get("purchase_value") if matched_interaction else None,
            interaction_context.get("purchase_value"),
        )
        if value is None and action == "purchase":
            value = feature_snapshot.get("price")
        attributed_purchase = action == "purchase"
        attributed_click = action in {"click", "add_to_cart", "purchase"}
        business_value = _first_numeric_value(
            matched_interaction.get("margin") if matched_interaction else None,
            matched_interaction.get("profit") if matched_interaction else None,
            matched_interaction.get("gross_margin") if matched_interaction else None,
            interaction_context.get("margin"),
            interaction_context.get("profit"),
            interaction_context.get("gross_margin"),
            value,
            feature_snapshot.get("price") if attributed_purchase else None,
        )
        if business_value is None:
            business_value = 0.0
        context["attributed_click"] = attributed_click
        context["attributed_purchase"] = attributed_purchase
        context["business_value"] = business_value

        sample = {
            "event_id": (
                matched_interaction.get("event_id")
                if matched_interaction
                else f"{impression_id}:{product_id}:impression"
            ),
            "schema_version": (
                matched_interaction.get("schema_version", 1)
                if matched_interaction
                else 1
            ),
            "request_id": item.get("request_id"),
            "user_id": user_id,
            "product_id": product_id,
            "action": action,
            "context": context,
            "timestamp": _timestamp(occurred_at, created_at),
            "occurred_at": _timestamp(occurred_at, created_at),
            "source": source,
            "candidate_scores": scores,
            "product_metadata": product_metadata,
            "value": business_value,
            "purchase_value": value,
            "business_value": business_value,
        }
        sample.update(scores)
        samples.append(sample)

    return samples


TWO_TOWER_POSITIVE_ACTIONS = {"click", "add_to_cart", "purchase", "favorite", "share"}


def build_two_tower_training_negatives_from_impression_records(
    impression_items: Iterable[Mapping[str, Any]],
    interactions: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Build weak retrieval negatives from returned no-click and rejected candidates."""
    positive_keys: Set[tuple[str, str, str]] = set()
    for interaction in interactions:
        action = str(interaction.get("action") or "").lower()
        if action not in TWO_TOWER_POSITIVE_ACTIONS:
            continue
        context = _safe_dict(interaction.get("context"))
        impression_id = str(context.get("impression_id") or "").strip()
        product_id = str(interaction.get("product_id") or "").strip()
        user_id = str(interaction.get("user_id") or "").strip()
        if impression_id and product_id and user_id:
            positive_keys.add((impression_id, product_id, user_id))

    negatives: List[Dict[str, Any]] = []
    emitted: Set[tuple[str, str, str, str]] = set()
    rejected_contexts: Dict[tuple[str, str], List[Mapping[str, Any]]] = {}

    for item in impression_items:
        impression_id = str(item.get("impression_id") or "").strip()
        product_id = str(item.get("product_id") or "").strip()
        user_id = str(item.get("user_id") or "").strip()
        if not impression_id or not product_id or not user_id:
            continue

        context = _safe_dict(item.get("context"))
        rejected_items = context.get("rejected_candidate_items")
        if isinstance(rejected_items, list):
            rejected_contexts.setdefault((impression_id, user_id), [])
            for rejected in rejected_items:
                if isinstance(rejected, Mapping):
                    rejected_contexts[(impression_id, user_id)].append(rejected)

        if (impression_id, product_id, user_id) in positive_keys:
            continue
        key = (impression_id, product_id, user_id, "impression_no_click")
        if key in emitted:
            continue
        emitted.add(key)
        negatives.append(
            {
                "user_id": user_id,
                "product_id": product_id,
                "source": "impression_no_click",
                "weight": 0.25,
                "exposed": True,
                "sample_prob": 1.0,
                "rank_position": item.get("position"),
                "context": {
                    "impression_id": impression_id,
                    "recommendation_source": item.get("source")
                    or _safe_dict(item.get("feature_snapshot")).get("candidate_source"),
                    "candidate_scores": _score_snapshot_from_item(item),
                },
            }
        )

    for (impression_id, user_id), rejected_items in rejected_contexts.items():
        for rejected in rejected_items:
            product_id = str(rejected.get("product_id") or "").strip()
            if not product_id:
                continue
            if (impression_id, product_id, user_id) in positive_keys:
                continue
            key = (impression_id, product_id, user_id, "ranker_rejected")
            if key in emitted:
                continue
            emitted.add(key)
            negatives.append(
                {
                    "user_id": user_id,
                    "product_id": product_id,
                    "source": "ranker_rejected",
                    "weight": 0.15,
                    "exposed": False,
                    "sample_prob": 1.0,
                    "rank_position": rejected.get("position")
                    or rejected.get("rank_position"),
                    "context": {
                        "impression_id": impression_id,
                        "recommendation_source": rejected.get("source")
                        or rejected.get("candidate_source"),
                        "candidate_scores": _safe_dict(rejected.get("scores")),
                    },
                }
            )

    return negatives


class Base(DeclarativeBase):
    """Base metadata for system store tables."""


class InteractionEvent(Base):
    __tablename__ = "interaction_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    request_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    product_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    context: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class RecommendationImpression(Base):
    __tablename__ = "recommendation_impressions"

    impression_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True
    )
    content_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True
    )
    model_version: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ranking_model_version: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    context: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class RecommendationImpressionItem(Base):
    __tablename__ = "recommendation_impression_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    impression_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    product_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    feature_snapshot: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    scores: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ContentJob(Base):
    __tablename__ = "content_jobs"

    content_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    storage_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True
    )
    priority: Mapped[str] = mapped_column(String(32), nullable=False, default="normal")
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class ContentFeatureArtifact(Base):
    """Durable source of truth for offline multimodal training features."""

    __tablename__ = "content_feature_artifacts"

    content_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    features: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class ProductCatalogSnapshot(Base):
    __tablename__ = "product_catalog_snapshot"

    product_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    snapshot: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    activation_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True
    )
    source_version: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class CatalogActivation(Base):
    __tablename__ = "catalog_activations"

    activation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_version: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True
    )
    expected_count: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    actual_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="staging")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class CatalogFeatureOutbox(Base):
    __tablename__ = "catalog_feature_outbox"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    activation_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    product_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    event_payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    event_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    claimed_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    claim_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    published_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class FeatureHistoryBackfillRun(Base):
    __tablename__ = "feature_history_backfill_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    range_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    range_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    phase: Mapped[str] = mapped_column(String(32), nullable=False, default="catalog")
    cursor_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cursor_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    counts: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    reconciliation: Mapped[Dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class PitMaterializationRun(Base):
    __tablename__ = "pit_materialization_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    cutoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    phase: Mapped[str] = mapped_column(String(32), nullable=False, default="export")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    export_attempt: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    flink_job_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    worker_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    snapshot_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    manifest_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    row_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    quarantine_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    training_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    training_worker_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    training_lease_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    trained_model_version: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True
    )


class ModelCheckpoint(Base):
    __tablename__ = "model_checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(Text, nullable=False)
    materialization_run_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


Index(
    "uq_model_checkpoints_pit_run",
    ModelCheckpoint.model_name,
    ModelCheckpoint.materialization_run_id,
    unique=True,
    postgresql_where=ModelCheckpoint.materialization_run_id.is_not(None),
)


Index("ix_interaction_events_occurred_at_desc", InteractionEvent.occurred_at.desc())
Index(
    "ix_interaction_events_action_occurred_at",
    InteractionEvent.action,
    InteractionEvent.occurred_at.desc(),
)
Index(
    "ix_interaction_events_user_sequence",
    InteractionEvent.user_id,
    InteractionEvent.occurred_at,
    InteractionEvent.event_id,
)
Index(
    "ix_interaction_events_positive_user_sequence",
    InteractionEvent.user_id,
    InteractionEvent.occurred_at.desc(),
    InteractionEvent.event_id.desc(),
    postgresql_where=InteractionEvent.action.in_(POSITIVE_SEQUENCE_ACTIONS),
)
Index(
    "ix_model_checkpoints_latest",
    ModelCheckpoint.model_name,
    ModelCheckpoint.created_at.desc(),
    ModelCheckpoint.id.desc(),
)
Index(
    "ix_recommendation_impressions_user_created",
    RecommendationImpression.user_id,
    RecommendationImpression.created_at.desc(),
)
Index(
    "ix_recommendation_impressions_created_at",
    RecommendationImpression.created_at.desc(),
)
Index(
    "ix_recommendation_impression_items_product_created",
    RecommendationImpressionItem.product_id,
    RecommendationImpressionItem.created_at.desc(),
)
Index(
    "ux_recommendation_impression_items_impression_product",
    RecommendationImpressionItem.impression_id,
    RecommendationImpressionItem.product_id,
    unique=True,
)
Index(
    "ix_catalog_feature_outbox_pending",
    CatalogFeatureOutbox.published_at,
    CatalogFeatureOutbox.claim_expires_at,
    CatalogFeatureOutbox.created_at,
)
Index(
    "ux_feature_history_backfill_active_range",
    FeatureHistoryBackfillRun.range_start,
    FeatureHistoryBackfillRun.range_end,
    unique=True,
    postgresql_where=FeatureHistoryBackfillRun.status == "active",
)
Index(
    "ix_pit_materialization_runs_status_lease",
    PitMaterializationRun.status,
    PitMaterializationRun.lease_expires_at,
    PitMaterializationRun.cutoff_at,
)
Index(
    "uq_pit_single_running_materialization",
    PitMaterializationRun.status,
    unique=True,
    postgresql_where=PitMaterializationRun.status == "running",
)


@dataclass
class DatabaseHealth:
    status: str
    response_time_ms: float
    error: Optional[str] = None


class SystemStore:
    """Operational persistence layer backed by Postgres."""

    def __init__(self, config: DatabaseConfig, observability=None):
        self.config = config
        self.observability = observability
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker | None = None
        self.is_connected = False
        self._retention_cleanup_task: asyncio.Task | None = None
        self._analytics_summary_cache: Dict[int, tuple[float, Dict[str, Any]]] = {}

    @property
    def enabled(self) -> bool:
        return self.config.enable

    async def initialize(self) -> None:
        if not self.enabled:
            return

        self.engine = create_async_engine(
            self.config.url,
            pool_pre_ping=True,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            echo=self.config.echo_sql,
        )
        self._install_metrics_listeners()
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

        if self.config.auto_create_schema:
            async with self.engine.begin() as conn:
                if self.config.interaction_events_partitioned:
                    await conn.run_sync(
                        lambda sync_conn: Base.metadata.create_all(
                            sync_conn,
                            tables=[
                                table
                                for table in Base.metadata.sorted_tables
                                if table.name != InteractionEvent.__tablename__
                            ],
                        )
                    )
                    await self._ensure_partitioned_interaction_events(conn)
                else:
                    await conn.run_sync(Base.metadata.create_all)
                await self._ensure_feature_lake_operational_schema(conn)
                await self._ensure_operational_indexes(conn)

        self.is_connected = True
        self._update_pool_metrics()
        if self.config.enable_retention_cleanup:
            await self.prune_interaction_events()
            await self.prune_recommendation_impressions()
            self._retention_cleanup_task = asyncio.create_task(
                self._run_periodic_retention_cleanup(),
                name="postgres-interaction-retention-cleanup",
            )

    async def _ensure_operational_indexes(self, conn) -> None:
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_interaction_events_occurred_at_desc "
                "ON interaction_events (occurred_at DESC)"
            )
        )

        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_interaction_events_action_occurred_at "
                "ON interaction_events (action, occurred_at DESC)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_interaction_events_user_sequence "
                "ON interaction_events (user_id, occurred_at, event_id)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_interaction_events_positive_user_sequence "
                "ON interaction_events (user_id, occurred_at DESC, event_id DESC) "
                "WHERE action IN ('view', 'click', 'add_to_cart', 'purchase')"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_model_checkpoints_latest "
                "ON model_checkpoints (model_name, created_at DESC, id DESC)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_recommendation_impressions_user_created "
                "ON recommendation_impressions (user_id, created_at DESC)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_recommendation_impressions_created_at "
                "ON recommendation_impressions (created_at DESC)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_recommendation_impression_items_impression "
                "ON recommendation_impression_items (impression_id)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_recommendation_impression_items_product_created "
                "ON recommendation_impression_items (product_id, created_at DESC)"
            )
        )
        await conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "ux_recommendation_impression_items_impression_product "
                "ON recommendation_impression_items (impression_id, product_id)"
            )
        )

    async def _ensure_feature_lake_operational_schema(self, conn) -> None:
        """Apply additive feature-lake columns for existing operational tables."""
        await conn.execute(
            text(
                "ALTER TABLE product_catalog_snapshot "
                "ADD COLUMN IF NOT EXISTS activation_id VARCHAR(64), "
                "ADD COLUMN IF NOT EXISTS source_version VARCHAR(255)"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE catalog_activations "
                "ADD COLUMN IF NOT EXISTS manifest_hash VARCHAR(64)"
            )
        )

    async def _ensure_partitioned_interaction_events(self, conn) -> None:
        relation_result = await conn.execute(
            text(
                """
                SELECT c.relkind
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = current_schema()
                  AND c.relname = 'interaction_events'
                """
            )
        )
        relkind = relation_result.scalar_one_or_none()
        if relkind == "r":
            raise RuntimeError(
                "interaction_events already exists as a non-partitioned table; "
                "run migrations/postgres/001_partition_interaction_events.sql before "
                "setting DATABASE_INTERACTION_EVENTS_PARTITIONED=true"
            )

        if relkind is None:
            await conn.execute(
                text(
                    """
                    CREATE TABLE interaction_events (
                        event_id VARCHAR(64) NOT NULL,
                        schema_version INTEGER NOT NULL DEFAULT 1,
                        request_id VARCHAR(64),
                        user_id VARCHAR(255) NOT NULL,
                        product_id VARCHAR(255) NOT NULL,
                        action VARCHAR(64) NOT NULL,
                        context JSON NOT NULL DEFAULT '{}'::json,
                        occurred_at TIMESTAMPTZ NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        PRIMARY KEY (event_id, occurred_at)
                    ) PARTITION BY RANGE (occurred_at)
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS interaction_events_default
                    PARTITION OF interaction_events DEFAULT
                    """
                )
            )
        elif relkind != "p":
            raise RuntimeError(
                f"interaction_events has unsupported Postgres relkind {relkind!r}"
            )

        await self._ensure_interaction_event_partitions(conn)

    async def _ensure_interaction_event_partitions(self, conn) -> None:
        backfill_months = max(0, int(self.config.partition_backfill_months))
        premake_months = max(1, int(self.config.partition_premake_months))
        await conn.execute(
            text(
                f"""
                DO $$
                DECLARE
                    partition_start DATE := date_trunc('month', now())::date - INTERVAL '{backfill_months} months';
                    partition_end DATE := date_trunc('month', now())::date + INTERVAL '{premake_months} months';
                    current_start DATE;
                    current_end DATE;
                    partition_name TEXT;
                BEGIN
                    current_start := partition_start;
                    WHILE current_start < partition_end LOOP
                        current_end := current_start + INTERVAL '1 month';
                        partition_name := format('interaction_events_%s', to_char(current_start, 'YYYY_MM'));
                        EXECUTE format(
                            'CREATE TABLE IF NOT EXISTS %I PARTITION OF interaction_events FOR VALUES FROM (%L) TO (%L)',
                            partition_name,
                            current_start,
                            current_end
                        );
                        current_start := current_end;
                    END LOOP;
                END $$;
                """
            )
        )

    async def close(self) -> None:
        if self._retention_cleanup_task:
            self._retention_cleanup_task.cancel()
            try:
                await self._retention_cleanup_task
            except asyncio.CancelledError:
                pass
            self._retention_cleanup_task = None
        if self.engine is not None:
            await self.engine.dispose()
        self.engine = None
        self.session_factory = None
        self.is_connected = False
        self._analytics_summary_cache.clear()

    def _install_metrics_listeners(self) -> None:
        if self.engine is None or self.observability is None:
            return

        @event.listens_for(self.engine.sync_engine, "before_cursor_execute")
        def before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            conn.info.setdefault("video_commerce_query_start_time", []).append(
                time.perf_counter()
            )

        @event.listens_for(self.engine.sync_engine, "after_cursor_execute")
        def after_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            started_stack = conn.info.get("video_commerce_query_start_time", [])
            started_at = started_stack.pop(-1) if started_stack else time.perf_counter()
            operation = _sql_operation(statement)
            self.observability.record_database_query(
                operation,
                time.perf_counter() - started_at,
                "success",
            )

        @event.listens_for(self.engine.sync_engine, "handle_error")
        def handle_error(exception_context):
            operation = _sql_operation(exception_context.statement)
            self.observability.record_database_query(operation, 0.0, "error")

    def _update_pool_metrics(self) -> None:
        if self.observability is not None:
            self.observability.update_database_pool_metrics(self)

    async def health_check(self) -> DatabaseHealth:
        if not self.enabled:
            return DatabaseHealth(status="healthy", response_time_ms=0.0)

        started_at = time.time()
        try:
            async with self.session_factory() as session:
                await session.execute(text("SELECT 1"))
            self._update_pool_metrics()
            return DatabaseHealth(
                status="healthy",
                response_time_ms=round((time.time() - started_at) * 1000, 2),
            )
        except Exception as exc:
            self._update_pool_metrics()
            return DatabaseHealth(
                status="unhealthy",
                response_time_ms=round((time.time() - started_at) * 1000, 2),
                error=str(exc),
            )

    async def prune_interaction_events(self) -> int:
        """Delete raw interaction events older than the configured retention window."""
        if not self.enabled:
            return 0
        retention_days = int(self.config.interaction_retention_days)
        if retention_days <= 0:
            return 0

        cutoff = _utc_now() - timedelta(days=retention_days)
        stmt = delete(InteractionEvent).where(InteractionEvent.occurred_at < cutoff)
        async with self.session_factory.begin() as session:
            result = await session.execute(stmt)
        deleted_count = int(result.rowcount or 0)
        self._update_pool_metrics()
        return deleted_count

    async def prune_recommendation_impressions(self) -> int:
        """Delete recommendation impression/slate rows older than retention."""
        if not self.enabled:
            return 0
        retention_days = int(self.config.impression_retention_days)
        if retention_days <= 0:
            return 0

        cutoff = _utc_now() - timedelta(days=retention_days)
        item_stmt = delete(RecommendationImpressionItem).where(
            RecommendationImpressionItem.created_at < cutoff
        )
        impression_stmt = delete(RecommendationImpression).where(
            RecommendationImpression.created_at < cutoff
        )
        async with self.session_factory.begin() as session:
            item_result = await session.execute(item_stmt)
            impression_result = await session.execute(impression_stmt)
        deleted_count = int(item_result.rowcount or 0) + int(
            impression_result.rowcount or 0
        )
        self._update_pool_metrics()
        return deleted_count

    async def _run_periodic_retention_cleanup(self) -> None:
        interval_seconds = max(60, int(self.config.retention_cleanup_interval_seconds))
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                deleted_count = await self.prune_interaction_events()
                impression_deleted_count = await self.prune_recommendation_impressions()
                if deleted_count or impression_deleted_count:
                    logger.info(
                        "postgres_retention_deleted",
                        extra={
                            "interaction_events_deleted_count": deleted_count,
                            "recommendation_impression_rows_deleted_count": (
                                impression_deleted_count
                            ),
                        },
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("interaction_events_retention_cleanup_failed: %s", exc)
                await asyncio.sleep(min(interval_seconds, 300))

    async def record_interaction_events_batch(
        self,
        interactions: Iterable[Dict[str, Any]],
    ) -> None:
        if not self.enabled:
            return

        rows = []
        for interaction in interactions:
            occurred_at = interaction.get("occurred_at")
            if isinstance(occurred_at, (int, float)):
                occurred_at = datetime.fromtimestamp(occurred_at, tz=timezone.utc)
            elif not isinstance(occurred_at, datetime):
                occurred_at = _utc_now()

            rows.append(
                {
                    "event_id": interaction["event_id"],
                    "schema_version": int(interaction.get("schema_version", 1)),
                    "request_id": interaction.get("request_id"),
                    "user_id": interaction["user_id"],
                    "product_id": interaction["product_id"],
                    "action": interaction["action"],
                    "context": interaction.get("context", {}),
                    "occurred_at": occurred_at,
                }
            )

        if not rows:
            return

        stmt = pg_insert(InteractionEvent).values(rows)
        if self.config.interaction_events_partitioned:
            stmt = stmt.on_conflict_do_nothing()
        else:
            stmt = stmt.on_conflict_do_nothing(
                index_elements=[InteractionEvent.event_id]
            )
        async with self.session_factory.begin() as session:
            await session.execute(stmt)
        self._update_pool_metrics()

    async def record_recommendation_impression(self, event: Mapping[str, Any]) -> None:
        """Persist one recommendation impression/slate event idempotently."""
        if not self.enabled:
            return

        metadata = _safe_dict(event.get("metadata"))
        impression_id = str(
            metadata.get("impression_id") or event.get("impression_id") or ""
        ).strip()
        user_id = str(event.get("user_id") or metadata.get("user_id") or "").strip()
        displayed_items = (
            metadata.get("displayed_items") or event.get("displayed_items") or []
        )
        if not impression_id or not user_id or not isinstance(displayed_items, list):
            return

        created_at = (
            _coerce_datetime(event.get("timestamp"))
            or _coerce_datetime(metadata.get("created_at"))
            or _utc_now()
        )
        session_id = metadata.get("session_id")
        context = _impression_context_snapshot(
            metadata.get("context") or metadata.get("request_context")
        )
        if session_id is None:
            session_id = context.get("session_id")
        if metadata.get("content_id") is not None:
            context.setdefault("content_id", metadata.get("content_id"))
        if metadata.get("item_snapshot_scope") is not None:
            context["item_snapshot_scope"] = str(metadata.get("item_snapshot_scope"))[
                :64
            ]
        rejected_candidate_items = metadata.get("rejected_candidate_items")
        if isinstance(rejected_candidate_items, list):
            bounded_rejected_items = []
            for raw_item in rejected_candidate_items[:MAX_REJECTED_CANDIDATE_ITEMS]:
                if not isinstance(raw_item, Mapping):
                    continue
                product_id = str(raw_item.get("product_id") or "").strip()
                if not product_id:
                    continue
                bounded_rejected_items.append(
                    {
                        "product_id": product_id,
                        "position": raw_item.get("position")
                        or raw_item.get("rank_position"),
                        "source": raw_item.get("source")
                        or raw_item.get("candidate_source"),
                        "candidate_source": raw_item.get("candidate_source")
                        or raw_item.get("source"),
                        "feature_snapshot": _safe_dict(
                            raw_item.get("feature_snapshot")
                        ),
                        "scores": _safe_dict(raw_item.get("scores")),
                    }
                )
            if bounded_rejected_items:
                context["rejected_candidate_items"] = bounded_rejected_items

        impression_row = {
            "impression_id": impression_id,
            "request_id": event.get("request_id") or metadata.get("request_id"),
            "user_id": user_id,
            "session_id": session_id,
            "content_id": metadata.get("content_id"),
            "model_version": metadata.get("model_version"),
            "ranking_model_version": metadata.get("ranking_model_version"),
            "context": context,
            "created_at": created_at,
        }

        item_rows = []
        for raw_item in displayed_items:
            if not isinstance(raw_item, Mapping):
                continue
            product_id = str(raw_item.get("product_id") or "").strip()
            if not product_id:
                continue
            feature_snapshot = _safe_dict(raw_item.get("feature_snapshot"))
            for key in ("price", "category", "brand"):
                if key in raw_item and key not in feature_snapshot:
                    feature_snapshot[key] = raw_item.get(key)
            source = raw_item.get("source") or raw_item.get("candidate_source")
            if source is not None:
                feature_snapshot.setdefault("candidate_source", source)
            item_rows.append(
                {
                    "impression_id": impression_id,
                    "product_id": product_id,
                    "position": int(raw_item.get("position") or 0),
                    "source": source,
                    "feature_snapshot": feature_snapshot,
                    "scores": _score_snapshot_from_item(raw_item),
                    "created_at": created_at,
                }
            )

        if not item_rows:
            return

        impression_stmt = pg_insert(RecommendationImpression).values(impression_row)
        impression_stmt = impression_stmt.on_conflict_do_update(
            index_elements=[RecommendationImpression.impression_id],
            set_={
                "request_id": impression_stmt.excluded.request_id,
                "user_id": impression_stmt.excluded.user_id,
                "session_id": impression_stmt.excluded.session_id,
                "content_id": impression_stmt.excluded.content_id,
                "model_version": impression_stmt.excluded.model_version,
                "ranking_model_version": impression_stmt.excluded.ranking_model_version,
                "context": impression_stmt.excluded.context,
                "created_at": impression_stmt.excluded.created_at,
            },
        )
        item_stmt = pg_insert(RecommendationImpressionItem).values(item_rows)
        item_stmt = item_stmt.on_conflict_do_update(
            index_elements=[
                RecommendationImpressionItem.impression_id,
                RecommendationImpressionItem.product_id,
            ],
            set_={
                "position": item_stmt.excluded.position,
                "source": item_stmt.excluded.source,
                "feature_snapshot": item_stmt.excluded.feature_snapshot,
                "scores": item_stmt.excluded.scores,
                "created_at": item_stmt.excluded.created_at,
            },
        )

        async with self.session_factory.begin() as session:
            await session.execute(impression_stmt)
            await session.execute(item_stmt)
        self._update_pool_metrics()

    async def get_ltr_training_impressions(
        self,
        *,
        limit: int = 50000,
        lookback_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Read impression-backed ranking samples with no-click negatives."""
        if not self.enabled:
            return []
        limit = max(0, int(limit))
        if limit <= 0:
            return []

        if lookback_days is None:
            lookback_days = int(self.config.ltr_impression_lookback_days)
        cutoff = None
        if int(lookback_days or 0) > 0:
            cutoff = _utc_now() - timedelta(days=int(lookback_days or 0))

        stmt = (
            select(RecommendationImpression)
            .order_by(RecommendationImpression.created_at.desc())
            .limit(limit)
        )
        if cutoff is not None:
            stmt = stmt.where(RecommendationImpression.created_at >= cutoff)

        async with self.session_factory() as session:
            result = await session.execute(stmt)
            impression_rows = result.scalars().all()

            impression_ids = {
                impression.impression_id for impression in impression_rows
            }
            item_rows = []
            interaction_rows = []
            if impression_ids:
                item_stmt = (
                    select(RecommendationImpressionItem)
                    .where(
                        RecommendationImpressionItem.impression_id.in_(
                            sorted(impression_ids)
                        )
                    )
                    .order_by(
                        RecommendationImpressionItem.impression_id.asc(),
                        RecommendationImpressionItem.position.asc(),
                    )
                )
                item_result = await session.execute(item_stmt)
                item_rows = item_result.scalars().all()

                user_ids = sorted(
                    {
                        impression.user_id
                        for impression in impression_rows
                        if impression.user_id
                    }
                )
                interaction_stmt = select(InteractionEvent).where(
                    InteractionEvent.context["impression_id"]
                    .as_string()
                    .in_(sorted(impression_ids))
                )
                if user_ids:
                    interaction_stmt = interaction_stmt.where(
                        InteractionEvent.user_id.in_(user_ids)
                    )
                if cutoff is not None:
                    interaction_stmt = interaction_stmt.where(
                        InteractionEvent.occurred_at >= cutoff
                    )
                interaction_result = await session.execute(interaction_stmt)
                interaction_rows = interaction_result.scalars().all()

        self._update_pool_metrics()

        impressions_by_id = {
            impression.impression_id: impression for impression in impression_rows
        }
        flattened_items: List[Dict[str, Any]] = []
        for item in item_rows:
            impression = impressions_by_id.get(item.impression_id)
            if impression is None:
                continue
            flattened_items.append(
                {
                    "impression_id": impression.impression_id,
                    "request_id": impression.request_id,
                    "user_id": impression.user_id,
                    "session_id": impression.session_id,
                    "content_id": impression.content_id,
                    "model_version": impression.model_version,
                    "ranking_model_version": impression.ranking_model_version,
                    "context": impression.context or {},
                    "created_at": impression.created_at,
                    "product_id": item.product_id,
                    "position": item.position,
                    "source": item.source,
                    "feature_snapshot": item.feature_snapshot or {},
                    "scores": item.scores or {},
                }
            )
        interactions = [
            {
                "event_id": row.event_id,
                "schema_version": row.schema_version,
                "request_id": row.request_id,
                "user_id": row.user_id,
                "product_id": row.product_id,
                "action": row.action,
                "context": row.context or {},
                "occurred_at": row.occurred_at,
            }
            for row in interaction_rows
        ]
        return build_ltr_training_samples_from_impression_records(
            flattened_items,
            interactions,
        )

    async def get_two_tower_training_impression_negatives(
        self,
        *,
        limit: int = 50000,
        lookback_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Read weak retrieval negatives from returned no-click and rejected candidates."""
        if not self.enabled:
            return []
        limit = max(0, int(limit))
        if limit <= 0:
            return []

        if lookback_days is None:
            lookback_days = int(self.config.ltr_impression_lookback_days)
        cutoff = None
        if int(lookback_days or 0) > 0:
            cutoff = _utc_now() - timedelta(days=int(lookback_days or 0))

        stmt = (
            select(RecommendationImpression)
            .order_by(RecommendationImpression.created_at.desc())
            .limit(limit)
        )
        if cutoff is not None:
            stmt = stmt.where(RecommendationImpression.created_at >= cutoff)

        async with self.session_factory() as session:
            result = await session.execute(stmt)
            impression_rows = result.scalars().all()

            impression_ids = {
                impression.impression_id for impression in impression_rows
            }
            item_rows = []
            interaction_rows = []
            if impression_ids:
                item_stmt = (
                    select(RecommendationImpressionItem)
                    .where(
                        RecommendationImpressionItem.impression_id.in_(
                            sorted(impression_ids)
                        )
                    )
                    .order_by(
                        RecommendationImpressionItem.impression_id.asc(),
                        RecommendationImpressionItem.position.asc(),
                    )
                )
                item_result = await session.execute(item_stmt)
                item_rows = item_result.scalars().all()

                user_ids = sorted(
                    {
                        impression.user_id
                        for impression in impression_rows
                        if impression.user_id
                    }
                )
                interaction_stmt = select(InteractionEvent).where(
                    InteractionEvent.context["impression_id"]
                    .as_string()
                    .in_(sorted(impression_ids))
                )
                if user_ids:
                    interaction_stmt = interaction_stmt.where(
                        InteractionEvent.user_id.in_(user_ids)
                    )
                if cutoff is not None:
                    interaction_stmt = interaction_stmt.where(
                        InteractionEvent.occurred_at >= cutoff
                    )
                interaction_result = await session.execute(interaction_stmt)
                interaction_rows = interaction_result.scalars().all()

        self._update_pool_metrics()

        impressions_by_id = {
            impression.impression_id: impression for impression in impression_rows
        }
        flattened_items: List[Dict[str, Any]] = []
        for item in item_rows:
            impression = impressions_by_id.get(item.impression_id)
            if impression is None:
                continue
            flattened_items.append(
                {
                    "impression_id": impression.impression_id,
                    "request_id": impression.request_id,
                    "user_id": impression.user_id,
                    "session_id": impression.session_id,
                    "content_id": impression.content_id,
                    "model_version": impression.model_version,
                    "ranking_model_version": impression.ranking_model_version,
                    "context": impression.context or {},
                    "created_at": impression.created_at,
                    "product_id": item.product_id,
                    "position": item.position,
                    "source": item.source,
                    "feature_snapshot": item.feature_snapshot or {},
                    "scores": item.scores or {},
                }
            )
        interactions = [
            {
                "event_id": row.event_id,
                "schema_version": row.schema_version,
                "request_id": row.request_id,
                "user_id": row.user_id,
                "product_id": row.product_id,
                "action": row.action,
                "context": row.context or {},
                "occurred_at": row.occurred_at,
            }
            for row in interaction_rows
        ]
        return build_two_tower_training_negatives_from_impression_records(
            flattened_items,
            interactions,
        )

    async def get_training_interactions(
        self, limit: int = 50000
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        async with self.session_factory() as session:
            result = await session.execute(
                select(InteractionEvent)
                .order_by(InteractionEvent.occurred_at.desc())
                .limit(limit)
            )
            rows = result.scalars().all()
        self._update_pool_metrics()

        interactions: List[Dict[str, Any]] = []
        for row in rows:
            interactions.append(
                {
                    "event_id": row.event_id,
                    "schema_version": row.schema_version,
                    "request_id": row.request_id,
                    "user_id": row.user_id,
                    "product_id": row.product_id,
                    "action": row.action,
                    "context": row.context or {},
                    "timestamp": row.occurred_at.timestamp(),
                    "occurred_at": row.occurred_at.timestamp(),
                }
            )
        return interactions

    async def get_user_training_sequences(
        self,
        *,
        max_users: int = 10000,
        max_events_per_user: int = 200,
        since: Optional[Any] = None,
        min_sequence_length: int = 2,
        actions: Optional[Iterable[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return bounded positive chronological interaction sequences per user."""
        if not self.enabled or max_users <= 0 or max_events_per_user <= 0:
            return {}

        selected_actions = tuple(actions or POSITIVE_SEQUENCE_ACTIONS)
        if not selected_actions:
            return {}

        conditions = [InteractionEvent.action.in_(selected_actions)]
        since_dt = _coerce_datetime(since)
        if since_dt is not None:
            conditions.append(InteractionEvent.occurred_at >= since_dt)

        min_sequence_length = max(1, int(min_sequence_length))
        eligible_users = (
            select(InteractionEvent.user_id)
            .where(*conditions)
            .group_by(InteractionEvent.user_id)
            .having(func.count(InteractionEvent.event_id) >= min_sequence_length)
            .order_by(
                func.max(InteractionEvent.occurred_at).desc(),
                InteractionEvent.user_id.asc(),
            )
            .limit(max_users)
            .subquery()
        )
        event_rank = (
            func.row_number()
            .over(
                partition_by=InteractionEvent.user_id,
                order_by=(
                    InteractionEvent.occurred_at.desc(),
                    InteractionEvent.event_id.desc(),
                ),
            )
            .label("event_rank")
        )
        ranked_events = (
            select(
                InteractionEvent.event_id,
                InteractionEvent.schema_version,
                InteractionEvent.request_id,
                InteractionEvent.user_id,
                InteractionEvent.product_id,
                InteractionEvent.action,
                InteractionEvent.context,
                InteractionEvent.occurred_at,
                event_rank,
            )
            .where(
                InteractionEvent.user_id.in_(select(eligible_users.c.user_id)),
                *conditions,
            )
            .subquery()
        )
        stmt = (
            select(ranked_events)
            .where(ranked_events.c.event_rank <= max_events_per_user)
            .order_by(
                ranked_events.c.user_id.asc(),
                ranked_events.c.occurred_at.asc(),
                ranked_events.c.event_id.asc(),
            )
        )

        async with self.session_factory() as session:
            result = await session.execute(stmt)
            rows = result.mappings().all()
        self._update_pool_metrics()

        return build_chronological_user_sequences(
            rows,
            max_events_per_user=max_events_per_user,
            min_sequence_length=min_sequence_length,
        )

    async def get_analytics_summary(self) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "total_interactions": 0,
                "unique_users": 0,
                "unique_products": 0,
                "action_counts": {},
                "ctr": 0.0,
                "conversion_rate": 0.0,
                "timestamp": time.time(),
                "source": "postgres",
            }

        window_hours = max(1, int(self.config.analytics_window_hours))
        ttl_seconds = max(0, int(self.config.analytics_summary_cache_ttl_seconds))
        now = time.time()
        if ttl_seconds > 0:
            cached = self._analytics_summary_cache.get(window_hours)
            if cached is not None:
                cached_at, cached_summary = cached
                if now - cached_at < ttl_seconds:
                    return _copy_analytics_summary(cached_summary)

        cutoff = _utc_now() - timedelta(hours=window_hours)

        async with self.session_factory() as session:
            totals_result = await session.execute(
                select(
                    func.count(InteractionEvent.event_id),
                    func.count(func.distinct(InteractionEvent.user_id)),
                    func.count(func.distinct(InteractionEvent.product_id)),
                ).where(InteractionEvent.occurred_at >= cutoff)
            )
            total_interactions, unique_users, unique_products = totals_result.one()

            action_rows = await session.execute(
                select(InteractionEvent.action, func.count(InteractionEvent.event_id))
                .where(InteractionEvent.occurred_at >= cutoff)
                .group_by(InteractionEvent.action)
            )

        action_counts = Counter({action: count for action, count in action_rows.all()})
        self._update_pool_metrics()
        clicks = action_counts.get("click", 0)
        purchases = action_counts.get("purchase", 0)
        views = action_counts.get("view", 0)
        ctr = clicks / max(views, 1)
        conversion_rate = purchases / max(clicks, 1)

        summary = {
            "total_interactions": int(total_interactions or 0),
            "unique_users": int(unique_users or 0),
            "unique_products": int(unique_products or 0),
            "action_counts": dict(action_counts),
            "ctr": round(ctr, 4),
            "conversion_rate": round(conversion_rate, 4),
            "timestamp": now,
            "source": "postgres",
            "window_hours": window_hours,
        }
        if ttl_seconds > 0:
            self._analytics_summary_cache[window_hours] = (
                now,
                _copy_analytics_summary(summary),
            )
        return summary

    async def upsert_content_job(
        self,
        content_id: str,
        filename: Optional[str],
        storage_path: Optional[str],
        user_id: Optional[str],
        priority: str,
        status: str,
        payload: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        stmt = pg_insert(ContentJob).values(
            content_id=content_id,
            filename=filename,
            storage_path=storage_path,
            user_id=user_id,
            priority=priority,
            status=status,
            error_message=error_message,
            payload=payload or {},
            updated_at=_utc_now(),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ContentJob.content_id],
            set_={
                "filename": filename,
                "storage_path": storage_path,
                "user_id": user_id,
                "priority": priority,
                "status": status,
                "error_message": error_message,
                "payload": payload or {},
                "updated_at": _utc_now(),
            },
        )
        async with self.session_factory.begin() as session:
            await session.execute(stmt)
        self._update_pool_metrics()

    async def upsert_content_feature_artifact(
        self,
        content_id: str,
        features: Dict[str, Any],
        *,
        schema_version: str,
    ) -> None:
        if not self.enabled:
            return
        stmt = pg_insert(ContentFeatureArtifact).values(
            content_id=content_id,
            schema_version=schema_version,
            features=features,
            updated_at=_utc_now(),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ContentFeatureArtifact.content_id],
            set_={
                "schema_version": schema_version,
                "features": features,
                "updated_at": _utc_now(),
            },
        )
        async with self.session_factory.begin() as session:
            await session.execute(stmt)
        self._update_pool_metrics()

    async def get_content_feature_artifact(
        self, content_id: str
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        async with self.session_factory() as session:
            row = await session.get(ContentFeatureArtifact, content_id)
        self._update_pool_metrics()
        return dict(row.features) if row is not None else None

    async def list_content_jobs_missing_feature_artifact(
        self, *, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Return completed durable uploads not yet backfilled to the new schema."""
        if not self.enabled:
            return []
        statement = (
            select(ContentJob)
            .outerjoin(
                ContentFeatureArtifact,
                ContentFeatureArtifact.content_id == ContentJob.content_id,
            )
            .where(
                ContentJob.status == "completed",
                ContentJob.storage_path.is_not(None),
                ContentFeatureArtifact.content_id.is_(None),
            )
            .order_by(ContentJob.created_at, ContentJob.content_id)
            .limit(max(1, int(limit)))
        )
        async with self.session_factory() as session:
            rows = (await session.execute(statement)).scalars().all()
        self._update_pool_metrics()
        return [
            {
                "content_id": row.content_id,
                "filename": row.filename,
                "storage_path": row.storage_path,
                "user_id": row.user_id,
                "priority": row.priority,
            }
            for row in rows
        ]

    async def update_content_job_status(
        self,
        content_id: str,
        status: str,
        *,
        error_message: Optional[str] = None,
        storage_path: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        updated_at = _utc_now()
        stmt = pg_insert(ContentJob).values(
            content_id=content_id,
            filename=None,
            storage_path=storage_path,
            user_id=None,
            priority="normal",
            status=status,
            error_message=error_message,
            payload=payload or {},
            updated_at=updated_at,
        )
        update_values = {
            "status": status,
            "error_message": error_message,
            "updated_at": updated_at,
        }
        if storage_path is not None:
            update_values["storage_path"] = storage_path
        if payload is not None:
            update_values["payload"] = payload
        stmt = stmt.on_conflict_do_update(
            index_elements=[ContentJob.content_id],
            set_=update_values,
        )
        async with self.session_factory.begin() as session:
            await session.execute(stmt)
        self._update_pool_metrics()

    async def get_content_job(self, content_id: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        async with self.session_factory() as session:
            row = await session.get(ContentJob, content_id)
        self._update_pool_metrics()
        if row is None:
            return None
        return {
            "content_id": row.content_id,
            "filename": row.filename,
            "storage_path": row.storage_path,
            "user_id": row.user_id,
            "priority": row.priority,
            "status": row.status,
            "error_message": row.error_message,
            "payload": row.payload or {},
            "created_at": row.created_at.timestamp() if row.created_at else None,
            "updated_at": row.updated_at.timestamp() if row.updated_at else None,
        }

    async def store_product_catalog_snapshot_batch(
        self,
        metadata_map: Dict[str, Dict[str, Any]],
    ) -> None:
        if not self.enabled or not metadata_map:
            return

        rows = [
            {
                "product_id": product_id,
                "snapshot": metadata,
                "updated_at": _utc_now(),
            }
            for product_id, metadata in metadata_map.items()
        ]
        stmt = pg_insert(ProductCatalogSnapshot).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[ProductCatalogSnapshot.product_id],
            set_={
                "snapshot": stmt.excluded.snapshot,
                "updated_at": _utc_now(),
            },
        )
        async with self.session_factory.begin() as session:
            await session.execute(stmt)
        self._update_pool_metrics()

    async def activate_product_catalog(
        self,
        source_version: str,
        metadata_map: Mapping[str, Mapping[str, Any]],
        *,
        event_time: Optional[float] = None,
        available_at: Optional[float] = None,
        batch_size: int = 500,
    ) -> str:
        """Atomically stage each catalog batch with its durable Kafka outbox rows."""
        if not self.enabled:
            raise RuntimeError("catalog activation requires Postgres system store")
        if not metadata_map:
            raise ValueError("catalog activation requires at least one product")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        activation_time = float(
            available_at if available_at is not None else time.time()
        )
        prepared = prepare_catalog_activation(
            source_version=source_version,
            metadata_map=metadata_map,
            event_time=float(event_time if event_time is not None else activation_time),
            available_at=activation_time,
        )
        activation_insert = (
            pg_insert(CatalogActivation)
            .values(
                activation_id=prepared.activation_id,
                source_version=prepared.source_version,
                expected_count=prepared.expected_count,
                manifest_hash=prepared.manifest_hash,
                actual_count=0,
                status="staging",
            )
            .on_conflict_do_nothing(index_elements=[CatalogActivation.activation_id])
        )
        async with self.session_factory.begin() as session:
            await session.execute(activation_insert)

        async with self.session_factory() as session:
            activation = await session.get(CatalogActivation, prepared.activation_id)
        if activation is None:
            raise RuntimeError("catalog activation row was not persisted")
        if (
            activation.source_version != prepared.source_version
            or activation.expected_count != prepared.expected_count
            or activation.manifest_hash != prepared.manifest_hash
        ):
            raise RuntimeError(
                "catalog activation retry does not match staged activation"
            )
        if activation.status == "complete":
            return prepared.activation_id

        for offset in range(0, prepared.expected_count, batch_size):
            snapshot_rows = [
                {
                    **row,
                    "activation_id": prepared.activation_id,
                    "source_version": prepared.source_version,
                }
                for row in prepared.snapshot_rows[offset : offset + batch_size]
            ]
            outbox_rows = prepared.outbox_rows[offset : offset + batch_size]
            snapshot_stmt = pg_insert(ProductCatalogSnapshot).values(snapshot_rows)
            snapshot_stmt = snapshot_stmt.on_conflict_do_update(
                index_elements=[ProductCatalogSnapshot.product_id],
                set_={
                    "snapshot": snapshot_stmt.excluded.snapshot,
                    "activation_id": snapshot_stmt.excluded.activation_id,
                    "source_version": snapshot_stmt.excluded.source_version,
                    "updated_at": snapshot_stmt.excluded.updated_at,
                },
            )
            outbox_stmt = (
                pg_insert(CatalogFeatureOutbox)
                .values(outbox_rows)
                .on_conflict_do_nothing(index_elements=[CatalogFeatureOutbox.event_id])
            )
            async with self.session_factory.begin() as session:
                await session.execute(snapshot_stmt)
                await session.execute(outbox_stmt)

        async with self.session_factory.begin() as session:
            result = await session.execute(
                select(
                    CatalogFeatureOutbox.event_id, CatalogFeatureOutbox.payload_hash
                ).where(CatalogFeatureOutbox.activation_id == prepared.activation_id)
            )
            actual_hashes = {
                event_id: payload_hash for event_id, payload_hash in result.all()
            }
            expected_hashes = {
                row["event_id"]: row["payload_hash"] for row in prepared.outbox_rows
            }
            if actual_hashes != expected_hashes:
                await session.execute(
                    update(CatalogActivation)
                    .where(CatalogActivation.activation_id == prepared.activation_id)
                    .values(actual_count=len(actual_hashes), updated_at=_utc_now())
                )
                raise RuntimeError("catalog activation outbox reconciliation failed")
            completed_at = _utc_now()
            await session.execute(
                update(CatalogActivation)
                .where(
                    CatalogActivation.activation_id == prepared.activation_id,
                    CatalogActivation.status == "staging",
                )
                .values(
                    actual_count=len(actual_hashes),
                    status="complete",
                    completed_at=completed_at,
                    updated_at=completed_at,
                )
            )
        self._update_pool_metrics()
        return prepared.activation_id

    async def claim_catalog_outbox(
        self,
        *,
        worker_id: str,
        batch_size: int,
        lease_seconds: int,
    ) -> List[Dict[str, Any]]:
        """Lease unpublished rows from completed activations without worker contention."""
        if not self.enabled:
            return []
        now = _utc_now()
        lease_until = now + timedelta(seconds=max(1, int(lease_seconds)))
        statement = (
            select(CatalogFeatureOutbox)
            .join(
                CatalogActivation,
                CatalogActivation.activation_id == CatalogFeatureOutbox.activation_id,
            )
            .where(
                CatalogActivation.status == "complete",
                CatalogFeatureOutbox.published_at.is_(None),
                or_(
                    CatalogFeatureOutbox.claim_expires_at.is_(None),
                    CatalogFeatureOutbox.claim_expires_at < now,
                ),
            )
            .order_by(CatalogFeatureOutbox.created_at, CatalogFeatureOutbox.event_id)
            .limit(max(1, int(batch_size)))
            .with_for_update(skip_locked=True, of=CatalogFeatureOutbox)
        )
        async with self.session_factory.begin() as session:
            result = await session.execute(statement)
            rows = list(result.scalars().all())
            for row in rows:
                row.claimed_by = worker_id
                row.claim_expires_at = lease_until
            events = [dict(row.event_payload or {}) for row in rows]
        self._update_pool_metrics()
        return events

    async def mark_catalog_outbox_published(
        self,
        event_id: str,
        *,
        worker_id: str,
    ) -> None:
        if not self.enabled:
            return
        async with self.session_factory.begin() as session:
            await session.execute(
                update(CatalogFeatureOutbox)
                .where(
                    CatalogFeatureOutbox.event_id == event_id,
                    CatalogFeatureOutbox.claimed_by == worker_id,
                    CatalogFeatureOutbox.published_at.is_(None),
                )
                .values(
                    published_at=_utc_now(),
                    attempts=CatalogFeatureOutbox.attempts + 1,
                    last_error=None,
                    claimed_by=None,
                    claim_expires_at=None,
                )
            )
        self._update_pool_metrics()

    async def mark_catalog_outbox_failed(
        self,
        event_id: str,
        error: str,
        *,
        worker_id: str,
    ) -> None:
        if not self.enabled:
            return
        async with self.session_factory.begin() as session:
            await session.execute(
                update(CatalogFeatureOutbox)
                .where(
                    CatalogFeatureOutbox.event_id == event_id,
                    CatalogFeatureOutbox.claimed_by == worker_id,
                    CatalogFeatureOutbox.published_at.is_(None),
                )
                .values(
                    attempts=CatalogFeatureOutbox.attempts + 1,
                    last_error=str(error)[:4096],
                    claimed_by=None,
                    claim_expires_at=None,
                )
            )
        self._update_pool_metrics()

    async def prune_catalog_outbox(self, *, retention_days: int = 7) -> int:
        if not self.enabled:
            return 0
        cutoff = _utc_now() - timedelta(days=max(1, int(retention_days)))
        async with self.session_factory.begin() as session:
            result = await session.execute(
                delete(CatalogFeatureOutbox).where(
                    CatalogFeatureOutbox.published_at.is_not(None),
                    CatalogFeatureOutbox.published_at < cutoff,
                )
            )
        self._update_pool_metrics()
        return int(result.rowcount or 0)

    async def get_catalog_outbox_stats(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"pending": 0, "oldest_age_seconds": 0.0}
        async with self.session_factory() as session:
            result = await session.execute(
                select(
                    func.count(CatalogFeatureOutbox.event_id),
                    func.min(CatalogFeatureOutbox.created_at),
                )
                .join(
                    CatalogActivation,
                    CatalogActivation.activation_id
                    == CatalogFeatureOutbox.activation_id,
                )
                .where(
                    CatalogActivation.status == "complete",
                    CatalogFeatureOutbox.published_at.is_(None),
                )
            )
            pending, oldest = result.one()
        oldest_age = max(0.0, (_utc_now() - oldest).total_seconds()) if oldest else 0.0
        return {"pending": int(pending or 0), "oldest_age_seconds": oldest_age}

    @staticmethod
    def _pit_materialization_run_dict(
        row: PitMaterializationRun, *, claimed: bool
    ) -> Dict[str, Any]:
        return {
            "run_id": row.run_id,
            "cutoff_ts": row.cutoff_at.timestamp(),
            "status": row.status,
            "phase": row.phase,
            "attempts": row.attempts,
            "export_attempt": row.export_attempt,
            "flink_job_id": row.flink_job_id,
            "worker_id": row.worker_id,
            "lease_expires_at": (
                row.lease_expires_at.timestamp() if row.lease_expires_at else None
            ),
            "snapshot_id": row.snapshot_id,
            "manifest_uri": row.manifest_uri,
            "row_count": row.row_count,
            "quarantine_count": row.quarantine_count,
            "last_error": row.last_error,
            "claimed": claimed,
        }

    async def claim_pit_materialization_run(
        self,
        *,
        run_id: str,
        cutoff_ts: float,
        worker_id: str,
        lease_seconds: int,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("PIT materialization orchestration requires Postgres")
        cutoff_at = _coerce_datetime(cutoff_ts)
        if cutoff_at is None:
            raise ValueError("PIT materialization cutoff must be a Unix timestamp")
        now = _utc_now()
        lease_until = now + timedelta(seconds=max(60, int(lease_seconds)))
        async with self.session_factory.begin() as session:
            await session.execute(select(func.pg_advisory_xact_lock(1_947_420_117)))
            active_result = await session.execute(
                select(PitMaterializationRun)
                .where(
                    PitMaterializationRun.run_id != run_id,
                    PitMaterializationRun.status == "running",
                )
                .order_by(PitMaterializationRun.cutoff_at)
                .with_for_update()
            )
            active = active_result.scalars().first()
            if active is not None:
                lease_valid = (
                    active.lease_expires_at is not None
                    and active.lease_expires_at >= now
                )
                if lease_valid:
                    return self._pit_materialization_run_dict(active, claimed=False)
                active.attempts += 1
                active.worker_id = worker_id
                active.lease_expires_at = lease_until
                active.last_error = None
                active.completed_at = None
                active.updated_at = now
                await session.flush()
                return self._pit_materialization_run_dict(active, claimed=True)
            await session.execute(
                pg_insert(PitMaterializationRun)
                .values(
                    run_id=run_id,
                    cutoff_at=cutoff_at,
                    status="pending",
                    phase="export",
                    attempts=0,
                )
                .on_conflict_do_nothing(index_elements=[PitMaterializationRun.run_id])
            )
            result = await session.execute(
                select(PitMaterializationRun)
                .where(PitMaterializationRun.run_id == run_id)
                .with_for_update()
            )
            row = result.scalar_one()
            if row.cutoff_at != cutoff_at:
                raise RuntimeError(
                    "PIT materialization run_id already exists with a different cutoff"
                )
            terminal = row.status in {"completed", "waiting_for_eligible_rows"}
            lease_held = (
                row.status == "running"
                and row.lease_expires_at is not None
                and row.lease_expires_at >= now
                and row.worker_id != worker_id
            )
            if terminal or lease_held:
                return self._pit_materialization_run_dict(row, claimed=False)
            row.status = "running"
            row.attempts += 1
            row.worker_id = worker_id
            row.lease_expires_at = lease_until
            row.last_error = None
            row.completed_at = None
            row.updated_at = now
            await session.flush()
            return self._pit_materialization_run_dict(row, claimed=True)

    async def mark_pit_materialization_phase(
        self,
        run_id: str,
        *,
        phase: str,
        worker_id: str,
        export_attempt: Optional[int] = None,
        flink_job_id: Optional[str] = None,
        lease_seconds: Optional[int] = None,
    ) -> None:
        if phase not in {"export", "flink", "manifest"}:
            raise ValueError(
                "PIT materialization phase must be export, flink, or manifest"
            )
        now = _utc_now()
        updates: Dict[str, Any] = {"phase": phase, "updated_at": now}
        if export_attempt is not None:
            updates["export_attempt"] = max(1, int(export_attempt))
        if flink_job_id is not None:
            normalized_job_id = str(flink_job_id).strip().lower()
            if len(normalized_job_id) != 32:
                raise ValueError("Flink job ID must be 32 hexadecimal characters")
            int(normalized_job_id, 16)
            updates["flink_job_id"] = normalized_job_id
        if lease_seconds is not None:
            updates["lease_expires_at"] = now + timedelta(
                seconds=max(60, int(lease_seconds))
            )
        async with self.session_factory.begin() as session:
            result = await session.execute(
                update(PitMaterializationRun)
                .where(
                    PitMaterializationRun.run_id == run_id,
                    PitMaterializationRun.status == "running",
                    PitMaterializationRun.worker_id == worker_id,
                    PitMaterializationRun.lease_expires_at >= now,
                )
                .values(**updates)
            )
            if int(result.rowcount or 0) != 1:
                raise RuntimeError("PIT materialization phase update lost its lease")

    async def renew_pit_materialization_lease(
        self, run_id: str, *, worker_id: str, lease_seconds: int
    ) -> None:
        now = _utc_now()
        async with self.session_factory.begin() as session:
            result = await session.execute(
                update(PitMaterializationRun)
                .where(
                    PitMaterializationRun.run_id == run_id,
                    PitMaterializationRun.status == "running",
                    PitMaterializationRun.worker_id == worker_id,
                    PitMaterializationRun.lease_expires_at >= now,
                )
                .values(
                    lease_expires_at=now
                    + timedelta(seconds=max(60, int(lease_seconds))),
                    updated_at=now,
                )
            )
            if int(result.rowcount or 0) != 1:
                raise RuntimeError("PIT materialization lease renewal lost ownership")

    async def complete_pit_materialization_run(
        self,
        run_id: str,
        *,
        worker_id: str,
        snapshot_id: str,
        manifest_uri: str,
        row_count: int,
        quarantine_count: int,
    ) -> None:
        now = _utc_now()
        async with self.session_factory.begin() as session:
            result = await session.execute(
                update(PitMaterializationRun)
                .where(
                    PitMaterializationRun.run_id == run_id,
                    PitMaterializationRun.status == "running",
                    PitMaterializationRun.worker_id == worker_id,
                    PitMaterializationRun.lease_expires_at >= now,
                )
                .values(
                    status="completed",
                    phase="manifest",
                    snapshot_id=str(snapshot_id),
                    manifest_uri=str(manifest_uri),
                    row_count=max(0, int(row_count)),
                    quarantine_count=max(0, int(quarantine_count)),
                    worker_id=None,
                    lease_expires_at=None,
                    last_error=None,
                    completed_at=now,
                    updated_at=now,
                )
            )
            if int(result.rowcount or 0) != 1:
                raise RuntimeError("PIT materialization completion lost its lease")

    async def mark_pit_materialization_waiting(
        self, run_id: str, *, worker_id: str
    ) -> None:
        now = _utc_now()
        async with self.session_factory.begin() as session:
            result = await session.execute(
                update(PitMaterializationRun)
                .where(
                    PitMaterializationRun.run_id == run_id,
                    PitMaterializationRun.status == "running",
                    PitMaterializationRun.worker_id == worker_id,
                    PitMaterializationRun.lease_expires_at >= now,
                )
                .values(
                    status="waiting_for_eligible_rows",
                    phase="export",
                    export_attempt=None,
                    flink_job_id=None,
                    worker_id=None,
                    lease_expires_at=None,
                    completed_at=now,
                    updated_at=now,
                )
            )
            if int(result.rowcount or 0) != 1:
                raise RuntimeError("PIT materialization waiting update lost its lease")

    async def fail_pit_materialization_run(
        self, run_id: str, *, worker_id: str, error: str
    ) -> None:
        now = _utc_now()
        async with self.session_factory.begin() as session:
            await session.execute(
                update(PitMaterializationRun)
                .where(
                    PitMaterializationRun.run_id == run_id,
                    PitMaterializationRun.status == "running",
                    PitMaterializationRun.worker_id == worker_id,
                )
                .values(
                    status="failed",
                    worker_id=None,
                    lease_expires_at=None,
                    last_error=str(error)[:4096],
                    updated_at=now,
                )
            )

    async def claim_pit_training_run(
        self, *, run_id: str, worker_id: str, lease_seconds: int
    ) -> Dict[str, Any]:
        now = _utc_now()
        lease_until = now + timedelta(seconds=max(300, int(lease_seconds)))
        async with self.session_factory.begin() as session:
            result = await session.execute(
                select(PitMaterializationRun)
                .where(PitMaterializationRun.run_id == run_id)
                .with_for_update()
            )
            row = result.scalar_one_or_none()
            if row is None or row.status != "completed":
                raise RuntimeError(
                    "PIT training requires a completed materialization run"
                )
            completed = row.training_status == "completed"
            held = (
                row.training_status == "running"
                and row.training_lease_expires_at is not None
                and row.training_lease_expires_at >= now
                and row.training_worker_id != worker_id
            )
            if completed or held:
                return {
                    "claimed": False,
                    "status": row.training_status,
                    "model_version": row.trained_model_version,
                }
            row.training_status = "running"
            row.training_worker_id = worker_id
            row.training_lease_expires_at = lease_until
            row.updated_at = now
            return {"claimed": True, "status": "running"}

    async def get_pit_operational_metrics(self) -> Dict[str, Any]:
        now = _utc_now()
        async with self.session_factory() as session:
            latest_result = await session.execute(
                select(PitMaterializationRun)
                .order_by(desc(PitMaterializationRun.cutoff_at))
                .limit(1)
            )
            latest = latest_result.scalar_one_or_none()
            running_result = await session.execute(
                select(PitMaterializationRun)
                .where(PitMaterializationRun.status == "running")
                .limit(1)
            )
            running = running_result.scalar_one_or_none()
            success_result = await session.execute(
                select(func.max(PitMaterializationRun.completed_at)).where(
                    PitMaterializationRun.status == "completed"
                )
            )
            last_success = success_result.scalar_one_or_none()
        return {
            "last_success_timestamp": (
                last_success.timestamp() if last_success is not None else 0.0
            ),
            "waiting_for_rows": bool(
                latest is not None and latest.status == "waiting_for_eligible_rows"
            ),
            "run_in_progress": running is not None,
            "lease_expired": bool(
                running is not None
                and running.lease_expires_at is not None
                and running.lease_expires_at < now
            ),
        }

    async def renew_pit_training_lease(
        self, *, run_id: str, worker_id: str, lease_seconds: int
    ) -> None:
        now = _utc_now()
        async with self.session_factory.begin() as session:
            result = await session.execute(
                update(PitMaterializationRun)
                .where(
                    PitMaterializationRun.run_id == run_id,
                    PitMaterializationRun.training_status == "running",
                    PitMaterializationRun.training_worker_id == worker_id,
                    PitMaterializationRun.training_lease_expires_at >= now,
                )
                .values(
                    training_lease_expires_at=now
                    + timedelta(seconds=max(300, int(lease_seconds))),
                    updated_at=now,
                )
            )
            if int(result.rowcount or 0) != 1:
                raise RuntimeError("PIT training lease renewal lost ownership")

    async def complete_pit_training_run(
        self, *, run_id: str, worker_id: str, model_version: str
    ) -> None:
        now = _utc_now()
        async with self.session_factory.begin() as session:
            result = await session.execute(
                update(PitMaterializationRun)
                .where(
                    PitMaterializationRun.run_id == run_id,
                    PitMaterializationRun.training_status == "running",
                    PitMaterializationRun.training_worker_id == worker_id,
                    PitMaterializationRun.training_lease_expires_at >= now,
                )
                .values(
                    training_status="completed",
                    training_worker_id=None,
                    training_lease_expires_at=None,
                    trained_model_version=str(model_version),
                    updated_at=now,
                )
            )
            if int(result.rowcount or 0) != 1:
                raise RuntimeError("PIT training completion lost its lease")

    async def fail_pit_training_run(self, *, run_id: str, worker_id: str) -> None:
        async with self.session_factory.begin() as session:
            await session.execute(
                update(PitMaterializationRun)
                .where(
                    PitMaterializationRun.run_id == run_id,
                    PitMaterializationRun.training_status == "running",
                    PitMaterializationRun.training_worker_id == worker_id,
                )
                .values(
                    training_status="failed",
                    training_worker_id=None,
                    training_lease_expires_at=None,
                    updated_at=_utc_now(),
                )
            )

    async def start_feature_history_backfill(
        self,
        *,
        run_id: str,
        range_start: float,
        range_end: float,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("feature history backfill requires Postgres")
        start_at = _coerce_datetime(range_start)
        end_at = _coerce_datetime(range_end)
        if start_at is None or end_at is None or start_at >= end_at:
            raise ValueError("backfill range must have range_start < range_end")
        stmt = (
            pg_insert(FeatureHistoryBackfillRun)
            .values(
                run_id=run_id,
                range_start=start_at,
                range_end=end_at,
                status="active",
                phase="catalog",
                counts={},
                reconciliation={},
            )
            .on_conflict_do_nothing()
        )
        async with self.session_factory.begin() as session:
            await session.execute(stmt)
        run = await self.get_feature_history_backfill_run(run_id)
        if run is None:
            raise RuntimeError("another active backfill already owns this range")
        if (
            run["range_start"] != start_at.timestamp()
            or run["range_end"] != end_at.timestamp()
        ):
            raise RuntimeError("backfill run_id already exists with a different range")
        return run

    async def get_feature_history_backfill_run(
        self, run_id: str
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        async with self.session_factory() as session:
            row = await session.get(FeatureHistoryBackfillRun, run_id)
        if row is None:
            return None
        return {
            "run_id": row.run_id,
            "range_start": row.range_start.timestamp(),
            "range_end": row.range_end.timestamp(),
            "status": row.status,
            "phase": row.phase,
            "cursor_time": row.cursor_time.timestamp() if row.cursor_time else None,
            "cursor_id": row.cursor_id,
            "counts": dict(row.counts or {}),
            "reconciliation": dict(row.reconciliation or {}),
            "last_error": row.last_error,
        }

    async def checkpoint_feature_history_backfill(
        self,
        run_id: str,
        *,
        phase: Optional[str] = None,
        cursor_time: Optional[float] = None,
        cursor_id: Optional[str] = None,
        counts: Optional[Mapping[str, Any]] = None,
        status: Optional[str] = None,
        reconciliation: Optional[Mapping[str, Any]] = None,
        last_error: Optional[str] = None,
    ) -> None:
        values: Dict[str, Any] = {"updated_at": _utc_now(), "last_error": last_error}
        if phase is not None:
            values["phase"] = phase
        if cursor_time is not None:
            values["cursor_time"] = _coerce_datetime(cursor_time)
        elif phase is not None:
            values["cursor_time"] = None
        if cursor_id is not None:
            values["cursor_id"] = cursor_id
        elif phase is not None:
            values["cursor_id"] = None
        if counts is not None:
            values["counts"] = dict(counts)
        if reconciliation is not None:
            values["reconciliation"] = dict(reconciliation)
        if status is not None:
            values["status"] = status
            if status == "complete":
                values["completed_at"] = _utc_now()
        async with self.session_factory.begin() as session:
            result = await session.execute(
                update(FeatureHistoryBackfillRun)
                .where(FeatureHistoryBackfillRun.run_id == run_id)
                .values(**values)
            )
        if not result.rowcount:
            raise RuntimeError(f"unknown feature history backfill run {run_id}")
        self._update_pool_metrics()

    async def get_backfill_interactions_page(
        self,
        *,
        range_start: Optional[float] = None,
        range_end: float,
        cursor_time: Optional[float],
        cursor_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        start_at = _coerce_datetime(range_start) if range_start is not None else None
        end_at = _coerce_datetime(range_end)
        statement = select(InteractionEvent).where(
            InteractionEvent.occurred_at < end_at
        )
        if start_at is not None:
            statement = statement.where(InteractionEvent.occurred_at >= start_at)
        cursor_at = _coerce_datetime(cursor_time)
        if cursor_at is not None:
            statement = statement.where(
                or_(
                    InteractionEvent.occurred_at > cursor_at,
                    (
                        (InteractionEvent.occurred_at == cursor_at)
                        & (InteractionEvent.event_id > str(cursor_id or ""))
                    ),
                )
            )
        statement = statement.order_by(
            InteractionEvent.occurred_at, InteractionEvent.event_id
        ).limit(max(1, int(limit)))
        async with self.session_factory() as session:
            result = await session.execute(statement)
            rows = result.scalars().all()
        return [
            {
                "event_id": row.event_id,
                "request_id": row.request_id,
                "user_id": row.user_id,
                "product_id": row.product_id,
                "action": row.action,
                "context": dict(row.context or {}),
                "occurred_at": row.occurred_at,
                "created_at": row.created_at,
            }
            for row in rows
        ]

    async def get_backfill_catalog_snapshot(self) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            latest_activation = (
                select(CatalogActivation.activation_id)
                .where(CatalogActivation.status == "complete")
                .order_by(CatalogActivation.completed_at.desc())
                .limit(1)
                .scalar_subquery()
            )
            result = await session.execute(
                select(ProductCatalogSnapshot)
                .where(ProductCatalogSnapshot.activation_id == latest_activation)
                .order_by(ProductCatalogSnapshot.product_id)
            )
            rows = result.scalars().all()
        return [
            {
                "product_id": row.product_id,
                "snapshot": dict(row.snapshot or {}),
                "source_version": row.source_version,
                "updated_at": row.updated_at,
            }
            for row in rows
        ]

    async def get_backfill_impressions_page(
        self,
        *,
        range_start: Optional[float] = None,
        range_end: float,
        cursor_time: Optional[float],
        cursor_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        start_at = _coerce_datetime(range_start) if range_start is not None else None
        end_at = _coerce_datetime(range_end)
        statement = select(RecommendationImpression).where(
            RecommendationImpression.created_at < end_at
        )
        if start_at is not None:
            statement = statement.where(RecommendationImpression.created_at >= start_at)
        cursor_at = _coerce_datetime(cursor_time)
        if cursor_at is not None:
            statement = statement.where(
                or_(
                    RecommendationImpression.created_at > cursor_at,
                    (
                        (RecommendationImpression.created_at == cursor_at)
                        & (
                            RecommendationImpression.impression_id
                            > str(cursor_id or "")
                        )
                    ),
                )
            )
        statement = statement.order_by(
            RecommendationImpression.created_at,
            RecommendationImpression.impression_id,
        ).limit(max(1, int(limit)))
        async with self.session_factory() as session:
            impression_result = await session.execute(statement)
            impressions = list(impression_result.scalars().all())
            impression_ids = [row.impression_id for row in impressions]
            items_by_impression: Dict[str, List[Dict[str, Any]]] = {
                impression_id: [] for impression_id in impression_ids
            }
            if impression_ids:
                item_result = await session.execute(
                    select(RecommendationImpressionItem)
                    .where(
                        RecommendationImpressionItem.impression_id.in_(impression_ids)
                    )
                    .order_by(
                        RecommendationImpressionItem.impression_id,
                        RecommendationImpressionItem.position,
                        RecommendationImpressionItem.id,
                    )
                )
                for item in item_result.scalars().all():
                    items_by_impression[item.impression_id].append(
                        {
                            "product_id": item.product_id,
                            "position": item.position,
                            "source": item.source,
                            "feature_snapshot": dict(item.feature_snapshot or {}),
                            "scores": dict(item.scores or {}),
                        }
                    )
        return [
            {
                "impression_id": row.impression_id,
                "request_id": row.request_id,
                "user_id": row.user_id,
                "session_id": row.session_id,
                "content_id": row.content_id,
                "model_version": row.model_version,
                "ranking_model_version": row.ranking_model_version,
                "context": dict(row.context or {}),
                "created_at": row.created_at,
                "displayed_items": items_by_impression.get(row.impression_id, []),
            }
            for row in impressions
        ]

    async def record_model_checkpoint(
        self,
        model_name: str,
        model_version: str,
        checkpoint_path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.enabled:
            return False

        record_payload = payload or {}
        materialization_run_id = (
            str(record_payload.get("feature_lake_materialization_run_id") or "").strip()
            or None
        )
        async with self.session_factory.begin() as session:
            result = await session.execute(
                pg_insert(ModelCheckpoint)
                .values(
                    model_name=model_name,
                    model_version=model_version,
                    checkpoint_path=checkpoint_path,
                    materialization_run_id=materialization_run_id,
                    payload=record_payload,
                )
                .on_conflict_do_nothing(
                    index_elements=[
                        ModelCheckpoint.model_name,
                        ModelCheckpoint.materialization_run_id,
                    ],
                    index_where=ModelCheckpoint.materialization_run_id.is_not(None),
                )
            )
        self._update_pool_metrics()
        return int(result.rowcount or 0) == 1

    async def get_latest_model_checkpoint(
        self, model_name: str
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        async with self.session_factory() as session:
            result = await session.execute(
                select(ModelCheckpoint)
                .where(ModelCheckpoint.model_name == model_name)
                .order_by(desc(ModelCheckpoint.created_at), desc(ModelCheckpoint.id))
                .limit(1)
            )
            row = result.scalar_one_or_none()
        self._update_pool_metrics()

        if row is None:
            return None

        return {
            "id": row.id,
            "model_name": row.model_name,
            "model_version": row.model_version,
            "checkpoint_path": row.checkpoint_path,
            "payload": row.payload or {},
            "created_at": row.created_at.timestamp() if row.created_at else None,
        }

    async def get_model_checkpoint_for_materialization_run(
        self, model_name: str, materialization_run_id: str
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        async with self.session_factory() as session:
            result = await session.execute(
                select(ModelCheckpoint).where(
                    ModelCheckpoint.model_name == model_name,
                    ModelCheckpoint.materialization_run_id
                    == str(materialization_run_id),
                )
            )
            row = result.scalar_one_or_none()
        if row is None:
            return None
        return {
            "id": row.id,
            "model_name": row.model_name,
            "model_version": row.model_version,
            "checkpoint_path": row.checkpoint_path,
            "payload": row.payload or {},
            "created_at": row.created_at.timestamp() if row.created_at else None,
        }


def _sql_operation(statement: Optional[str]) -> str:
    if not statement:
        return "unknown"
    return statement.strip().split(None, 1)[0].lower() or "unknown"


def _copy_analytics_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(summary)
    copied["action_counts"] = dict(summary.get("action_counts") or {})
    return copied
