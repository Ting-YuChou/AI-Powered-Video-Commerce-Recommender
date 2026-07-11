"""Versioned feature records and leakage-safe point-in-time joins."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class HistoricalFeatureRecord:
    """One immutable offline feature version.

    ``event_time`` says when the feature describes the entity; ``available_at``
    says when it was actually materialized and safe to use. Point-in-time joins
    must check both fields.
    """

    entity_type: str
    entity_id: str
    event_time: float
    available_at: float
    feature_definition_version: str
    values: Mapping[str, Any]
    source_event_id: str = ""


@dataclass(frozen=True)
class RankingObservation:
    """An impression-candidate anchor plus its historical feature bundle."""

    observation_id: str
    user_id: str
    product_id: str
    as_of_ts: float
    feature_definition_version: str
    user_features: Mapping[str, Any]
    product_metadata: Mapping[str, Any]
    context: Mapping[str, Any]

    @property
    def feature_bundle_hash(self) -> str:
        payload = {
            "as_of_ts": self.as_of_ts,
            "context": dict(self.context),
            "feature_definition_version": self.feature_definition_version,
            "product_id": self.product_id,
            "product_metadata": dict(self.product_metadata),
            "user_features": dict(self.user_features),
            "user_id": self.user_id,
        }
        serialized = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def as_training_row(
        self, *, action: str, label_context: Optional[Mapping[str, Any]] = None
    ) -> dict:
        context = dict(self.context)
        if label_context:
            context.update(label_context)
        return {
            "observation_id": self.observation_id,
            "user_id": self.user_id,
            "product_id": self.product_id,
            "action": action,
            "as_of_ts": self.as_of_ts,
            "event_time": self.as_of_ts,
            "timestamp": self.as_of_ts,
            "feature_definition_version": self.feature_definition_version,
            "feature_bundle_hash": self.feature_bundle_hash,
            "user_features": dict(self.user_features),
            "product_metadata": dict(self.product_metadata),
            "context": context,
        }


class PointInTimeFeatureJoiner:
    """Choose the last historical version safely available at an anchor time."""

    def __init__(self, feature_definition_version: str):
        self.feature_definition_version = feature_definition_version

    def latest_available(
        self,
        records: Iterable[HistoricalFeatureRecord],
        *,
        entity_type: str,
        entity_id: str,
        as_of_ts: float,
    ) -> Optional[HistoricalFeatureRecord]:
        cutoff = float(as_of_ts)
        eligible = [
            record
            for record in records
            if record.entity_type == entity_type
            and record.entity_id == entity_id
            and record.feature_definition_version == self.feature_definition_version
            and float(record.event_time) <= cutoff
            and float(record.available_at) <= cutoff
        ]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda record: (
                float(record.event_time),
                float(record.available_at),
                str(record.source_event_id),
            ),
        )

    def build_ranking_observation(
        self,
        *,
        observation_id: str,
        user_id: str,
        product_id: str,
        as_of_ts: float,
        records: Iterable[HistoricalFeatureRecord],
        context: Optional[Mapping[str, Any]] = None,
    ) -> RankingObservation:
        # Materialize once because the same iterable is used for user and item joins.
        historical_records = tuple(records)
        user_record = self.latest_available(
            historical_records,
            entity_type="user",
            entity_id=user_id,
            as_of_ts=as_of_ts,
        )
        item_record = self.latest_available(
            historical_records,
            entity_type="item",
            entity_id=product_id,
            as_of_ts=as_of_ts,
        )
        return RankingObservation(
            observation_id=observation_id,
            user_id=user_id,
            product_id=product_id,
            as_of_ts=float(as_of_ts),
            feature_definition_version=self.feature_definition_version,
            user_features=dict(user_record.values) if user_record else {},
            product_metadata=dict(item_record.values) if item_record else {},
            context=dict(context or {}),
        )
