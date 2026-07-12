"""Canonical append-only feature-history event contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional


FEATURE_HISTORY_PAYLOAD_SCHEMA_VERSION = 1
RANKING_LTR_FEATURE_DEFINITION_VERSION = "ranking_ltr_v1"


class FeatureHistoryContractError(ValueError):
    """Raised when an event is unsafe to materialize into feature history."""


@dataclass(frozen=True)
class FeatureHistoryEventRecord:
    event_id: str
    event_type: str
    entity_type: str
    entity_id: str
    event_time: float
    available_at: float
    source_event_id: str
    source_version: str
    feature_definition_version: str
    payload_schema_version: int
    payload_hash: str
    payload: Dict[str, Any]
    request_id: Optional[str] = None


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize a feature payload deterministically for hashing and transport."""
    try:
        return _canonical_value(dict(payload))
    except (TypeError, ValueError) as exc:
        raise FeatureHistoryContractError(
            f"payload is not canonical JSON: {exc}"
        ) from exc


def _canonical_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("floating-point values must be finite")
        decimal = Decimal(str(value))
        if decimal.is_zero():
            return "0"
        return format(decimal.normalize(), "f")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        fields = (
            f"{json.dumps(key, ensure_ascii=False)}:{_canonical_value(value[key])}"
            for key in sorted(value)
        )
        return "{" + ",".join(fields) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical_value(item) for item in value) + "]"
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def catalog_feature_event_id(product_id: str, source_version: str) -> str:
    product = _required_identifier("product_id", product_id)
    version = _required_identifier("source_version", source_version)
    return hashlib.sha256(f"{product}\x00{version}".encode("utf-8")).hexdigest()


def build_feature_history_event(
    *,
    event_type: str,
    entity_type: str,
    entity_id: str,
    event_time: float,
    available_at: float,
    source_event_id: str,
    source_version: str,
    payload: Mapping[str, Any],
    feature_definition_version: str = RANKING_LTR_FEATURE_DEFINITION_VERSION,
    request_id: Optional[str] = None,
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    history_payload = dict(payload)
    resolved_source_event_id = _required_identifier("source_event_id", source_event_id)
    return {
        "event_id": _required_identifier(
            "event_id", event_id or resolved_source_event_id
        ),
        "event_type": _required_identifier("event_type", event_type),
        "entity_type": _required_identifier("entity_type", entity_type),
        "entity_id": _required_identifier("entity_id", entity_id),
        "event_time": _finite_timestamp("event_time", event_time),
        "available_at": _finite_timestamp("available_at", available_at),
        "source_event_id": resolved_source_event_id,
        "source_version": _required_identifier("source_version", source_version),
        "feature_definition_version": _required_identifier(
            "feature_definition_version", feature_definition_version
        ),
        "payload_schema_version": FEATURE_HISTORY_PAYLOAD_SCHEMA_VERSION,
        "payload_hash": payload_sha256(history_payload),
        "payload": history_payload,
        "request_id": request_id,
    }


def build_catalog_feature_event(
    *,
    product_id: str,
    source_version: str,
    event_time: float,
    available_at: float,
    payload: Mapping[str, Any],
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    event_id = catalog_feature_event_id(product_id, source_version)
    return build_feature_history_event(
        event_type="catalog_feature",
        entity_type="item",
        entity_id=product_id,
        event_time=event_time,
        available_at=available_at,
        source_event_id=event_id,
        source_version=source_version,
        payload=payload,
        request_id=request_id,
        event_id=event_id,
    )


def parse_feature_history_event(event: Mapping[str, Any]) -> FeatureHistoryEventRecord:
    required_fields = (
        "event_id",
        "event_type",
        "entity_type",
        "entity_id",
        "event_time",
        "available_at",
        "source_event_id",
        "source_version",
        "feature_definition_version",
        "payload_schema_version",
        "payload_hash",
        "payload",
    )
    for field in required_fields:
        if field not in event:
            raise FeatureHistoryContractError(f"missing required field: {field}")
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        raise FeatureHistoryContractError("payload must be an object")
    expected_hash = payload_sha256(payload)
    if event.get("payload_hash") != expected_hash:
        raise FeatureHistoryContractError(
            "payload_hash does not match canonical payload"
        )
    return FeatureHistoryEventRecord(
        event_id=_required_identifier("event_id", event.get("event_id")),
        event_type=_required_identifier("event_type", event.get("event_type")),
        entity_type=_required_identifier("entity_type", event.get("entity_type")),
        entity_id=_required_identifier("entity_id", event.get("entity_id")),
        event_time=_finite_timestamp("event_time", event.get("event_time")),
        available_at=_finite_timestamp("available_at", event.get("available_at")),
        source_event_id=_required_identifier(
            "source_event_id", event.get("source_event_id")
        ),
        source_version=_required_identifier(
            "source_version", event.get("source_version")
        ),
        feature_definition_version=_required_identifier(
            "feature_definition_version", event.get("feature_definition_version")
        ),
        payload_schema_version=int(event.get("payload_schema_version")),
        payload_hash=expected_hash,
        payload=dict(payload),
        request_id=event.get("request_id"),
    )


def _required_identifier(name: str, value: Any) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise FeatureHistoryContractError(f"{name} must not be blank")
    return normalized


def _finite_timestamp(name: str, value: Any) -> float:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureHistoryContractError(f"{name} must be a Unix timestamp") from exc
    if not math.isfinite(timestamp):
        raise FeatureHistoryContractError(f"{name} must be finite")
    return timestamp
