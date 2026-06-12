"""Shared ranking features built from last-N two-tower item histories."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

RANKING_HISTORY_CONTEXT_KEY = "_ranking_history_embeddings"
RANKING_HISTORY_SCHEMA_VERSION = "ranking_history_embeddings_v1"
RANKING_HISTORY_ACTIONS = ("click", "cart", "purchase")
RAW_ACTION_TO_HISTORY_ACTION = {
    "click": "click",
    "add_to_cart": "cart",
    "cart": "cart",
    "purchase": "purchase",
}

_SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class RankingHistoryConfig:
    embedding_dim: int = 128
    click_last_n: int = 20
    cart_last_n: int = 20
    purchase_last_n: int = 20
    click_scale: float = 1.0
    cart_scale: float = 1.25
    purchase_scale: float = 1.75
    max_recency_days: float = 30.0

    def last_n_for(self, action: str) -> int:
        return max(0, int(getattr(self, f"{action}_last_n", 0)))

    def scale_for(self, action: str) -> float:
        return float(getattr(self, f"{action}_scale", 1.0))

    @property
    def max_last_n(self) -> int:
        return max(self.last_n_for(action) for action in RANKING_HISTORY_ACTIONS)


def ranking_history_config_from_settings(config: Any) -> RankingHistoryConfig:
    """Build a pure history-feature config from the service ranking settings."""
    return RankingHistoryConfig(
        embedding_dim=max(1, int(getattr(config, "history_embedding_dim", 128) or 128)),
        click_last_n=max(0, int(getattr(config, "history_click_last_n", 20) or 0)),
        cart_last_n=max(0, int(getattr(config, "history_cart_last_n", 20) or 0)),
        purchase_last_n=max(
            0,
            int(getattr(config, "history_purchase_last_n", 20) or 0),
        ),
        click_scale=float(getattr(config, "history_click_scale", 1.0) or 0.0),
        cart_scale=float(getattr(config, "history_cart_scale", 1.25) or 0.0),
        purchase_scale=float(getattr(config, "history_purchase_scale", 1.75) or 0.0),
    )


def _event_value(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(key, default)
    return getattr(event, key, default)


def _event_product_id(event: Any) -> str:
    return str(_event_value(event, "product_id", "") or "")


def _event_timestamp(event: Any) -> Optional[float]:
    for key in ("occurred_at", "timestamp", "created_at"):
        value = _event_value(event, key)
        if value is None:
            continue
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _normalize_embedding(value: Any, embedding_dim: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.shape != (embedding_dim,):
        return None
    return np.nan_to_num(vector, 0.0)


def merge_item_embedding_maps(*maps: Optional[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    """Merge item embedding maps while preserving later-map overrides."""
    merged: Dict[str, np.ndarray] = {}
    for item_map in maps:
        if not isinstance(item_map, Mapping):
            continue
        for product_id, embedding in item_map.items():
            if product_id is None or embedding is None:
                continue
            try:
                merged[str(product_id)] = np.asarray(
                    embedding,
                    dtype=np.float32,
                ).reshape(-1)
            except (TypeError, ValueError):
                continue
    return merged


def _event_sort_key(index: int, event: Any) -> tuple:
    timestamp = _event_timestamp(event)
    if timestamp is not None:
        return (0, timestamp, index)
    return (1, float(index), index)


def build_ranking_history_context(
    user_sequence: Sequence[Any],
    item_embedding_map: Mapping[str, Any],
    *,
    config: RankingHistoryConfig,
    candidate_product_ids: Optional[Iterable[str]] = None,
    current_time: Optional[float] = None,
    two_tower_model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Build separate last-N action vectors and candidate similarities."""
    current_time = float(current_time if current_time is not None else 0.0)
    if current_time <= 0.0:
        timestamps = [
            timestamp
            for timestamp in (_event_timestamp(event) for event in user_sequence or [])
            if timestamp is not None
        ]
        current_time = max(timestamps) if timestamps else 0.0

    grouped_events: Dict[str, List[tuple]] = defaultdict(list)
    for index, event in enumerate(user_sequence or []):
        action = RAW_ACTION_TO_HISTORY_ACTION.get(str(_event_value(event, "action", "")))
        if action:
            grouped_events[action].append((index, event))

    action_payloads: Dict[str, Dict[str, Any]] = {}
    for action in RANKING_HISTORY_ACTIONS:
        last_n = config.last_n_for(action)
        indexed_events = sorted(
            grouped_events.get(action, []),
            key=lambda item: _event_sort_key(item[0], item[1]),
        )
        events = [event for _, event in indexed_events]
        selected_events = events[-last_n:] if last_n > 0 else []
        embeddings = []
        latest_timestamp = None
        for event in selected_events:
            timestamp = _event_timestamp(event)
            if timestamp is not None:
                latest_timestamp = (
                    timestamp
                    if latest_timestamp is None
                    else max(latest_timestamp, timestamp)
                )
            embedding = _normalize_embedding(
                item_embedding_map.get(_event_product_id(event)),
                config.embedding_dim,
            )
            if embedding is not None:
                embeddings.append(embedding)

        if embeddings:
            vector = np.mean(np.vstack(embeddings), axis=0).astype(np.float32)
        else:
            vector = np.zeros(config.embedding_dim, dtype=np.float32)

        scaled_vector = vector * config.scale_for(action)
        selected_count = len(selected_events)
        covered_count = len(embeddings)
        if latest_timestamp is None or current_time <= 0.0:
            recency_days_capped = config.max_recency_days
        else:
            recency_days = max(0.0, (current_time - latest_timestamp) / _SECONDS_PER_DAY)
            recency_days_capped = min(recency_days, config.max_recency_days)

        action_payloads[action] = {
            "vector": scaled_vector.astype(np.float32).tolist(),
            "count": selected_count,
            "covered_count": covered_count,
            "count_ratio": (
                float(selected_count) / float(last_n) if last_n > 0 else 0.0
            ),
            "coverage_ratio": (
                float(covered_count) / float(selected_count)
                if selected_count > 0
                else 0.0
            ),
            "recency_days_capped": recency_days_capped,
            "has_signal": 1.0 if covered_count > 0 else 0.0,
            "scale": config.scale_for(action),
        }

    candidate_similarities: Dict[str, Dict[str, float]] = {}
    for product_id in candidate_product_ids or []:
        candidate_id = str(product_id)
        candidate_embedding = _normalize_embedding(
            item_embedding_map.get(candidate_id),
            config.embedding_dim,
        )
        candidate_similarities[candidate_id] = {}
        for action in RANKING_HISTORY_ACTIONS:
            if candidate_embedding is None:
                similarity = 0.0
            else:
                history_vector = np.asarray(
                    action_payloads[action]["vector"],
                    dtype=np.float32,
                )
                similarity = float(np.dot(history_vector, candidate_embedding))
            candidate_similarities[candidate_id][action] = similarity

    return {
        "schema_version": RANKING_HISTORY_SCHEMA_VERSION,
        "embedding_dim": config.embedding_dim,
        "two_tower_model_version": two_tower_model_version,
        "actions": action_payloads,
        "candidate_similarities": candidate_similarities,
    }


def extract_ranking_history_feature_vector(
    history_context: Optional[Mapping[str, Any]],
    product_id: Any,
    *,
    embedding_dim: int,
) -> np.ndarray:
    """Flatten history context for one candidate in stable action-block order."""
    values: List[float] = []
    context = history_context if isinstance(history_context, Mapping) else {}
    actions = context.get("actions") if isinstance(context, Mapping) else {}
    if not isinstance(actions, Mapping):
        actions = {}
    candidate_id = str(product_id or "")
    similarities = context.get("candidate_similarities", {})
    if not isinstance(similarities, Mapping):
        similarities = {}
    candidate_sims = similarities.get(candidate_id, {})
    if not isinstance(candidate_sims, Mapping):
        candidate_sims = {}

    for action in RANKING_HISTORY_ACTIONS:
        payload = actions.get(action, {})
        if not isinstance(payload, Mapping):
            payload = {}
        vector = _normalize_embedding(payload.get("vector"), embedding_dim)
        if vector is None:
            vector = np.zeros(embedding_dim, dtype=np.float32)
        values.extend(vector.tolist())
        values.extend(
            [
                float(payload.get("count_ratio", 0.0) or 0.0),
                float(payload.get("coverage_ratio", 0.0) or 0.0),
                float(payload.get("recency_days_capped", 0.0) or 0.0),
                float(payload.get("has_signal", 0.0) or 0.0),
                float(candidate_sims.get(action, 0.0) or 0.0),
            ]
        )

    return np.nan_to_num(np.asarray(values, dtype=np.float32), 0.0)


def build_training_history_contexts(
    samples: Sequence[Mapping[str, Any]],
    item_embedding_map: Mapping[str, Any],
    *,
    config: RankingHistoryConfig,
    two_tower_model_version: Optional[str] = None,
) -> Dict[int, Dict[str, Any]]:
    """Build per-sample contexts from prior same-user events only."""
    indexed_samples = list(enumerate(samples))
    indexed_samples.sort(
        key=lambda item: (
            str(item[1].get("user_id") or ""),
            _event_timestamp(item[1])
            if _event_timestamp(item[1]) is not None
            else float(item[0]),
            str(item[1].get("event_id") or item[0]),
        )
    )

    per_user_history: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    contexts: Dict[int, Dict[str, Any]] = {}
    group: List[tuple] = []
    current_group_key: Optional[tuple] = None

    def flush_group(group_items: List[tuple]) -> None:
        if not group_items:
            return
        for original_index, sample in group_items:
            user_id = str(sample.get("user_id") or "")
            product_id = str(sample.get("product_id") or "")
            contexts[original_index] = build_ranking_history_context(
                per_user_history[user_id],
                item_embedding_map,
                config=config,
                candidate_product_ids=[product_id] if product_id else [],
                current_time=_event_timestamp(sample),
                two_tower_model_version=two_tower_model_version,
            )
        for _, sample in group_items:
            user_id = str(sample.get("user_id") or "")
            per_user_history[user_id].append(sample)

    for original_index, sample in indexed_samples:
        user_id = str(sample.get("user_id") or "")
        timestamp = _event_timestamp(sample)
        group_key = (
            user_id,
            timestamp if timestamp is not None else float(original_index),
        )
        if current_group_key is not None and group_key != current_group_key:
            flush_group(group)
            group = []
        current_group_key = group_key
        group.append((original_index, sample))

    flush_group(group)

    return contexts


def history_context_profile(history_context: Mapping[str, Any]) -> Dict[str, Any]:
    actions = history_context.get("actions", {}) if isinstance(history_context, Mapping) else {}
    profile: Dict[str, Any] = {
        "history_embeddings_two_tower_model": (
            history_context.get("two_tower_model_version")
            if isinstance(history_context, Mapping)
            else None
        )
    }
    for action in RANKING_HISTORY_ACTIONS:
        payload = actions.get(action, {}) if isinstance(actions, Mapping) else {}
        if not isinstance(payload, Mapping):
            payload = {}
        prefix = f"history_embeddings_{action}"
        profile[f"{prefix}_count"] = int(payload.get("count", 0) or 0)
        profile[f"{prefix}_coverage_ratio"] = round(
            float(payload.get("coverage_ratio", 0.0) or 0.0),
            4,
        )
    return profile
