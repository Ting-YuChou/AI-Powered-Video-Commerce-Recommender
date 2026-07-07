"""Swing ItemCF recall helpers.

This module builds and serves a model-like item-to-item recall index from
positive interaction sequences. The request path only loads the artifact and
aggregates precomputed neighbors for recent user seed items.
"""

from __future__ import annotations

import gzip
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from video_commerce.common.models import CandidateProduct, InteractionType

POSITIVE_ACTIONS = {
    InteractionType.VIEW.value,
    InteractionType.CLICK.value,
    InteractionType.ADD_TO_CART.value,
    InteractionType.PURCHASE.value,
}

ACTION_WEIGHTS = {
    InteractionType.VIEW.value: 1.0,
    InteractionType.CLICK.value: 2.0,
    InteractionType.ADD_TO_CART.value: 3.0,
    InteractionType.PURCHASE.value: 5.0,
}


def _event_time(event: Mapping[str, Any]) -> float:
    try:
        return float(event.get("occurred_at", event.get("timestamp", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _action_weight(action: Any) -> float:
    return ACTION_WEIGHTS.get(str(action or "").lower(), 0.0)


@dataclass(frozen=True)
class SwingNeighbor:
    product_id: str
    score: float
    normalized_score: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SwingNeighbor":
        return cls(
            product_id=str(payload["product_id"]),
            score=float(payload.get("score", 0.0)),
            normalized_score=float(payload.get("normalized_score", 0.0)),
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "score": float(self.score),
            "normalized_score": float(self.normalized_score),
        }


class SwingItemCFIndex:
    """In-memory Swing item-to-item neighbor index."""

    def __init__(
        self,
        *,
        model_version: Optional[str] = None,
        neighbors: Optional[
            Mapping[str, Sequence[Mapping[str, Any] | SwingNeighbor]]
        ] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_version = model_version
        self.metadata = dict(metadata or {})
        self.neighbors: Dict[str, List[SwingNeighbor]] = {}
        for product_id, raw_neighbors in (neighbors or {}).items():
            parsed = [
                neighbor
                if isinstance(neighbor, SwingNeighbor)
                else SwingNeighbor.from_payload(neighbor)
                for neighbor in raw_neighbors
            ]
            parsed.sort(key=lambda item: (-item.normalized_score, item.product_id))
            self.neighbors[str(product_id)] = parsed

    @property
    def is_trained(self) -> bool:
        return bool(self.neighbors)

    def get_neighbors(self, product_id: str) -> List[SwingNeighbor]:
        return list(self.neighbors.get(str(product_id), []))

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "model_version": self.model_version,
            "metadata": dict(self.metadata),
            "neighbors": {
                product_id: [neighbor.to_payload() for neighbor in neighbors]
                for product_id, neighbors in sorted(self.neighbors.items())
            },
        }

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_payload()
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))

    @classmethod
    def load(cls, path: str) -> "SwingItemCFIndex":
        target = Path(path)
        with gzip.open(target, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            model_version=payload.get("model_version"),
            neighbors=payload.get("neighbors") or {},
            metadata=payload.get("metadata") or {},
        )


class SwingItemCFTrainer:
    """Build a Swing item-to-item index from positive user sequences."""

    def __init__(
        self,
        *,
        alpha: float = 5.0,
        max_neighbors_per_item: int = 100,
        max_items_per_user: int = 100,
        max_users_per_item: int = 500,
    ) -> None:
        self.alpha = max(float(alpha), 1e-6)
        self.max_neighbors_per_item = max(1, int(max_neighbors_per_item))
        self.max_items_per_user = max(1, int(max_items_per_user))
        self.max_users_per_item = max(1, int(max_users_per_item))

    def fit(
        self,
        user_sequences: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        model_version: Optional[str] = None,
    ) -> SwingItemCFIndex:
        normalized = {
            str(user_id): items
            for user_id, sequence in user_sequences.items()
            if (items := self._dedupe_user_items(sequence))
        }
        item_users: Dict[str, List[str]] = defaultdict(list)
        for user_id, items in normalized.items():
            for product_id in items:
                item_users[product_id].append(user_id)

        bounded_item_users = {
            product_id: sorted(users)[: self.max_users_per_item]
            for product_id, users in item_users.items()
            if len(users) >= 2
        }
        user_pairs: Set[Tuple[str, str]] = set()
        for users in bounded_item_users.values():
            user_pairs.update(combinations(users, 2))

        pair_scores: Dict[str, Counter[str]] = defaultdict(Counter)
        bounded_user_item_sets: Dict[str, Set[str]] = defaultdict(set)
        for product_id, users in bounded_item_users.items():
            for user_id in users:
                bounded_user_item_sets[user_id].add(product_id)
        for left_user, right_user in sorted(user_pairs):
            common_items = sorted(
                bounded_user_item_sets[left_user] & bounded_user_item_sets[right_user]
            )
            if len(common_items) < 2:
                continue
            contribution = 1.0 / (self.alpha + len(common_items))
            for source in common_items:
                for target in common_items:
                    if source != target:
                        pair_scores[source][target] += contribution

        neighbors: Dict[str, List[Dict[str, float | str]]] = {}
        for product_id, scores in pair_scores.items():
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            top = ranked[: self.max_neighbors_per_item]
            max_score = top[0][1] if top else 0.0
            if max_score <= 0.0:
                continue
            neighbors[product_id] = [
                {
                    "product_id": neighbor_id,
                    "score": float(score),
                    "normalized_score": float(score / max_score),
                }
                for neighbor_id, score in top
            ]

        return SwingItemCFIndex(
            model_version=model_version or f"swing-itemcf-{int(time.time())}",
            neighbors=neighbors,
            metadata={
                "alpha": self.alpha,
                "training_user_count": len(normalized),
                "item_count": len(neighbors),
                "max_neighbors_per_item": self.max_neighbors_per_item,
                "max_items_per_user": self.max_items_per_user,
                "max_users_per_item": self.max_users_per_item,
                "built_at": time.time(),
            },
        )

    def _dedupe_user_items(
        self, sequence: Sequence[Mapping[str, Any]]
    ) -> List[str]:
        best_by_product: Dict[str, Tuple[float, float]] = {}
        for event in sequence:
            product_id = str(event.get("product_id") or "").strip()
            action = str(event.get("action") or "").lower()
            if not product_id or action not in POSITIVE_ACTIONS:
                continue
            weight = _action_weight(action)
            occurred_at = _event_time(event)
            previous = best_by_product.get(product_id)
            if previous is None:
                best_by_product[product_id] = (weight, occurred_at)
                continue
            best_by_product[product_id] = (
                max(previous[0], weight),
                max(previous[1], occurred_at),
            )

        ranked = sorted(
            best_by_product.items(),
            key=lambda item: (-item[1][1], item[0]),
        )
        return [product_id for product_id, _ in ranked[: self.max_items_per_user]]


class SwingItemCFCandidateEngine:
    """Aggregate Swing neighbors for a user's recent positive seed items."""

    def __init__(
        self,
        index: Optional[SwingItemCFIndex] = None,
        *,
        max_seed_items: int = 20,
        score_weight: float = 1.0,
    ) -> None:
        self.index = index or SwingItemCFIndex()
        self.max_seed_items = max(1, int(max_seed_items))
        self.score_weight = max(0.0, float(score_weight))
        self.model_version = self.index.model_version

    @property
    def is_trained(self) -> bool:
        return self.index.is_trained

    def load_artifact(self, path: str) -> bool:
        target = Path(path)
        if not target.exists():
            self.index = SwingItemCFIndex()
            self.model_version = None
            return False
        self.index = SwingItemCFIndex.load(str(target))
        self.model_version = self.index.model_version
        return self.index.is_trained

    def get_candidates(
        self,
        user_interactions: Sequence[Mapping[str, Any]],
        *,
        k: int,
        exclude_items: Optional[Set[str]] = None,
        current_time: Optional[float] = None,
    ) -> List[CandidateProduct]:
        if k <= 0 or not self.is_trained:
            return []

        current_time = time.time() if current_time is None else float(current_time)
        exclude_items = set(exclude_items or set())
        seed_events = self._recent_seed_events(user_interactions)
        scores: Counter[str] = Counter()
        for seed in seed_events:
            seed_product = str(seed.get("product_id"))
            seed_weight = _action_weight(seed.get("action"))
            recency_weight = self._recency_weight(seed, current_time)
            if seed_weight <= 0.0 or recency_weight <= 0.0:
                continue
            for neighbor in self.index.get_neighbors(seed_product):
                if neighbor.product_id in exclude_items:
                    continue
                scores[neighbor.product_id] += (
                    seed_weight
                    * recency_weight
                    * neighbor.normalized_score
                    * self.score_weight
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [
            CandidateProduct(
                product_id=product_id,
                collaborative_score=float(score),
                combined_score=float(score),
                source="swing_itemcf",
            )
            for product_id, score in ranked
            if score > 0.0
        ]

    def _recent_seed_events(
        self, user_interactions: Sequence[Mapping[str, Any]]
    ) -> List[Mapping[str, Any]]:
        latest_by_product: Dict[str, Tuple[float, str, float]] = {}
        for event in user_interactions:
            product_id = str(event.get("product_id") or "").strip()
            action = str(event.get("action") or "").lower()
            if not product_id or action not in POSITIVE_ACTIONS:
                continue
            weight = _action_weight(action)
            occurred_at = _event_time(event)
            previous = latest_by_product.get(product_id)
            if previous is None:
                latest_by_product[product_id] = (weight, action, occurred_at)
                continue
            previous_weight, previous_action, previous_time = previous
            latest_by_product[product_id] = (
                max(previous_weight, weight),
                action if weight > previous_weight else previous_action,
                max(previous_time, occurred_at),
            )
        seed_events = [
            {
                "product_id": product_id,
                "action": action,
                "occurred_at": occurred_at,
                "timestamp": occurred_at,
            }
            for product_id, (_, action, occurred_at) in latest_by_product.items()
        ]
        ranked = sorted(
            seed_events,
            key=lambda event: (-_event_time(event), str(event.get("product_id"))),
        )
        return ranked[: self.max_seed_items]

    @staticmethod
    def _recency_weight(event: Mapping[str, Any], current_time: float) -> float:
        age_days = max((current_time - _event_time(event)) / 86400.0, 0.0)
        return 1.0 / (1.0 + age_days / 30.0)
