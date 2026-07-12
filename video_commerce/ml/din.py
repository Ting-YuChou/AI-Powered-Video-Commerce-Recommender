"""Point-in-time behavior sequences and DIN ranking components."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DIN_SEQUENCE_CONTRACT_VERSION = "din_sequence_v1"
DIN_SEQUENCE_CONTEXT_KEY = "_din_behavior_sequences"
DIN_ACTIONS = ("click", "cart", "purchase")
DIN_ACTION_MAP = {
    "click": "click",
    "add_to_cart": "cart",
    "cart": "cart",
    "purchase": "purchase",
}
DIN_DEFAULT_LAST_N = 60
DIN_DEFAULT_LOOKBACK_DAYS = 30


@dataclass(frozen=True)
class DINActionSequence:
    product_ids: Tuple[str, ...]
    event_times: Tuple[float, ...]
    event_ids: Tuple[Optional[str], ...]
    mask: Tuple[bool, ...]

    @property
    def length(self) -> int:
        return sum(self.mask)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_ids": list(self.product_ids),
            "event_times": list(self.event_times),
            "event_ids": list(self.event_ids),
            "mask": list(self.mask),
        }


@dataclass(frozen=True)
class DINBehaviorSequences:
    as_of_ts: float
    actions: Mapping[str, DINActionSequence]
    contract_version: str = DIN_SEQUENCE_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "as_of_ts": self.as_of_ts,
            "actions": {
                action: self.actions[action].to_dict() for action in DIN_ACTIONS
            },
        }


@dataclass(frozen=True)
class DINBatchInputs:
    """Structured DIN tensors with histories stored once per request."""

    candidate_indices: torch.Tensor
    request_history_indices: torch.Tensor
    request_history_recency: torch.Tensor
    request_history_mask: torch.Tensor
    candidate_to_request: torch.Tensor
    summary_features: torch.Tensor

    def expanded_histories(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mapping = self.candidate_to_request
        return (
            self.request_history_indices.index_select(0, mapping),
            self.request_history_recency.index_select(0, mapping),
            self.request_history_mask.index_select(0, mapping),
        )


def build_din_batch_inputs(
    bundles: Sequence[Any],
    product_index: Mapping[str, int],
    *,
    sequence_length: int = DIN_DEFAULT_LAST_N,
    device: Optional[torch.device] = None,
) -> DINBatchInputs:
    """Build candidate tensors while de-duplicating request-level histories."""
    target_device = device or torch.device("cpu")
    request_keys: Dict[Tuple[Any, ...], int] = {}
    histories: list[list[list[int]]] = []
    recencies: list[list[list[float]]] = []
    masks: list[list[list[bool]]] = []
    summaries: list[list[float]] = []
    candidates: list[int] = []
    candidate_to_request: list[int] = []
    for bundle in bundles:
        sequences = getattr(bundle, "behavior_sequences", None)
        key = (
            float(bundle.as_of_ts),
            build_din_freshness_token(sequences)["sequence_hash"]
            if sequences is not None
            else "empty",
        )
        request_index = request_keys.get(key)
        if request_index is None:
            request_index = len(histories)
            request_keys[key] = request_index
            action_indices: list[list[int]] = []
            action_recencies: list[list[float]] = []
            action_masks: list[list[bool]] = []
            for action in DIN_ACTIONS:
                sequence = sequences.actions[action] if sequences is not None else None
                products = (
                    list(sequence.product_ids[-sequence_length:]) if sequence else []
                )
                times = (
                    list(sequence.event_times[-sequence_length:]) if sequence else []
                )
                present = list(sequence.mask[-sequence_length:]) if sequence else []
                padding = sequence_length - len(products)
                products = [""] * padding + products
                times = [0.0] * padding + times
                present = [False] * padding + present
                indices = [
                    int(product_index.get(product_id, 0)) for product_id in products
                ]
                known_mask = [
                    bool(valid and index != 0) for valid, index in zip(present, indices)
                ]
                action_indices.append(indices)
                action_masks.append(known_mask)
                action_recencies.append(
                    [
                        math.log1p(
                            max(0.0, float(bundle.as_of_ts) - event_time) / 86400.0
                        )
                        if valid
                        else 0.0
                        for valid, event_time in zip(known_mask, times)
                    ]
                )
            histories.append(action_indices)
            recencies.append(action_recencies)
            masks.append(action_masks)
        sequences = getattr(bundle, "behavior_sequences", None)
        row_summary: list[float] = []
        for action_index, action in enumerate(DIN_ACTIONS):
            sequence = sequences.actions[action] if sequences is not None else None
            count = float(sequence.length if sequence else 0)
            known = float(sum(masks[request_index][action_index]))
            valid_recencies = [
                value
                for value, valid in zip(
                    recencies[request_index][action_index],
                    masks[request_index][action_index],
                )
                if valid
            ]
            row_summary.extend(
                [
                    count,
                    known / count if count else 0.0,
                    min(valid_recencies) if valid_recencies else 0.0,
                    1.0 if count else 0.0,
                ]
            )
        candidates.append(int(product_index.get(bundle.candidate.product_id, 0)))
        candidate_to_request.append(request_index)
        summaries.append(row_summary)
    for index, candidate_index in enumerate(candidates):
        if candidate_index == 0:
            summaries[index] = [0.0] * 12
    return DINBatchInputs(
        candidate_indices=torch.tensor(
            candidates, dtype=torch.long, device=target_device
        ),
        request_history_indices=torch.tensor(
            histories, dtype=torch.long, device=target_device
        ),
        request_history_recency=torch.tensor(
            recencies, dtype=torch.float32, device=target_device
        ),
        request_history_mask=torch.tensor(
            masks, dtype=torch.bool, device=target_device
        ),
        candidate_to_request=torch.tensor(
            candidate_to_request, dtype=torch.long, device=target_device
        ),
        summary_features=torch.tensor(
            summaries, dtype=torch.float32, device=target_device
        ),
    )


def _value(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(key, default)
    return getattr(event, key, default)


def _timestamp(event: Any, *keys: str) -> Optional[float]:
    for key in keys:
        raw = _value(event, key)
        if raw is None:
            continue
        if hasattr(raw, "timestamp"):
            raw = raw.timestamp()
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def build_din_behavior_sequences(
    events: Iterable[Any],
    *,
    as_of_ts: float,
    last_n: int = DIN_DEFAULT_LAST_N,
    lookback_days: int = DIN_DEFAULT_LOOKBACK_DAYS,
) -> DINBehaviorSequences:
    """Build fixed-length, point-in-time-safe action histories."""
    as_of = float(as_of_ts)
    if not math.isfinite(as_of) or as_of < 0:
        raise ValueError("as_of_ts must be a finite non-negative timestamp")
    if last_n <= 0:
        raise ValueError("last_n must be positive")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    lower_bound = as_of - float(lookback_days) * 86400.0
    grouped: Dict[str, list[tuple[float, str, str]]] = {
        action: [] for action in DIN_ACTIONS
    }
    for event in events or []:
        action = DIN_ACTION_MAP.get(str(_value(event, "action", "")).lower())
        product_id = str(_value(event, "product_id", "") or "").strip()
        event_time = _timestamp(event, "event_time", "occurred_at", "timestamp")
        available_at = _timestamp(event, "available_at")
        if available_at is None:
            available_at = event_time
        if (
            not action
            or not product_id
            or event_time is None
            or available_at is None
            or event_time < lower_bound
            or event_time >= as_of
            or available_at > as_of
        ):
            continue
        event_id = str(_value(event, "event_id", "") or "")
        grouped[action].append((event_time, event_id, product_id))

    actions: Dict[str, DINActionSequence] = {}
    for action in DIN_ACTIONS:
        selected = sorted(grouped[action], key=lambda row: (row[0], row[1]))[-last_n:]
        padding = last_n - len(selected)
        actions[action] = DINActionSequence(
            product_ids=("",) * padding + tuple(row[2] for row in selected),
            event_times=(0.0,) * padding + tuple(row[0] for row in selected),
            event_ids=(None,) * padding + tuple(row[1] or None for row in selected),
            mask=(False,) * padding + (True,) * len(selected),
        )
    return DINBehaviorSequences(as_of_ts=as_of, actions=actions)


def build_din_freshness_token(sequences: DINBehaviorSequences) -> Dict[str, Any]:
    """Return a stable cache-freshness token for all DIN action histories."""
    action_tokens: Dict[str, Dict[str, Any]] = {}
    for action in DIN_ACTIONS:
        sequence = sequences.actions[action]
        valid_indices = [
            index for index, present in enumerate(sequence.mask) if present
        ]
        latest = valid_indices[-1] if valid_indices else None
        action_tokens[action] = {
            "length": sequence.length,
            "latest_event_id": sequence.event_ids[latest]
            if latest is not None
            else None,
            "latest_event_time": (
                round(sequence.event_times[latest], 6) if latest is not None else 0.0
            ),
            "latest_product_id": (
                sequence.product_ids[latest] if latest is not None else None
            ),
        }
    canonical = json.dumps(
        {
            "contract_version": sequences.contract_version,
            "actions": action_tokens,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "contract_version": sequences.contract_version,
        "actions": action_tokens,
        "sequence_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def parse_din_behavior_sequences(
    payload: Mapping[str, Any],
    *,
    expected_as_of_ts: Optional[float] = None,
    last_n: int = DIN_DEFAULT_LAST_N,
    lookback_days: int = DIN_DEFAULT_LOOKBACK_DAYS,
) -> DINBehaviorSequences:
    """Validate a serialized DIN contract without silently repairing unsafe rows."""
    if not isinstance(payload, Mapping):
        raise ValueError("DIN behavior sequences must be an object")
    if payload.get("contract_version") != DIN_SEQUENCE_CONTRACT_VERSION:
        raise ValueError("unsupported DIN sequence contract version")
    try:
        as_of_ts = float(payload["as_of_ts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("DIN behavior sequences require a valid as_of_ts") from exc
    if expected_as_of_ts is not None and as_of_ts != float(expected_as_of_ts):
        raise ValueError("DIN sequence as_of_ts does not match feature bundle")
    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, Mapping) or set(raw_actions) != set(DIN_ACTIONS):
        raise ValueError("DIN behavior sequences require exactly three actions")
    lower_bound = as_of_ts - float(lookback_days) * 86400.0
    actions: Dict[str, DINActionSequence] = {}
    for action in DIN_ACTIONS:
        raw = raw_actions[action]
        if not isinstance(raw, Mapping):
            raise ValueError(f"DIN {action} sequence must be an object")
        product_ids = raw.get("product_ids")
        event_times = raw.get("event_times")
        event_ids = raw.get("event_ids")
        mask = raw.get("mask")
        if not all(
            isinstance(values, list)
            for values in (product_ids, event_times, event_ids, mask)
        ):
            raise ValueError(f"DIN {action} sequence fields must be arrays")
        if any(
            len(values) != last_n
            for values in (product_ids, event_times, event_ids, mask)
        ):
            raise ValueError(f"DIN {action} sequence fields must have length {last_n}")
        normalized_products = tuple(str(value or "") for value in product_ids)
        normalized_times = tuple(float(value) for value in event_times)
        normalized_ids = tuple(
            None if value is None else str(value) for value in event_ids
        )
        normalized_mask = tuple(bool(value) for value in mask)
        saw_valid = False
        previous_time = 0.0
        for index, present in enumerate(normalized_mask):
            if not present:
                if (
                    saw_valid
                    or normalized_products[index]
                    or normalized_times[index] != 0.0
                    or normalized_ids[index] is not None
                ):
                    raise ValueError(f"DIN {action} sequence is not left padded")
                continue
            saw_valid = True
            event_time = normalized_times[index]
            if not normalized_products[index] or not math.isfinite(event_time):
                raise ValueError(f"DIN {action} sequence has an invalid event")
            if event_time < lower_bound or event_time >= as_of_ts:
                raise ValueError(f"DIN {action} sequence leaks outside the PIT window")
            if event_time < previous_time:
                raise ValueError(f"DIN {action} sequence is not chronological")
            previous_time = event_time
        actions[action] = DINActionSequence(
            product_ids=normalized_products,
            event_times=normalized_times,
            event_ids=normalized_ids,
            mask=normalized_mask,
        )
    return DINBehaviorSequences(as_of_ts=as_of_ts, actions=actions)


class Dice(nn.Module):
    """Data-adaptive activation used by DIN local activation units."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(features))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        original_shape = values.shape
        flattened = values.reshape(-1, original_shape[-1])
        mean = flattened.mean(dim=0, keepdim=True)
        variance = (flattened - mean).pow(2).mean(dim=0, keepdim=True)
        probability = torch.sigmoid((flattened - mean) / torch.sqrt(variance + 1e-8))
        activated = (
            probability * flattened + (1.0 - probability) * self.alpha * flattened
        )
        return activated.reshape(original_shape).contiguous()


class ScalarLinear(nn.Module):
    """One-unit affine head without the platform-sensitive Linear(..., 1) kernel."""

    def __init__(self, input_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_features))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.weight, mean=0.0, std=input_features**-0.5)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sum(values * self.weight, dim=-1) + self.bias


class DeepInterestNetwork(nn.Module):
    """Shared action-aware DIN attention over frozen two-tower embeddings."""

    def __init__(
        self,
        item_embeddings: torch.Tensor,
        *,
        action_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        embeddings = torch.as_tensor(item_embeddings, dtype=torch.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != 128:
            raise ValueError("DIN item embeddings must have shape [items, 128]")
        if embeddings.shape[0] < 1:
            raise ValueError("DIN item embeddings require a padding row")
        embeddings = embeddings.clone()
        embeddings[0].zero_()
        self.item_embedding = nn.Embedding.from_pretrained(
            embeddings,
            freeze=True,
            padding_idx=0,
        )
        self.action_embedding = nn.Embedding(len(DIN_ACTIONS), action_embedding_dim)
        input_dim = 4 * embeddings.shape[1] + action_embedding_dim + 1
        self.attention_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            Dice(256),
            nn.Linear(256, 64),
            Dice(64),
            ScalarLinear(64),
        )
        self.interest_projection = nn.Sequential(
            nn.Linear(len(DIN_ACTIONS) * embeddings.shape[1], 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.last_attention_entropy = 0.0

    def forward(
        self,
        candidate_indices: torch.Tensor,
        history_indices: torch.Tensor,
        history_recency: torch.Tensor,
        history_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        if history_indices.ndim != 3 or history_indices.shape[1] != len(DIN_ACTIONS):
            raise ValueError("history_indices must have shape [batch, 3, sequence]")
        if history_recency.shape != history_indices.shape:
            raise ValueError("history_recency shape must match history_indices")
        if history_mask.shape != history_indices.shape:
            raise ValueError("history_mask shape must match history_indices")
        original_sequence_length = history_indices.shape[-1]
        first_active = 0
        active_columns = history_mask.bool().any(dim=0).any(dim=0)
        if not active_columns.any():
            empty_interest = torch.zeros(
                (candidate_indices.shape[0], 128),
                dtype=self.item_embedding.weight.dtype,
                device=candidate_indices.device,
            )
            if return_attention:
                return empty_interest, torch.zeros_like(history_recency)
            return empty_interest
        if active_columns.any():
            first_active = int(torch.nonzero(active_columns, as_tuple=False)[0].item())
            history_indices = history_indices[:, :, first_active:]
            history_recency = history_recency[:, :, first_active:]
            history_mask = history_mask[:, :, first_active:]
        candidate = self.item_embedding(candidate_indices).unsqueeze(1).unsqueeze(1)
        history = self.item_embedding(history_indices)
        candidate = candidate.expand_as(history)
        batch_size, _, sequence_length, _ = history.shape
        action_ids = torch.arange(
            len(DIN_ACTIONS), device=history.device, dtype=torch.long
        ).view(1, len(DIN_ACTIONS), 1)
        action_values = self.action_embedding(action_ids).expand(
            batch_size, -1, sequence_length, -1
        )
        local_features = torch.cat(
            [
                candidate,
                history,
                candidate - history,
                candidate * history,
                action_values,
                history_recency.to(history.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        logits = self.attention_mlp(
            local_features.reshape(-1, local_features.shape[-1])
        ).reshape(batch_size, len(DIN_ACTIONS), sequence_length)
        mask = history_mask.bool()
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(masked_logits, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        weights = torch.nan_to_num(weights, nan=0.0)
        normalizer = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(
            normalizer > 0, weights / normalizer.clamp_min(1e-12), weights
        )
        with torch.no_grad():
            entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=-1)
            valid_actions = mask.any(dim=-1)
            self.last_attention_entropy = (
                float(entropy[valid_actions].mean().cpu().item())
                if valid_actions.any()
                else 0.0
            )
        action_interests = torch.sum(weights.unsqueeze(-1) * history, dim=2)
        interest = self.interest_projection(action_interests.flatten(start_dim=1))
        candidate_present = candidate_indices.ne(0).to(interest.dtype).unsqueeze(-1)
        any_history = mask.any(dim=-1).any(dim=-1).to(interest.dtype).unsqueeze(-1)
        interest = interest * candidate_present * any_history
        if return_attention:
            if first_active:
                weights = F.pad(weights, (first_active, 0))
                weights = weights[..., :original_sequence_length]
            return interest, weights
        return interest


def save_din_embedding_sidecar(
    path: str,
    embedding_map: Mapping[str, Any],
    *,
    two_tower_model_version: str,
) -> Dict[str, Any]:
    """Atomically copy trained two-tower embeddings into ranking ownership."""
    product_ids = sorted(str(product_id) for product_id in embedding_map)
    if not product_ids:
        raise ValueError("DIN embedding sidecar cannot be empty")
    rows = [np.zeros(128, dtype=np.float32)]
    for product_id in product_ids:
        vector = np.asarray(embedding_map[product_id], dtype=np.float32)
        if vector.shape != (128,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid DIN embedding for product {product_id}")
        rows.append(vector)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    os.close(handle)
    temporary_path = temporary + ".npz"
    try:
        np.savez_compressed(
            temporary_path,
            product_ids=np.asarray([""] + product_ids),
            embeddings=np.vstack(rows),
            two_tower_model_version=np.asarray([two_tower_model_version]),
            contract_version=np.asarray([DIN_SEQUENCE_CONTRACT_VERSION]),
        )
        os.replace(temporary_path, target)
    finally:
        for leftover in (temporary, temporary_path):
            if os.path.exists(leftover):
                os.remove(leftover)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"sha256": digest, "item_count": len(product_ids), "path": str(target)}


def load_din_embedding_sidecar(
    path: str,
) -> Tuple[torch.Tensor, Dict[str, int], Dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    product_ids = [str(value) for value in data["product_ids"]]
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    if (
        not product_ids
        or product_ids[0] != ""
        or embeddings.shape != (len(product_ids), 128)
    ):
        raise ValueError("invalid DIN embedding sidecar shape or padding row")
    if not np.allclose(embeddings[0], 0.0) or not np.isfinite(embeddings).all():
        raise ValueError("invalid DIN embedding sidecar values")
    mapping = {
        product_id: index for index, product_id in enumerate(product_ids) if product_id
    }
    return (
        torch.from_numpy(embeddings.copy()),
        mapping,
        {
            "two_tower_model_version": str(data["two_tower_model_version"][0]),
            "contract_version": str(data["contract_version"][0]),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        },
    )
