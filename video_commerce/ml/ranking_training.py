"""Typed ranking examples and versioned training-label construction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from video_commerce.ml.ranking_features import FeatureBundle


RANKING_LABEL_DEFINITION_VERSION = "ranking_labels_v1"
_ATTRIBUTION_ACTIONS = {"view", "click", "add_to_cart", "purchase"}


@dataclass(frozen=True)
class AttributionFacts:
    attributed_action: str
    attributed_click: bool
    attributed_purchase: bool
    attributed_value: Optional[float] = None
    attributed_value_source: Optional[str] = None

    def __post_init__(self) -> None:
        action = str(self.attributed_action or "").strip().lower()
        object.__setattr__(self, "attributed_action", action)
        if action not in _ATTRIBUTION_ACTIONS:
            raise ValueError(f"unsupported attributed action: {action}")
        if self.attributed_purchase and not self.attributed_click:
            raise ValueError("purchase attribution must also be click-attributed")
        if action == "purchase" and not self.attributed_purchase:
            raise ValueError("purchase action requires purchase attribution")
        if action in {"click", "add_to_cart"} and not self.attributed_click:
            raise ValueError(f"{action} action requires click attribution")
        if self.attributed_value is not None:
            value = float(self.attributed_value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("attributed value must be finite and non-negative")
            if not self.attributed_purchase:
                raise ValueError("attributed value requires purchase attribution")
            if not str(self.attributed_value_source or "").strip():
                raise ValueError("attributed value requires a source")
            object.__setattr__(self, "attributed_value", value)


@dataclass(frozen=True)
class TrainingLabels:
    label_definition_version: str
    ctr: float
    cvr: float
    ctcvr: float
    cvr_mask: float
    business_value: float
    value_mask: float
    relevance: float


@dataclass(frozen=True)
class RankingTrainingExample:
    observation_id: str
    impression_id: str
    bundle: FeatureBundle
    attribution: AttributionFacts
    is_slate_sample: bool = True

    def __post_init__(self) -> None:
        if not str(self.observation_id or "").strip():
            raise ValueError("ranking training example requires observation_id")
        if not str(self.impression_id or "").strip():
            raise ValueError("PIT ranking training example requires impression_id")


class RankingLabelBuilder:
    """Produce model targets exclusively from finalized attribution facts."""

    version = RANKING_LABEL_DEFINITION_VERSION

    def build(self, facts: AttributionFacts) -> TrainingLabels:
        clicked = bool(facts.attributed_click)
        purchased = bool(facts.attributed_purchase)
        has_value = purchased and facts.attributed_value is not None
        value = float(facts.attributed_value) if has_value else 0.0
        relevance = {
            "view": 1.0,
            "click": 2.0,
            "add_to_cart": 3.0,
            "purchase": 4.0 + math.log1p(value),
        }[facts.attributed_action]
        labels = TrainingLabels(
            label_definition_version=self.version,
            ctr=1.0 if clicked else 0.0,
            cvr=1.0 if purchased else 0.0,
            ctcvr=1.0 if purchased else 0.0,
            cvr_mask=1.0 if clicked else 0.0,
            business_value=value,
            value_mask=1.0 if has_value else 0.0,
            relevance=relevance,
        )
        values = (
            labels.ctr,
            labels.cvr,
            labels.ctcvr,
            labels.cvr_mask,
            labels.business_value,
            labels.value_mask,
            labels.relevance,
        )
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise ValueError("ranking labels contain NaN or infinity")
        return labels


class TrainingTensorBuilder:
    """Assemble typed examples and versioned labels into model tensors."""

    def __init__(
        self,
        assembler,
        *,
        device: Optional[torch.device] = None,
        label_builder: Optional[RankingLabelBuilder] = None,
        value_bucket_id: Optional[Callable[[dict[str, Any]], int]] = None,
        fit_value_transform: Optional[Callable[[list[dict[str, Any]]], None]] = None,
        transform_value: Optional[Callable[[float, int], float]] = None,
    ) -> None:
        self.assembler = assembler
        self.device = device or torch.device("cpu")
        self.label_builder = label_builder or RankingLabelBuilder()
        self.value_bucket_id = value_bucket_id or (lambda _metadata: 0)
        self.fit_value_transform = fit_value_transform or (lambda _records: None)
        self.transform_value = transform_value or (lambda value, _bucket: value)

    def build(
        self, examples: Sequence[RankingTrainingExample]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not examples:
            return torch.empty(0, device=self.device), {}
        features = self.assembler.build_many([example.bundle for example in examples])
        built_labels = [
            self.label_builder.build(example.attribution) for example in examples
        ]
        bucket_ids = [
            int(self.value_bucket_id(dict(example.bundle.product_metadata)))
            for example in examples
        ]
        value_records = [
            {"business_value": label.business_value, "value_bucket": bucket_id}
            for label, bucket_id in zip(built_labels, bucket_ids)
            if label.value_mask > 0.0
        ]
        self.fit_value_transform(value_records)
        normalized_values = [
            self.transform_value(label.business_value, bucket_id)
            if label.value_mask > 0.0
            else 0.0
            for label, bucket_id in zip(built_labels, bucket_ids)
        ]
        group_mapping: dict[str, int] = {}
        group_ids = []
        for example in examples:
            group_id = group_mapping.setdefault(
                example.impression_id, len(group_mapping)
            )
            group_ids.append(group_id)

        def column(values: Sequence[float]) -> torch.Tensor:
            return torch.tensor(
                values, dtype=torch.float32, device=self.device
            ).unsqueeze(1)

        labels: Dict[str, torch.Tensor] = {
            "ctr": column([label.ctr for label in built_labels]),
            "cvr": column([label.cvr for label in built_labels]),
            "ctcvr": column([label.ctcvr for label in built_labels]),
            "cvr_mask": column([label.cvr_mask for label in built_labels]),
            "value": column(normalized_values),
            "business_value": column([label.business_value for label in built_labels]),
            "value_mask": column([label.value_mask for label in built_labels]),
            "value_bucket": torch.tensor(
                bucket_ids, dtype=torch.long, device=self.device
            ),
            "ranking_relevance": column([label.relevance for label in built_labels]),
            "pairwise_group": torch.tensor(
                group_ids, dtype=torch.long, device=self.device
            ),
            "ltr_group": torch.tensor(group_ids, dtype=torch.long, device=self.device),
            "ltr_is_slate_sample": torch.tensor(
                [example.is_slate_sample for example in examples],
                dtype=torch.bool,
                device=self.device,
            ),
        }
        labels["gmv"] = labels["value"]
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        if not torch.isfinite(feature_tensor).all() or any(
            not torch.isfinite(value).all()
            for value in labels.values()
            if value.is_floating_point()
        ):
            raise ValueError("training tensors contain NaN or infinity")
        return feature_tensor, labels
