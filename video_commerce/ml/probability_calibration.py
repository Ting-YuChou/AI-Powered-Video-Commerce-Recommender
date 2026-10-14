"""Versioned probability calibration shared by training and serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


CALIBRATION_POLICY_VERSION = "temperature_v1"
_EPSILON = 1e-7


class StreamingTemperatureFitter:
    """Fixed-size likelihood accumulator; memory does not grow with rows."""

    def __init__(self):
        self.temperatures = np.exp(np.linspace(np.log(0.05), np.log(20.0), 1024))
        self.losses = np.zeros(1024, dtype=np.float64)
        self.row_count = 0
        self.positives = 0

    def add(self, probabilities, labels):
        values = np.clip(_as_vector(probabilities), _EPSILON, 1.0 - _EPSILON)
        targets = _as_vector(labels)
        if values.shape != targets.shape or not np.isin(targets, [0.0, 1.0]).all():
            raise ValueError("calibration labels must be aligned binary values")
        logits = np.log(values / (1.0 - values))
        for index, temperature in enumerate(self.temperatures):
            scaled = logits / temperature
            self.losses[index] += np.sum(np.logaddexp(0.0, scaled) - targets * scaled)
        self.row_count += len(targets)
        self.positives += int(targets.sum())

    @property
    def has_both_classes(self):
        return 0 < self.positives < self.row_count

    def fit(self):
        if not self.has_both_classes:
            return 1.0
        return float(self.temperatures[int(np.argmin(self.losses))])


def _as_vector(values: Iterable[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("calibration values must be non-empty and finite")
    return vector


def _apply_temperature_numpy(
    probabilities: Iterable[float], temperature: float
) -> np.ndarray:
    values = np.clip(_as_vector(probabilities), _EPSILON, 1.0 - _EPSILON)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("calibration temperature must be finite and positive")
    logits = np.log(values / (1.0 - values)) / float(temperature)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def fit_temperature(probabilities: Iterable[float], labels: Iterable[float]) -> float:
    """Fit one deterministic scalar by minimizing binary negative log loss."""
    values = np.clip(_as_vector(probabilities), _EPSILON, 1.0 - _EPSILON)
    targets = _as_vector(labels)
    if values.shape != targets.shape or not np.isin(targets, [0.0, 1.0]).all():
        raise ValueError("calibration labels must be aligned binary values")
    fitter = StreamingTemperatureFitter()
    fitter.add(values, targets)
    return fitter.fit()


def reliability_table(
    probabilities: Iterable[float], labels: Iterable[float], *, bins: int = 15
) -> list[dict[str, float | int]]:
    """Build deterministic equal-frequency reliability bins."""
    values = _as_vector(probabilities)
    targets = _as_vector(labels)
    if values.shape != targets.shape:
        raise ValueError("probabilities and labels must have the same length")
    if bins < 1:
        raise ValueError("calibration bins must be positive")
    order = np.argsort(values, kind="mergesort")
    groups = np.array_split(order, min(int(bins), len(order)))
    table = []
    for index, group in enumerate(groups):
        if group.size == 0:
            continue
        predicted = float(np.mean(values[group]))
        observed = float(np.mean(targets[group]))
        table.append(
            {
                "bin": index,
                "count": int(group.size),
                "predicted": predicted,
                "observed": observed,
                "absolute_gap": abs(predicted - observed),
            }
        )
    return table


def expected_calibration_error(
    probabilities: Iterable[float], labels: Iterable[float], *, bins: int = 15
) -> float:
    table = reliability_table(probabilities, labels, bins=bins)
    total = sum(int(row["count"]) for row in table)
    return float(
        sum(float(row["absolute_gap"]) * int(row["count"]) for row in table)
        / max(total, 1)
    )


@dataclass(frozen=True)
class TemperatureCalibration:
    ctr_temperature: float = 1.0
    ctcvr_temperature: float = 1.0
    policy_version: str = CALIBRATION_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.policy_version != CALIBRATION_POLICY_VERSION:
            raise ValueError("unsupported probability calibration policy")
        for value in (self.ctr_temperature, self.ctcvr_temperature):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("calibration temperatures must be finite and positive")

    def _temperature(self, head: str) -> float:
        if head == "ctr":
            return float(self.ctr_temperature)
        if head == "ctcvr":
            return float(self.ctcvr_temperature)
        raise ValueError("calibration head must be ctr or ctcvr")

    def apply_numpy(self, probabilities: Iterable[float], *, head: str) -> np.ndarray:
        return _apply_temperature_numpy(probabilities, self._temperature(head))

    def apply_torch(self, probabilities: torch.Tensor, *, head: str) -> torch.Tensor:
        values = probabilities.clamp(_EPSILON, 1.0 - _EPSILON)
        logits = torch.logit(values) / self._temperature(head)
        return torch.sigmoid(logits)

    def to_manifest(self) -> dict[str, float | str]:
        return {
            "policy_version": self.policy_version,
            "ctr_temperature": float(self.ctr_temperature),
            "ctcvr_temperature": float(self.ctcvr_temperature),
        }

    @classmethod
    def from_manifest(cls, manifest: dict | None) -> "TemperatureCalibration":
        payload = dict(manifest or {})
        if not payload:
            return cls()
        return cls(
            ctr_temperature=float(payload.get("ctr_temperature", 1.0)),
            ctcvr_temperature=float(payload.get("ctcvr_temperature", 1.0)),
            policy_version=str(payload.get("policy_version") or ""),
        )
