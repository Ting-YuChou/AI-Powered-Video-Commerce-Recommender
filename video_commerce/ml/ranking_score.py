"""Versioned canonical ranking score shared by training and serving."""

from __future__ import annotations

import numpy as np
import torch


SCORE_POLICY_VERSION = "business-value-v1"


def canonical_score_numpy(
    *,
    ctcvr,
    predicted_value,
    raw_ranking_score,
    business_score_enabled: bool,
) -> np.ndarray:
    raw = np.asarray(raw_ranking_score)
    if not business_score_enabled:
        return raw
    probability = np.clip(np.asarray(ctcvr), 0.0, 1.0)
    return (probability * np.asarray(predicted_value)).astype(np.float32)


def canonical_score_torch(
    *,
    ctcvr: torch.Tensor,
    predicted_value: torch.Tensor,
    raw_ranking_score: torch.Tensor,
    business_score_enabled: bool,
) -> torch.Tensor:
    if not business_score_enabled:
        return raw_ranking_score
    return ctcvr.clamp(0.0, 1.0) * predicted_value
