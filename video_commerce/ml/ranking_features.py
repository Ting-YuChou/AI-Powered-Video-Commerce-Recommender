"""Shared ranking feature contract used by online serving and offline training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from video_commerce.common.models import CandidateProduct, UserFeatures

if TYPE_CHECKING:
    from video_commerce.ml.ranking import FeatureExtractor


RANKING_LTR_FEATURE_DEFINITION_VERSION = "ranking_ltr_v1"


@dataclass(frozen=True)
class FeatureBundle:
    """Named ranking inputs captured at one observation time."""

    user_features: UserFeatures
    product_metadata: Mapping[str, Any]
    context: Mapping[str, Any]
    candidate: CandidateProduct


class RankingFeatureAssembler:
    """Build one deterministic ranking vector from a named feature bundle."""

    def __init__(self, extractor: "FeatureExtractor") -> None:
        self.extractor = extractor

    def build(self, bundle: FeatureBundle, *, as_of_ts: float) -> np.ndarray:
        return self.extractor.create_ranking_features(
            bundle.user_features,
            dict(bundle.product_metadata),
            dict(bundle.context),
            bundle.candidate,
            as_of_ts=as_of_ts,
        )
