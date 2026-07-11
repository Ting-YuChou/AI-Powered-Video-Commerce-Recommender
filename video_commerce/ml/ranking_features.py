"""Shared ranking feature contract used by online serving and offline training."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import threading
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from video_commerce.common.feature_history_contracts import (
    RANKING_LTR_FEATURE_DEFINITION_VERSION,
)
from video_commerce.common.models import CandidateProduct, UserFeatures

if TYPE_CHECKING:
    from video_commerce.ml.ranking import FeatureExtractor


@dataclass(frozen=True)
class FeatureBundle:
    """Named ranking inputs captured at one observation time."""

    as_of_ts: float
    feature_definition_version: str
    user_features: UserFeatures
    product_metadata: Mapping[str, Any]
    context: Mapping[str, Any]
    candidate: CandidateProduct

    def __post_init__(self) -> None:
        if self.feature_definition_version != RANKING_LTR_FEATURE_DEFINITION_VERSION:
            raise ValueError(
                "unsupported ranking feature definition version: "
                f"{self.feature_definition_version}"
            )
        if not np.isfinite(float(self.as_of_ts)) or float(self.as_of_ts) < 0:
            raise ValueError("FeatureBundle as_of_ts must be a finite Unix timestamp")


class RankingFeatureAssembler:
    """Build one deterministic ranking vector from a named feature bundle."""

    version = "ranking_feature_assembler_v1"

    def __init__(
        self, extractor: "FeatureExtractor", *, product_feature_cache_size: int = 0
    ) -> None:
        self.extractor = extractor
        self._product_feature_cache_size = max(0, int(product_feature_cache_size))
        self._product_feature_cache: OrderedDict[
            tuple[str, tuple[Any, ...]], tuple[np.ndarray, float]
        ] = OrderedDict()
        self._product_feature_cache_lock = threading.RLock()

    def build(self, bundle: FeatureBundle) -> np.ndarray:
        as_of_ts = float(bundle.as_of_ts)
        return self._build_with_shared_segments(
            bundle,
            self.extractor.extract_user_features(bundle.user_features, as_of_ts),
            self.extractor.extract_context_features(dict(bundle.context), as_of_ts),
        )

    def _build_with_shared_segments(
        self,
        bundle: FeatureBundle,
        user_features: np.ndarray,
        context_features: np.ndarray,
    ) -> np.ndarray:
        as_of_ts = float(bundle.as_of_ts)
        metadata = dict(bundle.product_metadata)
        context = dict(bundle.context)
        vector = np.concatenate(
            [
                user_features,
                self._product_features(bundle.candidate.product_id, metadata, as_of_ts),
                context_features,
                self.extractor.extract_candidate_features(bundle.candidate),
                self.extractor.extract_history_embedding_features(
                    context, bundle.candidate
                ),
                self.extractor.extract_realtime_window_features(
                    bundle.user_features,
                    metadata,
                    context,
                    bundle.candidate,
                ),
            ]
        )
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.extractor.total_feature_dim,):
            raise ValueError(
                "ranking feature vector shape mismatch: "
                f"expected {(self.extractor.total_feature_dim,)}, got {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("ranking feature vector contains NaN or infinity")
        return vector

    def build_many(self, bundles: Sequence[FeatureBundle]) -> np.ndarray:
        if not bundles:
            return np.empty((0, self.extractor.total_feature_dim), dtype=np.float32)
        user_segments: dict[tuple[int, float], np.ndarray] = {}
        context_segments: dict[tuple[int, float], np.ndarray] = {}
        vectors = []
        for bundle in bundles:
            as_of_ts = float(bundle.as_of_ts)
            user_key = (id(bundle.user_features), as_of_ts)
            context_key = (id(bundle.context), as_of_ts)
            if user_key not in user_segments:
                user_segments[user_key] = self.extractor.extract_user_features(
                    bundle.user_features, as_of_ts
                )
            if context_key not in context_segments:
                context_segments[context_key] = self.extractor.extract_context_features(
                    dict(bundle.context), as_of_ts
                )
            vectors.append(
                self._build_with_shared_segments(
                    bundle,
                    user_segments[user_key],
                    context_segments[context_key],
                )
            )
        return np.vstack(vectors).astype(np.float32, copy=False)

    @staticmethod
    def _metadata_fingerprint(metadata: Mapping[str, Any]) -> tuple[Any, ...]:
        tags = metadata.get("tags", [])
        if isinstance(tags, (list, tuple, set)):
            normalized_tags = tuple(str(tag) for tag in tags)
        else:
            normalized_tags = (str(tags),)
        created_at = (
            "__fallback_now__"
            if metadata.get("_ranking_fallback_metadata") is True
            else metadata.get("created_at")
        )
        return (
            metadata.get("price", 1.0),
            metadata.get("rating", 3.0),
            metadata.get("num_reviews", 1),
            bool(metadata.get("in_stock", True)),
            created_at,
            normalized_tags,
            metadata.get("brand", ""),
        )

    def _product_features(
        self, product_id: str, metadata: Mapping[str, Any], as_of_ts: float
    ) -> np.ndarray:
        metadata_dict = dict(metadata)
        cache_key = (str(product_id), self._metadata_fingerprint(metadata_dict))
        with self._product_feature_cache_lock:
            cached = self._product_feature_cache.get(cache_key)
            if cached is not None:
                static_features, created_at = cached
                self._product_feature_cache.move_to_end(cache_key)
            else:
                static_features = None
        if static_features is None:
            if metadata_dict.get("_ranking_fallback_metadata") is True:
                metadata_dict["created_at"] = as_of_ts
            (
                static_features,
                created_at,
            ) = self.extractor.extract_static_product_features(metadata_dict)
            with self._product_feature_cache_lock:
                if self._product_feature_cache_size > 0:
                    self._product_feature_cache[cache_key] = (
                        static_features,
                        created_at,
                    )
                    if (
                        len(self._product_feature_cache)
                        > self._product_feature_cache_size
                    ):
                        self._product_feature_cache.popitem(last=False)
        elif metadata_dict.get("_ranking_fallback_metadata") is True:
            created_at = as_of_ts
        product_features = static_features.copy()
        product_features[4] = (as_of_ts - created_at) / 86400
        return product_features
