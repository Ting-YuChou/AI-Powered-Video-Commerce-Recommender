"""
AI-Powered Video Commerce Recommender - Neural Ranking Model
============================================================

This module implements a neural ranking model that takes candidates from the
recommendation engine and ranks them based on predicted user engagement,
conversion probability, and business value (GMV optimization).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import Dict, List, Any, Optional, Sequence, Tuple
import json
import threading
import time
from pathlib import Path
import pickle
import os
import tempfile
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score, roc_auc_score

# Local imports
from video_commerce.common.models import (
    CandidateProduct,
    ProductRecommendation,
    UserFeatures,
    RankingFeatures,
)
from video_commerce.common.config import RankingConfig
from video_commerce.ml.dcn import (
    DeepAndCrossNetwork,
    LowRankDeepAndCrossNetwork,
    RANKING_ARCHITECTURES,
    normalize_architecture,
)
from video_commerce.ml.ranking_history import (
    RANKING_HISTORY_ACTIONS,
    RANKING_HISTORY_CONTEXT_KEY,
    build_training_history_contexts,
    extract_ranking_history_feature_vector,
    ranking_history_config_from_settings,
)
from video_commerce.common.feature_history_contracts import (
    RANKING_LTR_FEATURE_DEFINITION_VERSION,
    RANKING_LTR_DIN_FEATURE_DEFINITION_VERSION,
)
from video_commerce.ml.ranking_features import FeatureBundle, RankingFeatureAssembler
from video_commerce.ml.ranking_training import (
    RANKING_LABEL_DEFINITION_VERSION,
    RankingTrainingExample,
    TrainingTensorBuilder,
)
from video_commerce.ml.temporal_multimodal import (
    CandidateMultimodalAttention,
    TemporalSequenceEncoder,
    TrimodalCandidateAttention,
    unpack_temporal_multimodal_context,
)
from video_commerce.ml.candidate_embedding_sidecar import (
    CandidateEmbeddingSidecar,
    write_candidate_embedding_sidecar,
)
from video_commerce.ml.din import (
    DIN_SEQUENCE_CONTEXT_KEY,
    DeepInterestNetwork,
    build_din_behavior_sequences,
    build_din_batch_inputs,
    load_din_embedding_sidecar,
    parse_din_behavior_sequences,
)

logger = logging.getLogger(__name__)

RANKING_FEATURE_SCHEMA_VERSION = "ranking_v3_00_temporal_multimodal"
RANKING_TRIMODAL_FEATURE_SCHEMA_VERSION = "ranking_v4_00_temporal_trimodal"
RANKING_DIN_FEATURE_SCHEMA_VERSION = "ranking_v3_din"
RANKING_TRAINING_DATA_SOURCE = "interaction_events_online_equivalent_features"
RANKING_OBJECTIVE_VERSION = "business_v1"
LEGACY_RANKING_OBJECTIVE_VERSION = "legacy_multi_objective"


class RankingTrainingCancelled(RuntimeError):
    """Raised by the synchronous trainer after cooperative cancellation."""


class RankingFeatureMatrix(np.ndarray):
    """Dense candidate matrix carrying non-duplicated structured DIN tensors."""

    din_inputs: Any = None

    def __new__(cls, values, *, din_inputs=None):
        instance = np.asarray(values, dtype=np.float32).view(cls)
        instance.din_inputs = din_inputs
        return instance

    def __array_finalize__(self, source):
        self.din_inputs = getattr(source, "din_inputs", None)


def _stable_hash_bucket(value: Any, buckets: int = 100) -> int:
    normalized = str(value or "").encode("utf-8")
    digest = hashlib.sha256(normalized).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % buckets


def _candidate_get(candidate: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(field_name, default)
    return getattr(candidate, field_name, default)


try:
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False
except Exception:
    pass


class MultiObjectiveRankingModel(nn.Module):
    """
    Multi-objective neural ranking model that predicts:
    - Click-through rate (CTR)
    - Conversion rate (CVR)
    - Gross Merchandise Value (GMV)
    - Overall ranking score
    """

    def __init__(
        self,
        input_dim: int,
        config: RankingConfig,
        *,
        architecture: Optional[str] = None,
        hidden_dims: Optional[List[int]] = None,
        cross_layers: Optional[int] = None,
        low_rank_dim: Optional[int] = None,
        din_item_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.din = (
            DeepInterestNetwork(din_item_embeddings)
            if din_item_embeddings is not None
            else None
        )
        model_input_dim = input_dim + (140 if self.din is not None else 0)
        self.architecture = normalize_architecture(
            architecture or getattr(config, "architecture", "dcn"),
            supported=RANKING_ARCHITECTURES,
        )
        self.cross_layers = max(
            0,
            int(
                cross_layers
                if cross_layers is not None
                else getattr(config, "cross_layers", 3)
            ),
        )
        self.low_rank_dim = max(
            1,
            int(
                low_rank_dim
                if low_rank_dim is not None
                else getattr(config, "low_rank_dim", 8)
            ),
        )

        # Shared bottom layers
        hidden_dims = list(config.hidden_dims if hidden_dims is None else hidden_dims)
        if not hidden_dims:
            hidden_dims = [64]
        self.hidden_dims = hidden_dims

        if self.architecture == "mlp":
            layers = []
            prev_dim = model_input_dim

            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(config.dropout_rate),
                    ]
                )
                prev_dim = hidden_dim

            self.shared_layers = nn.Sequential(*layers)
        elif self.architecture == "dcn":
            self.shared_layers = DeepAndCrossNetwork(
                model_input_dim,
                hidden_dims,
                hidden_dims[-1],
                cross_layers=self.cross_layers,
                dropout=config.dropout_rate,
                use_batch_norm=False,
            )
        else:
            self.shared_layers = LowRankDeepAndCrossNetwork(
                model_input_dim,
                hidden_dims,
                hidden_dims[-1],
                cross_layers=self.cross_layers,
                low_rank_dim=self.low_rank_dim,
                dropout=config.dropout_rate,
                use_batch_norm=False,
            )

        # Task-specific towers
        tower_input_dim = hidden_dims[-1]

        # CTR prediction tower
        self.ctr_tower = nn.Sequential(
            nn.Linear(tower_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        # CVR prediction tower
        self.cvr_tower = nn.Sequential(
            nn.Linear(tower_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        # GMV prediction tower
        self.gmv_tower = nn.Sequential(
            nn.Linear(tower_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        # Final ranking tower
        self.ranking_tower = nn.Sequential(
            nn.Linear(tower_input_dim + 3, 32),  # +3 for CTR, CVR, GMV predictions
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _collapse_scalar_head(x: torch.Tensor) -> torch.Tensor:
        """Collapse a 2-unit head into one scalar while avoiding Linear(..., 1)."""
        return x[:, 1:2] - x[:, 0:1]

    def forward(
        self,
        x,
        *,
        candidate_indices=None,
        history_indices=None,
        history_recency=None,
        history_mask=None,
        summary_features=None,
    ):
        """Forward pass through the model."""
        if self.din is not None:
            if any(
                value is None
                for value in (
                    candidate_indices,
                    history_indices,
                    history_recency,
                    history_mask,
                    summary_features,
                )
            ):
                raise ValueError("DIN ranking requires structured sequence tensors")
            interest = self.din(
                candidate_indices,
                history_indices,
                history_recency,
                history_mask,
            )
            if summary_features.shape != (x.shape[0], 12):
                raise ValueError("DIN summary_features must have shape [batch, 12]")
            x = torch.cat([x, interest, summary_features], dim=1)
        # Shared representation
        shared_features = self.shared_layers(x)

        # Task-specific predictions
        ctr_pred = torch.sigmoid(
            self._collapse_scalar_head(self.ctr_tower(shared_features))
        )
        cvr_pred = torch.sigmoid(
            self._collapse_scalar_head(self.cvr_tower(shared_features))
        )
        ctcvr_pred = ctr_pred * cvr_pred
        gmv_pred = self._collapse_scalar_head(self.gmv_tower(shared_features))

        # Combine for final ranking
        combined_features = torch.cat(
            [shared_features, ctr_pred, cvr_pred, gmv_pred], dim=1
        )
        ranking_score = self._collapse_scalar_head(
            self.ranking_tower(combined_features)
        )

        return {
            "ctr": ctr_pred,
            "cvr": cvr_pred,
            "ctcvr": ctcvr_pred,
            "gmv": gmv_pred,
            "ranking_score": ranking_score,
        }


class TemporalMultimodalRankingModel(nn.Module):
    """Candidate-conditioned temporal/OCR ranker requiring a new checkpoint."""

    def __init__(
        self,
        base_input_dim: int,
        config: RankingConfig,
        *,
        visual_token_dim: int = 128,
        clip_dim: int = 512,
        two_tower_dim: int = 128,
        multimodal_dim: int = 128,
    ) -> None:
        super().__init__()
        self.multimodal = CandidateMultimodalAttention(
            visual_input_dim=visual_token_dim,
            text_input_dim=clip_dim,
            candidate_embedding_dim=clip_dim,
            two_tower_dim=two_tower_dim,
            model_dim=multimodal_dim,
        )
        self.ranker = MultiObjectiveRankingModel(
            base_input_dim + multimodal_dim,
            config,
        )

    def forward(self, base_features: torch.Tensor, **multimodal_inputs):
        attention = self.multimodal(**multimodal_inputs)
        predictions = self.ranker(torch.cat([base_features, attention.fused], dim=-1))
        predictions["visual_attention"] = attention.visual_attention
        predictions["text_attention"] = attention.text_attention
        return predictions


class TemporalTrimodalRankingModel(nn.Module):
    """Joint visual/OCR/ASR temporal ranker with a near-zero warm-start residual."""

    def __init__(
        self,
        base_input_dim: int,
        config: RankingConfig,
        *,
        model_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        architecture: Optional[str] = None,
        hidden_dims: Optional[List[int]] = None,
        cross_layers: Optional[int] = None,
        low_rank_dim: Optional[int] = None,
        din_item_embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.base_input_dim = int(base_input_dim)
        encoder_options = {
            "model_dim": int(model_dim),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
            "dropout": float(dropout),
        }
        self.visual_encoder = TemporalSequenceEncoder(input_dim=512, **encoder_options)
        self.ocr_encoder = TemporalSequenceEncoder(input_dim=384, **encoder_options)
        self.asr_encoder = TemporalSequenceEncoder(input_dim=384, **encoder_options)
        self.candidate_attention = TrimodalCandidateAttention(
            model_dim=int(model_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
        )
        self.residual_projection = nn.Linear(int(model_dim), self.base_input_dim)
        self.residual_gate_logit = nn.Parameter(torch.tensor(-4.0))
        self.ranker = MultiObjectiveRankingModel(
            self.base_input_dim,
            config,
            architecture=architecture,
            hidden_dims=hidden_dims,
            cross_layers=cross_layers,
            low_rank_dim=low_rank_dim,
            din_item_embeddings=din_item_embeddings,
        )
        # Preserve the attributes used by checkpoint reload and architecture checks.
        self.architecture = self.ranker.architecture
        self.hidden_dims = self.ranker.hidden_dims
        self.cross_layers = self.ranker.cross_layers
        self.low_rank_dim = self.ranker.low_rank_dim

    def forward(
        self,
        base_features: torch.Tensor,
        *,
        visual_embeddings: Optional[torch.Tensor] = None,
        visual_starts: Optional[torch.Tensor] = None,
        visual_ends: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        ocr_embeddings: Optional[torch.Tensor] = None,
        ocr_starts: Optional[torch.Tensor] = None,
        ocr_ends: Optional[torch.Tensor] = None,
        ocr_mask: Optional[torch.Tensor] = None,
        asr_embeddings: Optional[torch.Tensor] = None,
        asr_starts: Optional[torch.Tensor] = None,
        asr_ends: Optional[torch.Tensor] = None,
        asr_mask: Optional[torch.Tensor] = None,
        candidate_image: Optional[torch.Tensor] = None,
        candidate_text: Optional[torch.Tensor] = None,
        candidate_two_tower: Optional[torch.Tensor] = None,
        candidate_presence: Optional[torch.Tensor] = None,
        candidate_indices: Optional[torch.Tensor] = None,
        history_indices: Optional[torch.Tensor] = None,
        history_recency: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
        summary_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        ranker_kwargs = {
            "candidate_indices": candidate_indices,
            "history_indices": history_indices,
            "history_recency": history_recency,
            "history_mask": history_mask,
            "summary_features": summary_features,
        }
        ranker_kwargs = {
            key: value for key, value in ranker_kwargs.items() if value is not None
        }
        if visual_embeddings is None:
            return self.ranker(base_features, **ranker_kwargs)
        visual = self.visual_encoder(
            visual_embeddings,
            visual_starts,
            visual_mask,
            end_timestamps_seconds=visual_ends,
        )
        ocr = self.ocr_encoder(
            ocr_embeddings,
            ocr_starts,
            ocr_mask,
            end_timestamps_seconds=ocr_ends,
        )
        asr = self.asr_encoder(
            asr_embeddings,
            asr_starts,
            asr_mask,
            end_timestamps_seconds=asr_ends,
        )
        attended = self.candidate_attention(
            visual_tokens=visual.tokens,
            visual_pooled=visual.pooled_embedding,
            visual_mask=visual_mask,
            ocr_tokens=ocr.tokens,
            ocr_pooled=ocr.pooled_embedding,
            ocr_mask=ocr_mask,
            asr_tokens=asr.tokens,
            asr_pooled=asr.pooled_embedding,
            asr_mask=asr_mask,
            candidate_image=candidate_image,
            candidate_text=candidate_text,
            candidate_two_tower=candidate_two_tower,
            candidate_presence=candidate_presence,
        )
        residual = self.residual_projection(attended.fused)
        augmented = base_features + torch.sigmoid(self.residual_gate_logit) * residual
        predictions = self.ranker(augmented, **ranker_kwargs)
        predictions.update(
            {
                "visual_attention": attended.visual_attention,
                "ocr_attention": attended.ocr_attention,
                "asr_attention": attended.asr_attention,
                "modality_gate": attended.modality_gate,
            }
        )
        return predictions


class FeatureExtractor:
    """
    Feature extractor that converts raw data into model-ready features.
    """

    WINDOW_FEATURE_NAMES = ("5m", "1h", "24h")
    WINDOW_FEATURE_METRICS = (
        "views",
        "clicks",
        "add_to_cart",
        "purchases",
        "click_through_rate",
        "conversion_rate",
    )
    WINDOW_FEATURE_ENTITIES = ("user", "product", "category")

    def __init__(
        self,
        enable_realtime_window_features: bool = False,
        enable_history_embeddings: bool = False,
        history_embedding_dim: int = 128,
    ):
        self.user_scaler = StandardScaler()
        self.context_scaler = StandardScaler()
        self.is_fitted = False
        self.enable_realtime_window_features = enable_realtime_window_features
        self.enable_history_embeddings = enable_history_embeddings
        self.history_embedding_dim = max(1, int(history_embedding_dim or 128))

        # Feature dimensions
        self.user_feature_dim = 10
        self.product_feature_dim = 8
        self.context_feature_dim = 6
        self.candidate_feature_dim = 4
        self.history_embedding_feature_dim = (
            len(RANKING_HISTORY_ACTIONS) * (self.history_embedding_dim + 5)
            if self.enable_history_embeddings
            else 0
        )
        self.realtime_window_feature_dim = (
            len(self.WINDOW_FEATURE_ENTITIES)
            * len(self.WINDOW_FEATURE_NAMES)
            * len(self.WINDOW_FEATURE_METRICS)
            if self.enable_realtime_window_features
            else 0
        )

        self.total_feature_dim = (
            self.user_feature_dim
            + self.product_feature_dim
            + self.context_feature_dim
            + self.candidate_feature_dim
            + self.history_embedding_feature_dim
            + self.realtime_window_feature_dim
        )

    def extract_user_features(
        self,
        user_features: UserFeatures,
        current_time: Optional[float] = None,
    ) -> np.ndarray:
        """Extract numerical features from user profile."""
        current_time = time.time() if current_time is None else float(current_time)
        features = np.array(
            [
                user_features.total_interactions / 1000,  # Normalize
                user_features.avg_session_length / 3600,  # Hours
                user_features.price_sensitivity,
                user_features.click_through_rate,
                user_features.conversion_rate,
                len(user_features.preferred_categories) / 10,  # Normalize
                (current_time - user_features.last_active) / 86400,  # Days since active
                1.0 if user_features.total_interactions > 100 else 0.0,  # Heavy user
                1.0 if user_features.conversion_rate > 0.05 else 0.0,  # High converter
                1.0
                if user_features.click_through_rate > 0.1
                else 0.0,  # High engagement
            ],
            dtype=np.float32,
        )

        return features

    def extract_product_features(
        self,
        product_metadata: Dict[str, Any],
        current_time: Optional[float] = None,
    ) -> np.ndarray:
        """Extract features from product metadata."""
        current_time = time.time() if current_time is None else float(current_time)
        features = np.array(
            [
                np.log1p(product_metadata.get("price", 1.0)),  # Log price
                product_metadata.get("rating", 3.0) / 5.0,  # Normalized rating
                np.log1p(product_metadata.get("num_reviews", 1)),  # Log review count
                1.0 if product_metadata.get("in_stock", True) else 0.0,  # Stock status
                (current_time - product_metadata.get("created_at", current_time))
                / 86400,  # Age in days
                len(product_metadata.get("tags", [])) / 10,  # Tag count normalized
                1.0
                if product_metadata.get("price", 0) > 100
                else 0.0,  # Premium product
                _stable_hash_bucket(product_metadata.get("brand", "")) / 100,
            ],
            dtype=np.float32,
        )

        return features

    def extract_static_product_features(
        self,
        product_metadata: Dict[str, Any],
    ) -> Tuple[np.ndarray, float]:
        """Extract product features whose values do not depend on request time."""
        created_at = float(product_metadata.get("created_at") or time.time())
        features = np.array(
            [
                np.log1p(product_metadata.get("price", 1.0)),
                product_metadata.get("rating", 3.0) / 5.0,
                np.log1p(product_metadata.get("num_reviews", 1)),
                1.0 if product_metadata.get("in_stock", True) else 0.0,
                0.0,  # Filled with age-in-days at request time.
                len(product_metadata.get("tags", [])) / 10,
                1.0 if product_metadata.get("price", 0) > 100 else 0.0,
                _stable_hash_bucket(product_metadata.get("brand", "")) / 100,
            ],
            dtype=np.float32,
        )
        return features, created_at

    def extract_context_features(
        self,
        context: Dict[str, Any],
        current_time: Optional[float] = None,
    ) -> np.ndarray:
        """Extract features from request context."""
        current_time = time.time() if current_time is None else float(current_time)
        dt = time.localtime(current_time)

        features = np.array(
            [
                dt.tm_hour / 24.0,  # Hour of day normalized
                dt.tm_wday / 6.0,  # Day of week normalized
                1.0 if dt.tm_wday >= 5 else 0.0,  # Weekend flag
                context.get("session_position", 1) / 20.0,  # Position in session
                1.0 if context.get("device") == "mobile" else 0.0,  # Mobile device
                context.get("time_on_page", 0)
                / 300.0,  # Time on page (normalized to 5 min)
            ],
            dtype=np.float32,
        )

        return features

    def extract_candidate_features(self, candidate: Any) -> np.ndarray:
        """Extract features from candidate product."""
        features = np.array(
            [
                _candidate_get(candidate, "collaborative_score") or 0.0,
                _candidate_get(candidate, "content_similarity_score") or 0.0,
                _candidate_get(candidate, "popularity_score") or 0.0,
                _candidate_get(candidate, "combined_score") or 0.0,
            ],
            dtype=float,
        )

        return features

    def extract_realtime_window_features(
        self,
        user_features: UserFeatures,
        product_metadata: Dict[str, Any],
        context: Dict[str, Any],
        candidate: Any,
    ) -> np.ndarray:
        """Extract optional Flink realtime window features from request context."""
        if not self.enable_realtime_window_features:
            return np.zeros(0, dtype=np.float32)

        realtime_context = context.get("_realtime_window_features") or {}
        product_id = str(_candidate_get(candidate, "product_id", ""))
        category = (
            product_metadata.get("category")
            or context.get("product_category")
            or context.get("category")
            or ""
        )
        entity_ids = {
            "user": user_features.user_id,
            "product": product_id,
            "category": str(category),
        }

        values: List[float] = []
        for entity_type in self.WINDOW_FEATURE_ENTITIES:
            entity_id = entity_ids.get(entity_type) or ""
            window_payloads = realtime_context.get(f"{entity_type}:{entity_id}", {})
            for window in self.WINDOW_FEATURE_NAMES:
                payload = window_payloads.get(window, {}) if entity_id else {}
                values.extend(
                    [
                        float(payload.get("views", 0)) / 1000.0,
                        float(payload.get("clicks", 0)) / 1000.0,
                        float(payload.get("add_to_cart", 0)) / 1000.0,
                        float(payload.get("purchases", 0)) / 1000.0,
                        float(payload.get("click_through_rate", 0.0)),
                        float(payload.get("conversion_rate", 0.0)),
                    ]
                )
        return np.nan_to_num(np.asarray(values, dtype=np.float32), 0.0)

    def extract_history_embedding_features(
        self,
        context: Dict[str, Any],
        candidate: Any,
    ) -> np.ndarray:
        """Extract optional last-N two-tower history features."""
        if not self.enable_history_embeddings:
            return np.zeros(0, dtype=np.float32)
        product_id = _candidate_get(candidate, "product_id", "")
        return extract_ranking_history_feature_vector(
            context.get(RANKING_HISTORY_CONTEXT_KEY),
            product_id,
            embedding_dim=self.history_embedding_dim,
        )

    def create_ranking_features(
        self,
        user_features: UserFeatures,
        product_metadata: Dict[str, Any],
        context: Dict[str, Any],
        candidate: CandidateProduct,
        *,
        as_of_ts: Optional[float] = None,
    ) -> np.ndarray:
        """Create complete feature vector for ranking."""
        try:
            # Extract individual feature groups
            current_time = time.time() if as_of_ts is None else float(as_of_ts)
            user_feats = self.extract_user_features(user_features, current_time)
            product_feats = self.extract_product_features(
                product_metadata, current_time
            )
            context_feats = self.extract_context_features(context, current_time)
            candidate_feats = self.extract_candidate_features(candidate)
            history_embedding_feats = self.extract_history_embedding_features(
                context,
                candidate,
            )
            realtime_window_feats = self.extract_realtime_window_features(
                user_features,
                product_metadata,
                context,
                candidate,
            )

            # Concatenate all features
            combined_features = np.concatenate(
                [
                    user_feats,
                    product_feats,
                    context_feats,
                    candidate_feats,
                    history_embedding_feats,
                    realtime_window_feats,
                ]
            )

            # Handle any NaN or infinite values
            combined_features = np.nan_to_num(combined_features, 0.0)

            return combined_features

        except Exception as e:
            logger.error(f"Error creating ranking features: {e}")
            # Return zero features as fallback
            return np.zeros(self.total_feature_dim, dtype=np.float32)


class RankingModel:
    """
    Neural ranking model wrapper that handles training, inference, and model management.
    """

    def __init__(self, config: RankingConfig, *, observability: Any = None):
        self.config = config
        self.observability = observability
        self.feature_extractor = FeatureExtractor(
            enable_realtime_window_features=config.realtime_window_features_enabled,
            enable_history_embeddings=config.history_embeddings_enabled,
            history_embedding_dim=config.history_embedding_dim,
        )
        self.feature_assembler = RankingFeatureAssembler(
            self.feature_extractor,
            product_feature_cache_size=getattr(
                config, "product_feature_cache_size", 50000
            ),
        )
        self.din_product_index: Dict[str, int] = {}
        self.din_item_embeddings: Optional[torch.Tensor] = None
        self.din_sidecar_metadata: Dict[str, Any] = {}
        if getattr(config, "din_enabled", False):
            sidecar_path = getattr(config, "din_embedding_sidecar_path", None)
            if sidecar_path and Path(sidecar_path).exists():
                (
                    self.din_item_embeddings,
                    self.din_product_index,
                    self.din_sidecar_metadata,
                ) = load_din_embedding_sidecar(sidecar_path)
            self.feature_assembler.version = "ranking_feature_assembler_v2_din"

        # Model components
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training state
        self.is_trained = False
        self.model_version = "1.0.0"
        self.last_training_time = 0
        self.feature_schema_version = (
            RANKING_TRIMODAL_FEATURE_SCHEMA_VERSION
            if getattr(config, "trimodal_enabled", False)
            else RANKING_DIN_FEATURE_SCHEMA_VERSION
            if getattr(config, "din_enabled", False)
            else RANKING_FEATURE_SCHEMA_VERSION
        )
        self.training_data_source = RANKING_TRAINING_DATA_SOURCE
        self.ranking_objective_version = RANKING_OBJECTIVE_VERSION
        self.value_transform_stats: Dict[
            str, Any
        ] = self._default_value_transform_stats()
        self.value_bucket_mapping: Dict[str, int] = {}

        # Performance tracking
        self.training_history = []
        self.inference_stats = {
            "total_inferences": 0,
            "avg_inference_time": 0.0,
            "batch_inference_time": 0.0,
        }
        self.loaded_model_path: Optional[str] = None
        self.loaded_checkpoint_mtime: float = 0.0
        self.checkpoint_reload_count = 0
        self._checkpoint_reload_lock = asyncio.Lock()
        self.torch_inference_available = True
        self._compiled_model: Optional[Any] = None
        self.torch_compile_error: Optional[str] = None
        self.torch_compile_warmup_ms: Optional[float] = None
        self.torch_compile_fallback_count = 0
        self.torch_compile_last_fallback_error: Optional[str] = None
        self.torch_compile_last_inference_path = "eager"
        self.untrained_fallback_count = 0
        self.candidate_embedding_sidecar: Optional[CandidateEmbeddingSidecar] = None
        self.candidate_sidecar_sha256: Optional[str] = None
        self.candidate_sidecar_model_version: Optional[str] = None
        self._candidate_sidecar_training_records: Optional[
            Dict[str, Dict[str, Any]]
        ] = None
        self._candidate_sidecar_path: Optional[str] = None
        self.enable_profiling_logs = False
        self.profiling_log_min_duration_ms = 250.0
        self._product_feature_cache_max_size = max(
            0,
            getattr(config, "product_feature_cache_size", 50000),
        )
        self._product_feature_cache = self.feature_assembler._product_feature_cache
        self._product_feature_cache_lock = (
            self.feature_assembler._product_feature_cache_lock
        )
        self.training_tensor_builder = TrainingTensorBuilder(
            self.feature_assembler,
            device=self.device,
            value_bucket_id=lambda metadata: self._value_bucket_id(metadata),
            fit_value_transform=self._fit_value_transform,
            transform_value=self._transform_business_value,
        )

        logger.info(f"RankingModel initialized on device: {self.device}")

    def configure_candidate_sidecar_for_training(
        self,
        records: Dict[str, Dict[str, Any]],
        *,
        path: str,
    ) -> None:
        self._candidate_sidecar_training_records = records
        self._candidate_sidecar_path = str(path)

    def _default_value_transform_stats(self) -> Dict[str, Any]:
        return {
            "global": {
                "count": 0,
                "clip": None,
                "mean": 0.0,
                "std": 1.0,
            },
            "buckets": {},
        }

    def _initialize_model(
        self,
        *,
        architecture: Optional[str] = None,
        hidden_dims: Optional[List[int]] = None,
        cross_layers: Optional[int] = None,
        low_rank_dim: Optional[int] = None,
    ) -> None:
        self.model, self.optimizer = self._build_model_instance(
            architecture=architecture,
            hidden_dims=hidden_dims,
            cross_layers=cross_layers,
            low_rank_dim=low_rank_dim,
        )
        self._clear_compiled_model()

    def _build_model_instance(
        self,
        *,
        architecture: Optional[str] = None,
        hidden_dims: Optional[List[int]] = None,
        cross_layers: Optional[int] = None,
        low_rank_dim: Optional[int] = None,
    ) -> Tuple[nn.Module, optim.Optimizer]:
        input_dim = self.feature_extractor.total_feature_dim
        model_options = {
            "architecture": architecture,
            "hidden_dims": hidden_dims,
            "cross_layers": cross_layers,
            "low_rank_dim": low_rank_dim,
        }
        if getattr(self.config, "trimodal_enabled", False):
            model = TemporalTrimodalRankingModel(
                input_dim,
                self.config,
                din_item_embeddings=self.din_item_embeddings,
                **model_options,
            ).to(self.device)
            base_parameters = list(model.ranker.parameters())
            base_parameter_ids = {id(parameter) for parameter in base_parameters}
            new_parameters = [
                parameter
                for parameter in model.parameters()
                if id(parameter) not in base_parameter_ids
            ]
            optimizer = optim.Adam(
                [
                    {
                        "params": base_parameters,
                        "lr": self.config.learning_rate * 0.1,
                    },
                    {"params": new_parameters, "lr": self.config.learning_rate},
                ],
                weight_decay=1e-5,
            )
        else:
            model = MultiObjectiveRankingModel(
                input_dim,
                self.config,
                din_item_embeddings=self.din_item_embeddings,
                **model_options,
            ).to(self.device)
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-5,
            )
        return model, optimizer

    def _torch_compile_status(self) -> Dict[str, Any]:
        return {
            "torch_compile_enabled": bool(
                getattr(self.config, "torch_compile_enabled", False)
            ),
            "torch_compile_active": self._compiled_model is not None,
            "torch_compile_backend": getattr(
                self.config,
                "torch_compile_backend",
                "inductor",
            ),
            "torch_compile_mode": getattr(self.config, "torch_compile_mode", "default"),
            "torch_compile_dynamic": bool(
                getattr(self.config, "torch_compile_dynamic", True)
            ),
            "torch_compile_error": self.torch_compile_error,
            "torch_compile_warmup_ms": self.torch_compile_warmup_ms,
            "torch_compile_fallback_count": self.torch_compile_fallback_count,
            "torch_compile_last_fallback_error": (
                self.torch_compile_last_fallback_error
            ),
            "torch_compile_last_inference_path": (
                self.torch_compile_last_inference_path
            ),
        }

    def _clear_compiled_model(self, error: Optional[str] = None) -> None:
        self._compiled_model = None
        self.torch_compile_error = error
        self.torch_compile_warmup_ms = None
        self.torch_compile_last_inference_path = "eager"

    def _compile_model_for_inference(self) -> None:
        self._clear_compiled_model()
        if not getattr(self.config, "torch_compile_enabled", False):
            return
        if self.model is None:
            self.torch_compile_error = "model_not_loaded"
            return
        if not hasattr(torch, "compile"):
            self.torch_compile_error = "torch.compile unavailable"
            logger.warning("ranking_torch_compile_unavailable")
            return

        backend = getattr(self.config, "torch_compile_backend", "inductor")
        mode = getattr(self.config, "torch_compile_mode", "default")
        dynamic = bool(getattr(self.config, "torch_compile_dynamic", True))
        warmup_batch_size = max(
            1,
            int(getattr(self.config, "batch_target_requests", 1) or 1),
        )
        was_training = self.model.training
        self.model.eval()

        try:
            compile_started = time.perf_counter()
            compiled_model = torch.compile(
                self.model,
                backend=backend,
                mode=mode,
                dynamic=dynamic,
                fullgraph=True,
            )
            dummy_features = torch.zeros(
                warmup_batch_size,
                self.feature_extractor.total_feature_dim,
                dtype=torch.float32,
                device=self.device,
            )
            with torch.inference_mode():
                compiled_model(dummy_features)
            warmup_ms = round((time.perf_counter() - compile_started) * 1000, 2)
            self._compiled_model = compiled_model
            self.torch_compile_error = None
            self.torch_compile_warmup_ms = warmup_ms
            self.torch_compile_last_inference_path = "compiled"
            logger.info(
                "ranking_torch_compile_ready",
                extra={
                    "backend": backend,
                    "mode": mode,
                    "dynamic": dynamic,
                    "fullgraph": True,
                    "warmup_batch_size": warmup_batch_size,
                    "device": str(self.device),
                    "warmup_ms": warmup_ms,
                },
            )
        except Exception as exc:
            self._clear_compiled_model(f"{type(exc).__name__}: {exc}")
            logger.warning(
                "ranking_torch_compile_failed",
                extra={
                    "backend": backend,
                    "mode": mode,
                    "dynamic": dynamic,
                    "fullgraph": True,
                    "warmup_batch_size": warmup_batch_size,
                    "device": str(self.device),
                    "exception_type": type(exc).__name__,
                    "exception_repr": repr(exc),
                },
            )
        finally:
            if was_training:
                self.model.train()

    def _load_shape_compatible_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> int:
        if self.model is None:
            return 0
        return self._load_shape_compatible_state_dict_into_model(self.model, state_dict)

    def _load_shape_compatible_state_dict_into_model(
        self,
        model: MultiObjectiveRankingModel,
        state_dict: Dict[str, torch.Tensor],
    ) -> int:
        current_state = model.state_dict()
        compatible: Dict[str, torch.Tensor] = {}
        skipped = []
        for key, value in state_dict.items():
            target_key = key
            current_value = current_state.get(target_key)
            if current_value is None and isinstance(
                model, TemporalTrimodalRankingModel
            ):
                target_key = f"ranker.{key}"
                current_value = current_state.get(target_key)
            if current_value is not None and tuple(current_value.shape) == tuple(
                value.shape
            ):
                compatible[target_key] = value
            else:
                skipped.append(key)

        if not compatible:
            logger.warning("No shape-compatible ranking checkpoint tensors found")
            return 0

        load_result = model.load_state_dict(compatible, strict=False)
        if skipped or load_result.missing_keys or load_result.unexpected_keys:
            logger.warning(
                "Ranking checkpoint partially loaded: "
                f"loaded={len(compatible)}, skipped={len(skipped)}, "
                f"missing={len(load_result.missing_keys)}, "
                f"unexpected={len(load_result.unexpected_keys)}"
            )
        return len(compatible)

    @staticmethod
    def _checkpoint_state_and_config(
        checkpoint: Any,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], bool]:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return (
                checkpoint["model_state_dict"],
                dict(checkpoint.get("config") or {}),
                False,
            )
        return checkpoint, {"architecture": "mlp"}, True

    def _validate_checkpoint_feature_schema(
        self,
        checkpoint_config: Dict[str, Any],
        *,
        model_path: str,
    ) -> None:
        strict_schema = bool(
            getattr(self.config, "history_embeddings_enabled", False)
            or getattr(self.config, "trimodal_enabled", False)
            or getattr(self.config, "din_enabled", False)
        )
        if not strict_schema:
            return
        expected_input_dim = self.feature_extractor.total_feature_dim
        expected_schema = self.feature_schema_version
        checkpoint_input_dim = checkpoint_config.get("input_dim")
        checkpoint_schema = checkpoint_config.get("feature_schema_version")
        if (
            checkpoint_input_dim != expected_input_dim
            or checkpoint_schema != expected_schema
        ):
            requirement = (
                "Ranking history embeddings require a freshly trained v2 checkpoint"
                if getattr(self.config, "history_embeddings_enabled", False)
                else "Ranking checkpoint feature schema is incompatible"
            )
            raise RuntimeError(
                f"{requirement}: "
                f"path={model_path}, checkpoint_input_dim={checkpoint_input_dim}, "
                f"expected_input_dim={expected_input_dim}, "
                f"checkpoint_feature_schema_version={checkpoint_schema}, "
                f"expected_feature_schema_version={expected_schema}"
            )

    def _validate_din_checkpoint(
        self,
        model: MultiObjectiveRankingModel,
        state_dict: Dict[str, torch.Tensor],
        checkpoint_config: Dict[str, Any],
        *,
        model_path: str,
    ) -> None:
        if not getattr(self.config, "din_enabled", False):
            return
        expected_config = {
            "din_enabled": True,
            "feature_schema_version": RANKING_DIN_FEATURE_SCHEMA_VERSION,
            "feature_definition_version": RANKING_LTR_DIN_FEATURE_DEFINITION_VERSION,
            "feature_assembler_version": "ranking_feature_assembler_v2_din",
        }
        for key, expected in expected_config.items():
            if checkpoint_config.get(key) != expected:
                raise RuntimeError(
                    f"DIN checkpoint {model_path} has incompatible {key}"
                )
        checkpoint_sidecar = checkpoint_config.get("din_sidecar_metadata") or {}
        for key in ("sha256", "contract_version", "two_tower_model_version"):
            if not checkpoint_sidecar.get(key) or checkpoint_sidecar.get(
                key
            ) != self.din_sidecar_metadata.get(key):
                raise RuntimeError(f"DIN checkpoint/sidecar lineage mismatch for {key}")
        expected_state = model.state_dict()
        omitted = {"din.item_embedding.weight"}
        required_keys = set(expected_state) - omitted
        if set(state_dict) != required_keys:
            missing = sorted(required_keys - set(state_dict))
            unexpected = sorted(set(state_dict) - required_keys)
            raise RuntimeError(
                "DIN checkpoint trainable state is incomplete: "
                f"missing={missing}, unexpected={unexpected}"
            )
        invalid_shapes = [
            key
            for key in required_keys
            if tuple(state_dict[key].shape) != tuple(expected_state[key].shape)
        ]
        if invalid_shapes:
            raise RuntimeError(
                f"DIN checkpoint tensor shapes are incompatible: {invalid_shapes}"
            )

    async def load_model(self, model_path: str = None):
        """Load or initialize the ranking model."""
        previous_din_state = (
            self.din_item_embeddings,
            self.din_product_index,
            self.din_sidecar_metadata,
        )
        try:
            if getattr(self.config, "din_enabled", False):
                sidecar_path = getattr(self.config, "din_embedding_sidecar_path", "")
                if not sidecar_path or not Path(sidecar_path).exists():
                    raise RuntimeError("DIN ranking embedding sidecar is unavailable")
                (
                    self.din_item_embeddings,
                    self.din_product_index,
                    self.din_sidecar_metadata,
                ) = load_din_embedding_sidecar(sidecar_path)
            resolved_model_path = model_path or self.loaded_model_path

            # Load pre-trained weights if available
            if resolved_model_path and Path(resolved_model_path).exists():
                logger.info(f"Loading model from {resolved_model_path}")
                checkpoint_path = Path(resolved_model_path)
                checkpoint = torch.load(resolved_model_path, map_location=self.device)
                (
                    state_dict,
                    checkpoint_config,
                    is_legacy_raw,
                ) = self._checkpoint_state_and_config(checkpoint)
                self._validate_checkpoint_feature_schema(
                    checkpoint_config,
                    model_path=str(checkpoint_path),
                )
                architecture = normalize_architecture(
                    checkpoint_config.get("architecture"),
                    default="mlp"
                    if is_legacy_raw
                    else getattr(self.config, "architecture", "dcn"),
                    supported=RANKING_ARCHITECTURES,
                )
                next_model, next_optimizer = self._build_model_instance(
                    architecture=architecture,
                    hidden_dims=checkpoint_config.get("hidden_dims"),
                    cross_layers=checkpoint_config.get("cross_layers"),
                    low_rank_dim=checkpoint_config.get("low_rank_dim"),
                )
                self._validate_din_checkpoint(
                    next_model,
                    state_dict,
                    checkpoint_config,
                    model_path=str(checkpoint_path),
                )
                loaded_tensors = self._load_shape_compatible_state_dict_into_model(
                    next_model,
                    state_dict,
                )
                if loaded_tensors == 0:
                    raise RuntimeError(
                        f"No compatible tensors found in ranking checkpoint {resolved_model_path}"
                    )
                next_model.eval()
                self.model = next_model
                self.optimizer = next_optimizer
                if getattr(self.config, "trimodal_enabled", False):
                    sidecar_sha256 = str(
                        checkpoint_config.get("candidate_sidecar_sha256") or ""
                    )
                    sidecar_model_version = str(
                        checkpoint_config.get("candidate_sidecar_model_version") or ""
                    )
                    if len(sidecar_sha256) != 64 or not sidecar_model_version:
                        raise RuntimeError(
                            "ranking_v4 checkpoint is missing its candidate sidecar lock"
                        )
                    sidecar_path = str(
                        Path(resolved_model_path).with_suffix(".candidates.npz")
                    )
                    self.candidate_embedding_sidecar = CandidateEmbeddingSidecar.load(
                        sidecar_path,
                        expected_sha256=sidecar_sha256,
                        expected_model_version=sidecar_model_version,
                    )
                    self.candidate_sidecar_sha256 = sidecar_sha256
                    self.candidate_sidecar_model_version = sidecar_model_version
                self.is_trained = True
                self.ranking_objective_version = str(
                    checkpoint_config.get(
                        "ranking_objective_version",
                        LEGACY_RANKING_OBJECTIVE_VERSION,
                    )
                )
                self.value_transform_stats = dict(
                    checkpoint_config.get("value_transform_stats")
                    or self._default_value_transform_stats()
                )
                self.value_bucket_mapping = dict(
                    checkpoint_config.get("value_bucket_mapping") or {}
                )
                self.loaded_checkpoint_mtime = checkpoint_path.stat().st_mtime
            elif getattr(self.config, "history_embeddings_enabled", False):
                raise RuntimeError(
                    "Ranking history embeddings require an existing freshly trained "
                    f"v2 checkpoint at {resolved_model_path}"
                )
            else:
                logger.info("Initializing new model")
                next_model, next_optimizer = self._build_model_instance(
                    architecture=getattr(self.config, "architecture", "dcn"),
                )
                next_model.eval()
                self.model = next_model
                self.optimizer = next_optimizer
                self.is_trained = False
                self.ranking_objective_version = RANKING_OBJECTIVE_VERSION
                self.value_transform_stats = self._default_value_transform_stats()
                self.value_bucket_mapping = {}
                self.loaded_checkpoint_mtime = 0.0

            # Set to evaluation mode initially
            self.loaded_model_path = resolved_model_path
            self._compile_model_for_inference()

            logger.info("Ranking model loaded successfully")

        except Exception as e:
            (
                self.din_item_embeddings,
                self.din_product_index,
                self.din_sidecar_metadata,
            ) = previous_din_state
            logger.error(f"Error loading ranking model: {e}")
            raise

    async def reload_model_if_updated(self, model_path: str = None) -> bool:
        """Reload the ranking checkpoint when a newer file is available."""
        resolved_model_path = model_path or self.loaded_model_path
        if not resolved_model_path:
            return False

        checkpoint_path = Path(resolved_model_path)
        if not checkpoint_path.exists():
            return False

        checkpoint_mtime = checkpoint_path.stat().st_mtime
        if checkpoint_mtime <= self.loaded_checkpoint_mtime:
            return False

        async with self._checkpoint_reload_lock:
            checkpoint_mtime = checkpoint_path.stat().st_mtime
            if checkpoint_mtime <= self.loaded_checkpoint_mtime:
                return False

            previous_mtime = self.loaded_checkpoint_mtime
            await self.load_model(str(checkpoint_path))
            if previous_mtime > 0:
                self.checkpoint_reload_count += 1
            logger.info(
                "ranking_checkpoint_reloaded",
                extra={
                    "model_path": str(checkpoint_path),
                    "checkpoint_mtime": checkpoint_mtime,
                    "reload_count": self.checkpoint_reload_count,
                },
            )
            return True

    def prepare_request_matrix(
        self,
        candidates: List[CandidateProduct],
        user_features: UserFeatures,
        context: Dict[str, Any],
        product_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[
        Optional[np.ndarray], List[Tuple[CandidateProduct, Dict[str, Any]]], float
    ]:
        """Prepare a request's ranking feature matrix and candidate metadata."""
        feature_matrix, prepared_requests, _ = self.prepare_batch_matrix(
            [
                {
                    "index": 0,
                    "candidates": candidates,
                    "user_features": user_features,
                    "context": context,
                    "product_metadata_map": product_metadata_map or {},
                }
            ]
        )
        if not prepared_requests:
            return None, [], 0.0
        prepared = prepared_requests[0]
        return (
            feature_matrix,
            prepared["valid_candidates"],
            prepared["feature_extraction_ms"],
        )

    def prepare_batch_matrix(
        self,
        requests: List[Dict[str, Any]],
    ) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], float]:
        """Prepare one feature matrix for multiple ranking requests."""
        feature_stage_started = time.perf_counter()
        total_dim = self.feature_extractor.total_feature_dim
        total_candidates = sum(
            len(request.get("candidates") or []) for request in requests
        )
        if total_candidates <= 0:
            return None, [], 0.0

        feature_matrix = np.empty((total_candidates, total_dim), dtype=np.float32)
        prepared_requests: List[Dict[str, Any]] = []
        all_bundles: List[FeatureBundle] = []
        row_count = 0

        for request_offset, request in enumerate(requests):
            request_started = time.perf_counter()
            candidates = request.get("candidates") or []
            user_features = request["user_features"]
            context = request.get("context") or {}
            current_time = float(context.get("_feature_as_of_ts", time.time()))
            raw_din_sequences = context.get(DIN_SEQUENCE_CONTEXT_KEY)
            din_sequences = (
                parse_din_behavior_sequences(
                    raw_din_sequences,
                    expected_as_of_ts=current_time,
                    last_n=int(getattr(self.config, "din_sequence_last_n", 60)),
                    lookback_days=int(
                        getattr(self.config, "din_sequence_lookback_days", 30)
                    ),
                )
                if getattr(self.config, "din_enabled", False)
                and raw_din_sequences is not None
                else None
            )
            if getattr(self.config, "din_enabled", False) and din_sequences is None:
                din_sequences = build_din_behavior_sequences(
                    [],
                    as_of_ts=current_time,
                    last_n=int(getattr(self.config, "din_sequence_last_n", 60)),
                    lookback_days=int(
                        getattr(self.config, "din_sequence_lookback_days", 30)
                    ),
                )
            product_metadata_map = request.get("product_metadata_map") or {}
            valid_candidates: List[Tuple[CandidateProduct, Dict[str, Any]]] = []
            row_start = row_count
            bundles: List[FeatureBundle] = []

            for candidate in candidates:
                try:
                    product_id = str(_candidate_get(candidate, "product_id", ""))
                    product_metadata = product_metadata_map.get(
                        product_id
                    ) or self._build_product_metadata(candidate, current_time)
                    bundles.append(
                        FeatureBundle(
                            as_of_ts=current_time,
                            feature_definition_version=(
                                RANKING_LTR_DIN_FEATURE_DEFINITION_VERSION
                                if getattr(self.config, "din_enabled", False)
                                else RANKING_LTR_FEATURE_DEFINITION_VERSION
                            ),
                            user_features=user_features,
                            product_metadata=product_metadata,
                            context=context,
                            candidate=candidate,
                            behavior_sequences=din_sequences,
                        )
                    )
                    valid_candidates.append((candidate, product_metadata))
                    row_count += 1
                except Exception as e:
                    logger.warning(
                        "Error extracting features for candidate %s: %s",
                        _candidate_get(candidate, "product_id"),
                        e,
                    )

            row_end = row_count
            if row_end > row_start:
                feature_matrix[row_start:row_end] = self.feature_assembler.build_many(
                    bundles
                )
                all_bundles.extend(bundles)

            prepared_requests.append(
                {
                    "index": request.get("index", request_offset),
                    "k": request.get("k"),
                    "batch_wait_ms": request.get("batch_wait_ms", 0.0),
                    "valid_candidates": valid_candidates,
                    "context": context,
                    "row_start": row_start,
                    "row_end": row_end,
                    "candidate_count": len(candidates),
                    "feature_extraction_ms": round(
                        (time.perf_counter() - request_started) * 1000,
                        2,
                    ),
                }
            )

        feature_extraction_ms = round(
            (time.perf_counter() - feature_stage_started) * 1000, 2
        )
        if row_count == 0:
            return None, prepared_requests, feature_extraction_ms

        if row_count < total_candidates:
            feature_matrix = feature_matrix[:row_count]
        np.nan_to_num(feature_matrix, nan=0.0, copy=False)
        if getattr(self.config, "din_enabled", False):
            if self.din_item_embeddings is None:
                raise RuntimeError("DIN ranking embedding sidecar is not loaded")
            feature_matrix = RankingFeatureMatrix(
                feature_matrix,
                din_inputs=build_din_batch_inputs(
                    all_bundles,
                    self.din_product_index,
                    sequence_length=int(
                        getattr(self.config, "din_sequence_last_n", 60)
                    ),
                    device=self.device,
                ),
            )
        return feature_matrix, prepared_requests, feature_extraction_ms

    def run_inference_batch(
        self,
        feature_matrix: np.ndarray,
        value_bucket_ids: Optional[Sequence[Optional[int]]] = None,
        multimodal_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Run one Torch forward pass for a feature matrix."""
        model = self.model
        if model is None:
            raise RuntimeError("Ranking model is not loaded")
        with torch.inference_mode():
            tensor_stage_started = time.perf_counter()
            features_tensor = torch.as_tensor(
                feature_matrix,
                dtype=torch.float32,
                device=self.device,
            )
            tensor_prep_ms = round(
                (time.perf_counter() - tensor_stage_started) * 1000, 2
            )

            model_stage_started = time.perf_counter()
            inference_model = (
                model if multimodal_inputs else (self._compiled_model or model)
            )
            inference_path = (
                "eager_trimodal"
                if multimodal_inputs
                else "compiled"
                if self._compiled_model is not None
                else "eager"
            )
            try:
                din_inputs = getattr(feature_matrix, "din_inputs", None)
                forward_inputs = dict(multimodal_inputs or {})
                if din_inputs is not None:
                    (
                        history_indices,
                        history_recency,
                        history_mask,
                    ) = din_inputs.expanded_histories()
                    forward_inputs.update(
                        {
                            "candidate_indices": din_inputs.candidate_indices,
                            "history_indices": history_indices,
                            "history_recency": history_recency,
                            "history_mask": history_mask,
                            "summary_features": din_inputs.summary_features,
                        }
                    )
                predictions = inference_model(features_tensor, **forward_inputs)
            except Exception as exc:
                if self._compiled_model is None:
                    raise
                fallback_error = f"{type(exc).__name__}: {exc}"
                self.torch_compile_fallback_count += 1
                self.torch_compile_last_fallback_error = fallback_error
                self._clear_compiled_model(fallback_error)
                logger.warning(
                    "ranking_torch_compile_inference_failed",
                    extra={
                        "exception_type": type(exc).__name__,
                        "exception_repr": repr(exc),
                        "fallback_count": self.torch_compile_fallback_count,
                    },
                )
                inference_path = "eager"
                predictions = model(features_tensor, **forward_inputs)
            model_forward_ms = round(
                (time.perf_counter() - model_stage_started) * 1000, 2
            )
            if multimodal_inputs and self.observability is not None:
                self.observability.record_trimodal_inference(model_forward_ms / 1000.0)
                gate = predictions.get("modality_gate")
                if gate is not None and gate.numel():
                    for index, modality in enumerate(("visual", "ocr", "asr")):
                        self.observability.record_trimodal_gate(
                            modality, float(gate[:, index].mean().item())
                        )
            self.torch_compile_last_inference_path = inference_path

        prediction_arrays = {
            key: value.detach().cpu().numpy().reshape(-1)
            for key, value in predictions.items()
        }
        self._add_business_predictions(
            prediction_arrays,
            value_bucket_ids=value_bucket_ids,
        )
        return prediction_arrays, {
            "tensor_prep_ms": tensor_prep_ms,
            "model_forward_ms": model_forward_ms,
            "inference_path": inference_path,
        }

    def build_serving_trimodal_inputs(
        self,
        context: Dict[str, Any],
        valid_candidates: Sequence[Tuple[CandidateProduct, Dict[str, Any]]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not isinstance(self.model, TemporalTrimodalRankingModel):
            return None
        packed = context.get("temporal_multimodal")
        if not isinstance(packed, dict):
            packed = {}
        decoded = unpack_temporal_multimodal_context(packed)
        batch_size = len(valid_candidates)
        capacities = {"visual": (16, 512), "ocr": (32, 384), "asr": (64, 384)}
        tensors: Dict[str, torch.Tensor] = {}
        for modality, (capacity, dimension) in capacities.items():
            embeddings = np.zeros((capacity, dimension), dtype=np.float32)
            starts = np.zeros(capacity, dtype=np.float32)
            ends = np.zeros(capacity, dtype=np.float32)
            mask = np.zeros(capacity, dtype=np.bool_)
            source = decoded.get(f"{modality}_embeddings")
            if source is not None:
                rows = min(capacity, int(source.shape[0]))
                if source.shape[1] != dimension:
                    raise ValueError(
                        f"{modality} embedding dimension must be {dimension}"
                    )
                embeddings[:rows] = source[:rows]
                starts[:rows] = decoded[f"{modality}_starts"][:rows]
                ends[:rows] = decoded[f"{modality}_ends"][:rows]
                mask[:rows] = decoded[f"{modality}_mask"][:rows]
            if self.observability is not None:
                self.observability.record_trimodal_presence(modality, bool(mask.any()))
            tensors[f"{modality}_embeddings"] = torch.as_tensor(
                np.repeat(embeddings[None, :, :], batch_size, axis=0),
                dtype=torch.float32,
                device=self.device,
            )
            for name, values, dtype in (
                ("starts", starts, torch.float32),
                ("ends", ends, torch.float32),
                ("mask", mask, torch.bool),
            ):
                tensors[f"{modality}_{name}"] = torch.as_tensor(
                    np.repeat(values[None, :], batch_size, axis=0),
                    dtype=dtype,
                    device=self.device,
                )
        candidate_image = np.zeros((batch_size, 512), dtype=np.float32)
        candidate_text = np.zeros((batch_size, 384), dtype=np.float32)
        candidate_two_tower = np.zeros((batch_size, 128), dtype=np.float32)
        candidate_presence = np.zeros((batch_size, 3), dtype=np.bool_)
        if self.candidate_embedding_sidecar is not None:
            for row, (candidate, _metadata) in enumerate(valid_candidates):
                embedding = self.candidate_embedding_sidecar.get(candidate.product_id)
                if embedding is None:
                    if self.observability is not None:
                        self.observability.record_candidate_sidecar_miss()
                    continue
                candidate_image[row] = embedding["image"]
                candidate_text[row] = embedding["text"]
                candidate_two_tower[row] = embedding["two_tower"]
                candidate_presence[row] = embedding["presence"]
        elif self.observability is not None:
            for _candidate in valid_candidates:
                self.observability.record_candidate_sidecar_miss()
        for name, values in (
            ("candidate_image", candidate_image),
            ("candidate_text", candidate_text),
            ("candidate_two_tower", candidate_two_tower),
            ("candidate_presence", candidate_presence),
        ):
            tensors[name] = torch.as_tensor(values, device=self.device)
        return tensors

    def _add_business_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        value_bucket_ids: Optional[Sequence[Optional[int]]] = None,
    ) -> None:
        if not getattr(self.config, "business_score_enabled", True):
            return
        raw_values = np.asarray(predictions.get("gmv", []), dtype=np.float64).reshape(
            -1
        )
        if raw_values.size == 0:
            return
        bucket_ids: List[Optional[int]]
        if value_bucket_ids is None:
            bucket_ids = [None] * int(raw_values.size)
        else:
            bucket_ids = [
                None if bucket_id is None else int(bucket_id)
                for bucket_id in value_bucket_ids
            ]
            if len(bucket_ids) != raw_values.size:
                logger.warning(
                    "ranking_value_bucket_id_count_mismatch",
                    extra={
                        "bucket_id_count": len(bucket_ids),
                        "prediction_count": int(raw_values.size),
                    },
                )
                bucket_ids = [None] * int(raw_values.size)
        predicted_values = np.asarray(
            [
                self._inverse_transform_business_value(value, bucket_id)
                for value, bucket_id in zip(raw_values, bucket_ids)
            ],
            dtype=np.float32,
        )
        predictions["predicted_value"] = predicted_values
        predictions["gmv"] = predicted_values
        ctr = np.asarray(
            predictions.get("ctr", np.zeros_like(predicted_values))
        ).reshape(-1)
        cvr = np.asarray(
            predictions.get("cvr", np.zeros_like(predicted_values))
        ).reshape(-1)
        ctcvr = np.asarray(predictions.get("ctcvr", ctr * cvr)).reshape(-1)
        predictions["ctcvr"] = np.clip(ctcvr, 0.0, 1.0).astype(np.float32)
        predictions["business_score"] = (
            predictions["ctcvr"] * predicted_values
        ).astype(np.float32)

    def build_recommendations_from_predictions(
        self,
        valid_candidates: List[Tuple[CandidateProduct, Dict[str, Any]]],
        predictions: Dict[str, np.ndarray],
        k: int,
    ) -> Tuple[List[ProductRecommendation], float]:
        """Convert raw model predictions into ranked recommendation objects."""
        response_stage_started = time.perf_counter()
        recommendations: List[ProductRecommendation] = []
        if not valid_candidates or k <= 0:
            return [], round((time.perf_counter() - response_stage_started) * 1000, 2)

        score_key = (
            "business_score"
            if getattr(self.config, "business_score_enabled", True)
            and "business_score" in predictions
            else "ranking_score"
        )
        ranking_scores = np.asarray(predictions[score_key]).reshape(-1)
        top_count = min(k, len(valid_candidates), ranking_scores.shape[0])
        if top_count <= 0:
            return [], round((time.perf_counter() - response_stage_started) * 1000, 2)

        if ranking_scores.shape[0] > top_count:
            top_indices = np.argpartition(-ranking_scores, top_count - 1)[:top_count]
            top_indices = top_indices[np.argsort(-ranking_scores[top_indices])]
        else:
            top_indices = np.argsort(-ranking_scores)

        for i in top_indices:
            candidate, metadata = valid_candidates[int(i)]
            ctr_score = float(predictions["ctr"][i])
            cvr_score = float(predictions["cvr"][i])
            gmv_source = predictions.get("predicted_value", predictions.get("gmv"))
            gmv_score = float(gmv_source[i])
            ranking_score = float(ranking_scores[i])

            if getattr(self.config, "business_score_enabled", True):
                confidence_score = max(0.0, min(ctr_score * cvr_score, 1.0))
            elif self.config.enable_multi_objective:
                confidence_score = (
                    ctr_score * self.config.ctr_weight
                    + cvr_score * self.config.cvr_weight
                    + (gmv_score / 100.0) * self.config.gmv_weight
                ) / (
                    self.config.ctr_weight
                    + self.config.cvr_weight
                    + self.config.gmv_weight
                )
            else:
                confidence_score = ranking_score

            reason = self._generate_explanation(
                candidate, ctr_score, cvr_score, gmv_score
            )
            recommendations.append(
                ProductRecommendation.construct(
                    product_id=str(_candidate_get(candidate, "product_id", "")),
                    title=metadata.get(
                        "title",
                        f"Product {_candidate_get(candidate, 'product_id', '')}",
                    ),
                    description="Recommended based on your preferences",
                    price=round(float(metadata.get("price", 0.0) or 0.0), 2),
                    currency="USD",
                    category=metadata.get("category", "General"),
                    brand=metadata.get("brand", "Unknown"),
                    rating=metadata.get("rating"),
                    confidence_score=min(confidence_score, 1.0),
                    ranking_score=ranking_score,
                    reason=reason,
                )
            )

        return recommendations, round(
            (time.perf_counter() - response_stage_started) * 1000,
            2,
        )

    def _build_product_metadata(
        self, candidate: Any, current_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Build deterministic fallback metadata when the cache misses."""
        product_id = str(_candidate_get(candidate, "product_id", ""))
        return {
            "title": f"Product {product_id}",
            "category": "General",
            "price": 0.0,
            "rating": 0.0,
            "num_reviews": 0,
            "in_stock": True,
            "created_at": current_time or time.time(),
            "tags": [],
            "brand": "Unknown",
            "_ranking_fallback_metadata": True,
        }

    @staticmethod
    def _product_metadata_fingerprint(
        product_metadata: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        return RankingFeatureAssembler._metadata_fingerprint(product_metadata)

    def _get_product_feature_vector(
        self,
        product_id: str,
        product_metadata: Dict[str, Any],
        current_time: float,
    ) -> np.ndarray:
        """Return product features while preserving request-time freshness for age."""
        return self.feature_assembler._product_features(
            product_id, product_metadata, current_time
        )

    async def rank_candidates(
        self,
        candidates: List[CandidateProduct],
        user_features: UserFeatures,
        context: Dict[str, Any],
        k: int = 10,
        include_profile: bool = False,
        product_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[ProductRecommendation]:
        if getattr(self.config, "offload_inference_to_thread", True):
            return await asyncio.to_thread(
                self._rank_candidates_sync,
                candidates,
                user_features,
                context,
                k,
                include_profile,
                product_metadata_map,
            )
        return self._rank_candidates_sync(
            candidates,
            user_features,
            context,
            k,
            include_profile,
            product_metadata_map,
        )

    def _rank_candidates_sync(
        self,
        candidates: List[CandidateProduct],
        user_features: UserFeatures,
        context: Dict[str, Any],
        k: int = 10,
        include_profile: bool = False,
        product_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[ProductRecommendation]:
        """
        Rank candidate products and return top-k recommendations.

        Args:
            candidates: List of candidate products from recommendation engine
            user_features: User profile and preferences
            context: Request context (device, time, etc.)
            k: Number of top recommendations to return

        Returns:
            Ranked list of product recommendations
        """
        if not candidates:
            logger.warning("No candidates to rank")
            return []

        if not self.is_trained:
            self.untrained_fallback_count += 1
            if self.observability is not None:
                self.observability.record_ranking_untrained_fallback()
            fallback = self._fallback_rank_candidates(candidates, k)
            if include_profile:
                return fallback, {
                    "path": "fallback_untrained",
                    "feature_extraction_ms": 0.0,
                    "tensor_prep_ms": 0.0,
                    "model_forward_ms": 0.0,
                    "response_build_ms": 0.0,
                    "total_ms": 0.0,
                    "inference_path": "fallback_untrained",
                    "candidate_count": len(candidates),
                    "ranked_count": len(fallback),
                }
            return fallback

        if not self.model:
            logger.error("Model not loaded")
            return []

        if not self.torch_inference_available:
            fallback = self._fallback_rank_candidates(candidates, k)
            if include_profile:
                return fallback, {
                    "path": "fallback",
                    "feature_extraction_ms": 0.0,
                    "tensor_prep_ms": 0.0,
                    "model_forward_ms": 0.0,
                    "response_build_ms": 0.0,
                    "total_ms": 0.0,
                    "inference_path": "fallback",
                    "candidate_count": len(candidates),
                    "ranked_count": len(fallback),
                }
            return fallback

        start_time = time.time()
        profile = {
            "path": "torch",
            "feature_extraction_ms": 0.0,
            "tensor_prep_ms": 0.0,
            "model_forward_ms": 0.0,
            "response_build_ms": 0.0,
            "total_ms": 0.0,
            "inference_path": "none",
            "candidate_count": len(candidates),
            "ranked_count": 0,
        }

        try:
            (
                feature_matrix,
                valid_candidates,
                feature_extraction_ms,
            ) = self.prepare_request_matrix(
                candidates,
                user_features,
                context,
                product_metadata_map=product_metadata_map,
            )
            profile["feature_extraction_ms"] = feature_extraction_ms

            if feature_matrix is None:
                logger.warning("No valid feature vectors created")
                if include_profile:
                    return [], profile
                return []

            predictions, inference_profile = self.run_inference_batch(
                feature_matrix,
                value_bucket_ids=self._value_bucket_ids_for_candidates(
                    valid_candidates
                ),
                multimodal_inputs=self.build_serving_trimodal_inputs(
                    context, valid_candidates
                ),
            )
            profile["tensor_prep_ms"] = inference_profile["tensor_prep_ms"]
            profile["model_forward_ms"] = inference_profile["model_forward_ms"]
            profile["inference_path"] = inference_profile["inference_path"]

            (
                top_recommendations,
                response_build_ms,
            ) = self.build_recommendations_from_predictions(
                valid_candidates,
                predictions,
                k,
            )
            profile["response_build_ms"] = response_build_ms
            profile["ranked_count"] = len(top_recommendations)

            # Update inference statistics
            inference_time = time.time() - start_time
            self.inference_stats["total_inferences"] += 1
            self.inference_stats["avg_inference_time"] = (
                self.inference_stats["avg_inference_time"]
                * (self.inference_stats["total_inferences"] - 1)
                + inference_time
            ) / self.inference_stats["total_inferences"]

            logger.debug(
                f"Ranked {len(candidates)} candidates -> {len(top_recommendations)} "
                f"recommendations in {inference_time:.3f}s"
            )
            profile["total_ms"] = round(inference_time * 1000, 2)
            if (
                self.enable_profiling_logs
                or profile["total_ms"] >= self.profiling_log_min_duration_ms
            ):
                logger.info("ranking_profile", extra=profile)

            if include_profile:
                return top_recommendations, profile
            return top_recommendations

        except Exception as e:
            logger.error(f"Error ranking candidates: {e}")
            self.torch_inference_available = False
            fallback = self._fallback_rank_candidates(candidates, k)
            if include_profile:
                profile["path"] = "fallback_on_error"
                profile["error"] = str(e)
                profile["ranked_count"] = len(fallback)
                profile["total_ms"] = round((time.time() - start_time) * 1000, 2)
                return fallback, profile
            return fallback

    def _fallback_rank_candidates(
        self,
        candidates: List[CandidateProduct],
        k: int,
    ) -> List[ProductRecommendation]:
        """Fallback ranking path when Torch inference is unavailable."""
        fallback_recommendations: List[ProductRecommendation] = []

        for candidate in candidates:
            collaborative = _candidate_get(candidate, "collaborative_score") or 0.0
            content_similarity = (
                _candidate_get(candidate, "content_similarity_score") or 0.0
            )
            popularity = _candidate_get(candidate, "popularity_score") or 0.0
            combined = _candidate_get(candidate, "combined_score")
            if combined is None:
                combined = (
                    0.5 * collaborative + 0.3 * content_similarity + 0.2 * popularity
                )
            confidence_score = max(0.0, min(combined, 1.0))
            product_id = str(_candidate_get(candidate, "product_id", ""))

            fallback_recommendations.append(
                ProductRecommendation(
                    product_id=product_id,
                    title=f"Product {product_id}",
                    description="Fallback ranking path used",
                    price=0.0,
                    currency="USD",
                    category="General",
                    brand="Unknown",
                    rating=None,
                    confidence_score=confidence_score,
                    ranking_score=combined,
                    reason="Fallback ranking based on candidate generation scores",
                )
            )

        fallback_recommendations.sort(key=lambda item: item.ranking_score, reverse=True)
        return fallback_recommendations[:k]

    def _generate_explanation(
        self,
        candidate: CandidateProduct,
        ctr_score: float,
        cvr_score: float,
        gmv_score: float,
    ) -> str:
        """Generate explanation for recommendation."""
        explanations = []

        collaborative_score = _candidate_get(candidate, "collaborative_score")
        content_similarity_score = _candidate_get(candidate, "content_similarity_score")
        popularity_score = _candidate_get(candidate, "popularity_score")

        if collaborative_score and collaborative_score > 0.5:
            explanations.append("users with similar interests liked this")

        if content_similarity_score and content_similarity_score > 0.5:
            explanations.append("matches the video content you're viewing")

        if popularity_score and popularity_score > 0.5:
            explanations.append("trending among other users")

        if ctr_score > 0.7:
            explanations.append("high engagement product")

        if cvr_score > 0.3:
            explanations.append("frequently purchased")

        if not explanations:
            explanations.append("recommended for you")

        return f"Based on your preferences - {explanations[0]}"

    async def train_model(
        self,
        training_data: Sequence[RankingTrainingExample],
        *,
        training_sample_source: str = "interaction_events",
        cancellation_event: Optional[threading.Event] = None,
    ):
        """Train from validated typed examples only."""
        if not all(isinstance(row, RankingTrainingExample) for row in training_data):
            raise TypeError("RankingModel.train_model requires typed training examples")
        cancel_requested = cancellation_event or threading.Event()
        training_task = asyncio.create_task(
            asyncio.to_thread(
                self._train_model_sync,
                training_data,
                training_sample_source,
                cancellation_event=cancel_requested,
            )
        )
        try:
            saved_model_path = await asyncio.shield(training_task)
            if saved_model_path:
                await self.save_model(saved_model_path)
        except asyncio.CancelledError:
            # asyncio cannot terminate a to_thread worker. Signal the sync loop
            # and wait for it to stop before the caller releases a durable
            # training lease or permits another trainer to take over.
            cancel_requested.set()
            try:
                await training_task
            except RankingTrainingCancelled:
                pass
            raise
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def _train_model_sync(
        self,
        training_data,
        training_sample_source: str = "interaction_events",
        *,
        user_features_map: Optional[Dict[str, Any]] = None,
        product_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
        item_embedding_map: Optional[Dict[str, Any]] = None,
        two_tower_model_version: Optional[str] = None,
        cancellation_event: Optional[threading.Event] = None,
    ) -> Optional[str]:
        if cancellation_event is not None and cancellation_event.is_set():
            raise RankingTrainingCancelled("ranking training was cancelled")
        if len(training_data) < self.config.training_min_samples:
            logger.warning("Insufficient data for ranking model training")
            return None

        target_architecture = normalize_architecture(
            getattr(self.config, "architecture", "dcn"),
            supported=RANKING_ARCHITECTURES,
        )
        if self.model is None:
            self._initialize_model(architecture=target_architecture)
        elif getattr(self.model, "architecture", "mlp") != target_architecture:
            previous_state = self.model.state_dict()
            self._initialize_model(architecture=target_architecture)
            self._load_shape_compatible_state_dict(previous_state)
        elif self.optimizer is None:
            previous_state = self.model.state_dict()
            self._initialize_model(
                architecture=target_architecture,
                hidden_dims=getattr(self.model, "hidden_dims", None),
                cross_layers=getattr(self.model, "cross_layers", None),
                low_rank_dim=getattr(self.model, "low_rank_dim", None),
            )
            self._load_shape_compatible_state_dict(previous_state)

        logger.info(f"Training ranking model on {len(training_data)} samples")

        # Prepare training data
        if not all(isinstance(row, RankingTrainingExample) for row in training_data):
            from video_commerce.ml.legacy_training_adapter import (
                LegacyTrainingDatasetAdapter,
            )

            training_data = LegacyTrainingDatasetAdapter(
                feature_store=None,
                vector_search=None,
                ranking_model=self,
                recommendation_engine=None,
            ).build_from_maps(
                training_data,
                user_features_map=user_features_map or {},
                product_metadata_map=product_metadata_map or {},
                item_embedding_map=item_embedding_map or {},
                two_tower_model_version=two_tower_model_version,
                training_sample_source=training_sample_source,
            )
        multimodal_tensors = None
        if getattr(self.config, "trimodal_enabled", False):
            (
                features,
                labels,
                multimodal_tensors,
            ) = self.training_tensor_builder.build_trimodal(
                training_data,
                apply_modality_dropout=True,
                modality_dropout_probability=float(
                    getattr(self.config, "trimodal_modality_dropout", 0.1)
                ),
            )
        else:
            features, labels = self._prepare_training_examples(training_data)

        if cancellation_event is not None and cancellation_event.is_set():
            raise RankingTrainingCancelled("ranking training was cancelled")

        if features.size(0) == 0:
            logger.warning("No valid training samples prepared")
            return None

        self._clear_compiled_model()
        self.model.train()

        for epoch in range(self.config.epochs):
            if cancellation_event is not None and cancellation_event.is_set():
                raise RankingTrainingCancelled("ranking training was cancelled")
            epoch_loss = 0.0
            num_batches = 0

            if isinstance(self.model, TemporalTrimodalRankingModel):
                base_trainable = epoch > 0
                for parameter in self.model.ranker.parameters():
                    parameter.requires_grad_(base_trainable)
                self.model.ranker.train(base_trainable)

            batches = (
                self._iter_trimodal_training_batches(
                    features, labels, multimodal_tensors
                )
                if multimodal_tensors is not None
                else (
                    (batch_features, batch_labels, None)
                    for batch_features, batch_labels in self._iter_training_batches(
                        features, labels
                    )
                )
            )
            for batch_features, batch_labels, batch_multimodal in batches:
                if cancellation_event is not None and cancellation_event.is_set():
                    raise RankingTrainingCancelled("ranking training was cancelled")
                self.optimizer.zero_grad()
                forward_inputs = dict(batch_multimodal or {})
                if getattr(self.config, "din_enabled", False):
                    forward_inputs.update(
                        {
                            "candidate_indices": batch_labels["_din_candidate_indices"],
                            "history_indices": batch_labels["_din_history_indices"],
                            "history_recency": batch_labels["_din_history_recency"],
                            "history_mask": batch_labels["_din_history_mask"],
                            "summary_features": batch_labels["_din_summary_features"],
                        }
                    )
                predictions = self.model(batch_features, **forward_inputs)
                loss = self._compute_loss(predictions, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if cancellation_event is not None and cancellation_event.is_set():
                raise RankingTrainingCancelled("ranking training was cancelled")

            avg_loss = epoch_loss / max(num_batches, 1)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

            if avg_loss < 0.01:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if cancellation_event is not None and cancellation_event.is_set():
            raise RankingTrainingCancelled("ranking training was cancelled")
        self.model.eval()
        self.training_history.append(
            self._training_quality_metrics(features, labels, loss=avg_loss)
        )
        self.is_trained = True
        self.last_training_time = time.time()
        self.ranking_objective_version = RANKING_OBJECTIVE_VERSION
        self.model_version = f"ranking-{int(self.last_training_time)}"
        if (
            isinstance(self.model, TemporalTrimodalRankingModel)
            and self._candidate_sidecar_training_records is not None
            and self._candidate_sidecar_path
        ):
            self.candidate_sidecar_sha256 = write_candidate_embedding_sidecar(
                self._candidate_sidecar_path,
                self._candidate_sidecar_training_records,
                model_version=self.model_version,
            )
            self.candidate_sidecar_model_version = self.model_version
            self.candidate_embedding_sidecar = CandidateEmbeddingSidecar.load(
                self._candidate_sidecar_path,
                expected_sha256=self.candidate_sidecar_sha256,
                expected_model_version=self.model_version,
            )
        if isinstance(self.model, TemporalTrimodalRankingModel):
            for parameter in self.model.ranker.parameters():
                parameter.requires_grad_(True)
        self._compile_model_for_inference()

        logger.info("Model training completed")
        return self.loaded_model_path

    def _training_quality_metrics(self, features, labels, *, loss: float):
        with torch.no_grad():
            if getattr(self.config, "din_enabled", False):
                predictions = self.model(
                    features,
                    candidate_indices=labels["_din_candidate_indices"],
                    history_indices=labels["_din_history_indices"],
                    history_recency=labels["_din_history_recency"],
                    history_mask=labels["_din_history_mask"],
                    summary_features=labels["_din_summary_features"],
                )
            else:
                predictions = self.model(features)
        ctr_labels = labels["ctr"].detach().cpu().numpy().reshape(-1)
        ctr_scores = predictions["ctr"].detach().cpu().numpy().reshape(-1)
        auc = (
            float(roc_auc_score(ctr_labels, ctr_scores))
            if len(set(ctr_labels)) > 1
            else None
        )
        scores = predictions["ranking_score"].detach().cpu().numpy().reshape(-1)
        relevance = labels["ranking_relevance"].detach().cpu().numpy().reshape(-1)
        groups = labels["ltr_group"].detach().cpu().numpy().reshape(-1)
        group_ndcg = []
        for group in np.unique(groups):
            selected = groups == group
            if int(selected.sum()) > 1:
                group_ndcg.append(
                    float(ndcg_score([relevance[selected]], [scores[selected]]))
                )
        return {
            "loss": float(loss),
            "auc": auc,
            "ndcg": float(np.mean(group_ndcg)) if group_ndcg else None,
            "attention_entropy": getattr(
                getattr(self.model, "din", None), "last_attention_entropy", None
            ),
        }

    def _prepare_training_examples(
        self, training_data: Sequence[RankingTrainingExample]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features, labels = self.training_tensor_builder.build(training_data)
        if getattr(self.config, "din_enabled", False):
            if self.din_item_embeddings is None:
                raise RuntimeError("DIN ranking embedding sidecar is not loaded")
            din_inputs = build_din_batch_inputs(
                [example.bundle for example in training_data],
                self.din_product_index,
                sequence_length=int(getattr(self.config, "din_sequence_last_n", 60)),
                device=self.device,
            )
            (
                history_indices,
                history_recency,
                history_mask,
            ) = din_inputs.expanded_histories()
            labels.update(
                {
                    "_din_candidate_indices": din_inputs.candidate_indices,
                    "_din_history_indices": history_indices,
                    "_din_history_recency": history_recency,
                    "_din_history_mask": history_mask,
                    "_din_summary_features": din_inputs.summary_features,
                }
            )
        return features, labels

    def _ltr_grouped_training_enabled(self) -> bool:
        return bool(
            (
                getattr(self.config, "ltr_pairwise_enabled", False)
                and getattr(self.config, "ltr_pairwise_weight", 0.0) > 0
            )
            or (
                getattr(self.config, "ltr_listwise_enabled", False)
                and getattr(self.config, "ltr_listwise_weight", 0.0) > 0
            )
        )

    def _iter_training_batches(
        self,
        features: torch.Tensor,
        labels: Dict[str, torch.Tensor],
    ):
        for index_tensor in self._iter_training_batch_indices(features, labels):
            yield features.index_select(0, index_tensor), {
                key: value.index_select(0, index_tensor)
                for key, value in labels.items()
            }

    def _iter_trimodal_training_batches(
        self,
        features: torch.Tensor,
        labels: Dict[str, torch.Tensor],
        multimodal: Dict[str, torch.Tensor],
    ):
        for index_tensor in self._iter_training_batch_indices(features, labels):
            yield (
                features.index_select(0, index_tensor),
                {
                    key: value.index_select(0, index_tensor)
                    for key, value in labels.items()
                },
                {
                    key: value.index_select(0, index_tensor)
                    for key, value in multimodal.items()
                },
            )

    def _iter_training_batch_indices(
        self,
        features: torch.Tensor,
        labels: Dict[str, torch.Tensor],
    ):
        batch_size = max(1, int(getattr(self.config, "batch_size", 1) or 1))
        row_count = int(features.size(0))
        if row_count <= 0:
            return
        group_tensor = labels.get("pairwise_group")
        if (
            not self._ltr_grouped_training_enabled()
            or group_tensor is None
            or int(group_tensor.numel()) != row_count
        ):
            for start in range(0, row_count, batch_size):
                yield torch.arange(
                    start,
                    min(start + batch_size, row_count),
                    dtype=torch.long,
                    device=features.device,
                )
            return
        grouped_indices: "OrderedDict[int, List[int]]" = OrderedDict()
        for index, group_id in enumerate(group_tensor.detach().cpu().reshape(-1)):
            grouped_indices.setdefault(int(group_id.item()), []).append(index)
        pending_indices: List[int] = []
        for indices in grouped_indices.values():
            if pending_indices and len(pending_indices) + len(indices) > batch_size:
                yield torch.tensor(
                    pending_indices, dtype=torch.long, device=features.device
                )
                pending_indices = []
            pending_indices.extend(indices)
        if pending_indices:
            yield torch.tensor(
                pending_indices, dtype=torch.long, device=features.device
            )

    def _prepare_training_data(
        self,
        training_data: List[Dict[str, Any]],
        *,
        user_features_map: Optional[Dict[str, Any]] = None,
        product_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
        item_embedding_map: Optional[Dict[str, Any]] = None,
        two_tower_model_version: Optional[str] = None,
        training_sample_source: str = "interaction_events",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare training data from interaction logs."""
        try:
            user_features_map = user_features_map or {}
            product_metadata_map = product_metadata_map or {}
            item_embedding_map = item_embedding_map or {}
            history_contexts: Dict[int, Dict[str, Any]] = {}
            if getattr(self.config, "history_embeddings_enabled", False):
                history_contexts = build_training_history_contexts(
                    training_data,
                    item_embedding_map,
                    config=ranking_history_config_from_settings(self.config),
                    two_tower_model_version=two_tower_model_version,
                )
            features = []
            ctr_labels = []
            cvr_labels = []
            ctcvr_labels = []
            cvr_masks = []
            value_labels = []
            business_value_labels = []
            value_masks = []
            value_bucket_ids = []
            value_records: List[Dict[str, Any]] = []
            relevance_labels = []
            group_ids = []
            listwise_group_ids = []
            slate_sample_flags = []
            group_mapping: Dict[str, int] = {}
            listwise_group_mapping: Dict[str, int] = {}

            for sample_index, sample in enumerate(training_data):
                product_id = sample.get("product_id")
                if not product_id:
                    continue

                user_features = self._training_user_features(sample, user_features_map)
                product_metadata = self._training_product_metadata(
                    sample, product_metadata_map
                )
                context = dict(sample.get("context") or {})
                history_context = history_contexts.get(sample_index)
                if history_context is not None:
                    context[RANKING_HISTORY_CONTEXT_KEY] = history_context
                candidate = self._training_candidate(sample)
                feature_vector = self.feature_assembler.build(
                    FeatureBundle(
                        as_of_ts=self._training_as_of_timestamp(sample),
                        feature_definition_version=(
                            RANKING_LTR_FEATURE_DEFINITION_VERSION
                        ),
                        user_features=user_features,
                        product_metadata=product_metadata,
                        context=context,
                        candidate=candidate,
                    )
                )
                feature_vector = np.asarray(feature_vector, dtype=np.float32)
                if feature_vector.shape != (self.feature_extractor.total_feature_dim,):
                    logger.warning(
                        "Skipping ranking training sample with invalid feature shape",
                        extra={
                            "product_id": product_id,
                            "shape": feature_vector.shape,
                        },
                    )
                    continue
                features.append(feature_vector)

                # Create labels based on interaction type
                action = str(sample.get("action", "view") or "view").lower()
                context = dict(sample.get("context") or {})
                is_clicked = self._training_clicked_label(sample, action)
                is_purchase = action == "purchase" or bool(
                    context.get("attributed_purchase")
                )

                # CTR label (clicked or not)
                ctr_label = 1.0 if is_clicked else 0.0
                ctr_labels.append(ctr_label)

                # CVR is conditional on click; unclicked rows are masked out.
                cvr_label = 1.0 if is_purchase else 0.0
                cvr_labels.append(cvr_label)
                ctcvr_labels.append(1.0 if is_purchase else 0.0)
                cvr_masks.append(1.0 if is_clicked else 0.0)

                # Business value is purchase-conditional; non-purchase rows are masked out.
                business_value = self._training_business_value_label(
                    sample,
                    product_metadata,
                    action,
                )
                value_bucket = self._value_bucket_id(product_metadata)
                business_value_labels.append(business_value)
                value_masks.append(1.0 if is_purchase else 0.0)
                value_bucket_ids.append(value_bucket)
                if is_purchase:
                    value_records.append(
                        {
                            "business_value": business_value,
                            "value_bucket": value_bucket,
                        }
                    )

                relevance_labels.append(
                    self._training_relevance_label(
                        sample,
                        product_metadata,
                        action,
                    )
                )
                group_key = self._training_pairwise_group_key(sample)
                if group_key not in group_mapping:
                    group_mapping[group_key] = len(group_mapping)
                group_ids.append(group_mapping[group_key])

                impression_id = sample.get("impression_id") or context.get(
                    "impression_id"
                )
                is_slate_sample = bool(
                    training_sample_source
                    in {"recommendation_impressions", "feature_lake_pit"}
                    and impression_id
                )
                if is_slate_sample:
                    listwise_group_key = f"impression:{impression_id}"
                else:
                    listwise_group_key = f"row:{len(listwise_group_ids)}"
                if listwise_group_key not in listwise_group_mapping:
                    listwise_group_mapping[listwise_group_key] = len(
                        listwise_group_mapping
                    )
                listwise_group_ids.append(listwise_group_mapping[listwise_group_key])
                slate_sample_flags.append(is_slate_sample)

            if not features:
                return torch.empty(0), {}

            self._fit_value_transform(value_records)
            value_labels = [
                self._transform_business_value(value, bucket_id) if mask > 0.0 else 0.0
                for value, bucket_id, mask in zip(
                    business_value_labels,
                    value_bucket_ids,
                    value_masks,
                )
            ]

            features_tensor = torch.tensor(np.vstack(features), dtype=torch.float32).to(
                self.device
            )
            labels = {
                "ctr": torch.tensor(ctr_labels, dtype=torch.float32)
                .to(self.device)
                .unsqueeze(1),
                "cvr": torch.tensor(cvr_labels, dtype=torch.float32)
                .to(self.device)
                .unsqueeze(1),
                "ctcvr": torch.tensor(ctcvr_labels, dtype=torch.float32)
                .to(self.device)
                .unsqueeze(1),
                "cvr_mask": torch.tensor(cvr_masks, dtype=torch.float32)
                .to(self.device)
                .unsqueeze(1),
                "value": torch.tensor(value_labels, dtype=torch.float32)
                .to(self.device)
                .unsqueeze(1),
                "business_value": torch.tensor(
                    business_value_labels, dtype=torch.float32
                )
                .to(self.device)
                .unsqueeze(1),
                "value_mask": torch.tensor(value_masks, dtype=torch.float32)
                .to(self.device)
                .unsqueeze(1),
                "value_bucket": torch.tensor(value_bucket_ids, dtype=torch.long).to(
                    self.device
                ),
                "ranking_relevance": torch.tensor(relevance_labels, dtype=torch.float32)
                .to(self.device)
                .unsqueeze(1),
                "pairwise_group": torch.tensor(group_ids, dtype=torch.long).to(
                    self.device
                ),
                "ltr_group": torch.tensor(listwise_group_ids, dtype=torch.long).to(
                    self.device
                ),
                "ltr_is_slate_sample": torch.tensor(
                    slate_sample_flags, dtype=torch.bool
                ).to(self.device),
            }
            labels["gmv"] = labels["value"]

            return features_tensor, labels

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return torch.empty(0), {}

    def _training_user_features(
        self,
        sample: Dict[str, Any],
        user_features_map: Dict[str, Any],
    ) -> UserFeatures:
        user_id = str(sample.get("user_id") or "unknown")
        raw_features = sample.get("user_features")
        if raw_features is None:
            raw_features = (sample.get("context") or {}).get("user_features")
        if raw_features is None:
            raw_features = user_features_map.get(user_id)
        if isinstance(raw_features, UserFeatures):
            return raw_features
        if isinstance(raw_features, dict):
            payload = dict(raw_features)
            payload.setdefault("user_id", user_id)
            try:
                return UserFeatures(**payload)
            except Exception as exc:
                logger.warning("Invalid user features for ranking training: %s", exc)
        return UserFeatures(
            user_id=user_id,
            last_active=self._training_as_of_timestamp(sample),
        )

    @staticmethod
    def _training_as_of_timestamp(sample: Dict[str, Any]) -> float:
        """Resolve a deterministic feature cutoff for one training row."""
        for key in ("as_of_ts", "event_time", "occurred_at", "timestamp"):
            value = sample.get(key)
            if value is None:
                continue
            if hasattr(value, "timestamp"):
                return float(value.timestamp())
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        # Missing timestamps must remain deterministic rather than leak wall-clock
        # time into a training vector. Ingest contracts should make this fallback rare.
        return 0.0

    def _training_product_metadata(
        self,
        sample: Dict[str, Any],
        product_metadata_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        product_id = str(sample.get("product_id") or "")
        context = sample.get("context") or {}
        metadata = dict(product_metadata_map.get(product_id) or {})
        embedded_metadata = sample.get("product_metadata") or context.get(
            "product_metadata"
        )
        if isinstance(embedded_metadata, dict):
            metadata.update(embedded_metadata)

        if "category" not in metadata:
            metadata["category"] = (
                context.get("product_category") or context.get("category") or "General"
            )
        if "price" not in metadata:
            metadata["price"] = self._first_float(
                sample.get("price"),
                context.get("price"),
                default=1.0,
            )
        metadata.setdefault("rating", 3.0)
        metadata.setdefault("num_reviews", 1)
        metadata.setdefault("in_stock", True)
        metadata.setdefault(
            "created_at",
            self._training_as_of_timestamp(sample),
        )
        metadata.setdefault("tags", [])
        metadata.setdefault("brand", context.get("brand", "Unknown"))
        return metadata

    def _training_candidate(self, sample: Dict[str, Any]) -> CandidateProduct:
        product_id = str(sample.get("product_id") or "")
        context = sample.get("context") or {}
        candidate_payload = sample.get("candidate") or sample.get("candidate_product")
        if isinstance(candidate_payload, CandidateProduct):
            return candidate_payload
        if isinstance(candidate_payload, dict):
            payload = dict(candidate_payload)
            payload.setdefault("product_id", product_id)
            payload.setdefault("source", "training_interaction")
            try:
                return CandidateProduct(**payload)
            except Exception as exc:
                logger.warning(
                    "Invalid candidate payload for ranking training: %s", exc
                )

        candidate_scores = (
            sample.get("candidate_scores") or context.get("candidate_scores") or {}
        )
        if not isinstance(candidate_scores, dict):
            candidate_scores = {}
        collaborative_score = self._optional_float(
            sample.get("collaborative_score"),
            candidate_scores.get("collaborative_score"),
            candidate_scores.get("cf_score"),
        )
        content_score = self._optional_float(
            sample.get("content_similarity_score"),
            candidate_scores.get("content_similarity_score"),
            candidate_scores.get("content_score"),
        )
        popularity_score = self._optional_float(
            sample.get("popularity_score"),
            candidate_scores.get("popularity_score"),
        )
        combined_score = self._first_float(
            sample.get("combined_score"),
            candidate_scores.get("combined_score"),
            default=0.0,
        )
        return CandidateProduct(
            product_id=product_id,
            collaborative_score=collaborative_score,
            content_similarity_score=content_score,
            popularity_score=popularity_score,
            combined_score=combined_score,
            source=str(
                sample.get("source")
                or context.get("candidate_source")
                or "training_interaction"
            ),
        )

    def _training_gmv_label(
        self,
        sample: Dict[str, Any],
        product_metadata: Dict[str, Any],
        action: str,
    ) -> float:
        return self._training_business_value_label(sample, product_metadata, action)

    def _training_clicked_label(self, sample: Dict[str, Any], action: str) -> bool:
        context = sample.get("context") or {}
        if bool(context.get("attributed_click")) or bool(
            context.get("attributed_purchase")
        ):
            return True
        return str(action or "").lower() in {"click", "add_to_cart", "purchase"}

    def _training_business_value_label(
        self,
        sample: Dict[str, Any],
        product_metadata: Dict[str, Any],
        action: str,
    ) -> float:
        context = sample.get("context") or {}
        if action != "purchase" and not bool(context.get("attributed_purchase")):
            return 0.0

        label_keys = list(
            getattr(
                self.config,
                "value_label_preference",
                [
                    "margin",
                    "profit",
                    "gross_margin",
                    "value",
                    "gmv",
                    "purchase_value",
                    "price",
                ],
            )
            or []
        )
        for key in label_keys:
            value = self._optional_float(
                sample.get(key),
                context.get(key),
                product_metadata.get(key),
            )
            if value is not None:
                return max(0.0, float(value))
        return 0.0

    def _value_bucket_ids_for_candidates(
        self,
        valid_candidates: List[Tuple[CandidateProduct, Dict[str, Any]]],
    ) -> List[Optional[int]]:
        return [
            self._value_bucket_id(product_metadata, create=False)
            for _, product_metadata in valid_candidates
        ]

    def _value_bucket_id(
        self,
        product_metadata: Dict[str, Any],
        *,
        create: bool = True,
    ) -> Optional[int]:
        key = self._value_bucket_key(product_metadata)
        bucket_id = self.value_bucket_mapping.get(key)
        if bucket_id is None and create:
            bucket_id = len(self.value_bucket_mapping)
            self.value_bucket_mapping[key] = bucket_id
        return bucket_id

    def _value_bucket_key(self, product_metadata: Dict[str, Any]) -> str:
        category = str(product_metadata.get("category") or "General").strip().lower()
        price = self._first_float(product_metadata.get("price"), default=0.0)
        thresholds = sorted(
            float(value)
            for value in getattr(self.config, "value_price_buckets", [50.0, 200.0])
        )
        price_bucket = 0
        for threshold in thresholds:
            if price >= threshold:
                price_bucket += 1
        return f"{category}:{price_bucket}"

    def _fit_value_transform(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            self.value_transform_stats = self._default_value_transform_stats()
            return

        by_bucket: Dict[int, List[float]] = {}
        all_values: List[float] = []
        for record in records:
            value = max(
                0.0, self._first_float(record.get("business_value"), default=0.0)
            )
            bucket_id = int(record.get("value_bucket", -1))
            all_values.append(value)
            by_bucket.setdefault(bucket_id, []).append(value)

        stats = {
            "global": self._compute_value_stats(all_values),
            "buckets": {},
        }
        min_bucket_purchases = max(
            1,
            int(getattr(self.config, "value_min_bucket_purchases", 20) or 20),
        )
        for bucket_id, values in by_bucket.items():
            if len(values) >= min_bucket_purchases:
                stats["buckets"][str(bucket_id)] = self._compute_value_stats(values)
        self.value_transform_stats = stats

    def _compute_value_stats(self, values: List[float]) -> Dict[str, Any]:
        if not values:
            return self._default_value_transform_stats()["global"]
        raw = np.asarray([max(0.0, float(value)) for value in values], dtype=np.float64)
        quantile = float(getattr(self.config, "value_clip_quantile", 0.99) or 0.99)
        clip = float(np.quantile(raw, quantile)) if raw.size else 0.0
        if not np.isfinite(clip) or clip <= 0.0:
            clip = float(np.max(raw)) if raw.size else 0.0
        clipped = np.minimum(raw, clip)
        logged = np.log1p(clipped)
        mean = float(np.mean(logged)) if logged.size else 0.0
        std = float(np.std(logged)) if logged.size else 1.0
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        return {
            "count": int(raw.size),
            "clip": clip,
            "mean": mean,
            "std": std,
        }

    def _stats_for_value_bucket(
        self, bucket_id: Optional[int] = None
    ) -> Dict[str, Any]:
        stats = self.value_transform_stats or self._default_value_transform_stats()
        if bucket_id is not None:
            bucket_stats = (stats.get("buckets") or {}).get(str(int(bucket_id)))
            if bucket_stats:
                return bucket_stats
        return stats.get("global") or self._default_value_transform_stats()["global"]

    def _transform_business_value(
        self, value: float, bucket_id: Optional[int]
    ) -> float:
        stats = self._stats_for_value_bucket(bucket_id)
        clip = stats.get("clip")
        raw = max(0.0, float(value))
        if clip is not None:
            raw = min(raw, max(0.0, float(clip)))
        logged = float(np.log1p(raw))
        return (logged - float(stats.get("mean", 0.0))) / float(
            stats.get("std", 1.0) or 1.0
        )

    def _inverse_transform_business_value(
        self,
        normalized_value: float,
        bucket_id: Optional[int] = None,
    ) -> float:
        stats = self._stats_for_value_bucket(bucket_id)
        if int(stats.get("count", 0) or 0) <= 0:
            return max(0.0, float(normalized_value))
        logged = float(normalized_value) * float(stats.get("std", 1.0) or 1.0) + float(
            stats.get("mean", 0.0)
        )
        value = max(0.0, float(np.expm1(logged)))
        clip = stats.get("clip")
        if clip is not None:
            value = min(value, max(0.0, float(clip)))
        return value

    def _training_relevance_label(
        self,
        sample: Dict[str, Any],
        product_metadata: Dict[str, Any],
        action: str,
    ) -> float:
        """Create an ordinal ranking relevance target for pairwise LTR."""
        normalized_action = str(action or "").lower()
        if normalized_action == "purchase":
            gmv_value = self._training_gmv_label(
                sample,
                product_metadata,
                normalized_action,
            )
            return 4.0 + float(np.log1p(max(gmv_value, 0.0)))
        if normalized_action == "add_to_cart":
            return 3.0
        if normalized_action == "click":
            return 2.0
        if normalized_action == "view":
            return 1.0
        return 0.0

    def _training_pairwise_group_key(self, sample: Dict[str, Any]) -> str:
        """Group samples for pairwise ranking by impression, request, session, then user/time."""
        context = sample.get("context") or {}
        impression_id = sample.get("impression_id") or context.get("impression_id")
        if impression_id:
            return f"impression:{impression_id}"

        request_id = (
            sample.get("request_id")
            or context.get("request_id")
            or context.get("recommendation_request_id")
        )
        if request_id:
            return f"request:{request_id}"

        session_id = sample.get("session_id") or context.get("session_id")
        if session_id:
            return f"session:{session_id}"

        user_id = str(sample.get("user_id") or "unknown")
        timestamp = self._optional_float(
            sample.get("as_of_ts"),
            sample.get("occurred_at"),
            sample.get("timestamp"),
        )
        if timestamp is None:
            timestamp = time.time()
        time_bucket = int(max(timestamp, 0.0) // 1800)
        return f"user_time:{user_id}:{time_bucket}"

    @staticmethod
    def _optional_float(*values: Any) -> Optional[float]:
        for value in values:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @classmethod
    def _first_float(cls, *values: Any, default: float) -> float:
        value = cls._optional_float(*values)
        return float(default if value is None else value)

    def _compute_loss(
        self, predictions: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute multi-objective loss."""
        try:
            # CTR loss (binary cross-entropy)
            ctr_loss = nn.BCELoss()(predictions["ctr"], labels["ctr"])

            # CVR loss is conditional on click; unclicked rows do not contribute.
            cvr_loss = self._masked_binary_cross_entropy(
                predictions["cvr"],
                labels["cvr"],
                labels.get("cvr_mask"),
            )
            ctcvr_predictions = predictions.get(
                "ctcvr",
                predictions["ctr"] * predictions["cvr"],
            )
            ctcvr_targets = labels.get("ctcvr", labels["cvr"])
            ctcvr_loss = self._binary_cross_entropy_with_pos_weight(
                ctcvr_predictions,
                ctcvr_targets,
                getattr(self.config, "ctcvr_pos_weight", None),
            )

            # Value loss is purchase-conditional on normalized business value.
            gmv_loss = self._masked_value_loss(
                predictions["gmv"],
                labels.get("value", labels.get("gmv")),
                labels.get("value_mask"),
            )

            # Combined loss with weights
            total_loss = (
                self.config.ctr_weight * ctr_loss
                + float(
                    getattr(self.config, "direct_cvr_weight", self.config.cvr_weight)
                )
                * cvr_loss
                + float(getattr(self.config, "ctcvr_weight", 0.0)) * ctcvr_loss
                + self.config.gmv_weight * gmv_loss
            )

            if (
                getattr(self.config, "ltr_pairwise_enabled", False)
                and getattr(self.config, "ltr_pairwise_weight", 0.0) > 0
            ):
                pairwise_loss = self._compute_pairwise_ltr_loss(
                    predictions["ranking_score"],
                    labels.get("ranking_relevance"),
                    labels.get("pairwise_group"),
                )
                total_loss = (
                    total_loss
                    + float(getattr(self.config, "ltr_pairwise_weight", 0.0))
                    * pairwise_loss
                )

            if (
                getattr(self.config, "ltr_listwise_enabled", False)
                and getattr(self.config, "ltr_listwise_weight", 0.0) > 0
            ):
                listwise_loss = self._compute_listwise_ltr_loss(
                    predictions["ranking_score"],
                    labels.get("ranking_relevance"),
                    labels.get("ltr_group"),
                    labels.get("ltr_is_slate_sample"),
                )
                total_loss = (
                    total_loss
                    + float(getattr(self.config, "ltr_listwise_weight", 0.0))
                    * listwise_loss
                )

            return total_loss

        except Exception as e:
            logger.error(f"Error computing loss: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _binary_cross_entropy_with_pos_weight(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pos_weight: Optional[float],
    ) -> torch.Tensor:
        losses = torch.nn.functional.binary_cross_entropy(
            predictions,
            targets,
            reduction="none",
        )
        if pos_weight is not None:
            weight = torch.where(
                targets > 0.0,
                torch.as_tensor(
                    float(pos_weight), device=losses.device, dtype=losses.dtype
                ),
                torch.ones((), device=losses.device, dtype=losses.dtype),
            )
            losses = losses * weight
        return losses.mean()

    def _masked_binary_cross_entropy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        losses = torch.nn.functional.binary_cross_entropy(
            predictions,
            targets,
            reduction="none",
        )
        if mask is None:
            return losses.mean()
        mask = mask.to(device=losses.device, dtype=losses.dtype)
        denominator = mask.sum()
        if float(denominator.detach().cpu().item()) <= 0.0:
            return losses.sum() * 0.0
        return (losses * mask).sum() / denominator

    def _masked_value_loss(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if targets is None:
            return predictions.sum() * 0.0
        value_loss = str(getattr(self.config, "value_loss", "huber") or "huber").lower()
        if value_loss == "mse":
            losses = torch.nn.functional.mse_loss(
                predictions,
                targets,
                reduction="none",
            )
        else:
            losses = torch.nn.functional.smooth_l1_loss(
                predictions,
                targets,
                reduction="none",
            )
        if mask is None:
            return losses.mean()
        mask = mask.to(device=losses.device, dtype=losses.dtype)
        denominator = mask.sum()
        if float(denominator.detach().cpu().item()) <= 0.0:
            return losses.sum() * 0.0
        return (losses * mask).sum() / denominator

    def _compute_listwise_ltr_loss(
        self,
        ranking_scores: torch.Tensor,
        relevance: Optional[torch.Tensor],
        group_ids: Optional[torch.Tensor],
        slate_sample_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute ListNet softmax cross-entropy over complete impression slates."""
        scores = ranking_scores.reshape(-1)
        if (
            relevance is None
            or group_ids is None
            or slate_sample_mask is None
            or scores.numel() < 2
        ):
            return scores.sum() * 0.0

        relevance = relevance.reshape(-1)
        group_ids = group_ids.reshape(-1)
        slate_sample_mask = slate_sample_mask.reshape(-1).bool()
        if (
            relevance.numel() != scores.numel()
            or group_ids.numel() != scores.numel()
            or slate_sample_mask.numel() != scores.numel()
        ):
            return scores.sum() * 0.0

        min_group_size = int(
            max(2, getattr(self.config, "ltr_listwise_min_group_size", 2) or 2)
        )
        min_gap = float(max(0.0, getattr(self.config, "ltr_min_relevance_gap", 0.0)))
        losses: List[torch.Tensor] = []
        candidate_group_ids = torch.unique(group_ids[slate_sample_mask], sorted=True)
        for group_id in candidate_group_ids:
            group_mask = (group_ids == group_id) & slate_sample_mask
            if int(group_mask.sum().item()) < min_group_size:
                continue

            group_relevance = relevance[group_mask]
            relevance_gap = group_relevance.max() - group_relevance.min()
            if float(relevance_gap.item()) < min_gap:
                continue

            group_scores = scores[group_mask]
            pred_log_probs = torch.nn.functional.log_softmax(group_scores, dim=0)
            target_probs = torch.nn.functional.softmax(
                group_relevance,
                dim=0,
            ).detach()
            losses.append(-(target_probs * pred_log_probs).sum())

        if not losses:
            return scores.sum() * 0.0
        return torch.stack(losses).mean()

    def _compute_pairwise_ltr_loss(
        self,
        ranking_scores: torch.Tensor,
        relevance: Optional[torch.Tensor],
        group_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute RankNet loss over higher/lower relevance pairs in each group."""
        scores = ranking_scores.reshape(-1)
        if relevance is None or group_ids is None or scores.numel() < 2:
            return scores.sum() * 0.0

        pos_indices, neg_indices = self._build_pairwise_ltr_pairs(
            relevance.reshape(-1),
            group_ids.reshape(-1),
        )
        if pos_indices.numel() == 0:
            return scores.sum() * 0.0

        score_diff = scores[pos_indices] - scores[neg_indices]
        return torch.nn.functional.softplus(-score_diff).mean()

    def _build_pairwise_ltr_pairs(
        self,
        relevance: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build deterministic capped positive/negative index pairs per group."""
        relevance = relevance.reshape(-1)
        group_ids = group_ids.reshape(-1)
        device = relevance.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        if relevance.numel() != group_ids.numel() or relevance.numel() < 2:
            return empty, empty

        max_pairs_per_group = int(
            max(0, getattr(self.config, "ltr_max_pairs_per_group", 0) or 0)
        )
        if max_pairs_per_group <= 0:
            return empty, empty
        min_gap = float(max(0.0, getattr(self.config, "ltr_min_relevance_gap", 0.0)))

        positive_chunks: List[torch.Tensor] = []
        negative_chunks: List[torch.Tensor] = []
        for group_id in torch.unique(group_ids, sorted=True):
            group_indices = torch.nonzero(
                group_ids == group_id,
                as_tuple=False,
            ).flatten()
            if group_indices.numel() < 2:
                continue

            group_relevance = relevance[group_indices]
            order = torch.argsort(group_relevance, descending=True, stable=True)
            sorted_indices = group_indices[order]
            sorted_relevance = group_relevance[order]
            pairs_for_group = 0

            for high_position in range(sorted_indices.numel() - 1):
                remaining = max_pairs_per_group - pairs_for_group
                if remaining <= 0:
                    break
                relevance_gap = (
                    sorted_relevance[high_position]
                    - sorted_relevance[high_position + 1 :]
                )
                valid_offsets = torch.nonzero(
                    relevance_gap >= min_gap,
                    as_tuple=False,
                ).flatten()
                if valid_offsets.numel() == 0:
                    continue

                valid_offsets = valid_offsets[:remaining]
                lower_indices = sorted_indices[high_position + 1 :][valid_offsets]
                higher_indices = sorted_indices[high_position].repeat(
                    lower_indices.numel()
                )
                positive_chunks.append(higher_indices)
                negative_chunks.append(lower_indices)
                pairs_for_group += int(lower_indices.numel())

        if not positive_chunks:
            return empty, empty
        return torch.cat(positive_chunks), torch.cat(negative_chunks)

    async def save_model(self, model_path: str):
        """Save the trained model to disk."""
        if self.model and self.is_trained:
            target = Path(model_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            state_dict = self.model.state_dict()
            if getattr(self.config, "din_enabled", False):
                state_dict = {
                    key: value
                    for key, value in state_dict.items()
                    if key != "din.item_embedding.weight"
                }
            checkpoint = {
                "model_state_dict": state_dict,
                "config": {
                    "architecture": self.model.architecture,
                    "cross_layers": self.model.cross_layers,
                    "low_rank_dim": self.model.low_rank_dim,
                    "hidden_dims": self.model.hidden_dims,
                    "input_dim": self.feature_extractor.total_feature_dim,
                    "feature_schema_version": self.feature_schema_version,
                    "feature_definition_version": (
                        RANKING_LTR_DIN_FEATURE_DEFINITION_VERSION
                        if getattr(self.config, "din_enabled", False)
                        else RANKING_LTR_FEATURE_DEFINITION_VERSION
                    ),
                    "label_definition_version": RANKING_LABEL_DEFINITION_VERSION,
                    "feature_assembler_version": self.feature_assembler.version,
                    "ranking_objective_version": self.ranking_objective_version,
                    "value_transform_stats": self.value_transform_stats,
                    "value_bucket_mapping": self.value_bucket_mapping,
                    "din_enabled": bool(getattr(self.config, "din_enabled", False)),
                    "din_sidecar_metadata": self.din_sidecar_metadata,
                    "candidate_sidecar_sha256": self.candidate_sidecar_sha256,
                    "candidate_sidecar_model_version": (
                        self.candidate_sidecar_model_version
                    ),
                },
            }
            file_descriptor, temporary_path = tempfile.mkstemp(
                prefix=target.name + ".", suffix=".tmp", dir=target.parent
            )
            try:
                with os.fdopen(file_descriptor, "wb") as handle:
                    torch.save(checkpoint, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, target)
            except Exception:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
                raise
            self.loaded_model_path = str(target)
            self.loaded_checkpoint_mtime = target.stat().st_mtime
            logger.info(f"Model saved to {target}")
            return str(target)
        raise RuntimeError("cannot save an untrained ranking model")

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        stats = {
            "is_trained": self.is_trained,
            "model_version": self.model_version,
            "last_training_time": self.last_training_time,
            "loaded_model_path": self.loaded_model_path,
            "loaded_checkpoint_mtime": self.loaded_checkpoint_mtime,
            "checkpoint_reload_count": self.checkpoint_reload_count,
            "device": str(self.device),
            "inference_stats": self.inference_stats.copy(),
            "model_parameters": sum(p.numel() for p in self.model.parameters())
            if self.model
            else 0,
            "feature_dim": self.feature_extractor.total_feature_dim,
            "untrained_fallback_count": self.untrained_fallback_count,
            "architecture": getattr(self.model, "architecture", None)
            if self.model
            else None,
            "low_rank_dim": getattr(self.model, "low_rank_dim", None)
            if self.model
            else None,
        }
        stats.update(self._torch_compile_status())
        return stats

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the ranking model."""
        try:
            status = {
                "status": "healthy" if self.model else "unhealthy",
                "model_loaded": self.model is not None,
                "is_trained": self.is_trained,
                "loaded_model_path": self.loaded_model_path,
                "loaded_checkpoint_mtime": self.loaded_checkpoint_mtime,
                "checkpoint_reload_count": self.checkpoint_reload_count,
                "device": str(self.device),
                "feature_dim": self.feature_extractor.total_feature_dim,
                "architecture": getattr(self.model, "architecture", None)
                if self.model
                else None,
                "low_rank_dim": getattr(self.model, "low_rank_dim", None)
                if self.model
                else None,
                "torch_inference_available": self.torch_inference_available,
                "untrained_fallback_count": self.untrained_fallback_count,
            }
            status.update(self._torch_compile_status())

            # Test inference if model is loaded
            if self.model:
                try:
                    dummy_features = torch.randn(
                        1, self.feature_extractor.total_feature_dim
                    ).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(dummy_features)

                    status["inference_test_passed"] = True
                    status["output_keys"] = list(outputs.keys())

                except Exception as test_error:
                    status["inference_test_passed"] = False
                    status["test_error"] = str(test_error)

            return status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e), "model_loaded": False}
