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
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from pathlib import Path
import pickle
import threading
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score, roc_auc_score

# Local imports
from models import (
    CandidateProduct,
    ProductRecommendation,
    UserFeatures,
    RankingFeatures,
)
from config import RankingConfig
from dcn import (
    DeepAndCrossNetwork,
    LowRankDeepAndCrossNetwork,
    RANKING_ARCHITECTURES,
    normalize_architecture,
)

logger = logging.getLogger(__name__)

RANKING_FEATURE_SCHEMA_VERSION = "ranking_v1_29"
RANKING_TRAINING_DATA_SOURCE = "interaction_events_online_equivalent_features"


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
    ):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
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
            prev_dim = input_dim

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
                input_dim,
                hidden_dims,
                hidden_dims[-1],
                cross_layers=self.cross_layers,
                dropout=config.dropout_rate,
                use_batch_norm=False,
            )
        else:
            self.shared_layers = LowRankDeepAndCrossNetwork(
                input_dim,
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

    def forward(self, x):
        """Forward pass through the model."""
        # Shared representation
        shared_features = self.shared_layers(x)

        # Task-specific predictions
        ctr_pred = torch.sigmoid(
            self._collapse_scalar_head(self.ctr_tower(shared_features))
        )
        cvr_pred = torch.sigmoid(
            self._collapse_scalar_head(self.cvr_tower(shared_features))
        )
        gmv_pred = torch.relu(
            self._collapse_scalar_head(self.gmv_tower(shared_features))
        )

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
            "gmv": gmv_pred,
            "ranking_score": ranking_score,
        }


class FeatureExtractor:
    """
    Feature extractor that converts raw data into model-ready features.
    """

    def __init__(self):
        self.user_scaler = StandardScaler()
        self.context_scaler = StandardScaler()
        self.is_fitted = False

        # Feature dimensions
        self.user_feature_dim = 10
        self.product_feature_dim = 8
        self.context_feature_dim = 6
        self.candidate_feature_dim = 4

        self.total_feature_dim = (
            self.user_feature_dim
            + self.product_feature_dim
            + self.context_feature_dim
            + self.candidate_feature_dim
        )

    def extract_user_features(
        self,
        user_features: UserFeatures,
        current_time: Optional[float] = None,
    ) -> np.ndarray:
        """Extract numerical features from user profile."""
        current_time = current_time or time.time()
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
        current_time = current_time or time.time()
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
        current_time = current_time or time.time()
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

    def create_ranking_features(
        self,
        user_features: UserFeatures,
        product_metadata: Dict[str, Any],
        context: Dict[str, Any],
        candidate: CandidateProduct,
    ) -> np.ndarray:
        """Create complete feature vector for ranking."""
        try:
            # Extract individual feature groups
            current_time = time.time()
            user_feats = self.extract_user_features(user_features, current_time)
            product_feats = self.extract_product_features(
                product_metadata, current_time
            )
            context_feats = self.extract_context_features(context, current_time)
            candidate_feats = self.extract_candidate_features(candidate)

            # Concatenate all features
            combined_features = np.concatenate(
                [user_feats, product_feats, context_feats, candidate_feats]
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

    def __init__(self, config: RankingConfig):
        self.config = config
        self.feature_extractor = FeatureExtractor()

        # Model components
        self.model: Optional[MultiObjectiveRankingModel] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training state
        self.is_trained = False
        self.model_version = "1.0.0"
        self.last_training_time = 0
        self.feature_schema_version = RANKING_FEATURE_SCHEMA_VERSION
        self.training_data_source = RANKING_TRAINING_DATA_SOURCE

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
        self.enable_profiling_logs = False
        self.profiling_log_min_duration_ms = 250.0
        self._product_feature_cache_max_size = max(
            0,
            getattr(config, "product_feature_cache_size", 50000),
        )
        self._product_feature_cache: OrderedDict[
            Tuple[str, Tuple[Any, ...]],
            Tuple[np.ndarray, float],
        ] = OrderedDict()
        self._product_feature_cache_lock = threading.RLock()

        logger.info(f"RankingModel initialized on device: {self.device}")

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

    def _build_model_instance(
        self,
        *,
        architecture: Optional[str] = None,
        hidden_dims: Optional[List[int]] = None,
        cross_layers: Optional[int] = None,
        low_rank_dim: Optional[int] = None,
    ) -> Tuple[MultiObjectiveRankingModel, optim.Optimizer]:
        input_dim = self.feature_extractor.total_feature_dim
        model = MultiObjectiveRankingModel(
            input_dim,
            self.config,
            architecture=architecture,
            hidden_dims=hidden_dims,
            cross_layers=cross_layers,
            low_rank_dim=low_rank_dim,
        ).to(self.device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5,
        )
        return model, optimizer

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
            current_value = current_state.get(key)
            if current_value is not None and tuple(current_value.shape) == tuple(
                value.shape
            ):
                compatible[key] = value
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

    async def load_model(self, model_path: str = None):
        """Load or initialize the ranking model."""
        try:
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
                self.is_trained = True
                self.loaded_checkpoint_mtime = checkpoint_path.stat().st_mtime
            else:
                logger.info("Initializing new model")
                next_model, next_optimizer = self._build_model_instance(
                    architecture=getattr(self.config, "architecture", "dcn"),
                )
                next_model.eval()
                self.model = next_model
                self.optimizer = next_optimizer
                self.is_trained = False
                self.loaded_checkpoint_mtime = 0.0

            # Set to evaluation mode initially
            self.loaded_model_path = resolved_model_path

            logger.info("Ranking model loaded successfully")

        except Exception as e:
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
        user_dim = self.feature_extractor.user_feature_dim
        product_dim = self.feature_extractor.product_feature_dim
        context_dim = self.feature_extractor.context_feature_dim
        candidate_dim = self.feature_extractor.candidate_feature_dim
        total_dim = self.feature_extractor.total_feature_dim
        product_start = user_dim
        context_start = product_start + product_dim
        candidate_start = context_start + context_dim
        total_candidates = sum(
            len(request.get("candidates") or []) for request in requests
        )
        if total_candidates <= 0:
            return None, [], 0.0

        feature_matrix = np.empty((total_candidates, total_dim), dtype=np.float32)
        prepared_requests: List[Dict[str, Any]] = []
        row_count = 0

        for request_offset, request in enumerate(requests):
            request_started = time.perf_counter()
            current_time = time.time()
            candidates = request.get("candidates") or []
            user_features = request["user_features"]
            context = request.get("context") or {}
            product_metadata_map = request.get("product_metadata_map") or {}
            valid_candidates: List[Tuple[CandidateProduct, Dict[str, Any]]] = []
            row_start = row_count
            user_feats = self.feature_extractor.extract_user_features(
                user_features,
                current_time,
            )
            context_feats = self.feature_extractor.extract_context_features(
                context,
                current_time,
            )
            product_rows: List[np.ndarray] = []
            candidate_rows: List[Tuple[float, float, float, float]] = []

            for candidate in candidates:
                try:
                    product_id = str(_candidate_get(candidate, "product_id", ""))
                    product_metadata = product_metadata_map.get(
                        product_id
                    ) or self._build_product_metadata(candidate, current_time)
                    product_feats = self._get_product_feature_vector(
                        product_id,
                        product_metadata,
                        current_time,
                    )
                    candidate_rows.append(
                        (
                            _candidate_get(candidate, "collaborative_score") or 0.0,
                            _candidate_get(candidate, "content_similarity_score")
                            or 0.0,
                            _candidate_get(candidate, "popularity_score") or 0.0,
                            _candidate_get(candidate, "combined_score") or 0.0,
                        )
                    )
                    product_rows.append(product_feats)
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
                feature_matrix[row_start:row_end, :user_dim] = user_feats
                feature_matrix[
                    row_start:row_end, product_start:context_start
                ] = np.asarray(product_rows, dtype=np.float32)
                feature_matrix[
                    row_start:row_end, context_start:candidate_start
                ] = context_feats
                feature_matrix[
                    row_start:row_end,
                    candidate_start : candidate_start + candidate_dim,
                ] = np.asarray(candidate_rows, dtype=np.float32)

            prepared_requests.append(
                {
                    "index": request.get("index", request_offset),
                    "k": request.get("k"),
                    "batch_wait_ms": request.get("batch_wait_ms", 0.0),
                    "valid_candidates": valid_candidates,
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
        return feature_matrix, prepared_requests, feature_extraction_ms

    def run_inference_batch(
        self,
        feature_matrix: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
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
            predictions = model(features_tensor)
            model_forward_ms = round(
                (time.perf_counter() - model_stage_started) * 1000, 2
            )

        prediction_arrays = {
            key: value.detach().cpu().numpy().reshape(-1)
            for key, value in predictions.items()
        }
        return prediction_arrays, {
            "tensor_prep_ms": tensor_prep_ms,
            "model_forward_ms": model_forward_ms,
        }

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

        ranking_scores = np.asarray(predictions["ranking_score"]).reshape(-1)
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
            gmv_score = float(predictions["gmv"][i])
            ranking_score = float(ranking_scores[i])

            if self.config.enable_multi_objective:
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
        tags = product_metadata.get("tags", [])
        if isinstance(tags, (list, tuple, set)):
            normalized_tags = tuple(str(tag) for tag in tags)
        else:
            normalized_tags = (str(tags),)
        created_at = (
            "__fallback_now__"
            if product_metadata.get("_ranking_fallback_metadata") is True
            else product_metadata.get("created_at")
        )
        return (
            product_metadata.get("price", 1.0),
            product_metadata.get("rating", 3.0),
            product_metadata.get("num_reviews", 1),
            bool(product_metadata.get("in_stock", True)),
            created_at,
            normalized_tags,
            product_metadata.get("brand", ""),
        )

    def _get_product_feature_vector(
        self,
        product_id: str,
        product_metadata: Dict[str, Any],
        current_time: float,
    ) -> np.ndarray:
        """Return product features while preserving request-time freshness for age."""
        cache_key = (
            product_id,
            self._product_metadata_fingerprint(product_metadata),
        )
        with self._product_feature_cache_lock:
            cached = self._product_feature_cache.get(cache_key)
            if cached is not None:
                static_features, created_at = cached
                self._product_feature_cache.move_to_end(cache_key)
            else:
                static_features = None

        if cached is None:
            (
                static_features,
                created_at,
            ) = self.feature_extractor.extract_static_product_features(product_metadata)
            with self._product_feature_cache_lock:
                if self._product_feature_cache_max_size > 0:
                    self._product_feature_cache[cache_key] = (
                        static_features,
                        created_at,
                    )
                    if (
                        len(self._product_feature_cache)
                        > self._product_feature_cache_max_size
                    ):
                        self._product_feature_cache.popitem(last=False)
        elif product_metadata.get("_ranking_fallback_metadata") is True:
            created_at = current_time

        product_features = static_features.copy()
        product_features[4] = (current_time - created_at) / 86400
        return product_features

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
        if not self.model:
            logger.error("Model not loaded")
            return []

        if not candidates:
            logger.warning("No candidates to rank")
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

            predictions, inference_profile = self.run_inference_batch(feature_matrix)
            profile["tensor_prep_ms"] = inference_profile["tensor_prep_ms"]
            profile["model_forward_ms"] = inference_profile["model_forward_ms"]

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
            combined = _candidate_get(candidate, "combined_score") or (
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
        training_data: List[Dict[str, Any]],
        *,
        user_features_map: Optional[Dict[str, Any]] = None,
        product_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Train the ranking model on user interaction data."""
        try:
            saved_model_path = await asyncio.to_thread(
                self._train_model_sync,
                training_data,
                user_features_map or {},
                product_metadata_map or {},
            )
            if saved_model_path:
                await self.save_model(saved_model_path)

        except Exception as e:
            logger.error(f"Error training model: {e}")

    def _train_model_sync(
        self,
        training_data: List[Dict[str, Any]],
        user_features_map: Optional[Dict[str, Any]] = None,
        product_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[str]:
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
        features, labels = self._prepare_training_data(
            training_data,
            user_features_map=user_features_map,
            product_metadata_map=product_metadata_map,
        )

        if features.size(0) == 0:
            logger.warning("No valid training samples prepared")
            return None

        self.model.train()

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, features.size(0), self.config.batch_size):
                batch_features = features[i : i + self.config.batch_size]
                batch_labels = {
                    key: value[i : i + self.config.batch_size]
                    for key, value in labels.items()
                }

                self.optimizer.zero_grad()
                predictions = self.model(batch_features)
                loss = self._compute_loss(predictions, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

            if avg_loss < 0.01:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        self.model.eval()
        self.is_trained = True
        self.last_training_time = time.time()
        self.model_version = f"ranking-{int(self.last_training_time)}"

        logger.info("Model training completed")
        return self.loaded_model_path

    def _prepare_training_data(
        self,
        training_data: List[Dict[str, Any]],
        *,
        user_features_map: Optional[Dict[str, Any]] = None,
        product_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare training data from interaction logs."""
        try:
            user_features_map = user_features_map or {}
            product_metadata_map = product_metadata_map or {}
            features = []
            ctr_labels = []
            cvr_labels = []
            gmv_labels = []

            for sample in training_data:
                product_id = sample.get("product_id")
                if not product_id:
                    continue

                user_features = self._training_user_features(sample, user_features_map)
                product_metadata = self._training_product_metadata(
                    sample, product_metadata_map
                )
                context = dict(sample.get("context") or {})
                candidate = self._training_candidate(sample)
                feature_vector = self.feature_extractor.create_ranking_features(
                    user_features,
                    product_metadata,
                    context,
                    candidate,
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
                action = sample.get("action", "view")

                # CTR label (clicked or not)
                ctr_label = (
                    1.0 if action in ["click", "purchase", "add_to_cart"] else 0.0
                )
                ctr_labels.append(ctr_label)

                # CVR label (converted or not)
                cvr_label = 1.0 if action == "purchase" else 0.0
                cvr_labels.append(cvr_label)

                # GMV label (purchase value)
                gmv_value = self._training_gmv_label(sample, product_metadata, action)
                gmv_labels.append(gmv_value)

            if not features:
                return torch.empty(0), {}

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
                "gmv": torch.tensor(gmv_labels, dtype=torch.float32)
                .to(self.device)
                .unsqueeze(1),
            }

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
        return UserFeatures(user_id=user_id)

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
            sample.get("occurred_at") or sample.get("timestamp") or time.time(),
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
        if action != "purchase":
            return 0.0
        context = sample.get("context") or {}
        return max(
            0.0,
            self._first_float(
                sample.get("value"),
                sample.get("gmv"),
                sample.get("purchase_value"),
                context.get("value"),
                context.get("gmv"),
                context.get("purchase_value"),
                product_metadata.get("price"),
                default=0.0,
            ),
        )

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

            # CVR loss (binary cross-entropy)
            cvr_loss = nn.BCELoss()(predictions["cvr"], labels["cvr"])

            # GMV loss (MSE)
            gmv_loss = nn.MSELoss()(predictions["gmv"], labels["gmv"])

            # Combined loss with weights
            total_loss = (
                self.config.ctr_weight * ctr_loss
                + self.config.cvr_weight * cvr_loss
                + self.config.gmv_weight * gmv_loss
            )

            return total_loss

        except Exception as e:
            logger.error(f"Error computing loss: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    async def save_model(self, model_path: str):
        """Save the trained model to disk."""
        try:
            if self.model and self.is_trained:
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    "model_state_dict": self.model.state_dict(),
                    "config": {
                        "architecture": self.model.architecture,
                        "cross_layers": self.model.cross_layers,
                        "low_rank_dim": self.model.low_rank_dim,
                        "hidden_dims": self.model.hidden_dims,
                        "input_dim": self.feature_extractor.total_feature_dim,
                        "feature_schema_version": self.feature_schema_version,
                    },
                }
                torch.save(checkpoint, model_path)
                self.loaded_model_path = model_path
                self.loaded_checkpoint_mtime = Path(model_path).stat().st_mtime
                logger.info(f"Model saved to {model_path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
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
            "architecture": getattr(self.model, "architecture", None)
            if self.model
            else None,
            "low_rank_dim": getattr(self.model, "low_rank_dim", None)
            if self.model
            else None,
        }

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
            }

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
