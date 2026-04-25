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
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score, roc_auc_score

# Local imports
from models import (
    CandidateProduct, ProductRecommendation, UserFeatures, 
    RankingFeatures
)
from config import RankingConfig

logger = logging.getLogger(__name__)

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
    
    def __init__(self, input_dim: int, config: RankingConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # Shared bottom layers
        hidden_dims = config.hidden_dims
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
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
            nn.Linear(16, 2)
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
        ctr_pred = torch.sigmoid(self._collapse_scalar_head(self.ctr_tower(shared_features)))
        cvr_pred = torch.sigmoid(self._collapse_scalar_head(self.cvr_tower(shared_features)))
        gmv_pred = torch.relu(self._collapse_scalar_head(self.gmv_tower(shared_features)))
        
        # Combine for final ranking
        combined_features = torch.cat([shared_features, ctr_pred, cvr_pred, gmv_pred], dim=1)
        ranking_score = self._collapse_scalar_head(self.ranking_tower(combined_features))
        
        return {
            'ctr': ctr_pred,
            'cvr': cvr_pred, 
            'gmv': gmv_pred,
            'ranking_score': ranking_score
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
            self.user_feature_dim + 
            self.product_feature_dim + 
            self.context_feature_dim + 
            self.candidate_feature_dim
        )
    
    def extract_user_features(self, user_features: UserFeatures) -> np.ndarray:
        """Extract numerical features from user profile."""
        features = np.array([
            user_features.total_interactions / 1000,  # Normalize
            user_features.avg_session_length / 3600,  # Hours
            user_features.price_sensitivity,
            user_features.click_through_rate,
            user_features.conversion_rate,
            len(user_features.preferred_categories) / 10,  # Normalize
            (time.time() - user_features.last_active) / 86400,  # Days since active
            1.0 if user_features.total_interactions > 100 else 0.0,  # Heavy user
            1.0 if user_features.conversion_rate > 0.05 else 0.0,  # High converter
            1.0 if user_features.click_through_rate > 0.1 else 0.0,  # High engagement
        ], dtype=np.float32)
        
        return features
    
    def extract_product_features(self, product_metadata: Dict[str, Any]) -> np.ndarray:
        """Extract features from product metadata."""
        features = np.array([
            np.log1p(product_metadata.get('price', 1.0)),  # Log price
            product_metadata.get('rating', 3.0) / 5.0,  # Normalized rating
            np.log1p(product_metadata.get('num_reviews', 1)),  # Log review count
            1.0 if product_metadata.get('in_stock', True) else 0.0,  # Stock status
            (time.time() - product_metadata.get('created_at', time.time())) / 86400,  # Age in days
            len(product_metadata.get('tags', [])) / 10,  # Tag count normalized
            1.0 if product_metadata.get('price', 0) > 100 else 0.0,  # Premium product
            hash(product_metadata.get('brand', '')) % 100 / 100,  # Brand embedding (simple)
        ], dtype=np.float32)
        
        return features
    
    def extract_context_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract features from request context."""
        current_time = time.time()
        dt = time.localtime(current_time)
        
        features = np.array([
            dt.tm_hour / 24.0,  # Hour of day normalized
            dt.tm_wday / 6.0,   # Day of week normalized
            1.0 if dt.tm_wday >= 5 else 0.0,  # Weekend flag
            context.get('session_position', 1) / 20.0,  # Position in session
            1.0 if context.get('device') == 'mobile' else 0.0,  # Mobile device
            context.get('time_on_page', 0) / 300.0,  # Time on page (normalized to 5 min)
        ], dtype=np.float32)
        
        return features
    
    def extract_candidate_features(self, candidate: CandidateProduct) -> np.ndarray:
        """Extract features from candidate product."""
        features = np.array([
            candidate.collaborative_score or 0.0,
            candidate.content_similarity_score or 0.0,
            candidate.popularity_score or 0.0,
            candidate.combined_score or 0.0,
        ], dtype=np.float32)
        
        return features
    
    def create_ranking_features(
        self,
        user_features: UserFeatures,
        product_metadata: Dict[str, Any],
        context: Dict[str, Any],
        candidate: CandidateProduct
    ) -> np.ndarray:
        """Create complete feature vector for ranking."""
        try:
            # Extract individual feature groups
            user_feats = self.extract_user_features(user_features)
            product_feats = self.extract_product_features(product_metadata)
            context_feats = self.extract_context_features(context)
            candidate_feats = self.extract_candidate_features(candidate)
            
            # Concatenate all features
            combined_features = np.concatenate([
                user_feats,
                product_feats, 
                context_feats,
                candidate_feats
            ])
            
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training state
        self.is_trained = False
        self.model_version = "1.0.0"
        self.last_training_time = 0
        
        # Performance tracking
        self.training_history = []
        self.inference_stats = {
            'total_inferences': 0,
            'avg_inference_time': 0.0,
            'batch_inference_time': 0.0
        }
        self.loaded_model_path: Optional[str] = None
        self.loaded_checkpoint_mtime: float = 0.0
        self.checkpoint_reload_count = 0
        self._checkpoint_reload_lock = asyncio.Lock()
        self.torch_inference_available = True
        self.enable_profiling_logs = False
        self.profiling_log_min_duration_ms = 250.0
        
        logger.info(f"RankingModel initialized on device: {self.device}")
    
    async def load_model(self, model_path: str = None):
        """Load or initialize the ranking model."""
        try:
            resolved_model_path = model_path or self.loaded_model_path

            # Initialize model
            input_dim = self.feature_extractor.total_feature_dim
            self.model = MultiObjectiveRankingModel(input_dim, self.config).to(self.device)
            
            # Load pre-trained weights if available
            if resolved_model_path and Path(resolved_model_path).exists():
                logger.info(f"Loading model from {resolved_model_path}")
                state_dict = torch.load(resolved_model_path, map_location=self.device)
                load_result = self.model.load_state_dict(state_dict, strict=False)
                if load_result.missing_keys or load_result.unexpected_keys:
                    logger.warning(
                        "Ranking checkpoint partially loaded due to head-shape mismatch: "
                        f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
                    )
                self.is_trained = True
                self.loaded_checkpoint_mtime = Path(resolved_model_path).stat().st_mtime
            else:
                logger.info("Initializing new model")
                self.is_trained = False
                self.loaded_checkpoint_mtime = 0.0
            
            # Initialize optimizer
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-5
            )
            
            # Set to evaluation mode initially
            self.model.eval()
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
    ) -> Tuple[Optional[np.ndarray], List[Tuple[CandidateProduct, Dict[str, Any]]], float]:
        """Prepare a request's ranking feature matrix and candidate metadata."""
        feature_stage_started = time.perf_counter()
        feature_vectors = []
        valid_candidates: List[Tuple[CandidateProduct, Dict[str, Any]]] = []
        product_metadata_map = product_metadata_map or {}

        for candidate in candidates:
            try:
                product_metadata = product_metadata_map.get(candidate.product_id) or self._build_product_metadata(candidate)
                features = self.feature_extractor.create_ranking_features(
                    user_features,
                    product_metadata,
                    context,
                    candidate,
                )
                feature_vectors.append(features)
                valid_candidates.append((candidate, product_metadata))
            except Exception as e:
                logger.warning(
                    f"Error extracting features for candidate {candidate.product_id}: {e}"
                )

        feature_extraction_ms = round(
            (time.perf_counter() - feature_stage_started) * 1000, 2
        )
        if not feature_vectors:
            return None, [], feature_extraction_ms

        return np.vstack(feature_vectors), valid_candidates, feature_extraction_ms

    def run_inference_batch(
        self,
        feature_matrix: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Run one Torch forward pass for a feature matrix."""
        with torch.no_grad():
            tensor_stage_started = time.perf_counter()
            features_tensor = torch.as_tensor(
                feature_matrix,
                dtype=torch.float32,
                device=self.device,
            )
            tensor_prep_ms = round((time.perf_counter() - tensor_stage_started) * 1000, 2)

            model_stage_started = time.perf_counter()
            predictions = self.model(features_tensor)
            model_forward_ms = round((time.perf_counter() - model_stage_started) * 1000, 2)

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
                    ctr_score * self.config.ctr_weight +
                    cvr_score * self.config.cvr_weight +
                    (gmv_score / 100.0) * self.config.gmv_weight
                ) / (
                    self.config.ctr_weight +
                    self.config.cvr_weight +
                    self.config.gmv_weight
                )
            else:
                confidence_score = ranking_score

            reason = self._generate_explanation(candidate, ctr_score, cvr_score, gmv_score)
            recommendations.append(
                ProductRecommendation(
                    product_id=candidate.product_id,
                    title=metadata.get("title", f"Product {candidate.product_id}"),
                    description="Recommended based on your preferences",
                    price=metadata.get("price", 0.0),
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

    def _build_product_metadata(self, candidate: CandidateProduct) -> Dict[str, Any]:
        """Build deterministic fallback metadata when the cache misses."""
        return {
            "title": f"Product {candidate.product_id}",
            "category": "General",
            "price": 0.0,
            "rating": 0.0,
            "num_reviews": 0,
            "in_stock": True,
            "created_at": time.time(),
            "tags": [],
            "brand": "Unknown",
        }
    
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
            feature_matrix, valid_candidates, feature_extraction_ms = self.prepare_request_matrix(
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

            top_recommendations, response_build_ms = self.build_recommendations_from_predictions(
                valid_candidates,
                predictions,
                k,
            )
            profile["response_build_ms"] = response_build_ms
            profile["ranked_count"] = len(top_recommendations)
            
            # Update inference statistics
            inference_time = time.time() - start_time
            self.inference_stats['total_inferences'] += 1
            self.inference_stats['avg_inference_time'] = (
                (self.inference_stats['avg_inference_time'] * (self.inference_stats['total_inferences'] - 1) +
                 inference_time) / self.inference_stats['total_inferences']
            )
            
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
            collaborative = candidate.collaborative_score or 0.0
            content_similarity = candidate.content_similarity_score or 0.0
            popularity = candidate.popularity_score or 0.0
            combined = candidate.combined_score or (
                0.5 * collaborative + 0.3 * content_similarity + 0.2 * popularity
            )
            confidence_score = max(0.0, min(combined, 1.0))

            fallback_recommendations.append(
                ProductRecommendation(
                    product_id=candidate.product_id,
                    title=f"Product {candidate.product_id}",
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
        gmv_score: float
    ) -> str:
        """Generate explanation for recommendation."""
        explanations = []
        
        if candidate.collaborative_score and candidate.collaborative_score > 0.5:
            explanations.append("users with similar interests liked this")
        
        if candidate.content_similarity_score and candidate.content_similarity_score > 0.5:
            explanations.append("matches the video content you're viewing")
        
        if candidate.popularity_score and candidate.popularity_score > 0.5:
            explanations.append("trending among other users")
        
        if ctr_score > 0.7:
            explanations.append("high engagement product")
        
        if cvr_score > 0.3:
            explanations.append("frequently purchased")
        
        if not explanations:
            explanations.append("recommended for you")
        
        return f"Based on your preferences - {explanations[0]}"
    
    async def train_model(self, training_data: List[Dict[str, Any]]):
        """Train the ranking model on user interaction data."""
        try:
            saved_model_path = await asyncio.to_thread(
                self._train_model_sync,
                training_data,
            )
            if saved_model_path:
                await self.save_model(saved_model_path)
            
        except Exception as e:
            logger.error(f"Error training model: {e}")

    def _train_model_sync(self, training_data: List[Dict[str, Any]]) -> Optional[str]:
        if not self.model or len(training_data) < self.config.training_min_samples:
            logger.warning("Insufficient data or model not loaded for training")
            return None

        logger.info(f"Training ranking model on {len(training_data)} samples")

        # Prepare training data
        features, labels = self._prepare_training_data(training_data)

        if features.size(0) == 0:
            logger.warning("No valid training samples prepared")
            return None

        self.model.train()

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, features.size(0), self.config.batch_size):
                batch_features = features[i:i+self.config.batch_size]
                batch_labels = {
                    key: value[i:i+self.config.batch_size]
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
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare training data from interaction logs."""
        try:
            features = []
            ctr_labels = []
            cvr_labels = []
            gmv_labels = []
            
            for sample in training_data:
                # Extract features (simplified - would use actual user/product data)
                feature_vector = np.random.normal(0, 1, self.feature_extractor.total_feature_dim)
                features.append(feature_vector)
                
                # Create labels based on interaction type
                action = sample.get('action', 'view')
                
                # CTR label (clicked or not)
                ctr_label = 1.0 if action in ['click', 'purchase', 'add_to_cart'] else 0.0
                ctr_labels.append(ctr_label)
                
                # CVR label (converted or not)  
                cvr_label = 1.0 if action == 'purchase' else 0.0
                cvr_labels.append(cvr_label)
                
                # GMV label (purchase value)
                gmv_value = sample.get('value', 0.0) if action == 'purchase' else 0.0
                gmv_labels.append(gmv_value)
            
            if not features:
                return torch.empty(0), {}
            
            features_tensor = torch.tensor(np.vstack(features), dtype=torch.float32).to(self.device)
            labels = {
                'ctr': torch.tensor(ctr_labels, dtype=torch.float32).to(self.device).unsqueeze(1),
                'cvr': torch.tensor(cvr_labels, dtype=torch.float32).to(self.device).unsqueeze(1),
                'gmv': torch.tensor(gmv_labels, dtype=torch.float32).to(self.device).unsqueeze(1)
            }
            
            return features_tensor, labels
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return torch.empty(0), {}
    
    def _compute_loss(self, predictions: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-objective loss."""
        try:
            # CTR loss (binary cross-entropy)
            ctr_loss = nn.BCELoss()(predictions['ctr'], labels['ctr'])
            
            # CVR loss (binary cross-entropy) 
            cvr_loss = nn.BCELoss()(predictions['cvr'], labels['cvr'])
            
            # GMV loss (MSE)
            gmv_loss = nn.MSELoss()(predictions['gmv'], labels['gmv'])
            
            # Combined loss with weights
            total_loss = (
                self.config.ctr_weight * ctr_loss +
                self.config.cvr_weight * cvr_loss +
                self.config.gmv_weight * gmv_loss
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
                torch.save(self.model.state_dict(), model_path)
                self.loaded_model_path = model_path
                self.loaded_checkpoint_mtime = Path(model_path).stat().st_mtime
                logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            'is_trained': self.is_trained,
            'model_version': self.model_version,
            'last_training_time': self.last_training_time,
            'loaded_model_path': self.loaded_model_path,
            'loaded_checkpoint_mtime': self.loaded_checkpoint_mtime,
            'checkpoint_reload_count': self.checkpoint_reload_count,
            'device': str(self.device),
            'inference_stats': self.inference_stats.copy(),
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'feature_dim': self.feature_extractor.total_feature_dim
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the ranking model."""
        try:
            status = {
                'status': 'healthy' if self.model else 'unhealthy',
                'model_loaded': self.model is not None,
                'is_trained': self.is_trained,
                'loaded_model_path': self.loaded_model_path,
                'loaded_checkpoint_mtime': self.loaded_checkpoint_mtime,
                'checkpoint_reload_count': self.checkpoint_reload_count,
                'device': str(self.device),
                'feature_dim': self.feature_extractor.total_feature_dim,
                'torch_inference_available': self.torch_inference_available,
            }
            
            # Test inference if model is loaded
            if self.model:
                try:
                    dummy_features = torch.randn(1, self.feature_extractor.total_feature_dim).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(dummy_features)
                    
                    status['inference_test_passed'] = True
                    status['output_keys'] = list(outputs.keys())
                    
                except Exception as test_error:
                    status['inference_test_passed'] = False
                    status['test_error'] = str(test_error)
            
            return status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model_loaded': False
            }
