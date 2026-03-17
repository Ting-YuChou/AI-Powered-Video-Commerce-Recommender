"""
AI-Powered Video Commerce Recommender - Two-Tower Retrieval Model
=================================================================

This module implements a Two-Tower (dual encoder) neural retrieval model
for collaborative filtering. It replaces the traditional NMF approach with
learned user and item embeddings trained via contrastive learning (InfoNCE),
and supports hard/mixed negative sampling for improved retrieval quality.

Architecture:
    UserTower:  user_id embedding + side features -> 128-dim user vector
    ItemTower:  item_id embedding + CLIP projection + side features -> 128-dim item vector
    Retrieval:  cosine similarity between user and item vectors via FAISS ANN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import faiss
import logging
import time
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature Encoders
# ---------------------------------------------------------------------------

NUM_CATEGORY_BUCKETS = 64
NUM_BRAND_BUCKETS = 128


def _hash_bucket(value: str, num_buckets: int) -> int:
    """Hash a string into a fixed bucket index."""
    return abs(hash(value)) % num_buckets


class UserFeatureEncoder:
    """Convert raw UserFeatures into a fixed-size numerical tensor.

    Output dimension: 10
    """

    FEATURE_DIM = 10

    @staticmethod
    def encode(user_features: Dict[str, Any]) -> np.ndarray:
        """Encode user features into a float32 array of length 10."""
        total_interactions = float(user_features.get("total_interactions", 0))
        avg_session = float(user_features.get("avg_session_length", 0.0))
        price_sens = float(user_features.get("price_sensitivity", 0.5))
        ctr = float(user_features.get("click_through_rate", 0.0))
        cvr = float(user_features.get("conversion_rate", 0.0))
        preferred_cats = user_features.get("preferred_categories", [])
        last_active = float(user_features.get("last_active", time.time()))

        hours_since_active = max((time.time() - last_active) / 3600.0, 0.0)

        features = np.array([
            np.log1p(total_interactions),
            avg_session / 3600.0,
            price_sens,
            ctr,
            cvr,
            len(preferred_cats) / 10.0,
            min(hours_since_active / 720.0, 1.0),  # cap at 30 days
            1.0 if total_interactions > 100 else 0.0,
            1.0 if cvr > 0.05 else 0.0,
            1.0 if ctr > 0.1 else 0.0,
        ], dtype=np.float32)

        return np.nan_to_num(features, 0.0)


class ItemFeatureEncoder:
    """Convert raw product metadata into a fixed-size numerical tensor.

    Output dimension: 8
    """

    FEATURE_DIM = 8

    @staticmethod
    def encode(metadata: Dict[str, Any]) -> np.ndarray:
        """Encode item features into a float32 array of length 8."""
        price = float(metadata.get("price", 0.0))
        rating = float(metadata.get("rating", 3.0))
        num_reviews = int(metadata.get("num_reviews", 0))
        in_stock = 1.0 if metadata.get("in_stock", True) else 0.0
        created_at = float(metadata.get("created_at", time.time()))
        tags = metadata.get("tags", [])
        category = str(metadata.get("category", "unknown"))
        brand = str(metadata.get("brand", "unknown"))

        age_days = max((time.time() - created_at) / 86400.0, 0.0)

        features = np.array([
            np.log1p(price),
            rating / 5.0,
            np.log1p(num_reviews),
            in_stock,
            min(age_days / 365.0, 1.0),
            len(tags) / 10.0,
            _hash_bucket(category, NUM_CATEGORY_BUCKETS) / NUM_CATEGORY_BUCKETS,
            _hash_bucket(brand, NUM_BRAND_BUCKETS) / NUM_BRAND_BUCKETS,
        ], dtype=np.float32)

        return np.nan_to_num(features, 0.0)


# ---------------------------------------------------------------------------
# Neural Network Towers
# ---------------------------------------------------------------------------

class UserTower(nn.Module):
    """User tower: maps user_id + side features to a normalized embedding."""

    def __init__(
        self,
        num_users: int,
        user_feat_dim: int = UserFeatureEncoder.FEATURE_DIM,
        id_embed_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]

        self.user_embedding = nn.Embedding(num_users + 1, id_embed_dim, padding_idx=0)

        layers: List[nn.Module] = []
        prev_dim = id_embed_dim + user_feat_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.01)

    def forward(self, user_ids: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids: (B,) int tensor of user indices
            user_features: (B, user_feat_dim) float tensor
        Returns:
            (B, output_dim) L2-normalized user embeddings
        """
        id_emb = self.user_embedding(user_ids)
        x = torch.cat([id_emb, user_features], dim=-1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class ItemTower(nn.Module):
    """Item tower: maps item_id + CLIP embedding + side features to a normalized embedding."""

    def __init__(
        self,
        num_items: int,
        clip_dim: int = 512,
        item_feat_dim: int = ItemFeatureEncoder.FEATURE_DIM,
        id_embed_dim: int = 64,
        clip_proj_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]

        self.item_embedding = nn.Embedding(num_items + 1, id_embed_dim, padding_idx=0)
        self.clip_projection = nn.Linear(clip_dim, clip_proj_dim)

        layers: List[nn.Module] = []
        prev_dim = id_embed_dim + clip_proj_dim + item_feat_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.01)

    def forward(
        self,
        item_ids: torch.Tensor,
        clip_embeddings: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            item_ids: (B,) int tensor of item indices
            clip_embeddings: (B, clip_dim) float tensor
            item_features: (B, item_feat_dim) float tensor
        Returns:
            (B, output_dim) L2-normalized item embeddings
        """
        id_emb = self.item_embedding(item_ids)
        clip_feat = self.clip_projection(clip_embeddings)
        x = torch.cat([id_emb, clip_feat, item_features], dim=-1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """Wrapper combining UserTower and ItemTower with InfoNCE loss."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        clip_dim: int = 512,
        output_dim: int = 128,
        temperature: float = 0.07,
        user_hidden_dims: Optional[List[int]] = None,
        item_hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.output_dim = output_dim

        self.user_tower = UserTower(
            num_users=num_users,
            output_dim=output_dim,
            hidden_dims=user_hidden_dims,
        )
        self.item_tower = ItemTower(
            num_items=num_items,
            clip_dim=clip_dim,
            output_dim=output_dim,
            hidden_dims=item_hidden_dims,
        )

    def encode_users(
        self, user_ids: torch.Tensor, user_features: torch.Tensor
    ) -> torch.Tensor:
        return self.user_tower(user_ids, user_features)

    def encode_items(
        self,
        item_ids: torch.Tensor,
        clip_embeddings: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_tower(item_ids, clip_embeddings, item_features)

    def forward(
        self,
        user_ids: torch.Tensor,
        user_features: torch.Tensor,
        pos_item_ids: torch.Tensor,
        pos_clip_embs: torch.Tensor,
        pos_item_feats: torch.Tensor,
        neg_item_ids: torch.Tensor,
        neg_clip_embs: torch.Tensor,
        neg_item_feats: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute InfoNCE loss over positive and negative pairs.

        Args:
            user_*: user side tensors (B, ...)
            pos_*: positive item tensors (B, ...)
            neg_*: negative item tensors (B, num_neg, ...)
        Returns:
            dict with 'loss' and 'accuracy' keys
        """
        batch_size = user_ids.size(0)
        num_neg = neg_item_ids.size(1)

        user_emb = self.encode_users(user_ids, user_features)  # (B, D)
        pos_emb = self.encode_items(pos_item_ids, pos_clip_embs, pos_item_feats)  # (B, D)

        # Flatten negatives for batch encoding
        neg_ids_flat = neg_item_ids.reshape(-1)
        neg_clip_flat = neg_clip_embs.reshape(-1, neg_clip_embs.size(-1))
        neg_feat_flat = neg_item_feats.reshape(-1, neg_item_feats.size(-1))
        neg_emb_flat = self.encode_items(neg_ids_flat, neg_clip_flat, neg_feat_flat)
        neg_emb = neg_emb_flat.reshape(batch_size, num_neg, -1)  # (B, K, D)

        # Positive scores: (B, 1)
        pos_scores = torch.sum(user_emb * pos_emb, dim=-1, keepdim=True) / self.temperature

        # Negative scores: (B, K)
        neg_scores = torch.bmm(neg_emb, user_emb.unsqueeze(-1)).squeeze(-1) / self.temperature

        # InfoNCE: log softmax over [pos, neg_1, ..., neg_K]
        logits = torch.cat([pos_scores, neg_scores], dim=-1)  # (B, 1+K)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        # Accuracy: fraction where positive has highest score
        accuracy = (logits.argmax(dim=-1) == 0).float().mean()

        return {"loss": loss, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# Negative Sampler
# ---------------------------------------------------------------------------

class NegativeSampler:
    """Mixed hard/random negative sampler using a FAISS index for hard mining."""

    def __init__(
        self,
        num_items: int,
        num_hard: int = 5,
        num_random: int = 10,
        hard_ratio_start: float = 0.1,
        hard_ratio_end: float = 0.5,
    ):
        self.num_items = num_items
        self.num_hard = num_hard
        self.num_random = num_random
        self.hard_ratio_start = hard_ratio_start
        self.hard_ratio_end = hard_ratio_end

        # FAISS index for hard-negative mining (set externally after training steps)
        self.item_index: Optional[faiss.Index] = None
        self.item_index_map: Dict[int, int] = {}  # faiss_idx -> item_idx

    @property
    def total_negatives(self) -> int:
        return self.num_hard + self.num_random

    def set_index(self, index: faiss.Index, index_map: Dict[int, int]):
        """Set the FAISS index for hard-negative mining."""
        self.item_index = index
        self.item_index_map = index_map

    def get_hard_ratio(self, epoch: int, total_epochs: int) -> float:
        """Curriculum learning: linearly increase hard negative ratio."""
        if total_epochs <= 1:
            return self.hard_ratio_end
        progress = min(epoch / max(total_epochs - 1, 1), 1.0)
        return self.hard_ratio_start + (self.hard_ratio_end - self.hard_ratio_start) * progress

    def sample(
        self,
        user_embedding: Optional[np.ndarray],
        positive_items: Set[int],
        epoch: int = 0,
        total_epochs: int = 1,
    ) -> List[int]:
        """Sample a mix of hard and random negatives.

        Args:
            user_embedding: user embedding for hard-negative mining (optional)
            positive_items: set of item indices the user interacted with
            epoch: current training epoch (for curriculum schedule)
            total_epochs: total training epochs
        Returns:
            list of negative item indices of length total_negatives
        """
        total_neg = self.total_negatives
        hard_ratio = self.get_hard_ratio(epoch, total_epochs)
        num_hard_actual = int(total_neg * hard_ratio)
        num_random_actual = total_neg - num_hard_actual

        negatives: List[int] = []

        # Hard negatives via FAISS
        if (
            num_hard_actual > 0
            and self.item_index is not None
            and user_embedding is not None
            and self.item_index.ntotal > 0
        ):
            query = user_embedding.reshape(1, -1).astype(np.float32)
            search_k = min(num_hard_actual * 5, self.item_index.ntotal)
            _, indices = self.item_index.search(query, search_k)

            for idx in indices[0]:
                if len(negatives) >= num_hard_actual:
                    break
                if idx == -1:
                    continue
                item_idx = self.item_index_map.get(int(idx))
                if item_idx is not None and item_idx not in positive_items:
                    negatives.append(item_idx)

        # Random negatives
        attempts = 0
        max_attempts = num_random_actual * 10
        while len(negatives) < total_neg and attempts < max_attempts:
            rand_idx = np.random.randint(1, self.num_items + 1)
            if rand_idx not in positive_items and rand_idx not in negatives:
                negatives.append(rand_idx)
            attempts += 1

        # Pad if needed
        while len(negatives) < total_neg:
            negatives.append(np.random.randint(1, max(self.num_items, 2)))

        return negatives[:total_neg]


# ---------------------------------------------------------------------------
# Two-Tower Trainer
# ---------------------------------------------------------------------------

# Interaction type weights for training signal strength
INTERACTION_WEIGHTS: Dict[str, float] = {
    "view": 1.0,
    "click": 2.0,
    "add_to_cart": 3.0,
    "purchase": 5.0,
    "favorite": 2.5,
    "share": 1.5,
}


class TwoTowerTrainer:
    """End-to-end trainer for the Two-Tower model.

    Handles data preparation, negative sampling, training loop,
    and exporting item embeddings to a FAISS index.
    """

    def __init__(
        self,
        clip_dim: int = 512,
        output_dim: int = 128,
        temperature: float = 0.07,
        learning_rate: float = 0.001,
        batch_size: int = 1024,
        epochs: int = 20,
        num_hard_negatives: int = 5,
        num_random_negatives: int = 10,
        hard_ratio_start: float = 0.1,
        hard_ratio_end: float = 0.5,
        user_hidden_dims: Optional[List[int]] = None,
        item_hidden_dims: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
    ):
        self.clip_dim = clip_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.user_hidden_dims = user_hidden_dims
        self.item_hidden_dims = item_hidden_dims
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Will be initialised in prepare()
        self.model: Optional[TwoTowerModel] = None
        self.negative_sampler: Optional[NegativeSampler] = None

        self.num_hard_negatives = num_hard_negatives
        self.num_random_negatives = num_random_negatives
        self.hard_ratio_start = hard_ratio_start
        self.hard_ratio_end = hard_ratio_end

        # Data structures populated by prepare()
        self.user_mapping: Dict[str, int] = {}
        self.item_mapping: Dict[str, int] = {}
        self.reverse_item_mapping: Dict[int, str] = {}
        self.user_positives: Dict[int, Set[int]] = defaultdict(set)

        # Per-item data arrays (populated in prepare)
        self._item_clip_embs: Optional[np.ndarray] = None
        self._item_side_feats: Optional[np.ndarray] = None
        self._user_side_feats: Dict[int, np.ndarray] = {}
        self.model_version: Optional[str] = None

        # Training samples: list of (user_idx, item_idx, weight)
        self._train_samples: List[Tuple[int, int, float]] = []

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare(
        self,
        interactions: List[Dict[str, Any]],
        product_metadata: Dict[str, Dict[str, Any]],
        product_clip_embeddings: Dict[str, np.ndarray],
        user_features_map: Dict[str, Dict[str, Any]],
    ):
        """Prepare training data from raw interactions and metadata.

        Args:
            interactions: list of {user_id, product_id, action, timestamp, ...}
            product_metadata: product_id -> metadata dict
            product_clip_embeddings: product_id -> CLIP embedding (512-dim)
            user_features_map: user_id -> user features dict
        """
        logger.info(f"Preparing training data from {len(interactions)} interactions")

        # Build mappings (index 0 reserved for padding)
        users: Set[str] = set()
        items: Set[str] = set()
        for interaction in interactions:
            uid = interaction.get("user_id")
            pid = interaction.get("product_id")
            if uid and pid:
                users.add(uid)
                items.add(pid)

        # Also include items with metadata but not in interactions
        for pid in product_metadata:
            items.add(pid)

        self.user_mapping = {u: idx + 1 for idx, u in enumerate(sorted(users))}
        self.item_mapping = {p: idx + 1 for idx, p in enumerate(sorted(items))}
        self.reverse_item_mapping = {idx: p for p, idx in self.item_mapping.items()}

        num_users = len(self.user_mapping)
        num_items = len(self.item_mapping)
        logger.info(f"Mapped {num_users} users, {num_items} items")

        # Build item feature arrays
        self._item_clip_embs = np.zeros((num_items + 1, self.clip_dim), dtype=np.float32)
        self._item_side_feats = np.zeros(
            (num_items + 1, ItemFeatureEncoder.FEATURE_DIM), dtype=np.float32
        )

        for product_id, idx in self.item_mapping.items():
            clip_emb = product_clip_embeddings.get(product_id)
            if clip_emb is not None:
                emb = np.asarray(clip_emb, dtype=np.float32).flatten()
                if emb.shape[0] == self.clip_dim:
                    self._item_clip_embs[idx] = emb

            meta = product_metadata.get(product_id, {})
            self._item_side_feats[idx] = ItemFeatureEncoder.encode(meta)

        # Build user feature arrays
        self._user_side_feats = {}
        for user_id, idx in self.user_mapping.items():
            uf = user_features_map.get(user_id, {})
            self._user_side_feats[idx] = UserFeatureEncoder.encode(uf)

        # Build training samples and positive sets
        self._train_samples = []
        self.user_positives = defaultdict(set)

        for interaction in interactions:
            uid = interaction.get("user_id")
            pid = interaction.get("product_id")
            action = interaction.get("action", "view")
            if uid not in self.user_mapping or pid not in self.item_mapping:
                continue
            u_idx = self.user_mapping[uid]
            p_idx = self.item_mapping[pid]
            weight = INTERACTION_WEIGHTS.get(action, 1.0)
            self._train_samples.append((u_idx, p_idx, weight))
            self.user_positives[u_idx].add(p_idx)

        logger.info(f"Prepared {len(self._train_samples)} training samples")

        # Initialise model
        self.model = TwoTowerModel(
            num_users=num_users,
            num_items=num_items,
            clip_dim=self.clip_dim,
            output_dim=self.output_dim,
            temperature=self.temperature,
            user_hidden_dims=self.user_hidden_dims,
            item_hidden_dims=self.item_hidden_dims,
        ).to(self.device)

        # Initialise negative sampler
        self.negative_sampler = NegativeSampler(
            num_items=num_items,
            num_hard=self.num_hard_negatives,
            num_random=self.num_random_negatives,
            hard_ratio_start=self.hard_ratio_start,
            hard_ratio_end=self.hard_ratio_end,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, existing_cf_index: Optional[faiss.Index] = None) -> Dict[str, Any]:
        """Run the full training loop.

        Args:
            existing_cf_index: optional FAISS index from previous training
                               (used for hard-negative mining from epoch 0)
        Returns:
            dict of training statistics
        """
        if self.model is None or not self._train_samples:
            logger.warning("Model or training data not prepared; skipping training")
            return {"status": "skipped"}

        # Set hard-neg index if available
        if existing_cf_index is not None and existing_cf_index.ntotal > 0:
            idx_map = {i: i + 1 for i in range(existing_cf_index.ntotal)}
            self.negative_sampler.set_index(existing_cf_index, idx_map)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.model.train()

        num_samples = len(self._train_samples)
        stats: Dict[str, Any] = {"epoch_losses": [], "epoch_accs": []}
        total_neg = self.negative_sampler.total_negatives
        start_time = time.time()

        for epoch in range(self.epochs):
            # Shuffle training data
            indices = np.random.permutation(num_samples)
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0

            for batch_start in range(0, num_samples, self.batch_size):
                batch_indices = indices[batch_start : batch_start + self.batch_size]
                actual_bs = len(batch_indices)

                # Prepare batch tensors
                user_ids_np = np.zeros(actual_bs, dtype=np.int64)
                user_feats_np = np.zeros(
                    (actual_bs, UserFeatureEncoder.FEATURE_DIM), dtype=np.float32
                )
                pos_item_ids_np = np.zeros(actual_bs, dtype=np.int64)
                pos_clip_np = np.zeros((actual_bs, self.clip_dim), dtype=np.float32)
                pos_feat_np = np.zeros(
                    (actual_bs, ItemFeatureEncoder.FEATURE_DIM), dtype=np.float32
                )
                neg_item_ids_np = np.zeros((actual_bs, total_neg), dtype=np.int64)
                neg_clip_np = np.zeros(
                    (actual_bs, total_neg, self.clip_dim), dtype=np.float32
                )
                neg_feat_np = np.zeros(
                    (actual_bs, total_neg, ItemFeatureEncoder.FEATURE_DIM), dtype=np.float32
                )

                for i, sample_idx in enumerate(batch_indices):
                    u_idx, p_idx, _ = self._train_samples[sample_idx]

                    user_ids_np[i] = u_idx
                    user_feats_np[i] = self._user_side_feats.get(
                        u_idx, np.zeros(UserFeatureEncoder.FEATURE_DIM, dtype=np.float32)
                    )

                    pos_item_ids_np[i] = p_idx
                    pos_clip_np[i] = self._item_clip_embs[p_idx]
                    pos_feat_np[i] = self._item_side_feats[p_idx]

                    # Sample negatives
                    neg_indices = self.negative_sampler.sample(
                        user_embedding=None,  # no user emb yet in early epochs
                        positive_items=self.user_positives.get(u_idx, set()),
                        epoch=epoch,
                        total_epochs=self.epochs,
                    )
                    for j, neg_idx in enumerate(neg_indices):
                        neg_item_ids_np[i, j] = neg_idx
                        if neg_idx < len(self._item_clip_embs):
                            neg_clip_np[i, j] = self._item_clip_embs[neg_idx]
                            neg_feat_np[i, j] = self._item_side_feats[neg_idx]

                # To tensors
                user_ids_t = torch.tensor(user_ids_np, device=self.device)
                user_feats_t = torch.tensor(user_feats_np, device=self.device)
                pos_ids_t = torch.tensor(pos_item_ids_np, device=self.device)
                pos_clip_t = torch.tensor(pos_clip_np, device=self.device)
                pos_feat_t = torch.tensor(pos_feat_np, device=self.device)
                neg_ids_t = torch.tensor(neg_item_ids_np, device=self.device)
                neg_clip_t = torch.tensor(neg_clip_np, device=self.device)
                neg_feat_t = torch.tensor(neg_feat_np, device=self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    user_ids_t, user_feats_t,
                    pos_ids_t, pos_clip_t, pos_feat_t,
                    neg_ids_t, neg_clip_t, neg_feat_t,
                )
                loss = outputs["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += outputs["accuracy"].item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            avg_acc = epoch_acc / max(num_batches, 1)
            stats["epoch_losses"].append(avg_loss)
            stats["epoch_accs"].append(avg_acc)

            if epoch % 5 == 0 or epoch == self.epochs - 1:
                logger.info(
                    f"Two-Tower epoch {epoch}/{self.epochs}: "
                    f"loss={avg_loss:.4f}, accuracy={avg_acc:.4f}"
                )

            # Update hard-neg index mid-training (every 5 epochs after epoch 5)
            if epoch > 0 and epoch % 5 == 0:
                try:
                    cf_index, idx_map = self.build_item_index()
                    self.negative_sampler.set_index(cf_index, idx_map)
                    logger.info(f"Updated hard-neg index at epoch {epoch}")
                except Exception as e:
                    logger.warning(f"Failed to update hard-neg index: {e}")

        self.model.eval()
        training_time = time.time() - start_time
        stats["training_time_s"] = training_time
        stats["num_users"] = len(self.user_mapping)
        stats["num_items"] = len(self.item_mapping)
        stats["num_samples"] = num_samples
        logger.info(f"Two-Tower training completed in {training_time:.1f}s")
        return stats

    # ------------------------------------------------------------------
    # Index building & embedding export
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_item_index(self) -> Tuple[faiss.Index, Dict[int, int]]:
        """Build a FAISS HNSW index from all item embeddings.

        Returns:
            (faiss_index, faiss_idx_to_item_idx mapping)
        """
        if self.model is None:
            raise RuntimeError("Model not initialised")

        self.model.eval()
        num_items = len(self.item_mapping)
        all_embeddings = np.zeros((num_items, self.output_dim), dtype=np.float32)

        batch_size = 2048
        item_indices = list(range(1, num_items + 1))

        for start in range(0, len(item_indices), batch_size):
            batch_idxs = item_indices[start : start + batch_size]
            ids_t = torch.tensor(batch_idxs, dtype=torch.long, device=self.device)
            clip_t = torch.tensor(
                self._item_clip_embs[batch_idxs], device=self.device
            )
            feat_t = torch.tensor(
                self._item_side_feats[batch_idxs], device=self.device
            )

            embs = self.model.encode_items(ids_t, clip_t, feat_t).cpu().numpy()
            for i, idx in enumerate(batch_idxs):
                all_embeddings[idx - 1] = embs[i]

        # Build HNSW index
        index = faiss.IndexHNSWFlat(self.output_dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
        index.add(all_embeddings)

        # faiss_idx -> item_idx (1-based)
        idx_map = {i: i + 1 for i in range(num_items)}

        logger.info(f"Built CF FAISS index with {index.ntotal} item embeddings")
        return index, idx_map

    @torch.no_grad()
    def encode_user(self, user_id: str, user_features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Encode a single user into the Two-Tower embedding space.

        Returns None if the model is not trained or user not in mapping.
        For unknown users, uses index 0 (padding) to get a generic embedding.
        """
        if self.model is None:
            return None

        self.model.eval()
        u_idx = self.user_mapping.get(user_id, 0)
        u_feat = np.asarray(UserFeatureEncoder.encode(user_features), dtype=np.float32)

        ids_t = torch.tensor([u_idx], dtype=torch.long, device=self.device)
        feat_t = torch.from_numpy(u_feat[None, :]).to(self.device)
        emb = self.model.encode_users(ids_t, feat_t).cpu().numpy()[0]
        return emb

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        """Save model weights and mappings to disk."""
        if self.model is None:
            logger.warning("No model to save")
            return

        if not self.model_version:
            self.model_version = f"cf-{int(time.time())}"

        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_version": self.model_version,
            "model_state_dict": self.model.state_dict(),
            "user_mapping": self.user_mapping,
            "item_mapping": self.item_mapping,
            "reverse_item_mapping": self.reverse_item_mapping,
            "config": {
                "clip_dim": self.clip_dim,
                "output_dim": self.output_dim,
                "temperature": self.temperature,
                "num_users": len(self.user_mapping),
                "num_items": len(self.item_mapping),
                "model_version": self.model_version,
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved Two-Tower checkpoint to {path}")

    def load_checkpoint(self, path: str) -> bool:
        """Load model weights and mappings from disk.

        Returns True on success.
        """
        p = Path(path)
        if not p.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)
            cfg = checkpoint["config"]
            self.model_version = (
                checkpoint.get("model_version")
                or cfg.get("model_version")
                or f"legacy-{int(p.stat().st_mtime)}"
            )

            self.user_mapping = checkpoint["user_mapping"]
            self.item_mapping = checkpoint["item_mapping"]
            self.reverse_item_mapping = checkpoint["reverse_item_mapping"]

            self.model = TwoTowerModel(
                num_users=cfg["num_users"],
                num_items=cfg["num_items"],
                clip_dim=cfg.get("clip_dim", self.clip_dim),
                output_dim=cfg.get("output_dim", self.output_dim),
                temperature=cfg.get("temperature", self.temperature),
                user_hidden_dims=self.user_hidden_dims,
                item_hidden_dims=self.item_hidden_dims,
            ).to(self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            # Re-initialise item feature arrays with empty defaults
            num_items = len(self.item_mapping)
            if self._item_clip_embs is None or len(self._item_clip_embs) != num_items + 1:
                self._item_clip_embs = np.zeros((num_items + 1, self.clip_dim), dtype=np.float32)
                self._item_side_feats = np.zeros(
                    (num_items + 1, ItemFeatureEncoder.FEATURE_DIM), dtype=np.float32
                )

            logger.info(f"Loaded Two-Tower checkpoint from {path} (version={self.model_version})")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
