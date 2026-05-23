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

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from video_commerce.ml.dcn import DeepAndCrossNetwork, normalize_architecture

logger = logging.getLogger(__name__)

NUM_CATEGORY_BUCKETS = 64
NUM_BRAND_BUCKETS = 128


def _hash_bucket(value: str, num_buckets: int) -> int:
    """Hash a string into a fixed bucket index."""
    digest = hashlib.sha256(str(value or "").encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % num_buckets


class UserFeatureEncoder:
    """Convert raw UserFeatures into a fixed-size numerical tensor."""

    FEATURE_DIM = 10

    @staticmethod
    def encode(
        user_features: Dict[str, Any],
        current_time: Optional[float] = None,
    ) -> np.ndarray:
        if current_time is None:
            current_time = time.time()
        total_interactions = float(user_features.get("total_interactions", 0))
        avg_session = float(user_features.get("avg_session_length", 0.0))
        price_sens = float(user_features.get("price_sensitivity", 0.5))
        ctr = float(user_features.get("click_through_rate", 0.0))
        cvr = float(user_features.get("conversion_rate", 0.0))
        preferred_cats = user_features.get("preferred_categories", [])
        last_active = float(user_features.get("last_active", current_time))

        hours_since_active = max((current_time - last_active) / 3600.0, 0.0)

        features = np.array(
            [
                np.log1p(total_interactions),
                avg_session / 3600.0,
                price_sens,
                ctr,
                cvr,
                len(preferred_cats) / 10.0,
                min(hours_since_active / 720.0, 1.0),
                1.0 if total_interactions > 100 else 0.0,
                1.0 if cvr > 0.05 else 0.0,
                1.0 if ctr > 0.1 else 0.0,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(features, 0.0)


class ItemFeatureEncoder:
    """Convert raw product metadata into a fixed-size numerical tensor."""

    FEATURE_DIM = 8

    @staticmethod
    def encode(metadata: Dict[str, Any]) -> np.ndarray:
        price = float(metadata.get("price", 0.0))
        rating = float(metadata.get("rating", 3.0))
        num_reviews = int(metadata.get("num_reviews", 0))
        in_stock = 1.0 if metadata.get("in_stock", True) else 0.0
        created_at = float(metadata.get("created_at", time.time()))
        tags = metadata.get("tags", [])
        category = str(metadata.get("category", "unknown"))
        brand = str(metadata.get("brand", "unknown"))

        age_days = max((time.time() - created_at) / 86400.0, 0.0)

        features = np.array(
            [
                np.log1p(price),
                rating / 5.0,
                np.log1p(num_reviews),
                in_stock,
                min(age_days / 365.0, 1.0),
                len(tags) / 10.0,
                _hash_bucket(category, NUM_CATEGORY_BUCKETS) / NUM_CATEGORY_BUCKETS,
                _hash_bucket(brand, NUM_BRAND_BUCKETS) / NUM_BRAND_BUCKETS,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(features, 0.0)


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
        architecture: str = "dcn",
        cross_layers: int = 3,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        self.architecture = normalize_architecture(architecture)
        self.cross_layers = max(0, int(cross_layers))

        self.user_embedding = nn.Embedding(num_users + 1, id_embed_dim, padding_idx=0)

        input_dim = id_embed_dim + user_feat_dim
        if self.architecture == "mlp":
            layers: List[nn.Module] = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.mlp = nn.Sequential(*layers)
        else:
            self.dcn = DeepAndCrossNetwork(
                input_dim,
                hidden_dims,
                output_dim,
                cross_layers=self.cross_layers,
                dropout=dropout,
                use_batch_norm=False,
            )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0.0, 0.01)

    def forward(self, user_ids: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        id_emb = self.user_embedding(user_ids)
        x = torch.cat([id_emb, user_features], dim=-1)
        if self.architecture == "mlp":
            x = self.mlp(x)
        else:
            x = self.dcn(x)
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
        architecture: str = "dcn",
        cross_layers: int = 3,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        self.architecture = normalize_architecture(architecture)
        self.cross_layers = max(0, int(cross_layers))

        self.item_embedding = nn.Embedding(num_items + 1, id_embed_dim, padding_idx=0)
        self.clip_projection = nn.Linear(clip_dim, clip_proj_dim)

        input_dim = id_embed_dim + clip_proj_dim + item_feat_dim
        if self.architecture == "mlp":
            layers: List[nn.Module] = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.mlp = nn.Sequential(*layers)
        else:
            self.dcn = DeepAndCrossNetwork(
                input_dim,
                hidden_dims,
                output_dim,
                cross_layers=self.cross_layers,
                dropout=dropout,
                use_batch_norm=False,
            )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0.0, 0.01)

    def forward(
        self,
        item_ids: torch.Tensor,
        clip_embeddings: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        id_emb = self.item_embedding(item_ids)
        clip_feat = self.clip_projection(clip_embeddings)
        x = torch.cat([id_emb, clip_feat, item_features], dim=-1)
        if self.architecture == "mlp":
            x = self.mlp(x)
        else:
            x = self.dcn(x)
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
        architecture: str = "dcn",
        cross_layers: int = 3,
    ):
        super().__init__()
        self.temperature = temperature
        self.output_dim = output_dim
        self.architecture = normalize_architecture(architecture)
        self.cross_layers = max(0, int(cross_layers))

        self.user_tower = UserTower(
            num_users=num_users,
            output_dim=output_dim,
            hidden_dims=user_hidden_dims,
            architecture=self.architecture,
            cross_layers=self.cross_layers,
        )
        self.item_tower = ItemTower(
            num_items=num_items,
            clip_dim=clip_dim,
            output_dim=output_dim,
            hidden_dims=item_hidden_dims,
            architecture=self.architecture,
            cross_layers=self.cross_layers,
        )

    def encode_users(self, user_ids: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
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
        batch_size = user_ids.size(0)
        num_neg = neg_item_ids.size(1)

        user_emb = self.encode_users(user_ids, user_features)
        pos_emb = self.encode_items(pos_item_ids, pos_clip_embs, pos_item_feats)

        neg_ids_flat = neg_item_ids.reshape(-1)
        neg_clip_flat = neg_clip_embs.reshape(-1, neg_clip_embs.size(-1))
        neg_feat_flat = neg_item_feats.reshape(-1, neg_item_feats.size(-1))
        neg_emb_flat = self.encode_items(neg_ids_flat, neg_clip_flat, neg_feat_flat)
        neg_emb = neg_emb_flat.reshape(batch_size, num_neg, -1)

        pos_scores = torch.sum(user_emb * pos_emb, dim=-1, keepdim=True) / self.temperature
        neg_scores = torch.bmm(neg_emb, user_emb.unsqueeze(-1)).squeeze(-1) / self.temperature

        logits = torch.cat([pos_scores, neg_scores], dim=-1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=-1) == 0).float().mean()

        return {"loss": loss, "accuracy": accuracy}


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
        self.item_index: Optional[faiss.Index] = None
        self.item_index_map: Dict[int, int] = {}

    @property
    def total_negatives(self) -> int:
        return self.num_hard + self.num_random

    def set_index(self, index: faiss.Index, index_map: Dict[int, int]):
        self.item_index = index
        self.item_index_map = index_map

    def get_hard_ratio(self, epoch: int, total_epochs: int) -> float:
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
        total_neg = self.total_negatives
        hard_ratio = self.get_hard_ratio(epoch, total_epochs)
        num_hard_actual = int(total_neg * hard_ratio)

        negatives: List[int] = []
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

        attempts = 0
        max_attempts = total_neg * 10
        while len(negatives) < total_neg and attempts < max_attempts:
            rand_idx = np.random.randint(1, self.num_items + 1)
            if rand_idx not in positive_items and rand_idx not in negatives:
                negatives.append(rand_idx)
            attempts += 1

        while len(negatives) < total_neg:
            negatives.append(np.random.randint(1, max(self.num_items, 2)))

        return negatives[:total_neg]


INTERACTION_WEIGHTS: Dict[str, float] = {
    "view": 1.0,
    "click": 2.0,
    "add_to_cart": 3.0,
    "purchase": 5.0,
    "favorite": 2.5,
    "share": 1.5,
}


class TwoTowerTrainer:
    """End-to-end trainer for the Two-Tower model."""

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
        architecture: str = "dcn",
        cross_layers: int = 3,
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
        self.architecture = normalize_architecture(architecture)
        self.cross_layers = max(0, int(cross_layers))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[TwoTowerModel] = None
        self.negative_sampler: Optional[NegativeSampler] = None

        self.num_hard_negatives = num_hard_negatives
        self.num_random_negatives = num_random_negatives
        self.hard_ratio_start = hard_ratio_start
        self.hard_ratio_end = hard_ratio_end

        self.user_mapping: Dict[str, int] = {}
        self.item_mapping: Dict[str, int] = {}
        self.reverse_item_mapping: Dict[int, str] = {}
        self.user_positives: Dict[int, Set[int]] = defaultdict(set)

        self._item_clip_embs: Optional[np.ndarray] = None
        self._item_side_feats: Optional[np.ndarray] = None
        self._last_item_embeddings: Optional[np.ndarray] = None
        self._user_side_feats: Dict[int, np.ndarray] = {}
        self._train_samples: List[Tuple[int, int, float]] = []

    def prepare(
        self,
        interactions: List[Dict[str, Any]],
        product_metadata: Dict[str, Dict[str, Any]],
        product_clip_embeddings: Dict[str, np.ndarray],
        user_features_map: Dict[str, Dict[str, Any]],
    ):
        logger.info(f"Preparing training data from {len(interactions)} interactions")

        users: Set[str] = set()
        items: Set[str] = set()
        for interaction in interactions:
            uid = interaction.get("user_id")
            pid = interaction.get("product_id")
            if uid and pid:
                users.add(uid)
                items.add(pid)

        for pid in product_metadata:
            items.add(pid)

        self.user_mapping = {u: idx + 1 for idx, u in enumerate(sorted(users))}
        self.item_mapping = {p: idx + 1 for idx, p in enumerate(sorted(items))}
        self.reverse_item_mapping = {idx: p for p, idx in self.item_mapping.items()}

        num_users = len(self.user_mapping)
        num_items = len(self.item_mapping)
        logger.info(f"Mapped {num_users} users, {num_items} items")

        self._item_clip_embs = np.zeros((num_items + 1, self.clip_dim), dtype=np.float32)
        self._item_side_feats = np.zeros((num_items + 1, ItemFeatureEncoder.FEATURE_DIM), dtype=np.float32)

        for product_id, idx in self.item_mapping.items():
            clip_emb = product_clip_embeddings.get(product_id)
            if clip_emb is not None:
                emb = np.asarray(clip_emb, dtype=np.float32).flatten()
                if emb.shape[0] == self.clip_dim:
                    self._item_clip_embs[idx] = emb
            meta = product_metadata.get(product_id, {})
            self._item_side_feats[idx] = ItemFeatureEncoder.encode(meta)

        self._user_side_feats = {}
        for user_id, idx in self.user_mapping.items():
            uf = user_features_map.get(user_id, {})
            self._user_side_feats[idx] = UserFeatureEncoder.encode(uf)

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

        self.model = TwoTowerModel(
            num_users=num_users,
            num_items=num_items,
            clip_dim=self.clip_dim,
            output_dim=self.output_dim,
            temperature=self.temperature,
            user_hidden_dims=self.user_hidden_dims,
            item_hidden_dims=self.item_hidden_dims,
            architecture=self.architecture,
            cross_layers=self.cross_layers,
        ).to(self.device)

        self.negative_sampler = NegativeSampler(
            num_items=num_items,
            num_hard=self.num_hard_negatives,
            num_random=self.num_random_negatives,
            hard_ratio_start=self.hard_ratio_start,
            hard_ratio_end=self.hard_ratio_end,
        )

    def train(self, existing_cf_index: Optional[faiss.Index] = None) -> Dict[str, Any]:
        if self.model is None or not self._train_samples:
            logger.warning("Model or training data not prepared; skipping training")
            return {"status": "skipped"}

        if (
            existing_cf_index is not None
            and existing_cf_index.ntotal > 0
            and existing_cf_index.ntotal == len(self.item_mapping)
        ):
            idx_map = {i: i + 1 for i in range(existing_cf_index.ntotal)}
            self.negative_sampler.set_index(existing_cf_index, idx_map)
        elif existing_cf_index is not None and existing_cf_index.ntotal > 0:
            logger.info(
                "Skipping warm hard-negative index because item count changed: "
                f"index_items={existing_cf_index.ntotal}, current_items={len(self.item_mapping)}"
            )

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.model.train()

        num_samples = len(self._train_samples)
        stats: Dict[str, Any] = {"epoch_losses": [], "epoch_accs": []}
        total_neg = self.negative_sampler.total_negatives
        start_time = time.time()

        for epoch in range(self.epochs):
            indices = np.random.permutation(num_samples)
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0

            for batch_start in range(0, num_samples, self.batch_size):
                batch_indices = indices[batch_start : batch_start + self.batch_size]
                actual_bs = len(batch_indices)

                user_ids_np = np.zeros(actual_bs, dtype=np.int64)
                user_feats_np = np.zeros((actual_bs, UserFeatureEncoder.FEATURE_DIM), dtype=np.float32)
                pos_item_ids_np = np.zeros(actual_bs, dtype=np.int64)
                pos_clip_np = np.zeros((actual_bs, self.clip_dim), dtype=np.float32)
                pos_feat_np = np.zeros((actual_bs, ItemFeatureEncoder.FEATURE_DIM), dtype=np.float32)
                neg_item_ids_np = np.zeros((actual_bs, total_neg), dtype=np.int64)
                neg_clip_np = np.zeros((actual_bs, total_neg, self.clip_dim), dtype=np.float32)
                neg_feat_np = np.zeros((actual_bs, total_neg, ItemFeatureEncoder.FEATURE_DIM), dtype=np.float32)

                for i, sample_idx in enumerate(batch_indices):
                    u_idx, p_idx, _ = self._train_samples[sample_idx]

                    user_ids_np[i] = u_idx
                    user_feats_np[i] = self._user_side_feats.get(
                        u_idx,
                        np.zeros(UserFeatureEncoder.FEATURE_DIM, dtype=np.float32),
                    )

                    pos_item_ids_np[i] = p_idx
                    pos_clip_np[i] = self._item_clip_embs[p_idx]
                    pos_feat_np[i] = self._item_side_feats[p_idx]

                user_embeddings_np: Optional[np.ndarray] = None
                hard_count = int(
                    total_neg * self.negative_sampler.get_hard_ratio(epoch, self.epochs)
                )
                if (
                    hard_count > 0
                    and self.negative_sampler.item_index is not None
                    and self.negative_sampler.item_index.ntotal > 0
                ):
                    was_training = self.model.training
                    self.model.eval()
                    with torch.no_grad():
                        user_ids_t = torch.tensor(user_ids_np, dtype=torch.long, device=self.device)
                        user_feats_t = torch.tensor(user_feats_np, device=self.device)
                        user_embeddings_np = self.model.encode_users(
                            user_ids_t,
                            user_feats_t,
                        ).cpu().numpy()
                    if was_training:
                        self.model.train()

                for i, sample_idx in enumerate(batch_indices):
                    u_idx, _, _ = self._train_samples[sample_idx]
                    neg_indices = self.negative_sampler.sample(
                        user_embedding=(
                            user_embeddings_np[i]
                            if user_embeddings_np is not None
                            else None
                        ),
                        positive_items=self.user_positives.get(u_idx, set()),
                        epoch=epoch,
                        total_epochs=self.epochs,
                    )
                    for j, neg_idx in enumerate(neg_indices):
                        neg_item_ids_np[i, j] = neg_idx
                        if neg_idx < len(self._item_clip_embs):
                            neg_clip_np[i, j] = self._item_clip_embs[neg_idx]
                            neg_feat_np[i, j] = self._item_side_feats[neg_idx]

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
                    user_ids_t,
                    user_feats_t,
                    pos_ids_t,
                    pos_clip_t,
                    pos_feat_t,
                    neg_ids_t,
                    neg_clip_t,
                    neg_feat_t,
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

            if epoch > 0 and epoch % 5 == 0:
                try:
                    cf_index, idx_map = self.build_item_index()
                    self.negative_sampler.set_index(cf_index, idx_map)
                    logger.info(f"Updated hard-neg index at epoch {epoch}")
                except Exception as exc:
                    logger.warning(f"Failed to update hard-neg index: {exc}")

        self.model.eval()
        training_time = time.time() - start_time
        stats["training_time_s"] = training_time
        stats["num_users"] = len(self.user_mapping)
        stats["num_items"] = len(self.item_mapping)
        stats["num_samples"] = num_samples
        logger.info(f"Two-Tower training completed in {training_time:.1f}s")
        return stats

    @torch.no_grad()
    def build_item_index(self) -> Tuple[faiss.Index, Dict[int, int]]:
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
            clip_t = torch.tensor(self._item_clip_embs[batch_idxs], device=self.device)
            feat_t = torch.tensor(self._item_side_feats[batch_idxs], device=self.device)

            embs = self.model.encode_items(ids_t, clip_t, feat_t).cpu().numpy()
            for i, idx in enumerate(batch_idxs):
                all_embeddings[idx - 1] = embs[i]

        index = faiss.IndexHNSWFlat(self.output_dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
        index.add(all_embeddings)
        self._last_item_embeddings = all_embeddings

        idx_map = {i: i + 1 for i in range(num_items)}
        logger.info(f"Built CF FAISS index with {index.ntotal} item embeddings")
        return index, idx_map

    def get_item_embedding_map(self) -> Dict[str, np.ndarray]:
        """Return product_id -> latest trained item embedding."""
        if self._last_item_embeddings is None:
            return {}
        embeddings: Dict[str, np.ndarray] = {}
        for product_id, item_idx in self.item_mapping.items():
            if 1 <= item_idx <= len(self._last_item_embeddings):
                embeddings[product_id] = np.asarray(
                    self._last_item_embeddings[item_idx - 1],
                    dtype=np.float32,
                )
        return embeddings

    def get_item_side_feature_map(self) -> Dict[str, np.ndarray]:
        if self._item_side_feats is None:
            return {}
        features: Dict[str, np.ndarray] = {}
        for product_id, item_idx in self.item_mapping.items():
            if 0 <= item_idx < len(self._item_side_feats):
                features[product_id] = np.asarray(
                    self._item_side_feats[item_idx],
                    dtype=np.float32,
                )
        return features

    def get_item_clip_available_map(self) -> Dict[str, bool]:
        if self._item_clip_embs is None:
            return {}
        available: Dict[str, bool] = {}
        for product_id, item_idx in self.item_mapping.items():
            if 0 <= item_idx < len(self._item_clip_embs):
                available[product_id] = bool(np.linalg.norm(self._item_clip_embs[item_idx]) > 0.0)
        return available

    @torch.no_grad()
    def encode_user(
        self,
        user_id: str,
        user_features: Dict[str, Any],
        current_time: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        self.model.eval()
        u_idx = self.user_mapping.get(user_id, 0)
        u_feat = UserFeatureEncoder.encode(user_features, current_time=current_time)
        ids_t = torch.tensor([u_idx], dtype=torch.long, device=self.device)
        feat_t = torch.as_tensor(
            np.expand_dims(u_feat, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        return self.model.encode_users(ids_t, feat_t).cpu().numpy()[0]

    def save_checkpoint(self, path: str):
        if self.model is None:
            logger.warning("No model to save")
            return

        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
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
                "architecture": self.model.architecture,
                "cross_layers": self.model.cross_layers,
                "user_hidden_dims": self.user_hidden_dims,
                "item_hidden_dims": self.item_hidden_dims,
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved Two-Tower checkpoint to {path}")

    def _load_shape_compatible_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        skip_prefixes: Tuple[str, ...] = (),
    ) -> int:
        if self.model is None:
            return 0

        current_state = self.model.state_dict()
        compatible: Dict[str, torch.Tensor] = {}
        skipped: List[str] = []

        for key, value in state_dict.items():
            if skip_prefixes and key.startswith(skip_prefixes):
                skipped.append(key)
                continue
            current_value = current_state.get(key)
            if current_value is not None and tuple(current_value.shape) == tuple(value.shape):
                compatible[key] = value
            else:
                skipped.append(key)

        if not compatible:
            logger.warning("No shape-compatible Two-Tower checkpoint tensors found")
            return 0

        load_result = self.model.load_state_dict(compatible, strict=False)
        if skipped or load_result.missing_keys or load_result.unexpected_keys:
            logger.info(
                "Two-Tower checkpoint warm-start loaded partially: "
                f"loaded={len(compatible)}, skipped={len(skipped)}, "
                f"missing={len(load_result.missing_keys)}, "
                f"unexpected={len(load_result.unexpected_keys)}"
            )
        return len(compatible)

    def warm_start_from_checkpoint(self, path: str) -> bool:
        """Load shape-compatible tensors into the already-prepared model."""
        checkpoint_path = Path(path)
        if self.model is None or not checkpoint_path.exists():
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            skip_prefixes: List[str] = []
            if checkpoint.get("user_mapping") != self.user_mapping:
                skip_prefixes.append("user_tower.user_embedding.")
            if checkpoint.get("item_mapping") != self.item_mapping:
                skip_prefixes.append("item_tower.item_embedding.")

            loaded = self._load_shape_compatible_state_dict(
                state_dict,
                skip_prefixes=tuple(skip_prefixes),
            )
            if loaded:
                logger.info(
                    "Warm-started Two-Tower training checkpoint",
                    extra={
                        "path": str(checkpoint_path),
                        "loaded_tensors": loaded,
                        "target_architecture": self.model.architecture,
                    },
                )
            return loaded > 0
        except Exception as exc:
            logger.warning(f"Failed to warm-start Two-Tower checkpoint: {exc}")
            return False

    def load_checkpoint(self, path: str) -> bool:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)
            cfg = checkpoint["config"]

            self.user_mapping = checkpoint["user_mapping"]
            self.item_mapping = checkpoint["item_mapping"]
            self.reverse_item_mapping = checkpoint["reverse_item_mapping"]
            architecture = normalize_architecture(cfg.get("architecture"), default="mlp")
            cross_layers = int(cfg.get("cross_layers", self.cross_layers))

            self.model = TwoTowerModel(
                num_users=cfg["num_users"],
                num_items=cfg["num_items"],
                clip_dim=cfg.get("clip_dim", self.clip_dim),
                output_dim=cfg.get("output_dim", self.output_dim),
                temperature=cfg.get("temperature", self.temperature),
                user_hidden_dims=cfg.get("user_hidden_dims", self.user_hidden_dims),
                item_hidden_dims=cfg.get("item_hidden_dims", self.item_hidden_dims),
                architecture=architecture,
                cross_layers=cross_layers,
            ).to(self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            num_items = len(self.item_mapping)
            if self._item_clip_embs is None or len(self._item_clip_embs) != num_items + 1:
                self._item_clip_embs = np.zeros((num_items + 1, self.clip_dim), dtype=np.float32)
                self._item_side_feats = np.zeros((num_items + 1, ItemFeatureEncoder.FEATURE_DIM), dtype=np.float32)

            logger.info(f"Loaded Two-Tower checkpoint from {path}")
            return True
        except Exception as exc:
            logger.error(f"Failed to load checkpoint: {exc}")
            return False
