"""
Sequential recommendation candidate source based on SASRec.

This module keeps SASRec isolated as a retrieval-only source. It consumes
positive chronological product sequences, persists a checkpoint plus vocabulary,
and returns CandidateProduct entries that are compatible with the existing
ranking feature schema.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch import nn, optim

from video_commerce.common.config import RecommendationConfig
from video_commerce.common.models import CandidateProduct

logger = logging.getLogger(__name__)


class SASRecModel(nn.Module):
    """Minimal SASRec-style next-item predictor."""

    def __init__(
        self,
        *,
        num_items: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embedding_dim, num_items + 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.item_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        padding_mask = input_ids.eq(0)
        encoded = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return self.output(encoded)


class SASRecCandidateEngine:
    """Train, load, and serve SASRec candidates."""

    def __init__(self, config: RecommendationConfig, device: Optional[torch.device] = None) -> None:
        self.config = config
        self.max_sequence_length = max(1, int(config.sasrec_max_sequence_length))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[SASRecModel] = None
        self.product_to_id: Dict[str, int] = {}
        self.id_to_product: Dict[int, str] = {}
        self.is_trained = False
        self.model_version: Optional[str] = None
        self.last_training_time: float = 0.0
        self.training_sample_count = 0

    async def train_model(
        self,
        sequences: Dict[str, List[Dict[str, Any]]],
        *,
        catalog_product_ids: Iterable[str],
    ) -> bool:
        return await asyncio.to_thread(
            self._train_model_sync,
            sequences,
            list(catalog_product_ids),
        )

    def _train_model_sync(
        self,
        sequences: Dict[str, List[Dict[str, Any]]],
        catalog_product_ids: List[str],
    ) -> bool:
        torch.manual_seed(42)
        np.random.seed(42)

        catalog = sorted({product_id for product_id in catalog_product_ids if product_id})
        if not catalog:
            logger.warning("Skipping SASRec training because catalog is empty")
            self.is_trained = False
            return False

        self.product_to_id = {product_id: idx + 1 for idx, product_id in enumerate(catalog)}
        self.id_to_product = {idx: product_id for product_id, idx in self.product_to_id.items()}
        samples = self._build_training_samples(sequences)
        self.training_sample_count = len(samples)
        if not samples:
            logger.warning("Skipping SASRec training because no usable sequences were found")
            self.is_trained = False
            return False

        self.model = SASRecModel(
            num_items=len(self.product_to_id),
            max_sequence_length=self.max_sequence_length,
            embedding_dim=int(self.config.sasrec_embedding_dim),
            num_heads=int(self.config.sasrec_num_heads),
            num_layers=int(self.config.sasrec_num_layers),
            dropout=float(self.config.sasrec_dropout),
        ).to(self.device)
        self.model.train()

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(self.config.sasrec_learning_rate),
            weight_decay=1e-6,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        batch_size = max(1, int(self.config.sasrec_batch_size))
        epochs = max(1, int(self.config.sasrec_epochs))
        started_at = time.time()

        inputs_np = np.stack([sample[0] for sample in samples]).astype(np.int64)
        labels_np = np.stack([sample[1] for sample in samples]).astype(np.int64)
        for epoch in range(epochs):
            permutation = np.random.permutation(len(samples))
            epoch_loss = 0.0
            batch_count = 0
            for start in range(0, len(samples), batch_size):
                batch_idx = permutation[start : start + batch_size]
                input_ids = torch.tensor(inputs_np[batch_idx], dtype=torch.long, device=self.device)
                labels = torch.tensor(labels_np[batch_idx], dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                batch_count += 1
            logger.debug(
                "SASRec epoch %s/%s loss=%.4f",
                epoch + 1,
                epochs,
                epoch_loss / max(batch_count, 1),
            )

        self.model.eval()
        self.is_trained = True
        self.last_training_time = time.time()
        self.model_version = f"sasrec-{int(self.last_training_time)}"
        logger.info(
            "SASRec trained in %.1fs with %s sequences and %s catalog items",
            time.time() - started_at,
            len(samples),
            len(self.product_to_id),
        )
        return True

    def _build_training_samples(
        self,
        sequences: Dict[str, List[Dict[str, Any]]],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        samples: List[Tuple[np.ndarray, np.ndarray]] = []
        min_sequence_length = max(2, int(self.config.sasrec_min_sequence_length))
        for sequence in sequences.values():
            item_ids = [
                self.product_to_id[event.get("product_id")]
                for event in sequence
                if event.get("product_id") in self.product_to_id
            ]
            if len(item_ids) < min_sequence_length:
                continue

            window = item_ids[-(self.max_sequence_length + 1) :]
            input_ids = window[:-1]
            label_ids = window[1:]
            if not input_ids or not label_ids:
                continue

            padded_inputs = np.zeros(self.max_sequence_length, dtype=np.int64)
            padded_labels = np.zeros(self.max_sequence_length, dtype=np.int64)
            length = min(len(input_ids), self.max_sequence_length)
            padded_inputs[:length] = input_ids[-length:]
            padded_labels[:length] = label_ids[-length:]
            samples.append((padded_inputs, padded_labels))
        return samples

    async def get_candidates(
        self,
        sequence: Sequence[Dict[str, Any]],
        *,
        k: int,
        exclude_items: Optional[Set[str]],
        catalog_product_ids: Iterable[str],
    ) -> List[CandidateProduct]:
        return await asyncio.to_thread(
            self._get_candidates_sync,
            list(sequence),
            k,
            set(exclude_items or set()),
            set(catalog_product_ids),
        )

    @torch.no_grad()
    def _get_candidates_sync(
        self,
        sequence: List[Dict[str, Any]],
        k: int,
        exclude_items: Set[str],
        catalog_product_ids: Set[str],
    ) -> List[CandidateProduct]:
        if not self.is_trained or self.model is None or k <= 0:
            return []

        item_ids = [
            self.product_to_id[product_id]
            for product_id in (event.get("product_id") for event in sequence)
            if product_id in self.product_to_id
        ]
        if not item_ids:
            return []

        tail = item_ids[-self.max_sequence_length :]
        input_array = np.zeros(self.max_sequence_length, dtype=np.int64)
        input_array[: len(tail)] = tail
        input_ids = torch.from_numpy(input_array[None, :]).long().to(self.device)
        logits = self.model(input_ids)[0, len(tail) - 1]
        logits[0] = -float("inf")

        for product_id in exclude_items:
            item_id = self.product_to_id.get(product_id)
            if item_id is not None:
                logits[item_id] = -float("inf")

        for item_id, product_id in self.id_to_product.items():
            if product_id not in catalog_product_ids:
                logits[item_id] = -float("inf")

        candidate_count = min(k, len(self.product_to_id))
        if candidate_count <= 0:
            return []

        scores, item_indices = torch.topk(logits, k=candidate_count)
        candidates: List[CandidateProduct] = []
        score_weight = float(self.config.sasrec_score_weight)
        for raw_score, item_id_tensor in zip(scores.detach().cpu(), item_indices.detach().cpu()):
            if len(candidates) >= k:
                break
            raw_value = float(raw_score)
            if not np.isfinite(raw_value):
                continue
            item_id = int(item_id_tensor)
            product_id = self.id_to_product.get(item_id)
            if not product_id or product_id in exclude_items or product_id not in catalog_product_ids:
                continue
            score = min(max(float(torch.sigmoid(torch.tensor(raw_value))) * score_weight, 0.0), 1.0)
            candidates.append(
                CandidateProduct(
                    product_id=product_id,
                    collaborative_score=score,
                    combined_score=score,
                    source="sasrec",
                )
            )
        return candidates

    def save_artifacts(self, checkpoint_path: str, vocab_path: str, metadata_path: str) -> None:
        if self.model is None or not self.is_trained:
            raise RuntimeError("SASRec model is not trained")

        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self._model_config_payload(),
            },
            checkpoint_path,
        )
        vocab_file = Path(vocab_path)
        metadata_file = Path(metadata_path)
        vocab_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        vocab_file.write_text(
            json.dumps(
                {
                    "product_to_id": self.product_to_id,
                    "id_to_product": {str(k): v for k, v in self.id_to_product.items()},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        metadata = {
            "model_version": self.model_version,
            "last_training_time": self.last_training_time,
            "training_sample_count": self.training_sample_count,
            "num_items": len(self.product_to_id),
            **self._model_config_payload(),
        }
        metadata_file.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

    def load_artifacts(self, checkpoint_path: str, vocab_path: str, metadata_path: str) -> bool:
        checkpoint_file = Path(checkpoint_path)
        vocab_file = Path(vocab_path)
        metadata_file = Path(metadata_path)
        if not checkpoint_file.exists() or not vocab_file.exists() or not metadata_file.exists():
            logger.warning("SASRec artifacts are incomplete; skipping load")
            self.is_trained = False
            return False

        try:
            vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            self.product_to_id = {str(k): int(v) for k, v in vocab.get("product_to_id", {}).items()}
            self.id_to_product = {int(k): str(v) for k, v in vocab.get("id_to_product", {}).items()}

            checkpoint = torch.load(str(checkpoint_file), map_location=self.device)
            model_config = checkpoint.get("config") or metadata
            self.max_sequence_length = int(model_config.get("max_sequence_length", self.max_sequence_length))
            self.model = SASRecModel(
                num_items=len(self.product_to_id),
                max_sequence_length=self.max_sequence_length,
                embedding_dim=int(model_config.get("embedding_dim", self.config.sasrec_embedding_dim)),
                num_heads=int(model_config.get("num_heads", self.config.sasrec_num_heads)),
                num_layers=int(model_config.get("num_layers", self.config.sasrec_num_layers)),
                dropout=float(model_config.get("dropout", self.config.sasrec_dropout)),
            ).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.model_version = metadata.get("model_version")
            self.last_training_time = float(metadata.get("last_training_time") or 0.0)
            self.training_sample_count = int(metadata.get("training_sample_count") or 0)
            self.is_trained = True
            logger.info("Loaded SASRec artifacts from %s", checkpoint_path)
            return True
        except Exception as exc:
            logger.error("Failed to load SASRec artifacts: %s", exc)
            self.is_trained = False
            return False

    def _model_config_payload(self) -> Dict[str, Any]:
        return {
            "max_sequence_length": self.max_sequence_length,
            "embedding_dim": int(self.config.sasrec_embedding_dim),
            "num_heads": int(self.config.sasrec_num_heads),
            "num_layers": int(self.config.sasrec_num_layers),
            "dropout": float(self.config.sasrec_dropout),
        }
