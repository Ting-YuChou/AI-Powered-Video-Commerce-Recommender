"""Frozen multilingual text embeddings shared by OCR, ASR, and catalog sidecars."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F


class MultilingualTextEmbedder:
    """Mean-pool a pinned E5 encoder with explicit query/passage prefixes."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: torch.device,
        cache_dir: Optional[str] = None,
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        self.model_id = str(model_id)
        self.revision = str(revision)
        self.device = device
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.model = model.eval() if model is not None else None

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, texts: Iterable[str], *, prefix: str) -> list[list[float]]:
        values = [str(text or "").strip() for text in texts]
        if not values:
            return []
        self.load()
        prefixed = [f"{prefix}: {text}" for text in values]
        inputs = self.tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
            mask = (
                inputs["attention_mask"]
                .unsqueeze(-1)
                .to(output.last_hidden_state.dtype)
            )
            pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(
                dim=1
            ).clamp_min(1.0)
            pooled = F.normalize(pooled, dim=-1)
        return pooled.cpu().to(torch.float32).tolist()
