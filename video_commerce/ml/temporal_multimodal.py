"""Trainable temporal pooling and candidate-conditioned multimodal attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# PyTorch 2.1's oneDNN kernels are unstable in the linux/arm64 backend image
# (even a small Linear can fail to create a matmul primitive). Ranking inference
# already uses the eager kernels for this reason; keep this standalone module
# safe when it is imported without ``video_commerce.ml.ranking`` first.
if torch.backends.mkldnn.is_available():
    torch.backends.mkldnn.enabled = False


def encode_float16_matrix(values: np.ndarray) -> bytes:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    return np.ascontiguousarray(matrix.astype("<f2", copy=False)).tobytes()


def decode_float16_matrix(payload: bytes, *, rows: int, columns: int) -> np.ndarray:
    rows = int(rows)
    columns = int(columns)
    if rows < 0 or columns <= 0:
        raise ValueError("matrix shape must be non-negative")
    expected = rows * columns * np.dtype("<f2").itemsize
    if len(payload) != expected:
        raise ValueError(
            f"float16 matrix byte length {len(payload)} does not match expected {expected}"
        )
    return np.frombuffer(payload, dtype="<f2").reshape(rows, columns).astype(np.float32)


@dataclass
class TemporalEncoderOutput:
    tokens: torch.Tensor
    pooled_embedding: torch.Tensor
    attention_weights: torch.Tensor


class TemporalContentEncoder(nn.Module):
    """Contextualize frozen frame embeddings while retaining their source space."""

    def __init__(
        self,
        *,
        input_dim: int = 512,
        model_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.model_dim),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(2, self.model_dim),
            nn.ReLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(num_heads),
            dim_feedforward=int(feedforward_dim),
            dropout=float(dropout),
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.pool_score_weight = nn.Parameter(torch.empty(self.model_dim))
        self.pool_score_bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.pool_score_weight, mean=0.0, std=self.model_dim**-0.5)
        self.residual_gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        timestamps_seconds: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> TemporalEncoderOutput:
        if frame_embeddings.ndim != 3:
            raise ValueError("frame_embeddings must have shape [batch, frames, dim]")
        if frame_embeddings.shape[-1] != self.input_dim:
            raise ValueError("frame embedding dimension does not match encoder input")
        valid_mask = valid_mask.to(dtype=torch.bool, device=frame_embeddings.device)
        timestamps_seconds = timestamps_seconds.to(
            dtype=frame_embeddings.dtype,
            device=frame_embeddings.device,
        )
        if valid_mask.shape != frame_embeddings.shape[:2]:
            raise ValueError("valid_mask shape must match frame batch and sequence")
        if timestamps_seconds.shape != valid_mask.shape:
            raise ValueError("timestamps shape must match valid_mask")
        if not torch.all(valid_mask.any(dim=1)):
            raise ValueError("each temporal sequence must include a valid frame")

        masked_timestamps = timestamps_seconds.masked_fill(~valid_mask, 0.0)
        duration = masked_timestamps.max(dim=1, keepdim=True).values.clamp_min(1.0)
        normalized_time = masked_timestamps / duration
        deltas = torch.zeros_like(masked_timestamps)
        deltas[:, 1:] = (
            masked_timestamps[:, 1:] - masked_timestamps[:, :-1]
        ).clamp_min(0.0)
        normalized_delta = torch.log1p(deltas) / torch.log1p(duration)
        time_features = torch.stack([normalized_time, normalized_delta], dim=-1)

        tokens = self.input_projection(frame_embeddings) + self.time_projection(
            time_features
        )
        tokens = self.encoder(tokens, src_key_padding_mask=~valid_mask)
        # TransformerEncoder can return a non-standard view on linux/arm64;
        # materialize it before the following projection.
        tokens = tokens.contiguous()
        scores = (
            torch.sum(tokens * self.pool_score_weight, dim=-1) + self.pool_score_bias
        ).masked_fill(~valid_mask, -torch.inf)
        weights = torch.softmax(scores, dim=-1).masked_fill(~valid_mask, 0.0)

        raw = F.normalize(frame_embeddings, dim=-1)
        weighted = torch.sum(raw * weights.unsqueeze(-1), dim=1)
        mask_values = valid_mask.unsqueeze(-1).to(raw.dtype)
        mean = torch.sum(raw * mask_values, dim=1) / mask_values.sum(dim=1).clamp_min(
            1.0
        )
        gate = torch.sigmoid(self.residual_gate_logit)
        pooled = F.normalize((1.0 - gate) * mean + gate * weighted, dim=-1)
        return TemporalEncoderOutput(
            tokens=tokens, pooled_embedding=pooled, attention_weights=weights
        )


@dataclass
class CandidateAttentionOutput:
    fused: torch.Tensor
    visual_attention: torch.Tensor
    text_attention: torch.Tensor


class CandidateMultimodalAttention(nn.Module):
    """Attend to video and OCR tokens with a candidate-specific query."""

    def __init__(
        self,
        *,
        visual_input_dim: int = 128,
        text_input_dim: int = 512,
        candidate_embedding_dim: Optional[int] = None,
        two_tower_dim: int = 128,
        model_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        candidate_embedding_dim = int(candidate_embedding_dim or text_input_dim)
        self.visual_projection = nn.Linear(int(visual_input_dim), self.model_dim)
        self.text_projection = nn.Linear(int(text_input_dim), self.model_dim)
        self.image_query_projection = nn.Linear(candidate_embedding_dim, self.model_dim)
        self.text_query_projection = nn.Linear(candidate_embedding_dim, self.model_dim)
        self.two_tower_projection = nn.Linear(int(two_tower_dim), self.model_dim)
        self.query_fusion = nn.Sequential(
            nn.Linear(self.model_dim * 3, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(),
        )
        self.visual_attention = nn.MultiheadAttention(
            self.model_dim, int(num_heads), dropout=float(dropout), batch_first=True
        )
        self.text_attention = nn.MultiheadAttention(
            self.model_dim, int(num_heads), dropout=float(dropout), batch_first=True
        )
        self.output_fusion = nn.Sequential(
            nn.Linear(self.model_dim * 2 + 2, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        *,
        visual_tokens: torch.Tensor,
        visual_mask: torch.Tensor,
        candidate_image: torch.Tensor,
        candidate_text: torch.Tensor,
        candidate_two_tower: torch.Tensor,
        ocr_tokens: Optional[torch.Tensor] = None,
        ocr_mask: Optional[torch.Tensor] = None,
    ) -> CandidateAttentionOutput:
        visual_mask = visual_mask.to(dtype=torch.bool, device=visual_tokens.device)
        query = self.query_fusion(
            torch.cat(
                [
                    self.image_query_projection(candidate_image),
                    self.text_query_projection(candidate_text),
                    self.two_tower_projection(candidate_two_tower),
                ],
                dim=-1,
            )
        ).unsqueeze(1)
        visual_values = self.visual_projection(visual_tokens)
        visual_output, visual_weights = self.visual_attention(
            query,
            visual_values,
            visual_values,
            key_padding_mask=~visual_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        visual_output = visual_output.squeeze(1)
        visual_weights = visual_weights.squeeze(1).masked_fill(~visual_mask, 0.0)

        batch_size = visual_tokens.shape[0]
        text_output = torch.zeros(
            batch_size,
            self.model_dim,
            dtype=visual_output.dtype,
            device=visual_output.device,
        )
        if ocr_tokens is None:
            ocr_tokens = torch.zeros(
                batch_size,
                1,
                self.text_projection.in_features,
                dtype=visual_tokens.dtype,
                device=visual_tokens.device,
            )
        if ocr_mask is None:
            ocr_mask = torch.zeros(
                batch_size,
                ocr_tokens.shape[1],
                dtype=torch.bool,
                device=visual_tokens.device,
            )
        else:
            ocr_mask = ocr_mask.to(dtype=torch.bool, device=visual_tokens.device)
        text_weights = torch.zeros(
            batch_size,
            ocr_tokens.shape[1],
            dtype=visual_output.dtype,
            device=visual_output.device,
        )
        rows_with_text = ocr_mask.any(dim=1)
        if rows_with_text.any():
            text_values = self.text_projection(ocr_tokens[rows_with_text])
            attended, weights = self.text_attention(
                query[rows_with_text],
                text_values,
                text_values,
                key_padding_mask=~ocr_mask[rows_with_text],
                need_weights=True,
                average_attn_weights=True,
            )
            text_output[rows_with_text] = attended.squeeze(1)
            text_weights[rows_with_text] = weights.squeeze(1).masked_fill(
                ~ocr_mask[rows_with_text], 0.0
            )

        visual_present = visual_mask.any(dim=1, keepdim=True).to(visual_output.dtype)
        text_present = ocr_mask.any(dim=1, keepdim=True).to(visual_output.dtype)
        fused = self.output_fusion(
            torch.cat(
                [visual_output, text_output, visual_present, text_present], dim=-1
            )
        )
        return CandidateAttentionOutput(
            fused=fused,
            visual_attention=visual_weights,
            text_attention=text_weights,
        )
