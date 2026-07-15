"""Trainable temporal pooling and candidate-conditioned multimodal attention."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Optional

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


class _ScalarScore(nn.Module):
    """Linear scalar projection without the broken ARM oneDNN Mx1 kernel."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(int(input_dim)))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.weight, mean=0.0, std=int(input_dim) ** -0.5)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sum(values * self.weight, dim=-1, keepdim=True) + self.bias


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


def _pack_temporal_records(
    records: list[tuple[Any, float, float]],
    *,
    dimension: int,
    capacity: int,
) -> dict[str, Any] | None:
    records = records[:capacity]
    if not records:
        return None
    matrix = np.asarray([record[0] for record in records], dtype=np.float32)
    if matrix.shape != (len(records), dimension):
        raise ValueError(f"temporal embedding dimension must be {dimension}")
    if not np.isfinite(matrix).all():
        raise ValueError("temporal embeddings must be finite")
    starts = [float(record[1]) for record in records]
    ends = [float(record[2]) for record in records]
    return {
        "data": base64.b64encode(encode_float16_matrix(matrix)).decode("ascii"),
        "rows": len(records),
        "columns": dimension,
        "starts": starts,
        "ends": ends,
        "mask": [True] * len(records),
    }


def pack_content_features(features: Any) -> dict[str, Any]:
    """Build the bounded ranking-v4 wire representation without transcript text."""
    context: dict[str, Any] = {"schema_version": "temporal_multimodal_v2"}
    visual = _pack_temporal_records(
        [
            (embedding, timestamp, timestamp)
            for embedding, timestamp in zip(
                list(features.frame_embeddings),
                list(features.frame_timestamps_seconds),
            )
        ],
        dimension=512,
        capacity=16,
    )
    ocr = _pack_temporal_records(
        [
            (
                track.get("text_embedding") or [],
                track.get("first_seen_seconds") or 0.0,
                track.get("last_seen_seconds")
                if track.get("last_seen_seconds") is not None
                else track.get("first_seen_seconds") or 0.0,
            )
            for track in list(features.ocr_tracks)
            if track.get("text_embedding")
        ],
        dimension=384,
        capacity=32,
    )
    segments = (
        list(features.audio_features.asr_segments)
        if features.audio_features is not None
        else []
    )
    asr = _pack_temporal_records(
        [
            (
                segment.get("text_embedding") or [],
                segment.get("start_seconds") or 0.0,
                segment.get("end_seconds")
                if segment.get("end_seconds") is not None
                else segment.get("start_seconds") or 0.0,
            )
            for segment in segments
            if segment.get("text_embedding")
        ],
        dimension=384,
        capacity=64,
    )
    for name, sequence in (("visual", visual), ("ocr", ocr), ("asr", asr)):
        if sequence is not None:
            context[name] = sequence
    return context


def unpack_temporal_multimodal_context(
    context: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Decode a validated v4 context into modality arrays and temporal masks."""
    output: dict[str, np.ndarray] = {}
    for modality in ("visual", "ocr", "asr"):
        sequence = context.get(modality)
        if not isinstance(sequence, dict):
            continue
        rows, columns = int(sequence["rows"]), int(sequence["columns"])
        packed = base64.b64decode(str(sequence["data"]), validate=True)
        output[f"{modality}_embeddings"] = decode_float16_matrix(
            packed, rows=rows, columns=columns
        )
        output[f"{modality}_starts"] = np.asarray(sequence["starts"], dtype=np.float32)
        output[f"{modality}_ends"] = np.asarray(sequence["ends"], dtype=np.float32)
        output[f"{modality}_mask"] = np.asarray(sequence["mask"], dtype=np.bool_)
    return output


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


class TemporalSequenceEncoder(nn.Module):
    """Encode one frozen modality sequence with real temporal span features."""

    def __init__(
        self,
        *,
        input_dim: int,
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
            nn.Linear(3, self.model_dim),
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
        self.pool_score = _ScalarScore(self.model_dim)
        self.mean_residual_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        embeddings: torch.Tensor,
        timestamps_seconds: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        end_timestamps_seconds: Optional[torch.Tensor] = None,
    ) -> TemporalEncoderOutput:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.input_dim:
            raise ValueError("embeddings must have shape [batch, sequence, input_dim]")
        mask = valid_mask.to(dtype=torch.bool, device=embeddings.device)
        starts = timestamps_seconds.to(dtype=embeddings.dtype, device=embeddings.device)
        ends = (
            end_timestamps_seconds.to(dtype=embeddings.dtype, device=embeddings.device)
            if end_timestamps_seconds is not None
            else starts
        )
        if mask.shape != embeddings.shape[:2] or starts.shape != mask.shape:
            raise ValueError("timestamp and mask shapes must match the sequence")
        if ends.shape != mask.shape:
            raise ValueError("end timestamp shape must match the sequence")

        starts = starts.masked_fill(~mask, 0.0)
        ends = torch.maximum(ends.masked_fill(~mask, 0.0), starts)
        duration = ends.max(dim=1, keepdim=True).values.clamp_min(1.0)
        deltas = torch.zeros_like(starts)
        deltas[:, 1:] = (starts[:, 1:] - starts[:, :-1]).clamp_min(0.0)
        temporal = torch.stack(
            [
                starts / duration,
                torch.log1p(deltas) / torch.log1p(duration),
                (ends - starts) / duration,
            ],
            dim=-1,
        )
        tokens = self.input_projection(embeddings) + self.time_projection(temporal)

        # Transformer attention cannot consume a row whose key mask is entirely
        # padding. Give such rows one temporary zero token, then erase them.
        safe_mask = mask.clone()
        missing_rows = ~safe_mask.any(dim=1)
        if safe_mask.shape[1] == 0:
            raise ValueError("temporal sequences require at least one padded slot")
        safe_mask[missing_rows, 0] = True
        tokens = tokens.masked_fill(~mask.unsqueeze(-1), 0.0)
        tokens = self.encoder(tokens, src_key_padding_mask=~safe_mask).contiguous()
        tokens = tokens.masked_fill(~mask.unsqueeze(-1), 0.0)

        scores = self.pool_score(tokens).squeeze(-1).masked_fill(~safe_mask, -torch.inf)
        weights = torch.softmax(scores, dim=-1).masked_fill(~mask, 0.0)
        learned = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        mask_values = mask.unsqueeze(-1).to(tokens.dtype)
        mean = torch.sum(tokens * mask_values, dim=1) / mask_values.sum(
            dim=1
        ).clamp_min(1.0)
        residual = torch.sigmoid(self.mean_residual_logit)
        pooled = (1.0 - residual) * learned + residual * mean
        pooled = pooled.masked_fill(missing_rows.unsqueeze(-1), 0.0)
        return TemporalEncoderOutput(tokens, pooled, weights)


@dataclass
class TrimodalAttentionOutput:
    fused: torch.Tensor
    modality_gate: torch.Tensor
    visual_attention: torch.Tensor
    ocr_attention: torch.Tensor
    asr_attention: torch.Tensor


class TrimodalCandidateAttention(nn.Module):
    """Candidate-conditioned cross-attention and presence-aware modality gating."""

    def __init__(
        self,
        *,
        model_dim: int = 128,
        candidate_image_dim: int = 512,
        candidate_text_dim: int = 384,
        candidate_two_tower_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.image_query = nn.Linear(int(candidate_image_dim), self.model_dim)
        self.text_query = nn.Linear(int(candidate_text_dim), self.model_dim)
        self.two_tower_query = nn.Linear(int(candidate_two_tower_dim), self.model_dim)
        self.query_fusion = nn.Sequential(
            nn.Linear(self.model_dim * 3, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.ReLU(),
        )
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.model_dim,
                    int(num_heads),
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(3)
            ]
        )
        self.modality_fusions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.model_dim * 2, self.model_dim),
                    nn.LayerNorm(self.model_dim),
                    nn.ReLU(),
                )
                for _ in range(3)
            ]
        )
        self.gate_scores = nn.ModuleList(
            [_ScalarScore(self.model_dim) for _ in range(3)]
        )
        self.residual_gate_logit = nn.Parameter(torch.tensor(-4.0))

    @staticmethod
    def _attend_present_rows(
        attention: nn.MultiheadAttention,
        query: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence, dim = tokens.shape
        output = tokens.new_zeros((batch, dim))
        weights = tokens.new_zeros((batch, sequence))
        present = mask.any(dim=1)
        if present.any():
            attended, row_weights = attention(
                query[present],
                tokens[present],
                tokens[present],
                key_padding_mask=~mask[present],
                need_weights=True,
                average_attn_weights=True,
            )
            output[present] = attended.squeeze(1)
            weights[present] = row_weights.squeeze(1).masked_fill(~mask[present], 0.0)
        return output, weights

    def forward(
        self,
        *,
        visual_tokens: torch.Tensor,
        visual_pooled: torch.Tensor,
        visual_mask: torch.Tensor,
        ocr_tokens: torch.Tensor,
        ocr_pooled: torch.Tensor,
        ocr_mask: torch.Tensor,
        asr_tokens: torch.Tensor,
        asr_pooled: torch.Tensor,
        asr_mask: torch.Tensor,
        candidate_image: torch.Tensor,
        candidate_text: torch.Tensor,
        candidate_two_tower: torch.Tensor,
        candidate_presence: Optional[torch.Tensor] = None,
    ) -> TrimodalAttentionOutput:
        masks = [
            visual_mask.to(dtype=torch.bool, device=visual_tokens.device),
            ocr_mask.to(dtype=torch.bool, device=visual_tokens.device),
            asr_mask.to(dtype=torch.bool, device=visual_tokens.device),
        ]
        candidates = [candidate_image, candidate_text, candidate_two_tower]
        if candidate_presence is None:
            candidate_presence = torch.ones(
                candidate_image.shape[0],
                3,
                dtype=torch.bool,
                device=visual_tokens.device,
            )
        else:
            candidate_presence = candidate_presence.to(
                dtype=torch.bool, device=visual_tokens.device
            )
        projected = [
            projection(value) * candidate_presence[:, index : index + 1]
            for index, (projection, value) in enumerate(
                zip(
                    (self.image_query, self.text_query, self.two_tower_query),
                    candidates,
                )
            )
        ]
        query = self.query_fusion(torch.cat(projected, dim=-1)).unsqueeze(1)
        token_groups = [visual_tokens, ocr_tokens, asr_tokens]
        pooled_groups = [visual_pooled, ocr_pooled, asr_pooled]
        attended_groups = []
        attention_weights = []
        fused_groups = []
        for attention, fusion, tokens, pooled, mask in zip(
            self.attentions, self.modality_fusions, token_groups, pooled_groups, masks
        ):
            attended, weights = self._attend_present_rows(
                attention, query, tokens, mask
            )
            attended_groups.append(attended)
            attention_weights.append(weights)
            fused_groups.append(fusion(torch.cat([pooled, attended], dim=-1)))

        presence = torch.stack([mask.any(dim=1) for mask in masks], dim=1)
        gate_logits = torch.cat(
            [score(value) for score, value in zip(self.gate_scores, fused_groups)],
            dim=1,
        ).masked_fill(~presence, -torch.inf)
        all_missing = ~presence.any(dim=1)
        gate_logits = gate_logits.masked_fill(all_missing.unsqueeze(1), 0.0)
        modality_gate = torch.softmax(gate_logits, dim=1).masked_fill(~presence, 0.0)
        fused = sum(
            value * modality_gate[:, index : index + 1]
            for index, value in enumerate(fused_groups)
        )
        fused = fused * torch.sigmoid(self.residual_gate_logit)
        fused = fused.masked_fill(all_missing.unsqueeze(1), 0.0)
        return TrimodalAttentionOutput(
            fused=fused,
            modality_gate=modality_gate,
            visual_attention=attention_weights[0],
            ocr_attention=attention_weights[1],
            asr_attention=attention_weights[2],
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
