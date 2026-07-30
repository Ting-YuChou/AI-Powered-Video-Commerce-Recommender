"""Visual temporal attention pooling for content-based retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from video_commerce.common.models import ContentFeatures
from video_commerce.ml.ranking_training import RankingTrainingExample
from video_commerce.ml.temporal_multimodal import TemporalSequenceEncoder


VISUAL_RETRIEVAL_SCHEMA_VERSION = "visual_retrieval_attention_v1"
_ACTION_WEIGHTS = {"click": 1.0, "add_to_cart": 2.0, "purchase": 3.0}
_LINEAGE_FIELDS = {
    "parent_ranker_checkpoint",
    "clip_model_id",
    "clip_revision",
    "product_index_version",
    "product_index_sha256",
    "catalog_activation_id",
    "pit_manifest_uri",
    "pit_sidecar_sha256",
    "training_cutoff",
}


@dataclass
class VisualRetrievalOutput:
    embedding: torch.Tensor
    attention_weights: torch.Tensor


@dataclass
class VisualRetrievalLoss:
    loss: torch.Tensor
    valid_negative_mask: torch.Tensor


@dataclass(frozen=True)
class VisualRetrievalTrainingExample:
    observation_id: str
    impression_id: str
    content_id: str
    product_id: str
    as_of_ts: float
    product_available_at: float
    frame_embeddings: tuple[tuple[float, ...], ...]
    frame_timestamps_seconds: tuple[float, ...]
    positive_embedding: tuple[float, ...]
    positive_weight: float


@dataclass(frozen=True)
class VisualRetrievalTrainingMetrics:
    training_examples: int
    validation_examples: int
    final_training_loss: float
    validation_loss: float


@dataclass(frozen=True)
class VisualRetrievalEvaluation:
    recall_at: dict[int, float]
    mrr: float
    catalog_coverage: float


class VisualRetrievalPooler(nn.Module):
    """Learn frame importance while keeping the query in frozen CLIP space."""

    def __init__(
        self,
        *,
        input_dim: int = 512,
        model_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        residual_gate_logit: float = -4.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.dropout = float(dropout)
        self.temporal_encoder = TemporalSequenceEncoder(
            input_dim=self.input_dim,
            model_dim=self.model_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
        )
        self.residual_gate_logit = nn.Parameter(
            torch.tensor(float(residual_gate_logit))
        )

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        starts_seconds: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        ends_seconds: Optional[torch.Tensor] = None,
    ) -> VisualRetrievalOutput:
        if frame_embeddings.ndim != 3 or frame_embeddings.shape[-1] != self.input_dim:
            raise ValueError(
                "frame_embeddings must have shape [batch, sequence, input_dim]"
            )
        mask = valid_mask.to(dtype=torch.bool, device=frame_embeddings.device)
        if mask.shape != frame_embeddings.shape[:2]:
            raise ValueError("valid_mask shape must match the frame sequence")
        normalized_frames = F.normalize(frame_embeddings, dim=-1)
        temporal_ends = ends_seconds
        if temporal_ends is None:
            temporal_ends = starts_seconds.clone()
            if temporal_ends.shape[1] > 1:
                adjacent = mask[:, :-1] & mask[:, 1:]
                temporal_ends[:, :-1] = torch.where(
                    adjacent,
                    starts_seconds[:, 1:],
                    starts_seconds[:, :-1],
                )
        temporal = self.temporal_encoder(
            normalized_frames,
            starts_seconds,
            mask,
            end_timestamps_seconds=temporal_ends,
        )
        mask_values = mask.unsqueeze(-1).to(normalized_frames.dtype)
        denominator = mask_values.sum(dim=1).clamp_min(1.0)
        mean_embedding = torch.sum(normalized_frames * mask_values, dim=1) / denominator
        attended_embedding = torch.sum(
            normalized_frames * temporal.attention_weights.unsqueeze(-1),
            dim=1,
        )
        gate = torch.sigmoid(self.residual_gate_logit)
        embedding = F.normalize(
            (1.0 - gate) * mean_embedding + gate * attended_embedding,
            dim=-1,
        )
        missing_rows = ~mask.any(dim=1)
        embedding = embedding.masked_fill(missing_rows.unsqueeze(-1), 0.0)
        return VisualRetrievalOutput(
            embedding=embedding,
            attention_weights=temporal.attention_weights,
        )


def build_visual_retrieval_training_examples(
    examples: Sequence[RankingTrainingExample],
) -> list[VisualRetrievalTrainingExample]:
    """Build positive retrieval pairs from PIT-resolved ranking observations."""
    result = []
    for example in examples:
        action = example.attribution.attributed_action
        if action not in _ACTION_WEIGHTS:
            continue
        content = example.multimodal_content
        candidate = dict(example.candidate_embeddings or {})
        if content is None or not content.frame_embeddings:
            continue
        image = candidate.get("image")
        if image is None:
            continue
        as_of_ts = float(example.bundle.as_of_ts)
        try:
            available_at = float(candidate["available_at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "visual retrieval positive requires versioned available_at"
            ) from exc
        if not math.isfinite(available_at) or available_at > as_of_ts:
            raise ValueError(
                "visual retrieval product embedding is future data for observation"
            )
        frames = content.frame_embeddings[:16]
        timestamps = content.frame_timestamps_seconds[:16]
        if len(frames) != len(timestamps):
            raise ValueError("visual frame embeddings and timestamps are misaligned")
        if len(image) != 512 or any(len(frame) != 512 for frame in frames):
            raise ValueError("visual retrieval CLIP embeddings must be 512-dimensional")
        if not all(math.isfinite(float(value)) for value in image):
            raise ValueError("visual retrieval product embedding must be finite")
        for frame, timestamp in zip(frames, timestamps):
            if not math.isfinite(float(timestamp)) or float(timestamp) < 0:
                raise ValueError("visual retrieval frame timestamp is invalid")
            if not all(math.isfinite(float(value)) for value in frame):
                raise ValueError("visual retrieval frame embedding must be finite")
        result.append(
            VisualRetrievalTrainingExample(
                observation_id=example.observation_id,
                impression_id=example.impression_id,
                content_id=content.content_id,
                product_id=example.bundle.candidate.product_id,
                as_of_ts=as_of_ts,
                product_available_at=available_at,
                frame_embeddings=tuple(tuple(map(float, frame)) for frame in frames),
                frame_timestamps_seconds=tuple(map(float, timestamps)),
                positive_embedding=tuple(map(float, image)),
                positive_weight=_ACTION_WEIGHTS[action],
            )
        )
    return result


def build_visual_retrieval_hard_negatives(
    examples: Sequence[RankingTrainingExample],
) -> dict[tuple[str, str], list[tuple[str, tuple[float, ...]]]]:
    """Collect PIT-safe unclicked impression candidates as explicit negatives."""
    positives_by_content: dict[str, set[str]] = {}
    for example in examples:
        if example.multimodal_content is not None and (
            example.attribution.attributed_action in _ACTION_WEIGHTS
        ):
            positives_by_content.setdefault(
                example.multimodal_content.content_id, set()
            ).add(example.bundle.candidate.product_id)
    result: dict[tuple[str, str], list[tuple[str, tuple[float, ...]]]] = {}
    for example in examples:
        content = example.multimodal_content
        candidate = dict(example.candidate_embeddings or {})
        if (
            content is None
            or example.attribution.attributed_click
            or candidate.get("image") is None
        ):
            continue
        available_at = float(candidate.get("available_at", float("inf")))
        if available_at > float(example.bundle.as_of_ts):
            raise ValueError("visual retrieval hard negative contains future data")
        product_id = example.bundle.candidate.product_id
        if product_id in positives_by_content.get(content.content_id, set()):
            continue
        image = tuple(map(float, candidate["image"]))
        if len(image) != 512:
            raise ValueError("visual retrieval hard negative must be 512-dimensional")
        result.setdefault((content.content_id, example.impression_id), []).append(
            (product_id, image)
        )
    return result


def add_product_index_hard_negatives(
    negatives: Mapping[tuple[str, str], Sequence[tuple[str, Sequence[float]]]],
    positives: Sequence[VisualRetrievalTrainingExample],
    *,
    product_index_bundle: Any,
    neighbors_per_content: int = 16,
) -> dict[tuple[str, str], list[tuple[str, tuple[float, ...]]]]:
    """Add FAISS-near and same-category negatives from one pinned activation."""
    result = {
        key: [
            (str(product_id), tuple(map(float, embedding)))
            for product_id, embedding in values
        ]
        for key, values in negatives.items()
    }
    positive_products: dict[str, set[str]] = {}
    for example in positives:
        positive_products.setdefault(example.content_id, set()).add(example.product_id)
    seen_keys: set[tuple[str, str]] = set()
    for example in positives:
        key = (example.content_id, example.impression_id)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        frames = np.asarray(example.frame_embeddings, dtype=np.float32)
        query = frames.mean(axis=0, keepdims=True)
        norm = float(np.linalg.norm(query))
        if norm <= 0:
            continue
        query /= norm
        count = min(
            int(product_index_bundle.index.ntotal),
            max(1, int(neighbors_per_content) * 3),
        )
        _, indices = product_index_bundle.index.search(query, count)
        candidate_ids = [
            product_index_bundle.product_index_map[int(index)]
            for index in indices[0]
            if int(index) >= 0
        ]
        positive_category = product_index_bundle.product_metadata.get(
            example.product_id, {}
        ).get("category")
        if positive_category:
            candidate_ids.extend(
                product_id
                for product_id, metadata in product_index_bundle.product_metadata.items()
                if metadata.get("category") == positive_category
            )
        existing = {product_id for product_id, _ in result.get(key, [])}
        for product_id in candidate_ids:
            metadata = product_index_bundle.product_metadata.get(product_id, {})
            available_at = float(
                metadata.get(
                    "available_at",
                    product_index_bundle.manifest.get(
                        "catalog_available_at", float("inf")
                    ),
                )
            )
            if (
                product_id in existing
                or product_id in positive_products.get(example.content_id, set())
                or not math.isfinite(available_at)
                or available_at > example.as_of_ts
            ):
                continue
            embedding = product_index_bundle.product_embeddings.get(product_id)
            if embedding is None:
                continue
            result.setdefault(key, []).append(
                (product_id, tuple(map(float, embedding)))
            )
            existing.add(product_id)
            if len(result[key]) >= neighbors_per_content:
                break
    return result


def warm_start_visual_retrieval_encoder(
    model: VisualRetrievalPooler,
    ranker_visual_encoder: nn.Module,
) -> None:
    """Copy only the visual temporal encoder; retrieval scorer then evolves alone."""
    try:
        model.temporal_encoder.load_state_dict(
            ranker_visual_encoder.state_dict(), strict=True
        )
    except RuntimeError as exc:
        raise ValueError(
            "ranker visual encoder is incompatible with retrieval pooler"
        ) from exc


def _collate_visual_retrieval_examples(
    examples: Sequence[VisualRetrievalTrainingExample],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("visual retrieval batch is empty")
    width = len(examples[0].positive_embedding)
    max_frames = min(16, max(len(example.frame_embeddings) for example in examples))
    frames = torch.zeros(len(examples), max_frames, width, device=device)
    starts = torch.zeros(len(examples), max_frames, device=device)
    mask = torch.zeros(len(examples), max_frames, dtype=torch.bool, device=device)
    positives = torch.zeros(len(examples), width, device=device)
    weights = torch.zeros(len(examples), device=device)
    for row, example in enumerate(examples):
        count = min(max_frames, len(example.frame_embeddings))
        frames[row, :count] = torch.as_tensor(
            example.frame_embeddings[:count], dtype=torch.float32, device=device
        )
        starts[row, :count] = torch.as_tensor(
            example.frame_timestamps_seconds[:count],
            dtype=torch.float32,
            device=device,
        )
        mask[row, :count] = True
        positives[row] = torch.as_tensor(
            example.positive_embedding, dtype=torch.float32, device=device
        )
        weights[row] = example.positive_weight
    return frames, starts, mask, positives, weights


def _time_group_split(
    examples: Sequence[VisualRetrievalTrainingExample],
    *,
    validation_fraction: float,
) -> tuple[list[VisualRetrievalTrainingExample], list[VisualRetrievalTrainingExample]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("visual retrieval validation fraction must be in (0, 1)")
    groups: dict[tuple[str, str], list[VisualRetrievalTrainingExample]] = {}
    for example in examples:
        groups.setdefault((example.content_id, example.impression_id), []).append(
            example
        )
    ordered = sorted(
        groups.values(),
        key=lambda values: max(example.as_of_ts for example in values),
    )
    validation_groups = max(1, int(math.ceil(len(ordered) * validation_fraction)))
    if validation_groups >= len(ordered):
        raise ValueError("visual retrieval requires at least two temporal groups")
    return (
        [example for group in ordered[:-validation_groups] for example in group],
        [example for group in ordered[-validation_groups:] for example in group],
    )


def train_visual_retrieval_pooler(
    model: VisualRetrievalPooler,
    examples: Sequence[VisualRetrievalTrainingExample],
    *,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    temperature: float = 0.07,
    validation_fraction: float = 0.2,
    seed: int = 17,
    hard_negatives: Mapping[tuple[str, str], Sequence[tuple[str, Sequence[float]]]]
    | None = None,
) -> VisualRetrievalTrainingMetrics:
    """Train the retrieval-only temporal scorer with a time-group holdout."""
    if epochs <= 0 or batch_size <= 1 or learning_rate <= 0:
        raise ValueError("visual retrieval training settings are invalid")
    training, validation = _time_group_split(
        examples, validation_fraction=validation_fraction
    )
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    positive_products_by_content: dict[str, set[str]] = {}
    for example in examples:
        positive_products_by_content.setdefault(example.content_id, set()).add(
            example.product_id
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    final_loss = float("nan")
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(training), generator=generator).tolist()
        for offset in range(0, len(order), batch_size):
            batch = [training[index] for index in order[offset : offset + batch_size]]
            if len(batch) < 2:
                continue
            (
                frames,
                starts,
                mask,
                positives,
                weights,
            ) = _collate_visual_retrieval_examples(batch, device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model(frames, starts, mask)
            negative_records = []
            for example in batch:
                negative_records.extend(
                    (hard_negatives or {}).get(
                        (example.content_id, example.impression_id), ()
                    )
                )
            negative_products = (
                torch.as_tensor(
                    [embedding for _, embedding in negative_records],
                    dtype=torch.float32,
                    device=device,
                )
                if negative_records
                else None
            )
            result = weighted_multi_positive_info_nce(
                output.embedding,
                positives,
                content_ids=[example.content_id for example in batch],
                positive_product_ids=[example.product_id for example in batch],
                positive_weights=weights,
                temperature=temperature,
                negative_products=negative_products,
                negative_product_ids=[product_id for product_id, _ in negative_records],
                positive_products_by_content=positive_products_by_content,
            )
            result.loss.backward()
            optimizer.step()
            final_loss = float(result.loss.detach().cpu())
    if not math.isfinite(final_loss):
        raise ValueError("visual retrieval training produced no valid optimizer step")

    validation_losses = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(validation), batch_size):
            batch = validation[offset : offset + batch_size]
            if len(batch) < 2:
                continue
            (
                frames,
                starts,
                mask,
                positives,
                weights,
            ) = _collate_visual_retrieval_examples(batch, device=device)
            output = model(frames, starts, mask)
            validation_losses.append(
                float(
                    weighted_multi_positive_info_nce(
                        output.embedding,
                        positives,
                        content_ids=[example.content_id for example in batch],
                        positive_product_ids=[example.product_id for example in batch],
                        positive_weights=weights,
                        temperature=temperature,
                    ).loss.cpu()
                )
            )
    validation_loss = (
        sum(validation_losses) / len(validation_losses)
        if validation_losses
        else float("nan")
    )
    return VisualRetrievalTrainingMetrics(
        training_examples=len(training),
        validation_examples=len(validation),
        final_training_loss=final_loss,
        validation_loss=validation_loss,
    )


def evaluate_visual_retrieval(
    queries: Any,
    *,
    relevant_product_ids: Sequence[set[str]],
    product_embeddings: Any,
    product_ids: Sequence[str],
    cutoffs: Sequence[int] = (50, 100, 200),
    expected_catalog_size: int | None = None,
) -> VisualRetrievalEvaluation:
    """Evaluate multiple query variants against one fixed real product index."""
    query_matrix = np.asarray(queries, dtype=np.float32)
    product_matrix = np.asarray(product_embeddings, dtype=np.float32)
    if (
        query_matrix.ndim != 2
        or product_matrix.ndim != 2
        or query_matrix.shape[1] != product_matrix.shape[1]
        or product_matrix.shape[0] != len(product_ids)
        or query_matrix.shape[0] != len(relevant_product_ids)
    ):
        raise ValueError("visual retrieval evaluation dimensions are inconsistent")
    if not cutoffs or any(int(value) <= 0 for value in cutoffs):
        raise ValueError("visual retrieval evaluation cutoffs must be positive")
    query_norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
    product_norms = np.linalg.norm(product_matrix, axis=1, keepdims=True)
    if np.any(query_norms <= 0) or np.any(product_norms <= 0):
        raise ValueError("visual retrieval evaluation embeddings must be non-zero")
    scores = (query_matrix / query_norms) @ (product_matrix / product_norms).T
    order = np.argsort(-scores, axis=1, kind="stable")
    recalls = {int(cutoff): 0.0 for cutoff in cutoffs}
    reciprocal_ranks = []
    for row, relevant in enumerate(relevant_product_ids):
        relevant = {str(value) for value in relevant}
        ranked = [str(product_ids[index]) for index in order[row]]
        for cutoff in recalls:
            recalls[cutoff] += float(
                bool(relevant.intersection(ranked[: min(cutoff, len(ranked))]))
            )
        rank = next(
            (
                index + 1
                for index, product_id in enumerate(ranked)
                if product_id in relevant
            ),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)
    denominator = max(1, len(relevant_product_ids))
    expected = (
        int(expected_catalog_size)
        if expected_catalog_size is not None
        else len(product_ids)
    )
    if expected <= 0:
        raise ValueError("expected catalog size must be positive")
    return VisualRetrievalEvaluation(
        recall_at={key: value / denominator for key, value in recalls.items()},
        mrr=sum(reciprocal_ranks) / denominator,
        catalog_coverage=len(set(map(str, product_ids))) / expected,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_lineage(lineage: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(lineage)
    if set(payload) != _LINEAGE_FIELDS:
        raise ValueError("visual retrieval checkpoint lineage is incomplete")
    for field in _LINEAGE_FIELDS - {"training_cutoff"}:
        value = str(payload[field] or "").strip()
        if not value:
            raise ValueError("visual retrieval checkpoint lineage is incomplete")
        payload[field] = value
    cutoff = float(payload["training_cutoff"])
    if not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError("visual retrieval training cutoff is invalid")
    payload["training_cutoff"] = cutoff
    for checksum_field in ("product_index_sha256", "pit_sidecar_sha256"):
        if len(payload[checksum_field]) != 64:
            raise ValueError("visual retrieval lineage checksum is invalid")
    return payload


def save_visual_retrieval_checkpoint(
    path: str | Path,
    *,
    model: VisualRetrievalPooler,
    model_version: str,
    lineage: Mapping[str, Any],
) -> str:
    """Atomically save a self-describing retrieval-only checkpoint."""
    version = str(model_version or "").strip()
    if not version:
        raise ValueError("visual retrieval model version is required")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": VISUAL_RETRIEVAL_SCHEMA_VERSION,
        "model_version": version,
        "lineage": _validated_lineage(lineage),
        "model_config": {
            "input_dim": model.input_dim,
            "model_dim": model.model_dim,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "feedforward_dim": model.feedforward_dim,
            "dropout": model.dropout,
        },
        "state_dict": model.state_dict(),
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return _file_sha256(target)


def load_visual_retrieval_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_lineage: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[VisualRetrievalPooler, dict[str, Any]]:
    source = Path(path)
    if expected_sha256 is not None and _file_sha256(source) != expected_sha256:
        raise ValueError("visual retrieval checkpoint checksum mismatch")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(source, map_location=map_location)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != VISUAL_RETRIEVAL_SCHEMA_VERSION
    ):
        raise ValueError("visual retrieval checkpoint schema mismatch")
    lineage = _validated_lineage(payload.get("lineage") or {})
    if expected_lineage is not None and lineage != _validated_lineage(expected_lineage):
        raise ValueError("visual retrieval checkpoint lineage mismatch")
    config = payload.get("model_config")
    state_dict = payload.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError("visual retrieval checkpoint is incomplete")
    try:
        model = VisualRetrievalPooler(**config)
        model.load_state_dict(state_dict, strict=True)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError("visual retrieval checkpoint model is incompatible") from exc
    metadata = {
        "schema_version": payload["schema_version"],
        "model_version": str(payload.get("model_version") or ""),
        "lineage": lineage,
        "model_config": dict(config),
    }
    if not metadata["model_version"]:
        raise ValueError("visual retrieval checkpoint model version is missing")
    return model, metadata


def select_content_retrieval_embedding(
    features: ContentFeatures,
    *,
    enabled: bool,
    canary_percent: float,
    expected_model_version: str,
    expected_product_index_version: str,
) -> tuple[list[float], str]:
    fallback = list(features.visual_embedding or [])
    percent = float(canary_percent)
    if not math.isfinite(percent) or percent < 0.0 or percent > 100.0:
        raise ValueError("visual retrieval canary percent must be in [0, 100]")
    attention = list(features.retrieval_visual_embedding or [])
    lineage_matches = (
        bool(attention)
        and len(attention) == len(fallback)
        and features.multimodal_schema_version == "temporal_multimodal_v3"
        and features.retrieval_model_version == expected_model_version
        and (features.retrieval_product_index_version == expected_product_index_version)
    )
    if not enabled or not lineage_matches or percent <= 0.0:
        return fallback, "mean"
    if 0.0 < percent < 100.0:
        digest = hashlib.sha256(features.content_id.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % 10000
        if bucket >= int(percent * 100):
            return fallback, "mean"
    return attention, "attention"


def upgrade_content_features_to_visual_retrieval_v3(
    features: ContentFeatures,
    *,
    model: VisualRetrievalPooler,
    model_version: str,
    product_index_version: str,
    clip_model_id: str,
    clip_revision: str,
    device: str | torch.device = "cpu",
) -> ContentFeatures:
    """Backfill v3 directly from stored frame tensors and real timestamps."""
    if not features.frame_embeddings:
        raise ValueError("visual retrieval backfill requires stored frame embeddings")
    if len(features.frame_embeddings) != len(features.frame_timestamps_seconds):
        raise ValueError("visual retrieval backfill frame timestamps are misaligned")
    target_device = torch.device(device)
    frames = torch.as_tensor(
        features.frame_embeddings[:16], dtype=torch.float32, device=target_device
    ).unsqueeze(0)
    starts = torch.as_tensor(
        features.frame_timestamps_seconds[:16],
        dtype=torch.float32,
        device=target_device,
    ).unsqueeze(0)
    mask = torch.ones(starts.shape, dtype=torch.bool, device=target_device)
    model = model.to(target_device).eval()
    with torch.no_grad():
        embedding = model(frames, starts, mask).embedding[0].cpu().tolist()
    return features.copy(
        deep=True,
        update={
            "multimodal_schema_version": "temporal_multimodal_v3",
            "retrieval_visual_embedding": embedding,
            "retrieval_model_version": str(model_version),
            "retrieval_product_index_version": str(product_index_version),
            "retrieval_clip_model": str(clip_model_id),
            "retrieval_clip_revision": str(clip_revision),
            "artifact_uri": None,
            "artifact_sha256": None,
            "artifact_schema_version": None,
            "artifact_created_at": None,
        },
    )


def weighted_multi_positive_info_nce(
    queries: torch.Tensor,
    products: torch.Tensor,
    *,
    content_ids: Sequence[str],
    positive_product_ids: Sequence[str] | None = None,
    positive_weights: torch.Tensor,
    temperature: float,
    negative_products: torch.Tensor | None = None,
    negative_product_ids: Sequence[str] = (),
    positive_products_by_content: Mapping[str, set[str]] | None = None,
) -> VisualRetrievalLoss:
    if queries.ndim != 2 or products.shape != queries.shape:
        raise ValueError("queries and products must share shape [batch, embedding]")
    batch_size = queries.shape[0]
    if len(content_ids) != batch_size or positive_weights.shape != (batch_size,):
        raise ValueError("content IDs and positive weights must match the batch")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    weights = positive_weights.to(dtype=queries.dtype, device=queries.device)
    if not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError("positive weights must be finite and greater than zero")

    content_values = [str(content_id) for content_id in content_ids]
    same_content = torch.tensor(
        [[left == right for right in content_values] for left in content_values],
        dtype=torch.bool,
        device=queries.device,
    )
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=queries.device)
    same_product = torch.zeros_like(same_content)
    if positive_product_ids is not None:
        if len(positive_product_ids) != batch_size:
            raise ValueError("positive product IDs must match the batch")
        product_values = [str(product_id) for product_id in positive_product_ids]
        same_product = torch.tensor(
            [[left == right for right in product_values] for left in product_values],
            dtype=torch.bool,
            device=queries.device,
        )
    valid_negative_mask = ~(same_content | same_product) | diagonal
    logits = (
        F.normalize(queries, dim=-1) @ F.normalize(products, dim=-1).transpose(0, 1)
    ) / float(temperature)
    logits = logits.masked_fill(~valid_negative_mask, -torch.inf)
    if negative_products is not None:
        if (
            negative_products.ndim != 2
            or negative_products.shape[1] != queries.shape[1]
            or negative_products.shape[0] != len(negative_product_ids)
        ):
            raise ValueError("explicit visual retrieval negatives are misaligned")
        positives_by_content = {
            str(content_id): {str(product_id) for product_id in product_ids}
            for content_id, product_ids in (positive_products_by_content or {}).items()
        }
        if positive_product_ids is not None and not positives_by_content:
            for content_id, product_id in zip(content_values, positive_product_ids):
                positives_by_content.setdefault(content_id, set()).add(str(product_id))
        explicit_mask = torch.tensor(
            [
                [
                    str(product_id) not in positives_by_content.get(content_id, set())
                    for product_id in negative_product_ids
                ]
                for content_id in content_values
            ],
            dtype=torch.bool,
            device=queries.device,
        )
        explicit_logits = (
            F.normalize(queries, dim=-1)
            @ F.normalize(negative_products.to(queries.device), dim=-1).transpose(0, 1)
        ) / float(temperature)
        explicit_logits = explicit_logits.masked_fill(~explicit_mask, -torch.inf)
        logits = torch.cat([logits, explicit_logits], dim=1)
        valid_negative_mask = torch.cat([valid_negative_mask, explicit_mask], dim=1)
    targets = torch.arange(batch_size, device=queries.device)
    row_losses = F.cross_entropy(logits, targets, reduction="none")
    loss = torch.sum(row_losses * weights) / weights.sum().clamp_min(1e-12)
    return VisualRetrievalLoss(
        loss=loss,
        valid_negative_mask=valid_negative_mask,
    )
