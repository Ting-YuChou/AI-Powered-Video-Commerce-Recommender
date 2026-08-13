"""Deterministic helpers for recommendation serving capacity tests.

The synthetic checkpoint produced here has random weights and is only suitable
for exercising the real serving/model-forward path.  It must not be used for
ranking-quality evaluation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

import torch

from video_commerce.common.config import RankingConfig
from video_commerce.common.models import CandidateProduct
from video_commerce.ml.ranking import RankingModel


def build_synthetic_catalog(
    *, count: int, seed: int
) -> Tuple[List[CandidateProduct], Dict[str, Dict[str, Any]]]:
    """Build stable candidates and metadata for serving-path load tests."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    candidates: List[CandidateProduct] = []
    metadata: Dict[str, Dict[str, Any]] = {}
    categories = ("electronics", "home", "beauty", "sports")
    for index in range(count):
        product_id = f"loadtest_product_{index:04d}"
        popularity = rng.random()
        candidates.append(
            CandidateProduct(
                product_id=product_id,
                popularity_score=popularity,
                combined_score=popularity,
                source="loadtest_trending_pool",
            )
        )
        metadata[product_id] = {
            "name": f"Load-test product {index}",
            "category": categories[index % len(categories)],
            "brand": f"loadtest_brand_{index % 8}",
            "price": round(10.0 + rng.random() * 490.0, 2),
            "rating": round(3.0 + rng.random() * 2.0, 2),
            "num_reviews": 10 + index,
            "active": True,
            "is_active": True,
            "in_stock": True,
            "deleted": False,
        }
    return candidates, metadata


async def create_synthetic_ranking_checkpoint(
    *,
    checkpoint_path: Path,
    ranking_config: RankingConfig,
    model_version: str,
    seed: int,
) -> Dict[str, Any]:
    """Save deterministic random weights in the normal trained checkpoint format."""
    if not model_version.strip():
        raise ValueError("model_version is required")
    torch.manual_seed(seed)
    model = RankingModel(ranking_config)
    await model.load_model(None)
    model.is_trained = True
    model.model_version = model_version
    await model.save_model(str(checkpoint_path))
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return {
        "model_version": model_version,
        "artifact_sha256": digest,
        "feature_schema_version": model.feature_schema_version,
        "architecture": getattr(model.model, "architecture", None),
        "model_parameters": sum(
            parameter.numel() for parameter in model.model.parameters()
        ),
        "random_seed": seed,
        "synthetic_weights": True,
        "purpose": "serving_capacity_only",
        "quality_evaluation_valid": False,
    }


def _metric_values(payload: Dict[str, Any], name: str) -> Dict[str, Any]:
    return dict((payload.get("metrics") or {}).get(name) or {})


def _rate(payload: Dict[str, Any], name: str) -> float:
    metric = _metric_values(payload, name)
    passes = float(metric.get("passes", 0) or 0)
    fails = float(metric.get("fails", 0) or 0)
    total = passes + fails
    return passes / total if total else 0.0


def _optional_float(values: Dict[str, Any], key: str) -> float | None:
    value = values.get(key)
    return float(value) if value is not None else None


def summarize_k6_result(
    path: Path, *, target_qps: int, duration: str, scenario: str
) -> Dict[str, Any]:
    """Extract comparable recommendation capacity fields from k6 summary JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    iterations = _metric_values(payload, "iterations")
    latency = _metric_values(payload, "recommendation_success_latency")
    if not latency:
        latency = _metric_values(payload, "http_req_duration")
    actual_qps = round(float(iterations.get("rate", 0) or 0), 3)
    error_rate = round(_rate(payload, "recommendation_errors"), 6)
    batch_requests = _metric_values(payload, "recommendation_batch_request_count")
    model_forward = _metric_values(payload, "recommendation_model_forward_latency")
    return {
        "source_file": str(path),
        "scenario": scenario,
        "target_qps": target_qps,
        "duration": duration,
        "actual_qps": actual_qps,
        "successful_qps": round(actual_qps * (1.0 - error_rate), 3),
        "iterations": int(iterations.get("count", 0) or 0),
        "dropped_iterations": int(
            _metric_values(payload, "dropped_iterations").get("count", 0) or 0
        ),
        "error_rate": error_rate,
        "five_xx_rate": round(_rate(payload, "recommendation_5xx"), 6),
        "overload_429_rate": round(_rate(payload, "recommendation_429"), 6),
        "p50_ms": float(latency.get("med", latency.get("p(50)", 0)) or 0),
        "p95_ms": float(latency.get("p(95)", 0) or 0),
        "p99_ms": _optional_float(latency, "p(99)"),
        "avg_ms": float(latency.get("avg", 0) or 0),
        "candidate_cache_rank_rate": round(
            _rate(payload, "candidate_cache_rank_responses"), 6
        ),
        "recommendation_cache_rate": round(
            _rate(payload, "recommendation_cache_responses"), 6
        ),
        "model_forward_evidence_rate": round(
            _rate(payload, "model_forward_responses"), 6
        ),
        "untrained_fallback_rate": round(
            _rate(payload, "ranking_untrained_fallback_responses"), 6
        ),
        "batch_requests_avg": float(batch_requests.get("avg", 0) or 0),
        "batch_requests_p95": float(batch_requests.get("p(95)", 0) or 0),
        "model_forward_avg_ms": float(model_forward.get("avg", 0) or 0),
        "model_forward_p95_ms": float(model_forward.get("p(95)", 0) or 0),
        "model_forward_p99_ms": _optional_float(model_forward, "p(99)"),
    }
