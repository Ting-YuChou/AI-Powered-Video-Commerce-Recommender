"""Shared internal ranking request parsing helpers."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

try:
    from pydantic.v1 import BaseModel, Field
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Field

from models import CandidateProduct, ProductRecommendation, UserFeatures


class RankRequest(BaseModel):
    request_id: Optional[str] = Field(None, description="Request id for tracing")
    deadline_unix_seconds: Optional[float] = Field(
        None,
        description="Optional internal wall-clock deadline for ranking completion",
    )
    candidates: List[CandidateProduct]
    user_features: UserFeatures
    context: Dict[str, Any] = Field(default_factory=dict)
    product_metadata_map: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    k: int = Field(..., ge=1, le=500)


class RankResponse(BaseModel):
    recommendations: List[ProductRecommendation]
    profile: Dict[str, Any]


def coerce_rank_payload(raw_payload: Any) -> RankRequest:
    if isinstance(raw_payload, RankRequest):
        return raw_payload
    if not isinstance(raw_payload, dict):
        raise HTTPException(status_code=400, detail="Rank request must be an object")

    try:
        k = int(raw_payload.get("k"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid k") from exc
    if k < 1 or k > 500:
        raise HTTPException(status_code=400, detail="Invalid k")

    raw_candidates = raw_payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise HTTPException(status_code=400, detail="candidates must be a list")

    user_features = coerce_user_features(raw_payload.get("user_features"))
    candidates = [coerce_candidate(item) for item in raw_candidates]
    context = raw_payload.get("context") or {}
    product_metadata_map = raw_payload.get("product_metadata_map") or {}
    if not isinstance(context, dict):
        raise HTTPException(status_code=400, detail="context must be an object")
    if not isinstance(product_metadata_map, dict):
        raise HTTPException(
            status_code=400,
            detail="product_metadata_map must be an object",
        )

    return RankRequest.construct(
        request_id=raw_payload.get("request_id"),
        deadline_unix_seconds=optional_float(raw_payload.get("deadline_unix_seconds")),
        candidates=candidates,
        user_features=user_features,
        context=context,
        product_metadata_map=product_metadata_map,
        k=k,
    )


def coerce_candidate(raw_candidate: Any) -> CandidateProduct:
    if isinstance(raw_candidate, CandidateProduct):
        return raw_candidate
    if not isinstance(raw_candidate, dict):
        raise HTTPException(status_code=400, detail="candidate must be an object")
    product_id = raw_candidate.get("product_id")
    if not product_id:
        raise HTTPException(status_code=400, detail="candidate.product_id is required")
    return CandidateProduct.construct(
        product_id=str(product_id),
        collaborative_score=optional_float(raw_candidate.get("collaborative_score")),
        content_similarity_score=optional_float(
            raw_candidate.get("content_similarity_score")
        ),
        popularity_score=optional_float(raw_candidate.get("popularity_score")),
        combined_score=float_or_default(raw_candidate.get("combined_score"), 0.0),
        source=str(raw_candidate.get("source") or "unknown"),
    )


def coerce_user_features(raw_user_features: Any) -> UserFeatures:
    if isinstance(raw_user_features, UserFeatures):
        return raw_user_features
    if not isinstance(raw_user_features, dict):
        raise HTTPException(status_code=400, detail="user_features must be an object")
    return UserFeatures.construct(
        user_id=str(raw_user_features.get("user_id") or ""),
        total_interactions=int(raw_user_features.get("total_interactions") or 0),
        avg_session_length=float_or_default(
            raw_user_features.get("avg_session_length"),
            0.0,
        ),
        preferred_categories=list(raw_user_features.get("preferred_categories") or []),
        price_sensitivity=float_or_default(
            raw_user_features.get("price_sensitivity"),
            0.5,
        ),
        click_through_rate=float_or_default(
            raw_user_features.get("click_through_rate"),
            0.0,
        ),
        conversion_rate=float_or_default(
            raw_user_features.get("conversion_rate"),
            0.0,
        ),
        last_active=float_or_default(raw_user_features.get("last_active"), time.time()),
        demographics=dict(raw_user_features.get("demographics") or {}),
    )


def optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float_or_default(value, 0.0)


def float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def model_payload(item: Any) -> Dict[str, Any]:
    raw = getattr(item, "__dict__", None)
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(item, "dict"):
        return item.dict()
    return dict(item)
