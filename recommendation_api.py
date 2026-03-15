"""
Dedicated recommendation-serving API.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Optional

from fastapi import Body, HTTPException

from config import Config
from feature_store import FeatureStore
from kafka_client import close_kafka, init_kafka
from models import RecommendationRequest, RecommendationResponse
from ranking import RankingModel
from recommender import RecommendationEngine
from service_common import (
    build_health_response,
    build_metrics_response,
    component_health,
    configure_service_logging,
    create_service_app,
)
from vector_search import VectorSearchEngine

logger = logging.getLogger(__name__)

app = create_service_app(
    title="Recommendation Service",
    description="Dedicated online recommendation serving service",
    service_name="recommendation-service",
)

feature_store: Optional[FeatureStore] = None
vector_search: Optional[VectorSearchEngine] = None
recommendation_engine: Optional[RecommendationEngine] = None
ranking_model: Optional[RankingModel] = None
kafka_manager = None


@app.on_event("startup")
async def startup_event():
    global feature_store, vector_search, recommendation_engine, ranking_model, kafka_manager

    runtime = app.state.runtime
    runtime.config = Config()
    configure_service_logging(runtime)

    feature_store = FeatureStore(runtime.config.redis_config, runtime.config.cache_config)
    await feature_store.initialize()

    vector_search = VectorSearchEngine(runtime.config.vector_config)
    await vector_search.load_index()

    recommendation_engine = RecommendationEngine(
        feature_store,
        vector_search,
        runtime.config.recommendation_config,
    )
    await recommendation_engine.load_serving_state()

    ranking_model = RankingModel(runtime.config.ranking_config)
    await ranking_model.load_model(runtime.config.model_config.ranking_model_path)

    if runtime.config.kafka_config.enable:
        try:
            kafka_manager = await init_kafka(runtime.config.kafka_config)
        except Exception as exc:
            logger.warning(f"Recommendation service Kafka init failed: {exc}")
            kafka_manager = None


@app.on_event("shutdown")
async def shutdown_event():
    if feature_store:
        await feature_store.close()
    if kafka_manager:
        await close_kafka()


@app.get("/")
async def root():
    return {
        "service": "recommendation-service",
        "version": "1.0.0",
        "health": "/health",
    }


@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest = Body(...)):
    start_time = time.time()

    try:
        cache_key = feature_store.generate_context_hash(
            {
                "content_id": request.content_id,
                "context": request.context,
                "k": request.k,
            }
        )
        cached = await feature_store.get_cached_recommendations(request.user_id, cache_key)
        if cached:
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=cached,
                metadata={
                    "total_candidates": len(cached),
                    "response_time_ms": int((time.time() - start_time) * 1000),
                    "model_version": "v1.0.0",
                    "cache_hit": True,
                    "content_processed": request.content_id is not None,
                },
            )

        content_features = None
        if request.content_id:
            content_features = await feature_store.get_content_features(request.content_id)

        user_features = await feature_store.get_user_features(request.user_id)
        candidates = await recommendation_engine.generate_candidates(
            user_id=request.user_id,
            content_features=content_features,
            context=request.context,
            k_per_source=min(request.k * 10, 500),
        )

        if not candidates:
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=[],
                metadata={
                    "total_candidates": 0,
                    "response_time_ms": int((time.time() - start_time) * 1000),
                    "fallback_reason": "no_candidates",
                    "cache_hit": False,
                },
            )

        ranked_recommendations = await ranking_model.rank_candidates(
            candidates=candidates,
            user_features=user_features,
            context=request.context,
            k=request.k,
        )
        response_time = time.time() - start_time

        await feature_store.cache_recommendations(
            request.user_id,
            cache_key,
            [recommendation.dict() for recommendation in ranked_recommendations],
        )
        await feature_store.log_recommendation_request(
            request.user_id,
            len(ranked_recommendations),
            response_time,
        )

        if kafka_manager:
            asyncio.create_task(
                kafka_manager.send_recommendation_event(
                    user_id=request.user_id,
                    recommendations=[item.product_id for item in ranked_recommendations],
                    response_time_ms=int(response_time * 1000),
                    metadata={
                        "content_id": request.content_id,
                        "candidate_count": len(candidates),
                    },
                )
            )

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=ranked_recommendations,
            metadata={
                "total_candidates": len(candidates),
                "response_time_ms": int(response_time * 1000),
                "model_version": "v1.0.0",
                "cache_hit": False,
                "content_processed": request.content_id is not None,
            },
        )

    except Exception as exc:
        logger.error(f"Recommendation request failed: {exc}")
        logger.error(traceback.format_exc())
        try:
            trending_recommendations = await recommendation_engine.get_trending_recommendations(request.k)
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=trending_recommendations,
                metadata={
                    "total_candidates": len(trending_recommendations),
                    "response_time_ms": int((time.time() - start_time) * 1000),
                    "fallback": True,
                    "error": str(exc),
                },
            )
        except Exception as fallback_exc:
            raise HTTPException(
                status_code=500,
                detail=f"Recommendation service unavailable: {fallback_exc}",
            ) from fallback_exc


@app.get("/health")
async def health_check():
    feature_store_health = await feature_store.health_check()
    recommendation_health = recommendation_engine.health_check()
    ranking_health = ranking_model.health_check()
    vector_health = vector_search.health_check()

    return build_health_response(
        {
            "feature_store": component_health(
                feature_store_health.get("status", "unhealthy"),
                feature_store_health.get("response_time_ms"),
                feature_store_health.get("error"),
            ),
            "recommendation_engine": component_health(
                recommendation_health.get("status", "unhealthy"),
                error_message=recommendation_health.get("error"),
            ),
            "ranking_model": component_health(
                ranking_health.get("status", "unhealthy"),
                error_message=ranking_health.get("error"),
            ),
            "vector_search": component_health(
                vector_health.get("status", "unhealthy"),
                error_message=vector_health.get("error"),
            ),
        },
        app.state.runtime.started_at,
    )


@app.get("/metrics")
async def metrics():
    return await build_metrics_response(
        app.state.runtime,
        feature_store=feature_store,
        kafka_manager=kafka_manager,
    )
