"""
AI-Powered Video Commerce Recommender - Main FastAPI Application
================================================================

This is the main FastAPI application that serves as the API gateway for the
video commerce recommendation system. It handles all HTTP endpoints and
orchestrates the recommendation pipeline.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request, Response, Body
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import time
import logging
import uuid
from typing import List, Optional, Dict, Any
import traceback

# Local imports
from video_commerce.common.models import (
    RecommendationRequest, 
    RecommendationResponse, 
    HealthResponse,
    ContentUploadResponse,
    UserInteractionRequest,
    AnalyticsResponse
)
from video_commerce.ml.content_processor import ContentProcessor
from video_commerce.ml.recommender import RecommendationEngine
from video_commerce.ml.ranking import RankingModel
from video_commerce.data_plane.feature_store import FeatureStore
from video_commerce.ml.vector_search import VectorSearchEngine
from video_commerce.common.health import HealthChecker
from video_commerce.common.config import Config
from video_commerce.data_plane.kafka_client import KafkaManager, init_kafka, close_kafka, get_kafka_manager
from video_commerce.common.observability import (
    ObservabilityManager,
    configure_logging,
    request_id_ctx_var,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Video Commerce Recommender",
    description="Production-ready AI system for video-based product recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (initialized on startup)
content_processor: Optional[ContentProcessor] = None
recommendation_engine: Optional[RecommendationEngine] = None
ranking_model: Optional[RankingModel] = None
feature_store: Optional[FeatureStore] = None
vector_search: Optional[VectorSearchEngine] = None
health_checker: Optional[HealthChecker] = None
kafka_manager: Optional[KafkaManager] = None
config: Optional[Config] = None
observability = ObservabilityManager()
health_monitor_task: Optional[asyncio.Task] = None


def _get_route_template(request: Request) -> str:
    route = request.scope.get("route")
    if route and getattr(route, "path", None):
        return route.path
    return request.url.path


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id_header = (
        config.monitoring_config.request_id_header
        if config
        else "X-Request-ID"
    )
    request_id = request.headers.get(request_id_header) or str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.request_started_at = time.perf_counter()
    request_id_token = request_id_ctx_var.set(request_id)
    observability.http_requests_in_progress.inc()

    try:
        response = await call_next(request)
    except Exception as exc:
        duration = time.perf_counter() - request.state.request_started_at
        route_path = _get_route_template(request)
        observability.record_exception(
            request.method,
            route_path,
            type(exc).__name__,
        )
        observability.record_request(request.method, route_path, 500, duration)
        logger.exception(
            "request_failed",
            extra={
                "method": request.method,
                "path": route_path,
                "status_code": 500,
                "duration_ms": round(duration * 1000, 2),
                "client_ip": request.client.host if request.client else None,
            },
        )
        raise
    else:
        duration = time.perf_counter() - request.state.request_started_at
        route_path = _get_route_template(request)
        observability.record_request(request.method, route_path, response.status_code, duration)

        if not (config and not config.monitoring_config.enable_request_logging):
            logger.info(
                "request_complete",
                extra={
                    "method": request.method,
                    "path": route_path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                    "client_ip": request.client.host if request.client else None,
                },
            )

        response.headers[request_id_header] = request_id
        return response
    finally:
        observability.http_requests_in_progress.dec()
        request_id_ctx_var.reset(request_id_token)

@app.on_event("startup")
async def startup_event():
    """Initialize all components on application startup."""
    global content_processor, recommendation_engine, ranking_model
    global feature_store, vector_search, health_checker, kafka_manager, config, health_monitor_task
    
    try:
        # Load configuration
        config = Config()
        configure_logging(config.monitoring_config)
        logger.info("Starting Video Commerce Recommender...")
        
        # Initialize feature store (Redis)
        feature_store = FeatureStore(config.redis_config)
        await feature_store.initialize()
        
        # Initialize Kafka (if enabled)
        if config.kafka_config.enable:
            logger.info("Initializing Kafka connection...")
            try:
                kafka_manager = await init_kafka(config.kafka_config)
                logger.info("Kafka connection established successfully")
            except Exception as kafka_error:
                logger.warning(f"Kafka initialization failed, continuing without Kafka: {kafka_error}")
                kafka_manager = None
        else:
            logger.info("Kafka is disabled in configuration")
        
        # Initialize content processor
        content_processor = ContentProcessor(config.model_config)
        if not content_processor.lazy_load:
            await content_processor.load_models()
        
        # Initialize vector search
        vector_search = VectorSearchEngine(config.vector_config)
        await vector_search.load_index()
        
        # Initialize recommendation engine
        recommendation_engine = RecommendationEngine(
            feature_store, vector_search, config.recommendation_config
        )
        await recommendation_engine.load_models()
        
        # Initialize ranking model
        ranking_model = RankingModel(config.ranking_config)
        await ranking_model.load_model()
        
        # Initialize health checker (include Kafka if available)
        health_checker = HealthChecker(
            feature_store, content_processor, recommendation_engine,
            ranking_model=ranking_model,
            vector_search=vector_search,
            kafka_manager=kafka_manager
        )
        health_monitor_task = health_checker.start_monitoring()
        
        # Load data if needed
        if config.load_sample_data:
            from video_commerce.ml import data
            if config.data_config.use_csv_dataset:
                # Load from CSV dataset
                logger.info("Loading data from CSV dataset...")
                await data.load_dataset_from_csv(
                    dataset_dir=config.data_config.dataset_dir,
                    feature_store=feature_store,
                    vector_search=vector_search,
                    limit_users=config.data_config.csv_limit_users,
                    limit_products=config.data_config.csv_limit_products,
                    limit_interactions=config.data_config.csv_limit_interactions,
                    limit_content=config.data_config.csv_limit_content
                )
            else:
                # Generate sample data
                await data.initialize_sample_data(feature_store, vector_search)
        
        # Schedule periodic model updates (Two-Tower retraining + trending)
        asyncio.create_task(_periodic_model_update())
        
        logger.info("All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        logger.error(traceback.format_exc())
        raise


async def _periodic_model_update():
    """Background task that periodically retrains the Two-Tower model and trending scores."""
    while True:
        try:
            await asyncio.sleep(3600)  # Check every hour
            if recommendation_engine:
                logger.info("Running periodic model update...")
                await recommendation_engine.update_models()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic model update: {e}")
            await asyncio.sleep(60)  # Retry after 1 minute on error


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Video Commerce Recommender...")

    global health_monitor_task

    if health_monitor_task:
        health_monitor_task.cancel()
        try:
            await health_monitor_task
        except asyncio.CancelledError:
            pass
        health_monitor_task = None
    
    # Close Kafka connection
    if kafka_manager:
        logger.info("Closing Kafka connection...")
        await close_kafka()
    
    if feature_store:
        await feature_store.close()
    
    logger.info("Shutdown complete.")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint - API information."""
    return {
        "message": "AI Video Commerce Recommender API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest = Body(...)):
    """
    Generate product recommendations based on user and content data.
    
    This is the main endpoint that orchestrates the entire recommendation pipeline:
    1. Extract content features (if content_id provided)
    2. Generate candidate products
    3. Rank candidates using the ranking model
    4. Return top-K recommendations
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing recommendation request for user: {request.user_id}")
        
        # Extract content features if content provided
        content_features = None
        if request.content_id:
            content_features = await feature_store.get_content_features(request.content_id)
            if not content_features:
                logger.warning(f"Content features not found for ID: {request.content_id}")
        
        # Get user features
        user_features = await feature_store.get_user_features(request.user_id)
        
        # Generate candidate products
        candidates = await recommendation_engine.generate_candidates(
            user_id=request.user_id,
            content_features=content_features,
            context=request.context,
            k_per_source=min(request.k * 10, 500)  # Generate more candidates than needed
        )
        
        if not candidates:
            logger.warning(f"No candidates generated for user: {request.user_id}")
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=[],
                metadata={
                    "total_candidates": 0,
                    "response_time_ms": int((time.time() - start_time) * 1000),
                    "fallback_reason": "no_candidates"
                }
            )
        
        # Rank candidates
        ranked_recommendations = await ranking_model.rank_candidates(
            candidates=candidates,
            user_features=user_features,
            context=request.context,
            k=request.k
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        response_time_ms = int(response_time * 1000)
        
        # Send recommendation event to Kafka (async, non-blocking)
        if kafka_manager and config.kafka_config.enable:
            # Fire and forget - don't wait for Kafka response
            asyncio.create_task(
                kafka_manager.send_recommendation_event(
                    user_id=request.user_id,
                    recommendations=[r.product_id for r in ranked_recommendations],
                    response_time_ms=response_time_ms,
                    metadata={
                        "total_candidates": len(candidates),
                        "model_version": config.model_version,
                        "content_id": request.content_id,
                        "context": request.context
                    }
                )
            )
        
        # Also log to Redis for backward compatibility
        await feature_store.log_recommendation_request(
            request.user_id, len(ranked_recommendations), response_time
        )
        
        logger.info(
            f"Generated {len(ranked_recommendations)} recommendations for user {request.user_id} "
            f"in {response_time:.3f}s"
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=ranked_recommendations,
            metadata={
                "total_candidates": len(candidates),
                "response_time_ms": response_time_ms,
                "model_version": config.model_version,
                "content_processed": request.content_id is not None,
                "kafka_enabled": kafka_manager is not None and config.kafka_config.enable
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing recommendation request: {e}")
        logger.error(traceback.format_exc())
        
        # Return fallback trending recommendations
        try:
            trending_recs = await recommendation_engine.get_trending_recommendations(request.k)
            response_time = time.time() - start_time
            
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=trending_recs,
                metadata={
                    "total_candidates": len(trending_recs),
                    "response_time_ms": int(response_time * 1000),
                    "fallback": True,
                    "error": str(e)
                }
            )
        except:
            raise HTTPException(status_code=500, detail=f"Recommendation service unavailable: {e}")

@app.post("/api/content/upload", response_model=ContentUploadResponse)
async def upload_content(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = None,
    priority: str = "normal"
):
    """
    Upload video content and extract features for recommendations.
    
    This endpoint handles video file uploads and triggers background processing
    to extract multi-modal features (visual, audio, text) from the content.
    
    When Kafka is enabled, video processing tasks are sent to Kafka for
    distributed processing by worker nodes, enabling:
    - Horizontal scaling of video processing
    - Task persistence and automatic retry on failure
    - Priority-based processing (high/normal/low)
    """
    try:
        # Validate file
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Validate priority
        if priority not in ["low", "normal", "high"]:
            priority = "normal"
        
        # Generate content ID
        content_id = f"content_{int(time.time())}_{file.filename}"
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            file_path = tmp_file.name
        
        # Set initial status
        await feature_store.update_content_status(content_id, "pending")
        
        # Check if Kafka is available for distributed processing
        if kafka_manager and config.kafka_config.enable:
            # Send to Kafka for distributed processing
            success = await kafka_manager.send_video_processing_task(
                content_id=content_id,
                file_path=file_path,
                user_id=user_id,
                priority=priority
            )
            
            if success:
                logger.info(f"Video task sent to Kafka: {content_id} (priority: {priority})")
                return ContentUploadResponse(
                    content_id=content_id,
                    filename=file.filename,
                    size_bytes=len(content),
                    status="queued",
                    message=f"Content uploaded and queued for processing (priority: {priority})."
                )
            else:
                logger.warning("Kafka send failed, falling back to local processing")
        
        # Fallback: Process content in background (local processing)
        background_tasks.add_task(
            process_uploaded_content, 
            content_id, 
            file_path, 
            user_id
        )
        
        logger.info(f"Content uploaded: {content_id} ({len(content)} bytes)")
        
        return ContentUploadResponse(
            content_id=content_id,
            filename=file.filename,
            size_bytes=len(content),
            status="processing",
            message="Content uploaded successfully. Processing in background."
        )
        
    except Exception as e:
        logger.error(f"Error uploading content: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

async def process_uploaded_content(content_id: str, file_path: str, user_id: Optional[str] = None):
    """Background task to process uploaded video content."""
    try:
        logger.info(f"Processing content: {content_id}")
        
        # Extract features from video
        features = await content_processor.process_video(file_path)
        
        # Store features
        await feature_store.store_content_features(content_id, features)
        
        # Update vector search index
        if features.get('visual_embedding'):
            await vector_search.add_content_embedding(content_id, features['visual_embedding'])
        
        # Log processing completion
        await feature_store.update_content_status(content_id, "completed")
        
        logger.info(f"Content processing completed: {content_id}")
        
    except Exception as e:
        logger.error(f"Error processing content {content_id}: {e}")
        await feature_store.update_content_status(content_id, "failed")
    finally:
        # Cleanup temporary file
        try:
            import os
            os.remove(file_path)
        except:
            pass

@app.post("/api/interactions")
async def log_user_interaction(request: UserInteractionRequest = Body(...)):
    """
    Log user interactions for model training and personalization.
    
    Records user actions (views, clicks, purchases) to improve future recommendations.
    
    When Kafka is enabled, interactions are sent to Kafka for async processing,
    providing much faster API response times (~2-5ms vs ~100-200ms).
    """
    try:
        # Check if Kafka is available for async processing
        if kafka_manager and config.kafka_config.enable:
            # Send to Kafka for async processing (fast path)
            success = await kafka_manager.send_user_interaction(
                user_id=request.user_id,
                product_id=request.product_id,
                action=request.action.value if hasattr(request.action, 'value') else str(request.action),
                context=request.context
            )
            
            if success:
                logger.debug(f"Interaction sent to Kafka: {request.user_id} -> {request.action} -> {request.product_id}")
                return {
                    "status": "success", 
                    "message": "Interaction queued for processing",
                    "processing_mode": "async"
                }
            else:
                # Fallback to sync processing if Kafka send fails
                logger.warning("Kafka send failed, falling back to sync processing")
        
        # Sync processing (fallback or when Kafka is disabled)
        await feature_store.log_user_interaction(
            user_id=request.user_id,
            product_id=request.product_id,
            action=request.action.value if hasattr(request.action, 'value') else str(request.action),
            context=request.context
        )
        
        # Update user features in real-time
        await feature_store.update_user_features(
            request.user_id, 
            request.action.value if hasattr(request.action, 'value') else str(request.action)
        )
        
        logger.info(f"Logged interaction: {request.user_id} -> {request.action} -> {request.product_id}")
        
        return {
            "status": "success", 
            "message": "Interaction logged successfully",
            "processing_mode": "sync"
        }
        
    except Exception as e:
        logger.error(f"Error logging interaction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log interaction: {e}")

@app.get("/api/content/{content_id}/status")
async def get_content_status(content_id: str):
    """Check the processing status of uploaded content."""
    try:
        status = await feature_store.get_content_status(content_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Content not found")
        
        return {
            "content_id": content_id,
            "status": status,
            "processed_at": await feature_store.get_content_processed_time(content_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting content status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get status")

@app.get("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Get system analytics and performance metrics."""
    try:
        analytics = await feature_store.get_analytics()
        return AnalyticsResponse(**analytics)
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Verifies that all system components are functioning properly.
    """
    try:
        health_status = await health_checker.check_system_health()
        
        status_code = 200 if health_status.status == "healthy" else 503
        
        return JSONResponse(
            status_code=status_code,
            content=health_status.dict()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "components": {},
                "error": str(e),
                "timestamp": time.time()
            }
        )

@app.get("/metrics")
async def get_metrics():
    """Expose Prometheus metrics."""
    try:
        if config and config.monitoring_config.enable_prometheus_metrics:
            await observability.collect_runtime_metrics(
                feature_store=feature_store,
                kafka_manager=kafka_manager,
            )
        return Response(
            content=observability.prometheus_payload(),
            media_type=observability.prometheus_content_type,
        )
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")


@app.get("/metrics/system")
async def get_system_metrics():
    """Get JSON system metrics for debugging and manual baseline analysis."""
    try:
        metrics = await feature_store.get_system_metrics()
        if health_checker:
            metrics["system_status"] = health_checker.get_system_status_summary()
        return metrics

    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail="System metrics unavailable")

@app.get("/kafka/health")
async def kafka_health():
    """
    Get Kafka connection health status.
    
    Returns detailed health information about the Kafka producer and consumers.
    """
    try:
        if not kafka_manager:
            return {
                "status": "disabled",
                "message": "Kafka is not enabled or failed to initialize",
                "enabled": config.kafka_config.enable if config else False
            }
        
        health = await kafka_manager.health_check()
        health['status'] = 'healthy' if health['producer']['connected'] else 'unhealthy'
        return health
        
    except Exception as e:
        logger.error(f"Error checking Kafka health: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/kafka/topics")
async def kafka_topics():
    """
    Get information about Kafka topics used by this application.
    
    Returns the topic names and their configurations.
    """
    if not config:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    
    return {
        "topics": {
            "user_interactions": {
                "name": config.kafka_config.user_interactions_topic,
                "description": "User interaction events (clicks, views, purchases)",
                "partition_key": "user_id"
            },
            "video_processing": {
                "name": config.kafka_config.video_processing_topic,
                "description": "Video processing tasks",
                "partition_key": "content_id"
            },
            "recommendation_events": {
                "name": config.kafka_config.recommendation_events_topic,
                "description": "Recommendation request events for analytics",
                "partition_key": "user_id"
            },
            "feature_updates": {
                "name": config.kafka_config.feature_updates_topic,
                "description": "Feature update notifications",
                "partition_key": "entity_id"
            }
        },
        "bootstrap_servers": config.kafka_config.bootstrap_servers,
        "consumer_group": config.kafka_config.consumer_group_id,
        "enabled": config.kafka_config.enable
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    request_id_header = (
        config.monitoring_config.request_id_header
        if config
        else "X-Request-ID"
    )
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url),
            "request_id": getattr(request.state, 'request_id', 'unknown'),
        },
        headers={request_id_header: getattr(request.state, 'request_id', 'unknown')},
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    request_id_header = (
        config.monitoring_config.request_id_header
        if config
        else "X-Request-ID"
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        },
        headers={request_id_header: getattr(request.state, 'request_id', 'unknown')},
    )

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
