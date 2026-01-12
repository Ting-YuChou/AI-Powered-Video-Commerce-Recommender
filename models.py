"""
AI-Powered Video Commerce Recommender - Data Models
===================================================

This module contains all Pydantic models used for API request/response validation,
data serialization, and internal data structures throughout the system.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union, TYPE_CHECKING
from enum import Enum
import time

# Enums for categorical fields
class InteractionType(str, Enum):
    """User interaction types with products."""
    VIEW = "view"
    CLICK = "click"
    PURCHASE = "purchase"
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    FAVORITE = "favorite"
    SHARE = "share"

class ContentStatus(str, Enum):
    """Content processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class HealthStatus(str, Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

# Request Models
class RecommendationRequest(BaseModel):
    """Request model for product recommendations."""
    user_id: str = Field(..., description="Unique user identifier")
    content_id: Optional[str] = Field(None, description="Video content identifier")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    k: int = Field(10, ge=1, le=100, description="Number of recommendations to return")
    
    # Context field examples
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_12345",
                "content_id": "video_67890",
                "context": {
                    "device": "mobile",
                    "time_of_day": "evening",
                    "location": "home",
                    "session_id": "sess_abc123"
                },
                "k": 10
            }
        }

class UserInteractionRequest(BaseModel):
    """Request model for logging user interactions."""
    user_id: str = Field(..., description="Unique user identifier")
    product_id: str = Field(..., description="Product identifier")
    action: InteractionType = Field(..., description="Type of interaction")
    context: Dict[str, Any] = Field(default_factory=dict, description="Interaction context")
    timestamp: Optional[float] = Field(default_factory=time.time, description="Interaction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_12345",
                "product_id": "prod_67890",
                "action": "click",
                "context": {
                    "recommendation_position": 2,
                    "page": "video_recommendations",
                    "session_id": "sess_abc123"
                }
            }
        }

# Response Models
class ProductRecommendation(BaseModel):
    """Single product recommendation item."""
    product_id: str = Field(..., description="Unique product identifier")
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Product price")
    currency: str = Field("USD", description="Currency code")
    image_url: Optional[str] = Field(None, description="Product image URL")
    category: Optional[str] = Field(None, description="Product category")
    brand: Optional[str] = Field(None, description="Product brand")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Product rating (0-5)")
    confidence_score: float = Field(..., ge=0, le=1, description="Recommendation confidence")
    ranking_score: float = Field(..., description="Internal ranking score")
    reason: Optional[str] = Field(None, description="Recommendation explanation")

    @validator('price', pre=True, always=True)
    def validate_price(cls, v):
        """Validate price is positive."""
        if v < 0:
            raise ValueError('Price must be non-negative')
        return round(float(v), 2)
    
    @validator('confidence_score', pre=True, always=True)
    def validate_confidence_score(cls, v):
        """Validate confidence score range."""
        score = float(v)
        if not (0 <= score <= 1):
            raise ValueError('Confidence score must be between 0 and 1')
        return score
    
    class Config:
        schema_extra = {
            "example": {
                "product_id": "prod_12345",
                "title": "Wireless Bluetooth Headphones",
                "description": "High-quality wireless headphones with noise cancellation",
                "price": 129.99,
                "currency": "USD",
                "image_url": "https://example.com/product_image.jpg",
                "category": "Electronics",
                "brand": "AudioTech",
                "rating": 4.5,
                "confidence_score": 0.85,
                "ranking_score": 0.92,
                "reason": "Based on your interest in audio products"
            }
        }

class RecommendationResponse(BaseModel):
    """Response model for product recommendations."""
    user_id: str = Field(..., description="User ID from request")
    recommendations: List[ProductRecommendation] = Field(..., description="List of recommendations")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_12345",
                "recommendations": [
                    {
                        "product_id": "prod_12345",
                        "title": "Wireless Headphones",
                        "price": 129.99,
                        "confidence_score": 0.85,
                        "ranking_score": 0.92
                    }
                ],
                "metadata": {
                    "total_candidates": 1000,
                    "response_time_ms": 145,
                    "model_version": "v1.0.0"
                }
            }
        }

class ContentUploadResponse(BaseModel):
    """Response model for content upload."""
    content_id: str = Field(..., description="Generated content identifier")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    status: ContentStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    upload_timestamp: float = Field(default_factory=time.time, description="Upload timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "content_id": "content_1642123456_video.mp4",
                "filename": "product_demo.mp4",
                "size_bytes": 15728640,
                "status": "processing",
                "message": "Content uploaded successfully. Processing in background.",
                "upload_timestamp": 1642123456.789
            }
        }

class AnalyticsResponse(BaseModel):
    """Response model for system analytics."""
    total_users: int = Field(..., description="Total number of users")
    total_recommendations: int = Field(..., description="Total recommendations served")
    total_interactions: int = Field(..., description="Total user interactions")
    average_response_time_ms: float = Field(..., description="Average API response time")
    recommendation_accuracy: Dict[str, float] = Field(..., description="Accuracy metrics")
    top_categories: List[Dict[str, Any]] = Field(..., description="Most popular categories")
    daily_stats: List[Dict[str, Any]] = Field(..., description="Daily statistics")
    
    class Config:
        schema_extra = {
            "example": {
                "total_users": 50000,
                "total_recommendations": 1000000,
                "total_interactions": 250000,
                "average_response_time_ms": 185.5,
                "recommendation_accuracy": {
                    "ctr": 0.12,
                    "conversion_rate": 0.034
                },
                "top_categories": [
                    {"category": "Electronics", "count": 15000},
                    {"category": "Fashion", "count": 12000}
                ],
                "daily_stats": [
                    {"date": "2024-01-01", "recommendations": 8500, "interactions": 2100}
                ]
            }
        }

# Health Check Models
class ComponentHealth(BaseModel):
    """Health status of individual system component."""
    status: HealthStatus = Field(..., description="Component health status")
    response_time_ms: Optional[float] = Field(None, description="Component response time")
    error_message: Optional[str] = Field(None, description="Error details if unhealthy")
    last_check: float = Field(default_factory=time.time, description="Last health check timestamp")

class HealthResponse(BaseModel):
    """Complete system health check response."""
    status: HealthStatus = Field(..., description="Overall system health")
    components: Dict[str, ComponentHealth] = Field(..., description="Individual component health")
    timestamp: float = Field(default_factory=time.time, description="Health check timestamp")
    version: str = Field("1.0.0", description="System version")
    uptime_seconds: Optional[float] = Field(None, description="System uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "components": {
                    "redis": {
                        "status": "healthy",
                        "response_time_ms": 2.1,
                        "last_check": 1642123456.789
                    },
                    "content_processor": {
                        "status": "healthy",
                        "response_time_ms": 45.2,
                        "last_check": 1642123456.789
                    }
                },
                "timestamp": 1642123456.789,
                "version": "1.0.0",
                "uptime_seconds": 86400.0
            }
        }

# Internal Data Models
class UserFeatures(BaseModel):
    """User feature representation for ML models."""
    user_id: str = Field(..., description="User identifier")
    total_interactions: int = Field(0, description="Total user interactions")
    avg_session_length: float = Field(0.0, description="Average session length in seconds")
    preferred_categories: List[str] = Field(default_factory=list, description="Preferred product categories")
    price_sensitivity: float = Field(0.5, ge=0, le=1, description="Price sensitivity score")
    click_through_rate: float = Field(0.0, ge=0, le=1, description="Historical CTR")
    conversion_rate: float = Field(0.0, ge=0, le=1, description="Historical conversion rate")
    last_active: float = Field(default_factory=time.time, description="Last activity timestamp")
    demographics: Dict[str, Any] = Field(default_factory=dict, description="User demographics")

class ContentFeatures(BaseModel):
    """Content feature representation extracted from videos."""
    content_id: str = Field(..., description="Content identifier")
    visual_embedding: List[float] = Field(..., description="CLIP visual embedding")
    audio_features: Optional[Dict[str, Any]] = Field(None, description="Audio analysis features")
    text_features: Optional[Dict[str, Any]] = Field(None, description="Extracted text features")
    duration_seconds: Optional[float] = Field(None, description="Video duration")
    detected_objects: List[str] = Field(default_factory=list, description="Detected objects in video")
    extracted_text: List[str] = Field(default_factory=list, description="OCR extracted text")
    product_mentions: List[str] = Field(default_factory=list, description="Mentioned products")
    category_scores: Dict[str, float] = Field(default_factory=dict, description="Content category scores")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    created_at: float = Field(default_factory=time.time, description="Feature extraction timestamp")

class ProductData(BaseModel):
    """Product catalog data model."""
    product_id: str = Field(..., description="Product identifier")
    title: str = Field(..., description="Product title")
    description: str = Field(..., description="Product description")
    price: float = Field(..., ge=0, description="Product price")
    currency: str = Field("USD", description="Currency code")
    category: str = Field(..., description="Product category")
    brand: str = Field(..., description="Product brand")
    image_url: Optional[str] = Field(None, description="Product image URL")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Product rating")
    num_reviews: Optional[int] = Field(0, description="Number of reviews")
    in_stock: bool = Field(True, description="Stock availability")
    tags: List[str] = Field(default_factory=list, description="Product tags")
    embedding: Optional[List[float]] = Field(None, description="Product embedding vector")
    created_at: float = Field(default_factory=time.time, description="Product creation timestamp")
    updated_at: float = Field(default_factory=time.time, description="Last update timestamp")

class CandidateProduct(BaseModel):
    """Candidate product with scores from different sources."""
    product_id: str = Field(..., description="Product identifier")
    collaborative_score: Optional[float] = Field(None, description="Collaborative filtering score")
    content_similarity_score: Optional[float] = Field(None, description="Content similarity score")
    popularity_score: Optional[float] = Field(None, description="Popularity/trending score")
    combined_score: float = Field(0.0, description="Combined candidate score")
    source: str = Field(..., description="Recommendation source (cf, content, trending)")

class RankingFeatures(BaseModel):
    """Feature vector for ranking model."""
    user_features: Dict[str, float] = Field(..., description="User feature vector")
    product_features: Dict[str, float] = Field(..., description="Product feature vector")
    context_features: Dict[str, float] = Field(..., description="Context feature vector")
    interaction_features: Dict[str, float] = Field(..., description="User-product interaction features")
    candidate_scores: Dict[str, float] = Field(..., description="Candidate generation scores")

class SystemMetrics(BaseModel):
    """System performance and business metrics."""
    api_metrics: Dict[str, Any] = Field(..., description="API performance metrics")
    ml_metrics: Dict[str, Any] = Field(..., description="ML model performance metrics")
    business_metrics: Dict[str, Any] = Field(..., description="Business KPIs")
    resource_metrics: Dict[str, Any] = Field(..., description="System resource usage")
    timestamp: float = Field(default_factory=time.time, description="Metrics timestamp")

# Remove these - validators are now in the ProductRecommendation class

# Configuration Models
class RedisConfig(BaseModel):
    """Redis configuration."""
    host: str = Field("localhost", description="Redis host")
    port: int = Field(6379, description="Redis port")
    db: int = Field(0, description="Redis database number")
    password: Optional[str] = Field(None, description="Redis password")
    decode_responses: bool = Field(True, description="Decode responses")
    max_connections: int = Field(100, description="Maximum connections")

class ModelConfig(BaseModel):
    """ML model configuration."""
    clip_model: str = Field("openai/clip-vit-large-patch14", description="CLIP model identifier")
    embedding_dim: int = Field(512, description="Embedding dimension")
    ranking_model_path: Optional[str] = Field(None, description="Ranking model file path")
    cache_dir: str = Field("/tmp/models", description="Model cache directory")
    device: str = Field("auto", description="Compute device (cpu/cuda/auto)")
    batch_size: int = Field(32, description="Processing batch size")

# Export all models
__all__ = [
    # Enums
    "InteractionType", "ContentStatus", "HealthStatus",
    # Request models
    "RecommendationRequest", "UserInteractionRequest",
    # Response models
    "ProductRecommendation", "RecommendationResponse", "ContentUploadResponse", 
    "AnalyticsResponse", "HealthResponse", "ComponentHealth",
    # Internal models
    "UserFeatures", "ContentFeatures", "ProductData", "CandidateProduct", 
    "RankingFeatures", "SystemMetrics",
    # Configuration models
    "RedisConfig", "ModelConfig"
]