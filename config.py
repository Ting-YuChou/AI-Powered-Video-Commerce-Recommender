"""
AI-Powered Video Commerce Recommender - Configuration Management
================================================================

This module handles all configuration settings for the video commerce recommender system.
It supports environment variables, configuration files, and provides sensible defaults
for development and production environments.
"""

import os
import json
from typing import Dict, Any, Optional, List
try:
    from pydantic.v1 import BaseSettings, Field, validator
except ImportError:
    from pydantic import BaseSettings, Field, validator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _load_secret_file_env() -> None:
    """Resolve `*_FILE` environment variables into their target values."""
    for env_name, file_path in list(os.environ.items()):
        if not env_name.endswith("_FILE"):
            continue
        target_name = env_name[:-5]
        if os.getenv(target_name):
            continue
        if not file_path:
            continue
        with open(file_path, "r", encoding="utf-8") as secret_handle:
            os.environ[target_name] = secret_handle.read().strip()

class RedisConfig(BaseSettings):
    """Redis database configuration."""
    host: str = Field("localhost", description="Redis host address")
    port: int = Field(6379, description="Redis port")
    db: int = Field(0, description="Redis database number")
    password: Optional[str] = Field(None, description="Redis password")
    decode_responses: bool = Field(True, description="Decode Redis responses")
    max_connections: int = Field(100, description="Maximum Redis connections")
    socket_timeout: float = Field(30, description="Socket timeout in seconds")
    socket_connect_timeout: float = Field(30, description="Socket connect timeout")
    retry_on_timeout: bool = Field(True, description="Retry on timeout")
    cache_host: Optional[str] = Field(
        None,
        description="Optional separate Redis host for recommendation caches and serving pools",
    )
    cache_port: Optional[int] = Field(None, description="Optional separate cache Redis port")
    cache_db: Optional[int] = Field(None, description="Optional separate cache Redis DB")
    cache_password: Optional[str] = Field(None, description="Optional separate cache Redis password")
    cache_max_connections: Optional[int] = Field(
        None,
        description="Optional separate cache Redis max connections",
    )
    cache_socket_timeout: Optional[float] = Field(
        None,
        description="Optional separate cache Redis socket timeout",
    )
    cache_socket_connect_timeout: Optional[float] = Field(
        None,
        description="Optional separate cache Redis connect timeout",
    )
    cache_retry_on_timeout: Optional[bool] = Field(
        None,
        description="Optional separate cache Redis retry-on-timeout setting",
    )
    
    class Config:
        env_prefix = "REDIS_"

class ModelConfig(BaseSettings):
    """Machine learning model configuration."""
    # CLIP model settings
    clip_model: str = Field(
        "openai/clip-vit-large-patch14", 
        description="CLIP model identifier"
    )
    embedding_dim: int = Field(512, description="Embedding dimension")
    
    # Model paths
    ranking_model_path: Optional[str] = Field(
        None, 
        description="Path to trained ranking model"
    )
    cf_model_path: Optional[str] = Field(
        None, 
        description="Path to collaborative filtering model"
    )
    
    # Processing settings
    cache_dir: str = Field("/tmp/models", description="Model cache directory")
    device: str = Field("auto", description="Compute device (cpu/cuda/auto)")
    batch_size: int = Field(32, description="Processing batch size")
    max_video_length: int = Field(300, description="Maximum video length in seconds")
    num_keyframes: int = Field(8, description="Number of keyframes to extract")
    
    # Performance settings
    enable_quantization: bool = Field(False, description="Enable model quantization")
    enable_gpu: bool = Field(True, description="Enable GPU acceleration if available")
    max_concurrent_requests: int = Field(10, description="Max concurrent ML requests")
    torch_num_threads: int = Field(
        0,
        description="Torch intra-op CPU thread count; 0 auto-detects from cgroup quota",
    )
    torch_num_interop_threads: int = Field(
        1,
        description="Torch inter-op CPU thread count; 0 auto-detects from cgroup quota",
    )
    
    class Config:
        env_prefix = "MODEL_"

class VectorConfig(BaseSettings):
    """Vector search configuration."""
    index_path: str = Field("/tmp/vector_index.faiss", description="FAISS index file path")
    embedding_dim: int = Field(512, description="Vector embedding dimension")
    index_type: str = Field("HNSW", description="FAISS index type (HNSW/IVF)")
    
    # HNSW parameters
    hnsw_m: int = Field(32, description="HNSW M parameter")
    hnsw_ef_construction: int = Field(200, description="HNSW efConstruction parameter")
    hnsw_ef_search: int = Field(50, description="HNSW efSearch parameter")
    
    # Search parameters
    search_k: int = Field(100, description="Number of candidates to retrieve")
    similarity_threshold: float = Field(0.1, description="Minimum similarity threshold")
    
    class Config:
        env_prefix = "VECTOR_"

class RecommendationConfig(BaseSettings):
    """Recommendation engine configuration."""
    # Legacy collaborative filtering (kept for backward-compat env vars)
    cf_factors: int = Field(64, description="Collaborative filtering factors (legacy)")
    cf_regularization: float = Field(0.1, description="CF regularization parameter (legacy)")
    cf_iterations: int = Field(50, description="CF training iterations (legacy)")
    
    # Candidate generation
    candidates_per_source: int = Field(100, description="Candidates per recommendation source")
    max_total_candidates: int = Field(500, description="Maximum total candidates")
    cold_start_random_candidate_cap: int = Field(
        50,
        description="Maximum random fallback candidates when no stronger retrieval source is available",
    )
    serving_trending_pool_size: int = Field(
        300,
        description="Number of products to precompute in the global trending pool",
    )
    serving_category_pool_size: int = Field(
        150,
        description="Number of products to precompute per category pool",
    )
    preferred_category_pool_count: int = Field(
        2,
        description="Maximum preferred categories to pull from serving pools per request",
    )
    max_live_cf_candidates: int = Field(
        40,
        description="Maximum collaborative candidates to fetch live per request",
    )
    max_live_content_candidates: int = Field(
        20,
        description="Maximum content-similar candidates to fetch live per request",
    )
    max_pool_trending_candidates: int = Field(
        30,
        description="Maximum precomputed trending candidates to merge per request",
    )
    max_pool_category_candidates: int = Field(
        30,
        description="Maximum precomputed category-pool candidates to merge per request",
    )
    max_random_candidates: int = Field(
        10,
        description="Maximum random fallback candidates to merge per request",
    )
    user_embedding_cache_size: int = Field(
        20000,
        description="Per-process LRU cache entries for Two-Tower user embeddings",
    )
    user_embedding_cache_time_bucket_seconds: float = Field(
        1.0,
        description="Time bucket used in Two-Tower user embedding cache keys",
    )
    interaction_history_timeout_ms: float = Field(
        50.0,
        description="Maximum time to spend reading recent user interactions on the serving path",
    )
    candidate_source_timeout_ms: float = Field(
        250.0,
        description="Maximum time to spend on one live candidate source on the serving path",
    )
    preload_product_metadata_on_startup: bool = Field(
        False,
        description="Preload product metadata into Redis during recommendation service startup",
    )
    publish_catalog_snapshot_on_startup: bool = Field(
        False,
        description="Publish product catalog snapshots to Postgres during recommendation service startup",
    )
    
    # Ranking weights
    cf_weight: float = Field(0.4, description="Collaborative filtering weight")
    content_weight: float = Field(0.3, description="Content similarity weight")
    popularity_weight: float = Field(0.3, description="Popularity weight")
    
    # Trending algorithm
    trending_decay_factor: float = Field(0.95, description="Time decay factor for trending")
    trending_window_hours: int = Field(24, description="Trending calculation window")
    
    # Diversity settings
    enable_diversity: bool = Field(True, description="Enable recommendation diversity")
    diversity_factor: float = Field(0.1, description="Diversity vs relevance trade-off")
    max_items_per_category: int = Field(3, description="Max recommendations per category")
    
    # Two-Tower model settings
    tt_embedding_dim: int = Field(128, description="Two-Tower output embedding dimension")
    tt_user_hidden_dims: List[int] = Field([256, 128], description="User tower hidden dimensions")
    tt_item_hidden_dims: List[int] = Field([256, 128], description="Item tower hidden dimensions")
    tt_learning_rate: float = Field(0.001, description="Two-Tower learning rate")
    tt_batch_size: int = Field(1024, description="Two-Tower training batch size")
    tt_epochs: int = Field(20, description="Two-Tower training epochs")
    tt_temperature: float = Field(0.07, description="InfoNCE temperature parameter")
    
    # Negative sampling settings
    tt_num_hard_negatives: int = Field(5, description="Hard negatives per positive sample")
    tt_num_random_negatives: int = Field(10, description="Random negatives per positive sample")
    tt_hard_negative_ratio_start: float = Field(0.1, description="Initial hard negative ratio (curriculum)")
    tt_hard_negative_ratio_end: float = Field(0.5, description="Final hard negative ratio (curriculum)")
    
    # CF FAISS index path
    cf_index_path: str = Field("/tmp/cf_vector_index.faiss", description="CF FAISS index file path")
    
    class Config:
        env_prefix = "RECOMMENDATION_"

class RankingConfig(BaseSettings):
    """Ranking model configuration."""
    model_type: str = Field("neural", description="Ranking model type")
    hidden_dims: List[int] = Field([256, 128, 64], description="Neural network hidden dimensions")
    dropout_rate: float = Field(0.2, description="Dropout rate")
    learning_rate: float = Field(0.001, description="Learning rate")
    enable_async_batching: bool = Field(
        True,
        description="Enable micro-batching queue for ranking inference",
    )
    batch_max_requests: int = Field(
        8,
        description="Maximum number of ranking requests to combine into one micro-batch",
    )
    batch_wait_ms: float = Field(
        5.0,
        description="Maximum time to wait for more ranking requests before dispatching a batch",
    )
    batch_queue_size: int = Field(
        2048,
        description="Maximum queued ranking requests per worker",
    )
    batch_runner_count: int = Field(
        1,
        description="Number of concurrent micro-batch runners per recommendation worker",
    )
    inference_executor_workers: int = Field(
        0,
        description="Dedicated per-process thread workers for ranking feature prep and inference; 0 uses asyncio's default executor",
    )
    offload_inference_to_thread: bool = Field(
        True,
        description="Run ranking feature preparation and inference off the asyncio event loop",
    )
    product_feature_cache_size: int = Field(
        50000,
        description="Per-process cache size for static product ranking feature components",
    )

    # Multi-objective settings
    enable_multi_objective: bool = Field(True, description="Enable multi-objective optimization")
    ctr_weight: float = Field(1.0, description="Click-through rate weight")
    cvr_weight: float = Field(2.0, description="Conversion rate weight")
    gmv_weight: float = Field(3.0, description="GMV optimization weight")
    
    # Training settings
    epochs: int = Field(100, description="Training epochs")
    batch_size: int = Field(1024, description="Training batch size")
    early_stopping_patience: int = Field(10, description="Early stopping patience")
    enable_periodic_training: bool = Field(
        False,
        description="Enable background retraining of the ranking model checkpoint",
    )
    training_min_samples: int = Field(
        100,
        description="Minimum interaction samples required before ranking retraining runs",
    )
    checkpoint_sync_interval_seconds: int = Field(
        60,
        description="How often recommendation workers check for a newer ranking checkpoint",
    )
    
    class Config:
        env_prefix = "RANKING_"

class APIConfig(BaseSettings):
    """API server configuration."""
    host: str = Field("0.0.0.0", description="API server host")
    port: int = Field(8000, description="API server port")
    debug: bool = Field(False, description="Debug mode")
    reload: bool = Field(False, description="Auto-reload on code changes")
    
    # Request limits
    max_request_size: int = Field(100 * 1024 * 1024, description="Max request size (100MB)")
    request_timeout: int = Field(30, description="Request timeout in seconds")
    max_concurrent_requests: int = Field(100, description="Max concurrent requests")
    
    # CORS settings
    cors_origins: List[str] = Field(["*"], description="CORS allowed origins")
    cors_credentials: bool = Field(True, description="CORS allow credentials")
    
    # Security
    api_key: Optional[str] = Field(None, description="API key for authentication")
    rate_limit_requests: int = Field(1000, description="Rate limit per minute")
    
    class Config:
        env_prefix = "API_"

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: Any):
            if field_name == "cors_origins" and isinstance(raw_val, str):
                value = raw_val.strip()
                if not value:
                    return []
                if value.startswith("["):
                    return json.loads(value)
                return [item.strip() for item in value.split(",") if item.strip()]
            return super().parse_env_var(field_name, raw_val)

class CacheConfig(BaseSettings):
    """Caching configuration."""
    enable_caching: bool = Field(True, description="Enable recommendation caching")
    default_ttl: int = Field(3600, description="Default cache TTL in seconds")
    user_features_ttl: int = Field(1800, description="User features cache TTL")
    content_features_ttl: int = Field(86400, description="Content features cache TTL")
    recommendations_ttl: int = Field(900, description="Recommendations cache TTL")
    candidate_ttl: int = Field(300, description="Candidate cache TTL in seconds")
    product_metadata_ttl: int = Field(86400, description="Product metadata cache TTL in seconds")
    serving_pool_ttl: int = Field(1800, description="Serving pool cache TTL in seconds")
    hot_path_read_timeout_ms: float = Field(
        150.0,
        description="Maximum time to spend on optional Redis reads in the recommendation request path",
    )
    background_write_timeout_ms: float = Field(
        250.0,
        description="Maximum time to spend on best-effort cache/analytics writes in the request path",
    )
    
    # Cache size limits
    max_cache_size: int = Field(10000, description="Maximum cache entries")
    cleanup_interval: int = Field(3600, description="Cache cleanup interval")
    
    # Intelligent caching
    adaptive_ttl: bool = Field(True, description="Enable adaptive TTL based on user activity")
    high_activity_ttl: int = Field(300, description="TTL for high-activity users")
    low_activity_ttl: int = Field(7200, description="TTL for low-activity users")
    
    class Config:
        env_prefix = "CACHE_"

class MonitoringConfig(BaseSettings):
    """Monitoring and logging configuration."""
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    structured_logging: bool = Field(True, description="Emit logs as JSON")
    request_id_header: str = Field("X-Request-ID", description="Request ID header name")
    
    # Metrics collection
    enable_metrics: bool = Field(True, description="Enable metrics collection")
    enable_prometheus_metrics: bool = Field(True, description="Expose Prometheus metrics")
    metrics_interval: int = Field(60, description="Metrics collection interval")
    
    # Health checks
    health_check_interval: int = Field(30, description="Health check interval")
    component_timeout: int = Field(5, description="Component health check timeout")
    
    # Performance monitoring
    slow_request_threshold: float = Field(1.0, description="Slow request threshold in seconds")
    enable_request_logging: bool = Field(True, description="Enable request logging")
    enable_profiling_logs: bool = Field(
        False,
        description="Emit detailed timing breakdown logs for critical request paths",
    )
    profiling_log_min_duration_ms: float = Field(
        250.0,
        description="Minimum end-to-end request duration before detailed timing logs are emitted",
    )
    worker_heartbeat_interval_seconds: int = Field(
        15,
        description="How often workers publish heartbeat records to Redis",
    )
    worker_heartbeat_ttl_seconds: int = Field(
        45,
        description="Heartbeat TTL for worker readiness evaluation",
    )
    
    # Alerting thresholds
    error_rate_threshold: float = Field(0.05, description="Error rate alert threshold")
    response_time_threshold: float = Field(500, description="Response time threshold (ms)")
    memory_threshold: float = Field(0.8, description="Memory usage threshold")
    
    class Config:
        env_prefix = "MONITORING_"

class KafkaConfig(BaseSettings):
    """Kafka message streaming configuration."""
    # Connection settings
    bootstrap_servers: str = Field("localhost:9092", description="Kafka bootstrap servers")
    enable: bool = Field(True, description="Enable Kafka integration")
    
    # Topic names
    user_interactions_topic: str = Field("user-interactions", description="User interactions topic")
    video_processing_topic: str = Field("video-processing-tasks", description="Video processing tasks topic")
    recommendation_events_topic: str = Field("recommendation-events", description="Recommendation events topic")
    feature_updates_topic: str = Field("feature-updates", description="Feature updates topic")
    
    # Producer settings
    producer_acks: str = Field("all", description="Producer acknowledgment level")
    producer_retries: int = Field(3, description="Number of producer retries")
    producer_batch_size: int = Field(16384, description="Producer batch size in bytes")
    producer_linger_ms: int = Field(10, description="Producer linger time in milliseconds")
    producer_compression_type: str = Field("gzip", description="Producer compression type")
    
    # Consumer settings
    consumer_group_id: str = Field("video-commerce-group", description="Consumer group ID")
    consumer_auto_offset_reset: str = Field("earliest", description="Auto offset reset policy")
    consumer_enable_auto_commit: bool = Field(False, description="Enable auto commit")
    consumer_auto_commit_interval_ms: int = Field(5000, description="Auto commit interval")
    consumer_max_poll_records: int = Field(500, description="Max records per poll")
    consumer_handler_retries: int = Field(3, description="Handler retry attempts before DLQ/fail")
    consumer_handler_retry_backoff_ms: int = Field(250, description="Handler retry backoff in milliseconds")
    dead_letter_enable: bool = Field(True, description="Publish poison messages to a dead-letter topic")
    dead_letter_topic: str = Field("dead-letter-events", description="Kafka dead-letter topic")
    
    # Timeout settings
    request_timeout_ms: int = Field(30000, description="Request timeout in milliseconds")
    session_timeout_ms: int = Field(10000, description="Session timeout in milliseconds")
    
    # Retry settings
    retry_backoff_ms: int = Field(100, description="Retry backoff in milliseconds")
    max_in_flight_requests: int = Field(5, description="Max in-flight requests per connection")
    
    class Config:
        env_prefix = "KAFKA_"


class SecurityConfig(BaseSettings):
    """Internal service authentication configuration."""
    auth_mode: str = Field(
        "api_key",
        description="Client auth mode: disabled, api_key, bearer, or api_key_or_bearer",
    )
    internal_service_key: Optional[str] = Field(
        None,
        description="Shared secret used for service-to-service HTTP calls",
    )
    internal_service_header: str = Field(
        "X-Internal-Service-Key",
        description="Header name used for service-to-service authentication",
    )
    oidc_enabled: bool = Field(
        False,
        description="Enable bearer token authentication backed by OIDC/JWT validation",
    )
    oidc_required: bool = Field(
        False,
        description="Require bearer auth when OIDC/JWT validation is enabled",
    )
    oidc_issuer: Optional[str] = Field(
        None,
        description="Expected issuer claim for bearer tokens",
    )
    oidc_audience: Optional[str] = Field(
        None,
        description="Expected audience claim for bearer tokens",
    )
    oidc_jwks_url: Optional[str] = Field(
        None,
        description="JWKS URL used to validate bearer tokens signed by an external IdP",
    )
    jwt_shared_secret: Optional[str] = Field(
        None,
        description="Fallback shared secret for HS256 bearer validation in local/test environments",
    )
    jwt_algorithms: List[str] = Field(
        ["RS256"],
        description="Allowed bearer token algorithms",
    )

    class Config:
        env_prefix = "SECURITY_"

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: Any):
            if field_name == "jwt_algorithms" and isinstance(raw_val, str):
                value = raw_val.strip()
                if not value:
                    return []
                if value.startswith("["):
                    return json.loads(value)
                return [item.strip() for item in value.split(",") if item.strip()]
            return super().parse_env_var(field_name, raw_val)


class ServiceTopologyConfig(BaseSettings):
    """Inter-service routing and deployment configuration."""
    gateway_host: str = Field("0.0.0.0", description="Gateway bind host")
    gateway_port: int = Field(8000, description="Gateway bind port")
    recommendation_host: str = Field("0.0.0.0", description="Recommendation service bind host")
    recommendation_port: int = Field(8001, description="Recommendation service bind port")
    interaction_host: str = Field("0.0.0.0", description="Interaction ingest service bind host")
    interaction_port: int = Field(8002, description="Interaction ingest service bind port")
    recommendation_service_url: str = Field(
        "http://recommendation-service:8001",
        description="Internal URL for the recommendation service",
    )
    interaction_ingest_service_url: str = Field(
        "http://interaction-ingest-service:8002",
        description="Internal URL for the interaction ingest service",
    )
    request_forward_timeout_seconds: float = Field(
        10.0,
        description="Gateway timeout when forwarding to internal services",
    )
    proxy_connect_timeout_seconds: float = Field(
        5.0,
        description="HTTP connect timeout from gateway to internal services",
    )
    proxy_read_timeout_seconds: float = Field(
        10.0,
        description="HTTP read timeout from gateway to internal services",
    )
    proxy_write_timeout_seconds: float = Field(
        10.0,
        description="HTTP write timeout from gateway to internal services",
    )
    proxy_pool_timeout_seconds: float = Field(
        5.0,
        description="HTTP connection-pool acquisition timeout in the gateway",
    )
    proxy_max_connections: int = Field(
        500,
        description="Maximum concurrent upstream HTTP connections per gateway worker",
    )
    proxy_max_keepalive_connections: int = Field(
        250,
        description="Maximum idle keepalive upstream HTTP connections per gateway worker",
    )
    proxy_keepalive_expiry_seconds: float = Field(
        30.0,
        description="Idle keepalive expiry for upstream HTTP connections in the gateway",
    )
    trainer_interval_seconds: int = Field(
        3600,
        description="Background model trainer interval in seconds",
    )
    gateway_workers: int = Field(2, description="Gateway worker process count")
    recommendation_workers: int = Field(2, description="Recommendation worker process count")
    interaction_workers: int = Field(2, description="Interaction ingest worker process count")

    class Config:
        env_prefix = "SERVICE_"


class DataConfig(BaseSettings):
    """Data management configuration."""
    # Sample data
    load_sample_data: bool = Field(True, description="Load sample data on startup")
    use_csv_dataset: bool = Field(False, description="Load data from CSV files in Dataset folder")
    dataset_dir: str = Field("Dataset", description="Directory containing CSV dataset files")
    sample_data_path: str = Field("data/sample_products.json", description="Sample data file path")
    sample_users: int = Field(1000, description="Number of sample users to generate")
    sample_interactions: int = Field(10000, description="Number of sample interactions")
    
    # CSV dataset loading limits (None = load all)
    csv_limit_users: Optional[int] = Field(None, description="Limit number of users to load from CSV")
    csv_limit_products: Optional[int] = Field(None, description="Limit number of products to load from CSV")
    csv_limit_interactions: Optional[int] = Field(None, description="Limit number of interactions to load from CSV")
    csv_limit_content: Optional[int] = Field(None, description="Limit number of content items to load from CSV")
    
    # Data storage
    upload_dir: str = Field("/tmp/uploads", description="File upload directory")
    max_file_size: int = Field(500 * 1024 * 1024, description="Max upload file size (500MB)")
    allowed_extensions: List[str] = Field(
        [".mp4", ".avi", ".mov", ".mkv", ".webm"],
        description="Allowed video file extensions"
    )
    allowed_mime_types: List[str] = Field(
        [
            "video/mp4",
            "video/quicktime",
            "video/x-msvideo",
            "video/x-matroska",
            "video/webm",
        ],
        description="Allowed upload MIME types",
    )
    upload_chunk_size_bytes: int = Field(
        1024 * 1024,
        description="Chunk size for streaming uploads to disk",
    )
    
    # Data processing
    cleanup_temp_files: bool = Field(True, description="Cleanup temporary files")
    temp_file_ttl: int = Field(3600, description="Temporary file TTL in seconds")
    
    class Config:
        env_prefix = "DATA_"


class ObjectStorageConfig(BaseSettings):
    """Remote object storage configuration for upload durability."""
    backend: str = Field(
        "local",
        description="Upload storage backend: local or s3",
    )
    endpoint_url: Optional[str] = Field(
        None,
        description="S3-compatible endpoint URL such as http://minio:9000",
    )
    region: str = Field("us-east-1", description="Object storage region")
    bucket: Optional[str] = Field(
        None,
        description="Bucket used for uploaded videos and related assets",
    )
    access_key_id: Optional[str] = Field(
        None,
        description="Access key ID for the object storage backend",
    )
    secret_access_key: Optional[str] = Field(
        None,
        description="Secret access key for the object storage backend",
    )
    prefix: str = Field(
        "uploads",
        description="Object key prefix for uploaded files",
    )
    create_bucket_on_startup: bool = Field(
        False,
        description="Create the configured bucket on startup if it does not already exist",
    )
    force_path_style: bool = Field(
        True,
        description="Force path-style S3 requests for compatibility with MinIO and similar stores",
    )
    connect_timeout_seconds: int = Field(
        5,
        description="S3 client connect timeout in seconds",
    )
    read_timeout_seconds: int = Field(
        60,
        description="S3 client read timeout in seconds",
    )
    max_attempts: int = Field(
        3,
        description="Maximum S3 client retry attempts",
    )
    checksum_algorithm: Optional[str] = Field(
        None,
        description="Optional S3 upload checksum algorithm, for example CRC32 or SHA256",
    )
    server_side_encryption: Optional[str] = Field(
        None,
        description="Optional S3 server-side encryption algorithm, for example AES256",
    )
    download_dir: str = Field(
        "/tmp/object-storage",
        description="Temporary directory used when materializing remote objects for processing",
    )

    class Config:
        env_prefix = "OBJECT_STORAGE_"


class DatabaseConfig(BaseSettings):
    """Postgres system-of-record configuration."""
    enable: bool = Field(False, description="Enable Postgres persistence layer")
    url: str = Field(
        "postgresql+asyncpg://video_commerce:video_commerce@postgres:5432/video_commerce",
        description="Async SQLAlchemy database URL",
    )
    pool_size: int = Field(5, description="Database connection pool size")
    max_overflow: int = Field(5, description="Database connection pool overflow")
    auto_create_schema: bool = Field(
        True,
        description="Create required tables automatically on startup",
    )
    analytics_window_hours: int = Field(
        24,
        description="Time window used by online Postgres analytics summaries",
    )
    interaction_retention_days: int = Field(
        90,
        description="Retention window for raw interaction_events rows",
    )
    enable_retention_cleanup: bool = Field(
        False,
        description="Enable periodic interaction_events retention cleanup",
    )
    interaction_events_partitioned: bool = Field(
        False,
        description="Set true after applying the interaction_events time-partition migration",
    )
    partition_backfill_months: int = Field(
        1,
        description="Months of historical interaction_events partitions to create on startup",
    )
    partition_premake_months: int = Field(
        6,
        description="Months of future interaction_events partitions to create on startup",
    )
    retention_cleanup_interval_seconds: int = Field(
        3600,
        description="How often to run interaction_events retention cleanup",
    )
    echo_sql: bool = Field(False, description="Enable SQLAlchemy SQL logging")

    class Config:
        env_prefix = "DATABASE_"

class Config:
    """Main configuration class that combines all configuration sections."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration from environment variables and optional config file."""
        self.config_file = config_file
        _load_secret_file_env()
        
        # Load configuration sections
        self.redis_config = RedisConfig()
        self.model_config = ModelConfig()
        self.vector_config = VectorConfig()
        self.recommendation_config = RecommendationConfig()
        self.ranking_config = RankingConfig()
        self.api_config = APIConfig()
        self.cache_config = CacheConfig()
        self.monitoring_config = MonitoringConfig()
        self.data_config = DataConfig()
        self.object_storage_config = ObjectStorageConfig()
        self.kafka_config = KafkaConfig()
        self.security_config = SecurityConfig()
        self.service_topology_config = ServiceTopologyConfig()
        self.database_config = DatabaseConfig()
        
        # Load additional config from file if provided
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)
        
        # Set up derived configurations
        self._setup_derived_config()
        
        # Validate configuration
        self._validate_config()
        
        logger.info("Configuration loaded successfully")
    
    def _load_config_file(self, config_file: str):
        """Load additional configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            # Override default values with file config
            for section, values in file_config.items():
                if hasattr(self, f"{section}_config"):
                    config_obj = getattr(self, f"{section}_config")
                    for key, value in values.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
            
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load config file {config_file}: {e}")
    
    def _setup_derived_config(self):
        """Set up derived configuration values."""
        # Ensure cache directory exists
        os.makedirs(self.model_config.cache_dir, exist_ok=True)
        os.makedirs(self.data_config.upload_dir, exist_ok=True)
        os.makedirs(self.object_storage_config.download_dir, exist_ok=True)

        if not self.model_config.ranking_model_path:
            self.model_config.ranking_model_path = str(
                Path(self.model_config.cache_dir) / "ranking_model.pt"
            )
        if (
            "VECTOR_INDEX_PATH" not in os.environ
            and self.vector_config.index_path == "/tmp/vector_index.faiss"
        ):
            self.vector_config.index_path = str(
                Path(self.model_config.cache_dir) / "vector_index.faiss"
            )
        if (
            "RECOMMENDATION_CF_INDEX_PATH" not in os.environ
            and self.recommendation_config.cf_index_path == "/tmp/cf_vector_index.faiss"
        ):
            self.recommendation_config.cf_index_path = str(
                Path(self.model_config.cache_dir) / "cf_vector_index.faiss"
            )
        
        # Set embedding dimension consistency
        if self.vector_config.embedding_dim != self.model_config.embedding_dim:
            logger.warning("Vector and model embedding dimensions don't match")
            self.vector_config.embedding_dim = self.model_config.embedding_dim
        
        # Environment-specific overrides
        if os.getenv("ENVIRONMENT") == "production":
            self.api_config.debug = False
            self.api_config.reload = False
            self.monitoring_config.log_level = "WARNING"
        elif os.getenv("ENVIRONMENT") == "development":
            self.api_config.debug = True
            self.api_config.reload = True
            self.monitoring_config.log_level = "DEBUG"
    
    def _validate_config(self):
        """Validate configuration values."""
        errors = []
        
        # Validate weights sum to reasonable values
        total_weight = (
            self.recommendation_config.cf_weight +
            self.recommendation_config.content_weight +
            self.recommendation_config.popularity_weight
        )
        if abs(total_weight - 1.0) > 0.1:
            errors.append(f"Recommendation weights should sum to ~1.0, got {total_weight}")
        
        # Validate cache TTL values
        if self.cache_config.high_activity_ttl >= self.cache_config.low_activity_ttl:
            errors.append("High activity TTL should be less than low activity TTL")
        
        # Validate batch sizes
        if self.model_config.batch_size <= 0:
            errors.append("Model batch size must be positive")
        
        # Validate file paths
        if not os.path.exists(os.path.dirname(self.model_config.cache_dir)):
            try:
                os.makedirs(os.path.dirname(self.model_config.cache_dir), exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create model cache directory: {e}")

        if self.object_storage_config.backend not in {"local", "s3"}:
            errors.append(
                f"Object storage backend must be 'local' or 's3', got {self.object_storage_config.backend}"
            )
        if self.object_storage_config.backend == "s3":
            if not self.object_storage_config.bucket:
                errors.append("Object storage bucket is required when OBJECT_STORAGE_BACKEND=s3")
            if self.object_storage_config.max_attempts <= 0:
                errors.append("Object storage max attempts must be positive")
            if os.getenv("ENVIRONMENT", "").lower() == "production":
                if not self.object_storage_config.access_key_id:
                    errors.append("Object storage access key is required in production when S3 is enabled")
                if not self.object_storage_config.secret_access_key:
                    errors.append("Object storage secret key is required in production when S3 is enabled")
                if self.object_storage_config.access_key_id == "minioadmin":
                    errors.append("Default MinIO access key is not allowed in production")
                if self.object_storage_config.secret_access_key == "minioadmin":
                    errors.append("Default MinIO secret key is not allowed in production")
        if self.security_config.auth_mode not in {"disabled", "api_key", "bearer", "api_key_or_bearer"}:
            errors.append(
                "Security auth mode must be disabled, api_key, bearer, or api_key_or_bearer"
            )
        if self.database_config.enable and not self.database_config.url:
            errors.append("DATABASE_URL or DATABASE_URL_FILE is required when DATABASE_ENABLE=true")

        if os.getenv("ENVIRONMENT", "").lower() == "production":
            if self.security_config.auth_mode == "disabled":
                errors.append("SECURITY_AUTH_MODE=disabled is not allowed in production")
            if self.security_config.auth_mode in {"api_key", "api_key_or_bearer"} and not self.api_config.api_key:
                errors.append("API_API_KEY or API_API_KEY_FILE is required in production")
            if not self.security_config.internal_service_key:
                errors.append("SECURITY_INTERNAL_SERVICE_KEY or SECURITY_INTERNAL_SERVICE_KEY_FILE is required in production")
            if self.security_config.internal_service_key == "change-me-in-production":
                errors.append("Default internal service key is not allowed in production")
            if not self.redis_config.password:
                errors.append("REDIS_PASSWORD or REDIS_PASSWORD_FILE is required in production")
            if self.redis_config.cache_host and not (
                self.redis_config.cache_password or self.redis_config.password
            ):
                errors.append("REDIS_CACHE_PASSWORD, REDIS_CACHE_PASSWORD_FILE, or REDIS_PASSWORD is required in production when cache Redis is separate")
            if "video_commerce:video_commerce@" in self.database_config.url:
                errors.append("Default Postgres credentials are not allowed in production")
        
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError("Invalid configuration")
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        auth = f":{self.redis_config.password}@" if self.redis_config.password else ""
        return f"redis://{auth}{self.redis_config.host}:{self.redis_config.port}/{self.redis_config.db}"
    
    def get_model_device(self) -> str:
        """Get the appropriate device for ML models."""
        if self.model_config.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() and self.model_config.enable_gpu else "cpu"
            except ImportError:
                return "cpu"
        return self.model_config.device
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for logging/debugging)."""
        config_dict = {}
        for attr_name in dir(self):
            if attr_name.endswith('_config') and not attr_name.startswith('_'):
                config_obj = getattr(self, attr_name)
                if hasattr(config_obj, '__dict__'):
                    config_dict[attr_name] = config_obj.__dict__.copy()
                    # Remove sensitive information
                    if 'password' in config_dict[attr_name]:
                        config_dict[attr_name]['password'] = "***"
                    if 'api_key' in config_dict[attr_name]:
                        config_dict[attr_name]['api_key'] = "***"
                    if 'internal_service_key' in config_dict[attr_name]:
                        config_dict[attr_name]['internal_service_key'] = "***"
                    if 'secret_access_key' in config_dict[attr_name]:
                        config_dict[attr_name]['secret_access_key'] = "***"
                    if 'access_key_id' in config_dict[attr_name]:
                        config_dict[attr_name]['access_key_id'] = "***"
                    if 'jwt_shared_secret' in config_dict[attr_name]:
                        config_dict[attr_name]['jwt_shared_secret'] = "***"
        
        return config_dict
    
    @property
    def model_version(self) -> str:
        """Get model version string."""
        return "v1.0.0"
    
    @property
    def load_sample_data(self) -> bool:
        """Whether to load sample data on startup."""
        return self.data_config.load_sample_data and not os.getenv("NO_SAMPLE_DATA")

# Global configuration instance
_config_instance: Optional[Config] = None

def get_config(config_file: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    global _config_instance
    
    if _config_instance is None:
        # Look for config file in standard locations
        if config_file is None:
            for path in ["config.json", "config/config.json", "/etc/recommender/config.json"]:
                if os.path.exists(path):
                    config_file = path
                    break
        
        _config_instance = Config(config_file)
    
    return _config_instance

def reset_config():
    """Reset the global configuration instance (useful for testing)."""
    global _config_instance
    _config_instance = None

# Convenience function for getting specific config sections
def get_redis_config() -> RedisConfig:
    """Get Redis configuration."""
    return get_config().redis_config

def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return get_config().model_config

def get_api_config() -> APIConfig:
    """Get API configuration."""
    return get_config().api_config

def get_kafka_config() -> KafkaConfig:
    """Get Kafka configuration."""
    return get_config().kafka_config


def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_config().database_config

# Environment detection
def is_production() -> bool:
    """Check if running in production environment."""
    return os.getenv("ENVIRONMENT", "").lower() == "production"

def is_development() -> bool:
    """Check if running in development environment."""
    return os.getenv("ENVIRONMENT", "").lower() in ("development", "dev", "")

def is_testing() -> bool:
    """Check if running in test environment."""
    return os.getenv("ENVIRONMENT", "").lower() in ("test", "testing")
