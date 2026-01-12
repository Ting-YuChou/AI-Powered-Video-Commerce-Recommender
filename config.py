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
from pydantic import BaseSettings, Field, validator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RedisConfig(BaseSettings):
    """Redis database configuration."""
    host: str = Field("localhost", description="Redis host address")
    port: int = Field(6379, description="Redis port")
    db: int = Field(0, description="Redis database number")
    password: Optional[str] = Field(None, description="Redis password")
    decode_responses: bool = Field(True, description="Decode Redis responses")
    max_connections: int = Field(100, description="Maximum Redis connections")
    socket_timeout: int = Field(30, description="Socket timeout in seconds")
    socket_connect_timeout: int = Field(30, description="Socket connect timeout")
    retry_on_timeout: bool = Field(True, description="Retry on timeout")
    
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
    # Collaborative filtering
    cf_factors: int = Field(64, description="Collaborative filtering factors")
    cf_regularization: float = Field(0.1, description="CF regularization parameter")
    cf_iterations: int = Field(50, description="CF training iterations")
    
    # Candidate generation
    candidates_per_source: int = Field(100, description="Candidates per recommendation source")
    max_total_candidates: int = Field(500, description="Maximum total candidates")
    
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
    
    class Config:
        env_prefix = "RECOMMENDATION_"

class RankingConfig(BaseSettings):
    """Ranking model configuration."""
    model_type: str = Field("neural", description="Ranking model type")
    hidden_dims: List[int] = Field([256, 128, 64], description="Neural network hidden dimensions")
    dropout_rate: float = Field(0.2, description="Dropout rate")
    learning_rate: float = Field(0.001, description="Learning rate")
    
    # Multi-objective settings
    enable_multi_objective: bool = Field(True, description="Enable multi-objective optimization")
    ctr_weight: float = Field(1.0, description="Click-through rate weight")
    cvr_weight: float = Field(2.0, description="Conversion rate weight")
    gmv_weight: float = Field(3.0, description="GMV optimization weight")
    
    # Training settings
    epochs: int = Field(100, description="Training epochs")
    batch_size: int = Field(1024, description="Training batch size")
    early_stopping_patience: int = Field(10, description="Early stopping patience")
    
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

class CacheConfig(BaseSettings):
    """Caching configuration."""
    enable_caching: bool = Field(True, description="Enable recommendation caching")
    default_ttl: int = Field(3600, description="Default cache TTL in seconds")
    user_features_ttl: int = Field(1800, description="User features cache TTL")
    content_features_ttl: int = Field(86400, description="Content features cache TTL")
    recommendations_ttl: int = Field(900, description="Recommendations cache TTL")
    
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
    
    # Metrics collection
    enable_metrics: bool = Field(True, description="Enable metrics collection")
    metrics_interval: int = Field(60, description="Metrics collection interval")
    
    # Health checks
    health_check_interval: int = Field(30, description="Health check interval")
    component_timeout: int = Field(5, description="Component health check timeout")
    
    # Performance monitoring
    slow_request_threshold: float = Field(1.0, description="Slow request threshold in seconds")
    enable_request_logging: bool = Field(True, description="Enable request logging")
    
    # Alerting thresholds
    error_rate_threshold: float = Field(0.05, description="Error rate alert threshold")
    response_time_threshold: float = Field(500, description="Response time threshold (ms)")
    memory_threshold: float = Field(0.8, description="Memory usage threshold")
    
    class Config:
        env_prefix = "MONITORING_"

class DataConfig(BaseSettings):
    """Data management configuration."""
    # Sample data
    load_sample_data: bool = Field(True, description="Load sample data on startup")
    sample_data_path: str = Field("data/sample_products.json", description="Sample data file path")
    sample_users: int = Field(1000, description="Number of sample users to generate")
    sample_interactions: int = Field(10000, description="Number of sample interactions")
    
    # Data storage
    upload_dir: str = Field("/tmp/uploads", description="File upload directory")
    max_file_size: int = Field(500 * 1024 * 1024, description="Max upload file size (500MB)")
    allowed_extensions: List[str] = Field(
        [".mp4", ".avi", ".mov", ".mkv"], 
        description="Allowed video file extensions"
    )
    
    # Data processing
    cleanup_temp_files: bool = Field(True, description="Cleanup temporary files")
    temp_file_ttl: int = Field(3600, description="Temporary file TTL in seconds")
    
    class Config:
        env_prefix = "DATA_"

class Config:
    """Main configuration class that combines all configuration sections."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration from environment variables and optional config file."""
        self.config_file = config_file
        
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