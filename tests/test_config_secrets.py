import pytest

from video_commerce.common.config import Config, reset_config


def test_config_reads_secret_file_env(monkeypatch, tmp_path):
    secret_file = tmp_path / "api_key.txt"
    secret_file.write_text("from-file-secret\n", encoding="utf-8")

    monkeypatch.delenv("API_API_KEY", raising=False)
    monkeypatch.setenv("API_API_KEY_FILE", str(secret_file))

    reset_config()
    config = Config()

    assert config.api_config.api_key == "from-file-secret"

    reset_config()


def test_config_defaults_index_paths_under_model_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("VECTOR_INDEX_PATH", raising=False)
    monkeypatch.delenv("RECOMMENDATION_CF_INDEX_PATH", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    for env_name in (
        "REDIS_CACHE_HOST",
        "REDIS_CACHE_PORT",
        "REDIS_CACHE_DB",
        "REDIS_CACHE_PASSWORD",
        "REDIS_CACHE_PASSWORD_FILE",
        "REDIS_CACHE_MAX_CONNECTIONS",
        "REDIS_CACHE_SOCKET_TIMEOUT",
        "REDIS_CACHE_SOCKET_CONNECT_TIMEOUT",
        "SERVICE_RANKING_COORDINATOR_HOST",
        "SERVICE_RANKING_COORDINATOR_PORT",
        "SERVICE_RANKING_COORDINATOR_CLIENT_POOL_SIZE",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "models"))

    reset_config()
    config = Config()

    assert config.vector_config.index_path == str(
        tmp_path / "models" / "vector_index.faiss"
    )
    assert config.recommendation_config.cf_index_path == str(
        tmp_path / "models" / "cf_vector_index.faiss"
    )
    assert config.cache_config.hot_path_read_timeout_ms == 150.0
    assert config.cache_config.candidate_cache_race_timeout_ms == 5.0
    assert config.cache_config.recommendation_cache_race_timeout_ms == 5.0
    assert config.cache_config.background_task_queue_size == 8192
    assert config.recommendation_config.known_user_snapshot_enabled is True
    assert config.recommendation_config.content_features_snapshot_enabled is True
    assert config.recommendation_config.speech_category_candidates_enabled is False
    assert config.model_config.speech_to_text_enabled is False
    assert config.model_config.speech_to_text_model == "Qwen/Qwen3-ASR-0.6B"
    assert config.kafka_config.consumer_max_poll_interval_ms == 600000
    assert config.service_topology_config.ranking_single_coordinator_enabled is True
    assert config.service_topology_config.ranking_coordinator_host == ""
    assert config.service_topology_config.ranking_coordinator_port == 8013
    assert config.service_topology_config.ranking_coordinator_client_pool_size == 128
    assert config.service_topology_config.ranking_runner_endpoint_missing_refreshes == 3
    assert (
        config.service_topology_config.ranking_runner_endpoint_missing_grace_seconds
        == 30.0
    )
    assert config.database_config.analytics_summary_cache_ttl_seconds == 15
    assert config.database_config.training_sequence_lookback_days == 90
    assert config.database_config.ltr_impression_lookback_days == 30
    assert config.database_config.impression_retention_days == 90
    assert config.recommendation_config.preload_product_metadata_on_startup is False
    assert config.recommendation_config.publish_catalog_snapshot_on_startup is False
    assert config.recommendation_config.impression_logging_enabled is True
    assert config.recommendation_config.impression_max_items == 100
    assert config.redis_config.cache_host is None

    reset_config()


def test_config_reads_database_optimization_env(monkeypatch):
    monkeypatch.setenv("DATABASE_ANALYTICS_SUMMARY_CACHE_TTL_SECONDS", "0")
    monkeypatch.setenv("DATABASE_TRAINING_SEQUENCE_LOOKBACK_DAYS", "30")
    monkeypatch.setenv("DATABASE_LTR_IMPRESSION_LOOKBACK_DAYS", "14")
    monkeypatch.setenv("DATABASE_IMPRESSION_RETENTION_DAYS", "45")
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    reset_config()
    config = Config()

    assert config.database_config.analytics_summary_cache_ttl_seconds == 0
    assert config.database_config.training_sequence_lookback_days == 30
    assert config.database_config.ltr_impression_lookback_days == 14
    assert config.database_config.impression_retention_days == 45

    reset_config()


def test_config_reads_recommendation_impression_logging_env(monkeypatch):
    monkeypatch.setenv("RECOMMENDATION_IMPRESSION_LOGGING_ENABLED", "false")
    monkeypatch.setenv("RECOMMENDATION_IMPRESSION_MAX_ITEMS", "25")
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    reset_config()
    config = Config()

    assert config.recommendation_config.impression_logging_enabled is False
    assert config.recommendation_config.impression_max_items == 25

    reset_config()


def test_config_reads_separate_cache_redis_env(monkeypatch):
    monkeypatch.setenv("REDIS_CACHE_HOST", "redis-cache")
    monkeypatch.setenv("REDIS_CACHE_DB", "2")
    monkeypatch.setenv("REDIS_CACHE_SOCKET_TIMEOUT", "0.25")
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    reset_config()
    config = Config()

    assert config.redis_config.cache_host == "redis-cache"
    assert config.redis_config.cache_db == 2
    assert config.redis_config.cache_socket_timeout == 0.25

    reset_config()


def test_feature_pipeline_defaults_to_official_flink(monkeypatch):
    monkeypatch.delenv("FEATURE_PIPELINE_MODE", raising=False)
    monkeypatch.delenv("FLINK_FEATURE_OUTPUT_NAMESPACE", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    reset_config()
    config = Config()

    assert config.feature_pipeline_config.mode == "flink"
    assert config.feature_pipeline_config.flink_feature_output_namespace == "official"

    reset_config()


def test_config_reads_flink_feature_pipeline_env(monkeypatch):
    monkeypatch.setenv("FEATURE_PIPELINE_MODE", "flink_shadow")
    monkeypatch.setenv("FLINK_FEATURE_OUTPUT_NAMESPACE", "shadow")
    monkeypatch.setenv("FLINK_CHECKPOINT_DIR", "file:///tmp/flink-checkpoints")
    monkeypatch.setenv("FLINK_WATERMARK_OUT_OF_ORDERNESS_SECONDS", "120")
    monkeypatch.setenv("FLINK_ALLOWED_LATENESS_SECONDS", "300")
    monkeypatch.setenv("RANKING_REALTIME_WINDOW_FEATURES_ENABLED", "true")
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    reset_config()
    config = Config()

    assert config.feature_pipeline_config.mode == "flink_shadow"
    assert config.feature_pipeline_config.flink_feature_output_namespace == "shadow"
    assert (
        config.feature_pipeline_config.flink_checkpoint_dir
        == "file:///tmp/flink-checkpoints"
    )
    assert (
        config.feature_pipeline_config.flink_watermark_out_of_orderness_seconds == 120
    )
    assert config.feature_pipeline_config.flink_allowed_lateness_seconds == 300
    assert config.ranking_config.realtime_window_features_enabled is True

    reset_config()


def test_config_reads_official_flink_cutover_env(monkeypatch):
    monkeypatch.setenv("FEATURE_PIPELINE_MODE", "flink")
    monkeypatch.setenv("FLINK_FEATURE_OUTPUT_NAMESPACE", "official")
    monkeypatch.setenv("RANKING_REALTIME_WINDOW_FEATURES_ENABLED", "true")
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    reset_config()
    config = Config()

    assert config.feature_pipeline_config.mode == "flink"
    assert config.feature_pipeline_config.flink_feature_output_namespace == "official"
    assert config.ranking_config.realtime_window_features_enabled is True

    reset_config()


def test_config_reads_ranking_torch_compile_env(monkeypatch):
    for env_name in (
        "RANKING_TORCH_COMPILE_ENABLED",
        "RANKING_TORCH_COMPILE_BACKEND",
        "RANKING_TORCH_COMPILE_MODE",
        "RANKING_TORCH_COMPILE_DYNAMIC",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    reset_config()
    config = Config()

    assert config.ranking_config.torch_compile_enabled is False
    assert config.ranking_config.torch_compile_backend == "inductor"
    assert config.ranking_config.torch_compile_mode == "default"
    assert config.ranking_config.torch_compile_dynamic is True

    monkeypatch.setenv("RANKING_TORCH_COMPILE_ENABLED", "true")
    monkeypatch.setenv("RANKING_TORCH_COMPILE_BACKEND", "eager")
    monkeypatch.setenv("RANKING_TORCH_COMPILE_MODE", "reduce-overhead")
    monkeypatch.setenv("RANKING_TORCH_COMPILE_DYNAMIC", "false")

    reset_config()
    config = Config()

    assert config.ranking_config.torch_compile_enabled is True
    assert config.ranking_config.torch_compile_backend == "eager"
    assert config.ranking_config.torch_compile_mode == "reduce-overhead"
    assert config.ranking_config.torch_compile_dynamic is False

    reset_config()


def test_config_reads_ranking_ltr_env(monkeypatch):
    for env_name in (
        "RANKING_LTR_PAIRWISE_ENABLED",
        "RANKING_LTR_PAIRWISE_WEIGHT",
        "RANKING_LTR_MAX_PAIRS_PER_GROUP",
        "RANKING_LTR_MIN_RELEVANCE_GAP",
        "RANKING_LTR_LISTWISE_ENABLED",
        "RANKING_LTR_LISTWISE_WEIGHT",
        "RANKING_LTR_LISTWISE_MIN_GROUP_SIZE",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    reset_config()
    config = Config()

    assert config.ranking_config.ltr_pairwise_enabled is False
    assert config.ranking_config.ltr_pairwise_weight == 0.1
    assert config.ranking_config.ltr_max_pairs_per_group == 2048
    assert config.ranking_config.ltr_min_relevance_gap == 0.5
    assert config.ranking_config.ltr_listwise_enabled is True
    assert config.ranking_config.ltr_listwise_weight == 0.1
    assert config.ranking_config.ltr_listwise_min_group_size == 2

    monkeypatch.setenv("RANKING_LTR_PAIRWISE_ENABLED", "true")
    monkeypatch.setenv("RANKING_LTR_PAIRWISE_WEIGHT", "0.25")
    monkeypatch.setenv("RANKING_LTR_MAX_PAIRS_PER_GROUP", "128")
    monkeypatch.setenv("RANKING_LTR_MIN_RELEVANCE_GAP", "1.0")
    monkeypatch.setenv("RANKING_LTR_LISTWISE_ENABLED", "true")
    monkeypatch.setenv("RANKING_LTR_LISTWISE_WEIGHT", "0.35")
    monkeypatch.setenv("RANKING_LTR_LISTWISE_MIN_GROUP_SIZE", "3")

    reset_config()
    config = Config()

    assert config.ranking_config.ltr_pairwise_enabled is True
    assert config.ranking_config.ltr_pairwise_weight == 0.25
    assert config.ranking_config.ltr_max_pairs_per_group == 128
    assert config.ranking_config.ltr_min_relevance_gap == 1.0
    assert config.ranking_config.ltr_listwise_enabled is True
    assert config.ranking_config.ltr_listwise_weight == 0.35
    assert config.ranking_config.ltr_listwise_min_group_size == 3

    reset_config()


def test_explicit_monitoring_log_level_wins_over_development_env(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("MONITORING_LOG_LEVEL", "INFO")

    reset_config()
    config = Config()

    assert config.monitoring_config.log_level == "INFO"

    reset_config()


def test_production_config_rejects_insecure_defaults(monkeypatch):
    for env_name in (
        "API_API_KEY",
        "API_API_KEY_FILE",
        "SECURITY_INTERNAL_SERVICE_KEY",
        "SECURITY_INTERNAL_SERVICE_KEY_FILE",
        "REDIS_PASSWORD",
        "REDIS_PASSWORD_FILE",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("SECURITY_AUTH_MODE", "api_key")

    reset_config()
    with pytest.raises(ValueError):
        Config()

    reset_config()


def test_production_config_accepts_secret_files(monkeypatch, tmp_path):
    api_key_file = tmp_path / "api_key.txt"
    internal_key_file = tmp_path / "internal_key.txt"
    redis_password_file = tmp_path / "redis_password.txt"
    api_key_file.write_text("api-key\n", encoding="utf-8")
    internal_key_file.write_text("internal-key\n", encoding="utf-8")
    redis_password_file.write_text("redis-password\n", encoding="utf-8")

    for env_name in (
        "API_API_KEY",
        "SECURITY_INTERNAL_SERVICE_KEY",
        "REDIS_PASSWORD",
        "OBJECT_STORAGE_BACKEND",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("SECURITY_AUTH_MODE", "api_key")
    monkeypatch.setenv("API_API_KEY_FILE", str(api_key_file))
    monkeypatch.setenv("SECURITY_INTERNAL_SERVICE_KEY_FILE", str(internal_key_file))
    monkeypatch.setenv("REDIS_PASSWORD_FILE", str(redis_password_file))
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://video_commerce:strong-password@postgres:5432/video_commerce",
    )

    reset_config()
    config = Config()

    assert config.api_config.api_key == "api-key"
    assert config.security_config.internal_service_key == "internal-key"
    assert config.redis_config.password == "redis-password"

    reset_config()
