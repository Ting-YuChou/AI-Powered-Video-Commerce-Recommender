import pytest

from config import Config, reset_config


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
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "models"))

    reset_config()
    config = Config()

    assert config.vector_config.index_path == str(tmp_path / "models" / "vector_index.faiss")
    assert config.recommendation_config.cf_index_path == str(
        tmp_path / "models" / "cf_vector_index.faiss"
    )

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
