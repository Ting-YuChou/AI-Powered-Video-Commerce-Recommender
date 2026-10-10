import pytest

from video_commerce.common.config import Config, RecommendationConfig, reset_config


def test_retrieval_runtime_defaults_keep_local_legacy_compatible():
    config = RecommendationConfig()

    assert config.retrieval_training_source == "legacy"
    assert config.retrieval_release_gate_mode == "observe"
    assert config.online_retraining_enabled is True
    assert config.retrieval_holdout_days == 7


def test_production_requires_pit_enforced_and_disables_online_training(
    monkeypatch, caplog
):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("VECTOR_BOOTSTRAP_MODE", "required")
    monkeypatch.setenv("MODEL_RELEASE_GATE_MODE", "enforced")
    monkeypatch.setenv("RETRIEVAL_TRAINING_SOURCE", "legacy")
    monkeypatch.setenv("RETRIEVAL_RELEASE_GATE_MODE", "observe")
    monkeypatch.setenv("RECOMMENDATION_ONLINE_RETRAINING_ENABLED", "true")
    reset_config()

    with pytest.raises(ValueError, match="Invalid configuration"):
        Config()
    assert "RETRIEVAL_TRAINING_SOURCE=pit" in caplog.text
    assert "RETRIEVAL_RELEASE_GATE_MODE=enforced" in caplog.text
    assert "RECOMMENDATION_ONLINE_RETRAINING_ENABLED=false" in caplog.text

    reset_config()
