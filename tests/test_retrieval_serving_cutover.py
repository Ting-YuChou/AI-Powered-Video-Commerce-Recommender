import asyncio
import sys
import types

try:
    import faiss  # noqa: F401
except ModuleNotFoundError:
    fake_faiss = types.ModuleType("faiss")
    fake_faiss.Index = object
    sys.modules["faiss"] = fake_faiss

from video_commerce.common.config import RecommendationConfig
from video_commerce.ml.recommender import RecommendationEngine


def test_load_models_skips_legacy_training_when_online_retraining_is_disabled():
    engine = RecommendationEngine.__new__(RecommendationEngine)
    engine.config = RecommendationConfig(online_retraining_enabled=False)
    calls = []

    async def load():
        calls.append("load")

    async def train():
        calls.append("train")

    engine._try_load_cf_index = load
    engine._update_models_from_interactions = train

    asyncio.run(engine.load_models())

    assert calls == ["load"]
    assert engine.is_initialized is True


def test_periodic_update_is_noop_when_online_retraining_is_disabled():
    engine = RecommendationEngine.__new__(RecommendationEngine)
    engine.config = RecommendationConfig(online_retraining_enabled=False)
    engine.last_model_update = 0
    calls = []

    async def train():
        calls.append("train")

    engine._update_models_from_interactions = train

    asyncio.run(engine.update_models())

    assert calls == []


def test_enforced_required_retrieval_propagates_missing_active_release():
    engine = RecommendationEngine.__new__(RecommendationEngine)
    engine.config = RecommendationConfig(
        retrieval_release_gate_mode="enforced",
        retrieval_required=True,
        cf_index_path="/missing/index.faiss",
    )

    class Artifacts:
        async def sync_selected_two_tower_artifacts(self, **kwargs):
            raise RuntimeError("active Two-Tower release is required")

    engine.artifact_manager = Artifacts()

    try:
        asyncio.run(engine._try_load_cf_index())
    except RuntimeError as exc:
        assert "active Two-Tower" in str(exc)
    else:
        raise AssertionError("enforced retrieval readiness failure was swallowed")
