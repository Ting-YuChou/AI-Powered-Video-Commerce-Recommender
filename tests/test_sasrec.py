import pytest

from config import CacheConfig, RecommendationConfig
from models import CandidateProduct, UserFeatures
from recommender import RecommendationEngine
from sasrec import SASRecCandidateEngine


def _event(product_id, action="view", timestamp=1.0):
    return {
        "user_id": "u1",
        "product_id": product_id,
        "action": action,
        "timestamp": timestamp,
        "occurred_at": timestamp,
        "event_id": f"e-{product_id}-{timestamp}",
    }


def _small_sasrec_config(**overrides):
    values = {
        "enable_sasrec": True,
        "sasrec_max_sequence_length": 3,
        "sasrec_embedding_dim": 8,
        "sasrec_num_heads": 2,
        "sasrec_num_layers": 1,
        "sasrec_dropout": 0.0,
        "sasrec_batch_size": 4,
        "sasrec_epochs": 2,
        "sasrec_learning_rate": 0.01,
        "sasrec_min_sequence_length": 2,
    }
    values.update(overrides)
    return RecommendationConfig(**values)


def test_sasrec_training_samples_pad_and_truncate_unknown_items():
    engine = SASRecCandidateEngine(_small_sasrec_config(sasrec_max_sequence_length=3))
    engine.product_to_id = {"p1": 1, "p2": 2, "p3": 3, "p4": 4}

    samples = engine._build_training_samples(
        {
            "u1": [
                _event("unknown", timestamp=0),
                _event("p1", timestamp=1),
                _event("p2", timestamp=2),
                _event("p3", timestamp=3),
                _event("p4", timestamp=4),
            ]
        }
    )

    assert len(samples) == 1
    input_ids, label_ids = samples[0]
    assert input_ids.tolist() == [1, 2, 3]
    assert label_ids.tolist() == [2, 3, 4]


@pytest.mark.asyncio
async def test_sasrec_train_save_load_and_infer_candidates(tmp_path):
    config = _small_sasrec_config()
    engine = SASRecCandidateEngine(config)
    sequences = {
        "u1": [_event("p1", timestamp=1), _event("p2", timestamp=2), _event("p3", timestamp=3)],
        "u2": [_event("p1", timestamp=1), _event("p2", timestamp=2), _event("p4", timestamp=3)],
    }
    catalog = {"p1", "p2", "p3", "p4"}

    trained = await engine.train_model(sequences, catalog_product_ids=catalog)

    assert trained is True
    assert engine.is_trained is True

    checkpoint = tmp_path / "sasrec.pt"
    vocab = tmp_path / "sasrec_vocab.json"
    metadata = tmp_path / "sasrec_metadata.json"
    engine.save_artifacts(str(checkpoint), str(vocab), str(metadata))

    loaded = SASRecCandidateEngine(config)
    assert loaded.load_artifacts(str(checkpoint), str(vocab), str(metadata)) is True

    candidates = await loaded.get_candidates(
        [_event("p1", timestamp=1), _event("p2", timestamp=2)],
        k=2,
        exclude_items={"p1", "p2"},
        catalog_product_ids=catalog,
    )

    assert candidates
    assert all(candidate.source == "sasrec" for candidate in candidates)
    assert all(candidate.product_id not in {"p1", "p2"} for candidate in candidates)
    assert all(0.0 <= candidate.collaborative_score <= 1.0 for candidate in candidates)


@pytest.mark.asyncio
async def test_sasrec_returns_empty_for_cold_or_unknown_sequence():
    engine = SASRecCandidateEngine(_small_sasrec_config())

    assert await engine.get_candidates([], k=5, exclude_items=set(), catalog_product_ids={"p1"}) == []

    trained = await engine.train_model(
        {"u1": [_event("p1", timestamp=1), _event("p2", timestamp=2)]},
        catalog_product_ids={"p1", "p2"},
    )
    assert trained is True
    assert await engine.get_candidates(
        [_event("unknown", timestamp=1)],
        k=5,
        exclude_items=set(),
        catalog_product_ids={"p1", "p2"},
    ) == []


class FakeFeatureStore:
    def __init__(self, interactions=None):
        self.cache_config = CacheConfig(hot_path_read_timeout_ms=50)
        self.interactions = interactions or []

    async def get_user_interactions(self, user_id, limit=100):
        return list(self.interactions)

    async def get_user_features(self, user_id, cache_default=False):
        return UserFeatures(user_id=user_id)

    async def get_trending_pool(self, k, exclude_items=None):
        exclude_items = exclude_items or set()
        if "trend" in exclude_items:
            return []
        return [
            CandidateProduct(
                product_id="trend",
                popularity_score=0.4,
                combined_score=0.4,
                source="trending_pool",
            )
        ][:k]

    async def get_category_pool(self, category, k, exclude_items=None):
        return []


class FakeVectorSearch:
    embedding_dim = 8
    product_metadata = {
        "trend": {"category": "cat"},
        "seq": {"category": "cat"},
    }

    async def search_similar_products(self, embedding, k=10):
        return []

    async def get_random_products(self, k=10):
        return []


class FakeSASRec:
    is_trained = True
    model_version = "sasrec-test"
    product_to_id = {"seq": 1}
    model = None

    async def get_candidates(self, sequence, *, k, exclude_items, catalog_product_ids):
        return [
            CandidateProduct(
                product_id="seq",
                collaborative_score=0.9,
                combined_score=0.9,
                source="sasrec",
            )
        ][:k]


class FailingSASRec(FakeSASRec):
    async def get_candidates(self, sequence, *, k, exclude_items, catalog_product_ids):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_generate_candidates_skips_sasrec_when_disabled():
    engine = RecommendationEngine(
        FakeFeatureStore(interactions=[_event("seen")]),
        FakeVectorSearch(),
        RecommendationConfig(enable_sasrec=False, candidates_per_source=2, max_total_candidates=10),
    )

    candidates, profile = await engine.generate_candidates("u1", include_profile=True)

    assert [candidate.source for candidate in candidates] == ["trending_pool"]
    assert profile["source_counts"]["sasrec"] == 0


@pytest.mark.asyncio
async def test_generate_candidates_merges_sasrec_source_when_enabled():
    engine = RecommendationEngine(
        FakeFeatureStore(interactions=[_event("seen")]),
        FakeVectorSearch(),
        RecommendationConfig(enable_sasrec=True, candidates_per_source=2, max_total_candidates=10),
    )
    engine.sasrec_engine = FakeSASRec()

    candidates, profile = await engine.generate_candidates("u1", include_profile=True)

    assert "sasrec" in {candidate.source for candidate in candidates}
    assert profile["source_counts"]["sasrec"] == 1


@pytest.mark.asyncio
async def test_generate_candidates_falls_back_when_sasrec_errors():
    engine = RecommendationEngine(
        FakeFeatureStore(interactions=[_event("seen")]),
        FakeVectorSearch(),
        RecommendationConfig(enable_sasrec=True, candidates_per_source=2, max_total_candidates=10),
    )
    engine.sasrec_engine = FailingSASRec()

    candidates, profile = await engine.generate_candidates("u1", include_profile=True)

    assert [candidate.source for candidate in candidates] == ["trending_pool"]
    assert profile["source_counts"]["sasrec"] == 0
