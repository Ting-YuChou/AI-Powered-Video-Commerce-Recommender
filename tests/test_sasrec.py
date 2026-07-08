import time

import numpy as np
import pytest

from video_commerce.common.config import CacheConfig, RecommendationConfig
from video_commerce.common.models import CandidateProduct, ContentFeatures, UserFeatures
from video_commerce.ml import sasrec as sasrec_module
from video_commerce.ml.recommender import RecommendationEngine
from video_commerce.ml.sasrec import SASRecCandidateEngine


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


def test_sasrec_config_rejects_invalid_attention_shape():
    with pytest.raises(ValueError, match="sasrec_embedding_dim"):
        _small_sasrec_config(sasrec_embedding_dim=8, sasrec_num_heads=3)


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
async def test_sasrec_training_failure_preserves_existing_state(monkeypatch):
    config = _small_sasrec_config()
    engine = SASRecCandidateEngine(config)
    trained = await engine.train_model(
        {"u1": [_event("p1", timestamp=1), _event("p2", timestamp=2)]},
        catalog_product_ids={"p1", "p2"},
    )
    assert trained is True
    original_model = engine.model
    original_product_to_id = dict(engine.product_to_id)
    original_id_to_product = dict(engine.id_to_product)
    original_sample_count = engine.training_sample_count

    class RaisingSASRecModel:
        def __init__(self, **kwargs):
            raise RuntimeError("model init failed")

    monkeypatch.setattr(sasrec_module, "SASRecModel", RaisingSASRecModel)

    with pytest.raises(RuntimeError, match="model init failed"):
        await engine.train_model(
            {"u1": [_event("p3", timestamp=1), _event("p4", timestamp=2)]},
            catalog_product_ids={"p3", "p4"},
        )

    assert engine.is_trained is True
    assert engine.model is original_model
    assert engine.product_to_id == original_product_to_id
    assert engine.id_to_product == original_id_to_product
    assert engine.training_sample_count == original_sample_count


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
        self.interaction_reads = 0
        self.interaction_limits = []

    async def get_user_interactions(self, user_id, limit=100):
        self.interaction_reads += 1
        self.interaction_limits.append(limit)
        return list(self.interactions[:limit])

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


class TrainingFeatureStore(FakeFeatureStore):
    async def get_all_user_features_map(self):
        return {}


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


class RecordingSequenceSystemStore:
    def __init__(self):
        self.calls = []

    async def get_user_training_sequences(self, **kwargs):
        self.calls.append(kwargs)
        return {}


class FakeArtifactManager:
    def __init__(self, system_store):
        self.system_store = system_store


class TrainingSequenceSystemStore:
    async def get_training_interactions(self, limit=50000):
        return [
            _event("p1", timestamp=1),
            _event("p2", timestamp=2),
            _event("p3", timestamp=3),
        ]

    async def get_user_training_sequences(self, **kwargs):
        return {
            "u1": [
                _event("p1", timestamp=1),
                _event("p2", timestamp=2),
                _event("p3", timestamp=3),
            ]
        }


class NonTrainingCFEngine:
    is_trained = False

    def __init__(self):
        self.calls = []

    async def train_model(
        self,
        interactions,
        user_features_map=None,
        external_negatives=None,
    ):
        self.calls.append(
            {
                "interactions": list(interactions),
                "user_features_map": dict(user_features_map or {}),
                "external_negatives": list(external_negatives or []),
            }
        )


class RaisingTrainingSASRec:
    is_trained = False

    async def train_model(self, sequences, *, catalog_product_ids):
        raise RuntimeError("sasrec training failed")


class RecordingTrendingEngine:
    def __init__(self):
        self.calls = []

    async def update_trending_scores(self, interactions):
        self.calls.append(list(interactions))


class RecordingCategoryFeatureStore(FakeFeatureStore):
    def __init__(self):
        super().__init__(interactions=[])
        self.category_calls = []

    async def get_category_pool(self, category, k, exclude_items=None):
        self.category_calls.append(category)
        return [
            CandidateProduct(
                product_id=f"category-{category}",
                popularity_score=0.5,
                combined_score=0.5,
                source="category_pool",
            )
        ][:k]


class FakeNewItemVectorSearch:
    embedding_dim = 8

    def __init__(self):
        self.embedding_lookup_count = 0
        self.catalog_version = 1
        self.last_updated = 1
        self.product_metadata = {
            "fresh": {
                "category": "cat",
                "active": True,
                "in_stock": True,
                "created_at": time.time(),
            }
        }
        self.product_embeddings = {
            "fresh": np.ones(8, dtype=np.float32),
        }

    def get_product_embedding(self, product_id):
        self.embedding_lookup_count += 1
        return self.product_embeddings.get(product_id)

    def get_catalog_version_context(self):
        return {
            "catalog_version": self.catalog_version,
            "last_updated": self.last_updated,
            "product_count": len(self.product_metadata),
        }

    async def search_similar_products(self, embedding, k=10):
        return []

    async def get_random_products(self, k=10):
        return []


class FakeSASRec:
    def __init__(self, candidates=None):
        self.is_trained = True
        self.model_version = "sasrec-test"
        self.product_to_id = {"seq": 1}
        self.model = None
        self.calls = []
        self.candidates = candidates or [
            CandidateProduct(
                product_id="seq",
                collaborative_score=0.9,
                combined_score=0.9,
                source="sasrec",
            )
        ]

    async def get_candidates(self, sequence, *, k, exclude_items, catalog_product_ids):
        self.calls.append(
            {
                "sequence": list(sequence),
                "exclude_items": set(exclude_items),
                "catalog_product_ids": set(catalog_product_ids),
            }
        )
        return self.candidates[:k]


class FailingSASRec(FakeSASRec):
    async def get_candidates(self, sequence, *, k, exclude_items, catalog_product_ids):
        raise RuntimeError("boom")


class FakeSwingItemCF:
    def __init__(self):
        self.is_trained = True
        self.model_version = "swing-test"
        self.calls = []

    def get_candidates(self, user_interactions, *, k, exclude_items, current_time=None):
        self.calls.append(
            {
                "user_interactions": list(user_interactions),
                "exclude_items": set(exclude_items),
            }
        )
        return [
            CandidateProduct(
                product_id="swing",
                collaborative_score=0.7,
                combined_score=0.7,
                source="swing_itemcf",
            )
        ][:k]


@pytest.mark.asyncio
async def test_generate_candidates_skips_sasrec_when_disabled():
    feature_store = FakeFeatureStore(interactions=[_event("seen")])
    engine = RecommendationEngine(
        feature_store,
        FakeVectorSearch(),
        RecommendationConfig(
            enable_sasrec=False,
            candidates_per_source=2,
            max_total_candidates=10,
            serving_recent_interaction_limit=0,
        ),
    )

    candidates, profile = await engine.generate_candidates("u1", include_profile=True)

    assert [candidate.source for candidate in candidates] == ["trending_pool"]
    assert profile["source_counts"]["sasrec"] == 0
    assert feature_store.interaction_reads == 0


@pytest.mark.asyncio
async def test_generate_candidates_merges_swing_itemcf_when_enabled():
    feature_store = FakeFeatureStore(interactions=[_event("seed", action="click")])
    engine = RecommendationEngine(
        feature_store,
        FakeVectorSearch(),
        RecommendationConfig(
            enable_swing_itemcf=True,
            enable_sasrec=False,
            candidates_per_source=2,
            max_total_candidates=10,
            serving_recent_interaction_limit=0,
            swing_itemcf_serving_interaction_limit=25,
        ),
    )
    engine.swing_itemcf_engine = FakeSwingItemCF()

    candidates, profile = await engine.generate_candidates("u1", include_profile=True)

    assert "swing_itemcf" in {candidate.source for candidate in candidates}
    assert profile["source_counts"]["swing_itemcf"] == 1
    assert feature_store.interaction_reads == 1
    assert feature_store.interaction_limits == [25]
    assert engine.swing_itemcf_engine.calls[0]["exclude_items"] == {"seed"}


@pytest.mark.asyncio
async def test_generate_candidates_reads_larger_swing_window_before_seed_dedupe():
    interactions = [
        _event("duplicate", action="view", timestamp=10),
        _event("duplicate", action="click", timestamp=9),
        _event("noise", action="report", timestamp=8),
        _event("older-positive", action="purchase", timestamp=7),
    ]
    feature_store = FakeFeatureStore(interactions=interactions)
    engine = RecommendationEngine(
        feature_store,
        FakeVectorSearch(),
        RecommendationConfig(
            enable_swing_itemcf=True,
            enable_sasrec=False,
            candidates_per_source=2,
            max_total_candidates=10,
            serving_recent_interaction_limit=0,
            swing_itemcf_max_seed_items=1,
            swing_itemcf_serving_interaction_limit=4,
        ),
    )
    engine.swing_itemcf_engine = FakeSwingItemCF()

    await engine.generate_candidates("u1", include_profile=True)

    assert feature_store.interaction_limits == [4]
    observed = engine.swing_itemcf_engine.calls[0]["user_interactions"]
    assert [event["product_id"] for event in observed] == [
        "duplicate",
        "duplicate",
        "noise",
        "older-positive",
    ]


@pytest.mark.asyncio
async def test_sasrec_training_sequences_use_configured_lookback():
    system_store = RecordingSequenceSystemStore()
    engine = RecommendationEngine(
        FakeFeatureStore(interactions=[]),
        FakeVectorSearch(),
        RecommendationConfig(enable_sasrec=True),
        artifact_manager=FakeArtifactManager(system_store),
        training_sequence_lookback_days=90,
    )

    started_at = time.time()
    await engine._update_sasrec_from_sequences()
    finished_at = time.time()

    assert len(system_store.calls) == 1
    since = system_store.calls[0]["since"]
    assert started_at - (90 * 86400) <= since <= finished_at - (90 * 86400)


@pytest.mark.asyncio
async def test_sasrec_training_sequences_can_disable_lookback():
    system_store = RecordingSequenceSystemStore()
    engine = RecommendationEngine(
        FakeFeatureStore(interactions=[]),
        FakeVectorSearch(),
        RecommendationConfig(enable_sasrec=True),
        artifact_manager=FakeArtifactManager(system_store),
        training_sequence_lookback_days=0,
    )

    await engine._update_sasrec_from_sequences()

    assert system_store.calls[0]["since"] is None


@pytest.mark.asyncio
async def test_sasrec_training_failure_does_not_abort_model_update():
    system_store = TrainingSequenceSystemStore()
    engine = RecommendationEngine(
        TrainingFeatureStore(interactions=[]),
        FakeVectorSearch(),
        RecommendationConfig(enable_sasrec=True),
        artifact_manager=FakeArtifactManager(system_store),
    )
    engine.cf_engine = NonTrainingCFEngine()
    engine.sasrec_engine = RaisingTrainingSASRec()
    engine.trending_engine = RecordingTrendingEngine()
    refresh_calls = []

    async def refresh_serving_pools():
        refresh_calls.append(True)

    engine.refresh_serving_pools = refresh_serving_pools

    await engine._update_models_from_interactions()

    assert len(engine.cf_engine.calls) == 1
    assert len(engine.trending_engine.calls) == 1
    assert refresh_calls == [True]


@pytest.mark.asyncio
async def test_generate_candidates_reads_recent_interactions_when_configured():
    feature_store = FakeFeatureStore(interactions=[_event("trend")])
    engine = RecommendationEngine(
        feature_store,
        FakeVectorSearch(),
        RecommendationConfig(
            enable_sasrec=False,
            candidates_per_source=2,
            max_total_candidates=10,
            serving_recent_interaction_limit=10,
        ),
    )

    candidates, _ = await engine.generate_candidates("u1", include_profile=True)

    assert candidates == []
    assert feature_store.interaction_reads == 1


@pytest.mark.asyncio
async def test_content_and_random_retrieval_use_bounded_executor():
    class ExecutorCFEngine:
        is_trained = False

        def __init__(self):
            self.calls = []

        async def run_in_retrieval_executor(self, func, *args, **kwargs):
            self.calls.append(func.__name__)
            return func(*args, **kwargs)

    class SyncVectorSearch:
        embedding_dim = 2
        product_metadata = {
            "content": {"category": "cat"},
            "random": {"category": "cat"},
        }

        def search_similar_products_sync(self, embedding, k=10, filter_categories=None):
            return [
                CandidateProduct(
                    product_id="content",
                    content_similarity_score=0.9,
                    combined_score=0.9,
                    source="content_similarity",
                )
            ][:k]

        def get_random_products_sync(self, k=10):
            return [
                CandidateProduct(
                    product_id="random",
                    popularity_score=0.2,
                    combined_score=0.2,
                    source="random",
                )
            ][:k]

    engine = RecommendationEngine(
        FakeFeatureStore(interactions=[]),
        SyncVectorSearch(),
        RecommendationConfig(
            enable_sasrec=False,
            candidates_per_source=2,
            max_total_candidates=2,
            max_live_content_candidates=1,
            max_random_candidates=1,
            max_pool_trending_candidates=0,
        ),
    )
    executor_cf = ExecutorCFEngine()
    engine.cf_engine = executor_cf

    candidates, _ = await engine.generate_candidates(
        "u1",
        content_features=ContentFeatures(content_id="c1", visual_embedding=[1.0, 0.0]),
        include_profile=True,
        user_interactions=[],
    )

    assert [candidate.product_id for candidate in candidates] == ["content", "random"]
    assert executor_cf.calls == [
        "search_similar_products_sync",
        "get_random_products_sync",
    ]


@pytest.mark.asyncio
async def test_speech_categories_feed_category_pools_after_explicit_context():
    feature_store = RecordingCategoryFeatureStore()
    engine = RecommendationEngine(
        feature_store,
        FakeVectorSearch(),
        RecommendationConfig(
            enable_sasrec=False,
            speech_category_candidates_enabled=True,
            preferred_category_pool_count=3,
            candidates_per_source=2,
            max_total_candidates=10,
        ),
    )
    features = ContentFeatures(
        content_id="spoken",
        visual_embedding=[],
        category_scores={"electronics": 0.8},
        audio_features={
            "has_audio": True,
            "audio_transcript": "手機 headphones",
            "transcription_status": "completed",
            "speech_categories": ["electronics"],
        },
    )

    _, profile = await engine.generate_candidates(
        "u1",
        content_features=features,
        context={"category": "request-category"},
        user_features=UserFeatures(user_id="u1", preferred_categories=["user-category"]),
        user_interactions=[],
        include_profile=True,
    )

    assert feature_store.category_calls == [
        "request-category",
        "electronics",
        "user-category",
    ]
    assert profile["preferred_categories"] == feature_store.category_calls
    assert profile["has_transcript"] is True
    assert profile["speech_category_candidates_used"] is True


@pytest.mark.asyncio
async def test_speech_categories_do_not_change_candidates_when_flag_disabled():
    feature_store = RecordingCategoryFeatureStore()
    engine = RecommendationEngine(
        feature_store,
        FakeVectorSearch(),
        RecommendationConfig(
            enable_sasrec=False,
            speech_category_candidates_enabled=False,
            candidates_per_source=2,
            max_total_candidates=10,
        ),
    )

    _, profile = await engine.generate_candidates(
        "u1",
        content_features=ContentFeatures(
            content_id="spoken",
            visual_embedding=[],
            category_scores={"electronics": 0.8},
            audio_features={
                "audio_transcript": "手機",
                "transcription_status": "completed",
                "speech_categories": ["electronics"],
            },
        ),
        user_features=UserFeatures(user_id="u1"),
        user_interactions=[],
        include_profile=True,
    )

    assert feature_store.category_calls == []
    assert profile["has_transcript"] is True
    assert profile["speech_category_candidates_used"] is False


@pytest.mark.asyncio
async def test_failed_asr_does_not_enable_content_categories_when_speech_flag_enabled():
    feature_store = RecordingCategoryFeatureStore()
    engine = RecommendationEngine(
        feature_store,
        FakeVectorSearch(),
        RecommendationConfig(
            enable_sasrec=False,
            speech_category_candidates_enabled=True,
            preferred_category_pool_count=3,
            candidates_per_source=2,
            max_total_candidates=10,
        ),
    )

    _, profile = await engine.generate_candidates(
        "u1",
        content_features=ContentFeatures(
            content_id="failed-speech",
            visual_embedding=[],
            category_scores={"beauty": 0.9, "electronics": 0.8},
            audio_features={
                "transcription_status": "degraded",
                "speech_categories": [],
            },
        ),
        user_features=UserFeatures(user_id="u1", preferred_categories=["user-category"]),
        user_interactions=[],
        include_profile=True,
    )

    assert feature_store.category_calls == ["user-category"]
    assert profile["speech_category_candidates_used"] is False


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
async def test_generate_candidates_passes_chronological_positive_sequence_to_sasrec():
    interactions = [
        _event("p3", action="purchase", timestamp=3.0),
        _event("noisy", action="remove_from_cart", timestamp=2.5),
        _event("p2", action="click", timestamp=2.0),
        _event("p1", action="view", timestamp=1.0),
    ]
    sasrec = FakeSASRec()
    engine = RecommendationEngine(
        FakeFeatureStore(interactions=interactions),
        FakeVectorSearch(),
        RecommendationConfig(
            enable_sasrec=True,
            candidates_per_source=2,
            max_total_candidates=10,
            sasrec_max_sequence_length=10,
        ),
    )
    engine.sasrec_engine = sasrec

    _, profile = await engine.generate_candidates("u1", include_profile=True)

    assert [event["product_id"] for event in sasrec.calls[0]["sequence"]] == ["p1", "p2", "p3"]
    assert "noisy" not in [event["product_id"] for event in sasrec.calls[0]["sequence"]]
    assert "noisy" in sasrec.calls[0]["exclude_items"]
    assert profile["sasrec_sequence_length"] == 3


class FakeCFEngine:
    is_trained = True

    async def get_user_recommendations(self, user_id, k, exclude_items, user_features=None):
        return [
            CandidateProduct(
                product_id="seq",
                collaborative_score=0.2,
                combined_score=0.2,
                source="cf",
            )
        ][:k]


@pytest.mark.asyncio
async def test_generate_candidates_profiles_sasrec_overlap_after_merge():
    sasrec = FakeSASRec()
    engine = RecommendationEngine(
        FakeFeatureStore(interactions=[_event("seen")]),
        FakeVectorSearch(),
        RecommendationConfig(enable_sasrec=True, candidates_per_source=2, max_total_candidates=10),
    )
    engine.cf_engine = FakeCFEngine()
    engine.sasrec_engine = sasrec

    candidates, profile = await engine.generate_candidates("u1", include_profile=True)

    seq_candidate = next(candidate for candidate in candidates if candidate.product_id == "seq")
    assert seq_candidate.source == "cf+sasrec"
    assert profile["source_counts"]["sasrec"] == 1
    assert profile["source_overlap_counts"]["sasrec"] == 1
    assert profile["merged_source_counts"]["cf"] == 1
    assert profile["merged_source_counts"]["sasrec"] == 1


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


@pytest.mark.asyncio
async def test_generate_candidates_includes_explicit_new_item_source():
    engine = RecommendationEngine(
        FakeFeatureStore(interactions=[]),
        FakeNewItemVectorSearch(),
        RecommendationConfig(
            candidates_per_source=2,
            max_total_candidates=10,
            max_new_item_candidates=1,
        ),
    )
    engine.cf_engine.refresh_new_item_candidates()

    candidates, profile = await engine.generate_candidates("u1", include_profile=True)

    assert "new_item" in {candidate.source for candidate in candidates}
    assert profile["source_counts"]["new_item"] == 1


@pytest.mark.asyncio
async def test_generate_candidates_uses_precomputed_new_item_pool_without_catalog_scan():
    vector_search = FakeNewItemVectorSearch()
    engine = RecommendationEngine(
        FakeFeatureStore(interactions=[]),
        vector_search,
        RecommendationConfig(
            candidates_per_source=2,
            max_total_candidates=10,
            max_new_item_candidates=1,
        ),
    )
    engine.cf_engine.refresh_new_item_candidates()
    assert vector_search.embedding_lookup_count == 1
    vector_search.embedding_lookup_count = 0

    candidates, profile = await engine.generate_candidates("u1", include_profile=True)

    assert "new_item" in {candidate.source for candidate in candidates}
    assert profile["source_counts"]["new_item"] == 1
    assert vector_search.embedding_lookup_count == 0


@pytest.mark.asyncio
async def test_generate_candidates_filters_expired_new_item_snapshot_without_catalog_scan():
    vector_search = FakeNewItemVectorSearch()
    engine = RecommendationEngine(
        FakeFeatureStore(interactions=[]),
        vector_search,
        RecommendationConfig(
            candidates_per_source=2,
            max_total_candidates=10,
            max_new_item_candidates=1,
            cf_cold_start_max_age_days=30,
        ),
    )
    engine.cf_engine.refresh_new_item_candidates()
    vector_search.product_metadata["fresh"]["created_at"] = time.time() - 60 * 86400
    vector_search.embedding_lookup_count = 0

    candidates, profile = await engine.generate_candidates("u1", include_profile=True)

    assert "new_item" not in {candidate.source for candidate in candidates}
    assert profile["source_counts"]["new_item"] == 0
    assert vector_search.embedding_lookup_count == 0


def test_new_item_pool_refreshes_on_ttl_or_catalog_change():
    vector_search = FakeNewItemVectorSearch()
    engine = RecommendationEngine(
        FakeFeatureStore(interactions=[]),
        vector_search,
        RecommendationConfig(
            max_new_item_candidates=1,
            new_item_pool_refresh_interval_seconds=10,
        ),
    )
    engine.cf_engine.refresh_new_item_candidates()

    assert not engine.cf_engine.should_refresh_new_item_candidates(
        engine.cf_engine.new_item_pool_refreshed_at + 1
    )

    vector_search.catalog_version += 1
    assert engine.cf_engine.should_refresh_new_item_candidates(
        engine.cf_engine.new_item_pool_refreshed_at + 1
    )

    engine.cf_engine.refresh_new_item_candidates()
    assert engine.cf_engine.should_refresh_new_item_candidates(
        engine.cf_engine.new_item_pool_refreshed_at + 11
    )
