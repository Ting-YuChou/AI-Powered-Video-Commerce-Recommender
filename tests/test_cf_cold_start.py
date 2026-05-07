import time

import faiss
import numpy as np
import pytest

from cf_cold_start import (
    ContentToCFAdapter,
    build_content_feature,
    build_hybrid_synthetic_embedding,
    is_cold_start_eligible,
    load_item_embedding_sidecar,
    metadata_affinity,
    save_item_embedding_sidecar,
)
from config import RecommendationConfig, VectorConfig
from recommender import RecommendationEngine, TwoTowerRetrievalEngine
from vector_search import VectorSearchEngine


def test_content_to_cf_adapter_predicts_normalized_cf_embedding():
    features = np.eye(3, dtype=np.float32)
    targets = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    adapter = ContentToCFAdapter.fit(features, targets, ridge_alpha=1e-3)
    predictions = adapter.predict(features)

    assert predictions.shape == (3, 2)
    np.testing.assert_allclose(np.linalg.norm(predictions, axis=1), np.ones(3), rtol=1e-5)


def test_content_to_cf_adapter_round_trips_metadata(tmp_path):
    path = tmp_path / "adapter.npz"
    adapter = ContentToCFAdapter.fit(
        np.eye(3, dtype=np.float32),
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        ridge_alpha=1e-3,
    )

    adapter.save(str(path), metadata={"two_tower_model_version": "two-tower-test"})
    loaded = ContentToCFAdapter.load(str(path))

    assert loaded is not None
    assert loaded.metadata["two_tower_model_version"] == "two-tower-test"


def test_hybrid_synthetic_embedding_blends_neighbor_and_adapter_priors():
    synthetic, confidence, weights = build_hybrid_synthetic_embedding(
        neighbor_embeddings=[
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ],
        neighbor_similarities=[0.9, 0.4],
        adapter_embedding=np.array([0.0, 1.0], dtype=np.float32),
        query_metadata={"category": "shoes", "brand": "a", "price": 100.0},
        neighbor_metadatas=[
            {"category": "shoes", "brand": "a", "price": 100.0},
            {"category": "bags", "brand": "b", "price": 500.0},
        ],
        neighbor_weight=0.65,
        softmax_temperature=0.07,
        configured_neighbor_count=2,
    )

    assert synthetic.shape == (2,)
    np.testing.assert_allclose(np.linalg.norm(synthetic), 1.0, rtol=1e-5)
    assert weights[0] > weights[1]
    assert 0.0 < confidence <= 1.0


def test_metadata_affinity_boosts_matching_category_and_penalizes_price_mismatch():
    matching = metadata_affinity(
        {"category": "shoes", "brand": "a", "price": 100.0},
        {"category": "shoes", "brand": "a", "price": 110.0},
    )
    mismatched = metadata_affinity(
        {"category": "shoes", "brand": "a", "price": 100.0},
        {"category": "bags", "brand": "b", "price": 500.0},
    )

    assert matching > 1.0
    assert mismatched < matching


def test_cold_start_eligibility_filters_trained_inactive_and_missing_clip_products():
    now = time.time()
    metadata = {"active": True, "in_stock": True, "created_at": now}
    clip = np.array([1.0, 0.0], dtype=np.float32)

    assert is_cold_start_eligible(
        product_id="new",
        metadata=metadata,
        clip_embedding=clip,
        trained_item_ids={"old"},
        current_time=now,
        max_age_days=30,
        max_interactions=5,
    )
    assert not is_cold_start_eligible(
        product_id="old",
        metadata=metadata,
        clip_embedding=clip,
        trained_item_ids={"old"},
        current_time=now,
        max_age_days=30,
        max_interactions=5,
    )
    assert not is_cold_start_eligible(
        product_id="new",
        metadata={"active": False, "in_stock": True, "created_at": now},
        clip_embedding=clip,
        trained_item_ids={"old"},
        current_time=now,
        max_age_days=30,
        max_interactions=5,
    )
    assert not is_cold_start_eligible(
        product_id="new",
        metadata=metadata,
        clip_embedding=None,
        trained_item_ids={"old"},
        current_time=now,
        max_age_days=30,
        max_interactions=5,
    )


def test_cold_start_eligibility_requires_recency_and_low_interactions():
    now = time.time()
    clip = np.array([1.0, 0.0], dtype=np.float32)

    assert not is_cold_start_eligible(
        product_id="old_zero_interactions",
        metadata={
            "active": True,
            "in_stock": True,
            "created_at": now - 60 * 86400,
            "interaction_count": 0,
        },
        clip_embedding=clip,
        trained_item_ids=set(),
        current_time=now,
        max_age_days=30,
        max_interactions=5,
    )
    assert not is_cold_start_eligible(
        product_id="fresh_many_interactions",
        metadata={
            "active": True,
            "in_stock": True,
            "created_at": now,
            "interaction_count": 10,
        },
        clip_embedding=clip,
        trained_item_ids=set(),
        current_time=now,
        max_age_days=30,
        max_interactions=5,
    )


def test_item_embedding_sidecar_round_trips_metadata(tmp_path):
    path = tmp_path / "items.npz"
    save_item_embedding_sidecar(
        str(path),
        embedding_map={"p1": np.array([1.0, 0.0]), "p2": np.array([0.0, 2.0])},
        clip_available={"p1": True, "p2": False},
        item_features={
            "p1": np.ones(8, dtype=np.float32),
            "p2": np.zeros(8, dtype=np.float32),
        },
        model_version="two-tower-test",
    )

    embeddings, clip_available, item_features, version = load_item_embedding_sidecar(str(path))

    assert version == "two-tower-test"
    assert clip_available == {"p1": True, "p2": False}
    np.testing.assert_allclose(embeddings["p2"], np.array([0.0, 1.0], dtype=np.float32))
    assert item_features["p1"].shape == (8,)


def test_two_tower_retrieval_returns_confidence_scaled_synthetic_candidate():
    vector_search = VectorSearchEngine(VectorConfig(embedding_dim=2))
    engine = TwoTowerRetrievalEngine(
        RecommendationConfig(tt_embedding_dim=2, enable_cf_cold_start_bootstrap=True),
        vector_search,
    )
    engine.is_trained = True
    engine.model_version = "two-tower-test"
    engine.item_mapping = {"old": 1}
    engine.trained_item_embeddings = {"old": np.array([0.0, 1.0], dtype=np.float32)}
    engine.synthetic_item_embeddings = {"new": np.array([1.0, 0.0], dtype=np.float32)}
    engine.synthetic_item_metadata = {"new": {"confidence": 0.5}}
    engine._rebuild_serving_cf_index()
    engine._get_user_embedding = lambda user_id, user_features, current_time: np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    candidates = engine._get_user_recommendations_sync("u1", k=2)
    synthetic = next(candidate for candidate in candidates if candidate.product_id == "new")

    assert synthetic.source == "cf_cold_start"
    assert 0.0 < synthetic.collaborative_score <= 0.5


def test_two_tower_load_sidecars_rejects_mismatched_embedding_version(tmp_path):
    vector_search = VectorSearchEngine(VectorConfig(embedding_dim=2))
    engine = TwoTowerRetrievalEngine(
        RecommendationConfig(
            tt_embedding_dim=2,
            cf_index_path=str(tmp_path / "cf.faiss"),
            enable_cf_cold_start_bootstrap=True,
        ),
        vector_search,
    )
    save_item_embedding_sidecar(
        engine._embedding_sidecar_path(),
        embedding_map={"old": np.array([1.0, 0.0], dtype=np.float32)},
        clip_available={"old": True},
        item_features={"old": np.zeros(8, dtype=np.float32)},
        model_version="two-tower-old",
    )
    adapter = ContentToCFAdapter.fit(
        np.eye(2, dtype=np.float32),
        np.eye(2, dtype=np.float32),
    )
    adapter.save(
        engine._adapter_path(),
        metadata={"two_tower_model_version": "two-tower-old"},
    )
    engine.model_version = "two-tower-new"

    engine._load_cold_start_sidecars()

    assert engine.trained_item_embeddings == {}
    assert engine.content_to_cf_adapter is None


def test_two_tower_load_sidecars_rejects_mismatched_adapter_version(tmp_path):
    vector_search = VectorSearchEngine(VectorConfig(embedding_dim=2))
    engine = TwoTowerRetrievalEngine(
        RecommendationConfig(
            tt_embedding_dim=2,
            cf_index_path=str(tmp_path / "cf.faiss"),
            enable_cf_cold_start_bootstrap=True,
        ),
        vector_search,
    )
    save_item_embedding_sidecar(
        engine._embedding_sidecar_path(),
        embedding_map={"old": np.array([1.0, 0.0], dtype=np.float32)},
        clip_available={"old": True},
        item_features={"old": np.zeros(8, dtype=np.float32)},
        model_version="two-tower-new",
    )
    adapter = ContentToCFAdapter.fit(
        np.eye(2, dtype=np.float32),
        np.eye(2, dtype=np.float32),
    )
    adapter.save(
        engine._adapter_path(),
        metadata={"two_tower_model_version": "two-tower-old"},
    )
    engine.model_version = "two-tower-new"

    engine._load_cold_start_sidecars()

    assert engine.trained_item_embeddings == {}
    assert engine.content_to_cf_adapter is None


def test_failed_adapter_fit_removes_stale_adapter_before_persist(tmp_path):
    vector_search = VectorSearchEngine(VectorConfig(embedding_dim=2))
    engine = TwoTowerRetrievalEngine(
        RecommendationConfig(
            tt_embedding_dim=2,
            cf_index_path=str(tmp_path / "cf.faiss"),
            enable_cf_cold_start_bootstrap=True,
        ),
        vector_search,
    )
    stale_adapter = ContentToCFAdapter.fit(
        np.eye(2, dtype=np.float32),
        np.eye(2, dtype=np.float32),
    )
    stale_adapter.save(
        engine._adapter_path(),
        metadata={"two_tower_model_version": "two-tower-old"},
    )
    engine.content_to_cf_adapter = stale_adapter
    engine.trained_item_embeddings = {"old": np.array([1.0, 0.0], dtype=np.float32)}
    engine.trained_item_clip_available = {"old": False}
    engine.trained_item_features = {"old": np.zeros(8, dtype=np.float32)}

    engine._save_cold_start_sidecars(
        "two-tower-new",
        product_clip_embeddings={},
        product_metadata={"old": {}},
    )

    assert not (tmp_path / "cf.cf_adapter.npz").exists()
    assert (tmp_path / "cf.cf_embeddings.npz").exists()
    assert engine.content_to_cf_adapter is None


def test_trained_item_replaces_stale_synthetic_metadata_in_retrieval():
    vector_search = VectorSearchEngine(VectorConfig(embedding_dim=2))
    engine = TwoTowerRetrievalEngine(
        RecommendationConfig(tt_embedding_dim=2, enable_cf_cold_start_bootstrap=True),
        vector_search,
    )
    engine.is_trained = True
    engine.model_version = "two-tower-test"
    engine.item_mapping = {"new": 1}
    engine.trained_item_embeddings = {"new": np.array([1.0, 0.0], dtype=np.float32)}
    engine.synthetic_item_embeddings = {"new": np.array([1.0, 0.0], dtype=np.float32)}
    engine.synthetic_item_metadata = {"new": {"confidence": 0.1}}
    engine._rebuild_serving_cf_index()
    engine._get_user_embedding = lambda user_id, user_features, current_time: np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    candidates = engine._get_user_recommendations_sync("u1", k=1)

    assert candidates[0].product_id == "new"
    assert candidates[0].source == "collaborative_filtering"
    assert candidates[0].collaborative_score > 0.0


def test_two_tower_l2_search_scores_nearest_items_highest():
    vector_search = VectorSearchEngine(VectorConfig(embedding_dim=2))
    engine = TwoTowerRetrievalEngine(
        RecommendationConfig(tt_embedding_dim=2),
        vector_search,
    )
    engine.is_trained = True
    engine.model_version = "two-tower-test"
    engine.item_mapping = {"near": 1, "far": 2}
    engine.cf_index = faiss.IndexFlatL2(2)
    engine.cf_index.add(
        np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    engine.cf_index_map = {0: "near", 1: "far"}
    engine._get_user_embedding = lambda user_id, user_features, current_time: np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    candidates = engine._get_user_recommendations_sync("u1", k=2)
    scores = {candidate.product_id: candidate.collaborative_score for candidate in candidates}

    assert scores["near"] > scores["far"]
    np.testing.assert_allclose(scores["near"], 1.0, rtol=1e-6)
    np.testing.assert_allclose(scores["far"], 0.0, rtol=1e-6)


@pytest.mark.asyncio
async def test_failed_checkpoint_load_does_not_swap_live_cf_state(tmp_path):
    vector_search = VectorSearchEngine(VectorConfig(embedding_dim=2))
    config = RecommendationConfig(
        tt_embedding_dim=2,
        cf_index_path=str(tmp_path / "cf.faiss"),
        enable_cf_cold_start_bootstrap=True,
    )
    engine = RecommendationEngine(object(), vector_search, config)
    old_cf_engine = engine.cf_engine
    old_cf_engine.is_trained = True
    old_cf_engine.model_version = "two-tower-old"
    old_cf_engine.cf_index_map = {0: "old"}

    replacement_index = vector_search.create_cf_index(2)
    replacement_index.add(np.array([[1.0, 0.0]], dtype=np.float32))
    VectorSearchEngine.save_cf_index(
        replacement_index,
        config.cf_index_path,
        {"index_map": {"0": "new"}},
    )

    loaded = await engine._try_load_cf_index()

    assert loaded is False
    assert engine.cf_engine is old_cf_engine
    assert engine.cf_engine.model_version == "two-tower-old"
    assert engine.cf_engine.cf_index_map == {0: "old"}


@pytest.mark.asyncio
async def test_vector_search_save_load_preserves_real_product_embeddings(tmp_path):
    config = VectorConfig(
        embedding_dim=2,
        index_path=str(tmp_path / "catalog.faiss"),
    )
    engine = VectorSearchEngine(config)
    await engine._create_empty_index()
    await engine.add_product_embedding(
        "p1",
        np.array([3.0, 0.0], dtype=np.float32),
        {"active": True, "in_stock": True, "created_at": time.time()},
    )
    await engine.add_product_embedding(
        "p2",
        np.array([0.0, 4.0], dtype=np.float32),
        {"active": True, "in_stock": True, "created_at": time.time()},
    )

    await engine.save_index()
    loaded = VectorSearchEngine(config)
    await loaded.load_index()

    assert (tmp_path / "catalog.embeddings.npz").exists()
    np.testing.assert_allclose(
        loaded.get_product_embedding("p1"),
        np.array([1.0, 0.0], dtype=np.float32),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        loaded.get_product_embedding("p2"),
        np.array([0.0, 1.0], dtype=np.float32),
        rtol=1e-6,
    )
