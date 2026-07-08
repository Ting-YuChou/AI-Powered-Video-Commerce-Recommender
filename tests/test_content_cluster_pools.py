from types import SimpleNamespace

import numpy as np
import pytest

from video_commerce.common.config import CacheConfig, RecommendationConfig, VectorConfig
from video_commerce.common.models import CandidateProduct, ContentFeatures, UserFeatures
from video_commerce.data_plane.feature_store import FeatureStore
from video_commerce.ml.content_clusters import (
    ContentClusterArtifact,
    build_content_cluster_artifact,
    load_content_cluster_artifact,
    save_content_cluster_artifact,
)
from video_commerce.ml.recommender import RecommendationEngine
from video_commerce.ml.vector_search import VectorSearchEngine
from video_commerce.services.recommendation import api as recommendation_api_module


class RecordingPoolStore:
    def __init__(self):
        self.cache_config = CacheConfig(hot_path_read_timeout_ms=50)
        self.trending_pool = []
        self.category_pools = {}
        self.cluster_pools = {}
        self.cluster_calls = []
        self.cluster_pool_versions = []

    async def get_user_interactions(self, user_id, limit=100):
        return []

    async def get_user_features(self, user_id, cache_default=False):
        return UserFeatures(user_id=user_id)

    async def get_trending_pool(self, limit, pool_name="global", exclude_items=None):
        return []

    async def get_category_pool(self, category, limit, exclude_items=None):
        return []

    async def get_cluster_pool(
        self,
        cluster_id,
        limit,
        exclude_items=None,
        pool_version=None,
    ):
        self.cluster_calls.append((cluster_id, limit, pool_version))
        exclude_items = exclude_items or set()
        candidates = self.cluster_pools.get(cluster_id) or self.cluster_pools.get(
            str(cluster_id),
            [],
        )
        return [
            CandidateProduct(**candidate.dict())
            for candidate in candidates
            if candidate.product_id not in exclude_items
        ][:limit]

    async def store_trending_pool(self, candidates, pool_name="global"):
        self.trending_pool = list(candidates)

    async def store_category_pools(self, pools):
        self.category_pools = dict(pools)

    async def store_cluster_pools(self, pools, pool_version=None):
        self.cluster_pools = dict(pools)
        self.cluster_pool_versions.append(pool_version)


class NoopCFEngine:
    is_trained = False

    def refresh_new_item_candidates(self):
        return None

    def get_new_item_candidates(self, k, exclude_items=None):
        return []


class ClusterVectorSearch:
    embedding_dim = 2

    def __init__(self):
        self.product_metadata = {
            "p1": {"category": "cat", "rating": 5.0, "content_cluster_id": 7},
            "p2": {"category": "cat", "rating": 4.0, "content_cluster_id": 8},
        }
        self.content_cluster_model_version = "cluster-test"

    def get_product_cluster_id(self, product_id):
        metadata = self.product_metadata.get(product_id) or {}
        return metadata.get("content_cluster_id")

    def assign_content_embedding_clusters(self, embedding, *, limit=1):
        return [7][:limit]

    async def search_similar_products(self, embedding, k=10):
        return []

    async def get_random_products(self, k=10):
        return []

    def get_catalog_version_context(self):
        return {"catalog_version": 1, "last_updated": 1, "product_count": 2}


def test_content_cluster_artifact_build_save_load_and_assign(tmp_path):
    embeddings = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.9, 0.1], dtype=np.float32),
        "c": np.array([0.0, 1.0], dtype=np.float32),
        "d": np.array([0.1, 0.9], dtype=np.float32),
    }

    artifact = build_content_cluster_artifact(
        embeddings,
        num_clusters=2,
        source_catalog_context={"catalog_version": 3},
    )
    metadata_path = tmp_path / "clusters.json"
    centroids_path = tmp_path / "clusters.npz"
    save_content_cluster_artifact(
        artifact,
        metadata_path=str(metadata_path),
        centroids_path=str(centroids_path),
    )

    loaded = load_content_cluster_artifact(
        metadata_path=str(metadata_path),
        centroids_path=str(centroids_path),
    )

    assert loaded.cluster_model_version == artifact.cluster_model_version
    assert loaded.num_clusters == 2
    assert loaded.product_count == 4
    assert loaded.product_cluster_map == artifact.product_cluster_map


def test_content_cluster_artifact_rejects_mismatched_centroids(tmp_path):
    artifact = build_content_cluster_artifact(
        {
            "a": np.array([1.0, 0.0], dtype=np.float32),
            "b": np.array([0.0, 1.0], dtype=np.float32),
        },
        num_clusters=2,
    )
    metadata_path = tmp_path / "clusters.json"
    centroids_path = tmp_path / "clusters.npz"
    save_content_cluster_artifact(
        artifact,
        metadata_path=str(metadata_path),
        centroids_path=str(centroids_path),
    )
    centroids_path.write_bytes(b"not-the-published-centroids")

    with pytest.raises(ValueError, match="checksum"):
        load_content_cluster_artifact(
            metadata_path=str(metadata_path),
            centroids_path=str(centroids_path),
        )


def test_vector_search_loads_cluster_artifact_and_assigns_request_embedding(tmp_path):
    artifact = build_content_cluster_artifact(
        {
            "a": np.array([1.0, 0.0], dtype=np.float32),
            "b": np.array([0.0, 1.0], dtype=np.float32),
        },
        num_clusters=2,
    )
    metadata_path = tmp_path / "clusters.json"
    centroids_path = tmp_path / "clusters.npz"
    save_content_cluster_artifact(
        artifact,
        metadata_path=str(metadata_path),
        centroids_path=str(centroids_path),
    )
    vector_search = VectorSearchEngine(
        VectorConfig(index_path=str(tmp_path / "vector.faiss"), embedding_dim=2)
    )
    vector_search.product_metadata = {"a": {}, "b": {}}

    assert vector_search.load_content_cluster_artifact(
        metadata_path=str(metadata_path),
        centroids_path=str(centroids_path),
    )

    cluster_id = vector_search.get_product_cluster_id("a")
    assert cluster_id is not None
    assert vector_search.product_metadata["a"]["content_cluster_id"] == cluster_id
    assert vector_search.assign_content_embedding_clusters([1.0, 0.0], limit=1) == [
        cluster_id
    ]


def test_vector_search_replaces_cluster_artifact_without_metadata_fallback(tmp_path):
    vector_search = VectorSearchEngine(
        VectorConfig(index_path=str(tmp_path / "vector.faiss"), embedding_dim=2)
    )
    vector_search.product_metadata = {
        "p1": {},
        "p2": {},
        "p3": {"content_cluster_id": 99},
    }

    first_artifact = _manual_cluster_artifact(
        "cluster-a",
        {"p1": 0, "p2": 1},
    )
    second_artifact = _manual_cluster_artifact(
        "cluster-b",
        {"p1": 1},
    )

    vector_search.apply_content_cluster_artifact(first_artifact)
    assert vector_search.get_product_cluster_id("p2") == 1

    vector_search.apply_content_cluster_artifact(second_artifact)

    assert vector_search.get_product_cluster_id("p1") == 1
    assert vector_search.get_product_cluster_id("p2") is None
    assert "content_cluster_id" not in vector_search.product_metadata["p2"]
    assert vector_search.get_product_cluster_id("p3") is None


@pytest.mark.asyncio
async def test_feature_store_cluster_pool_returns_candidate_clones():
    store = FeatureStore.__new__(FeatureStore)
    store._cluster_pool_memory_cache = {
        "7": [
            CandidateProduct(
                product_id="clustered",
                combined_score=0.6,
                source="cluster_pool",
            )
        ]
    }

    first = await FeatureStore.get_cluster_pool(store, 7, 1)
    first[0].combined_score = 99.0
    second = await FeatureStore.get_cluster_pool(store, "7", 1)

    assert second[0].combined_score == 0.6


@pytest.mark.asyncio
async def test_feature_store_cluster_pool_keys_are_versioned():
    store = FeatureStore.__new__(FeatureStore)
    store._cluster_pool_memory_cache = {
        "cluster-v1:7": [
            CandidateProduct(
                product_id="old",
                combined_score=0.4,
                source="cluster_pool",
            )
        ],
        "cluster-v2:7": [
            CandidateProduct(
                product_id="new",
                combined_score=0.6,
                source="cluster_pool",
            )
        ],
    }

    old_pool = await FeatureStore.get_cluster_pool(
        store,
        7,
        1,
        pool_version="cluster-v1",
    )
    new_pool = await FeatureStore.get_cluster_pool(
        store,
        7,
        1,
        pool_version="cluster-v2",
    )

    assert [candidate.product_id for candidate in old_pool] == ["old"]
    assert [candidate.product_id for candidate in new_pool] == ["new"]


@pytest.mark.asyncio
async def test_refresh_serving_pools_creates_cluster_pools_when_enabled():
    feature_store = RecordingPoolStore()
    engine = RecommendationEngine(
        feature_store,
        ClusterVectorSearch(),
        RecommendationConfig(
            enable_content_cluster_pools=True,
            serving_cluster_pool_size=1,
        ),
    )
    engine.cf_engine = NoopCFEngine()

    await engine.refresh_serving_pools()

    assert sorted(feature_store.cluster_pools.keys()) == [7, 8]
    assert feature_store.cluster_pools[7][0].source == "cluster_pool"
    assert feature_store.cluster_pool_versions == ["cluster-test"]


@pytest.mark.asyncio
async def test_generate_candidates_merges_cluster_pool_and_profiles_source():
    feature_store = RecordingPoolStore()
    feature_store.cluster_pools = {
        7: [
            CandidateProduct(
                product_id="cluster-7",
                popularity_score=0.8,
                combined_score=0.8,
                source="cluster_pool",
            )
        ]
    }
    engine = RecommendationEngine(
        feature_store,
        ClusterVectorSearch(),
        RecommendationConfig(
            enable_content_cluster_pools=True,
            candidates_per_source=2,
            max_total_candidates=10,
            max_pool_cluster_candidates=2,
            max_pool_trending_candidates=0,
            max_random_candidates=0,
        ),
    )
    engine.cf_engine = NoopCFEngine()

    candidates, profile = await engine.generate_candidates(
        "u1",
        content_features=ContentFeatures(
            content_id="content",
            visual_embedding=[1.0, 0.0],
        ),
        user_features=UserFeatures(user_id="u1"),
        user_interactions=[],
        include_profile=True,
    )

    assert [candidate.product_id for candidate in candidates] == ["cluster-7"]
    assert profile["preferred_clusters"] == [7]
    assert profile["source_counts"]["cluster_pool"] == 1
    assert profile["cluster_pool_ms"] >= 0.0
    assert feature_store.cluster_calls == [(7, 2, "cluster-test")]


@pytest.mark.asyncio
async def test_local_cluster_artifact_reload_rebuilds_versioned_pools(tmp_path):
    feature_store = RecordingPoolStore()
    vector_search = VectorSearchEngine(
        VectorConfig(index_path=str(tmp_path / "vector.faiss"), embedding_dim=2)
    )
    vector_search.product_metadata = {
        "p1": {"category": "cat", "rating": 5.0},
        "p2": {"category": "cat", "rating": 4.0},
    }
    metadata_path = tmp_path / "clusters.json"
    centroids_path = tmp_path / "clusters.npz"
    config = RecommendationConfig(
        enable_content_cluster_pools=True,
        content_cluster_metadata_path=str(metadata_path),
        content_cluster_centroids_path=str(centroids_path),
        serving_cluster_pool_size=10,
    )
    engine = RecommendationEngine(feature_store, vector_search, config)
    engine.cf_engine = NoopCFEngine()

    save_content_cluster_artifact(
        _manual_cluster_artifact("cluster-a", {"p1": 0, "p2": 1}),
        metadata_path=str(metadata_path),
        centroids_path=str(centroids_path),
    )

    assert await engine.sync_serving_artifacts_if_updated() is True
    assert engine.loaded_content_cluster_version == "cluster-a"
    assert sorted(feature_store.cluster_pools.keys()) == [0, 1]
    assert feature_store.cluster_pool_versions[-1] == "cluster-a"

    save_content_cluster_artifact(
        _manual_cluster_artifact("cluster-b", {"p1": 1, "p2": 1}),
        metadata_path=str(metadata_path),
        centroids_path=str(centroids_path),
    )

    assert await engine.sync_serving_artifacts_if_updated() is True
    assert engine.loaded_content_cluster_version == "cluster-b"
    assert sorted(feature_store.cluster_pools.keys()) == [1]
    assert {
        candidate.product_id for candidate in feature_store.cluster_pools[1]
    } == {"p1", "p2"}
    assert feature_store.cluster_pool_versions[-1] == "cluster-b"


@pytest.mark.asyncio
async def test_missing_cluster_artifact_disables_cluster_candidates_without_failure(tmp_path):
    feature_store = RecordingPoolStore()
    vector_search = ClusterVectorSearch()
    vector_search.assign_content_embedding_clusters = lambda embedding, limit=1: []
    engine = RecommendationEngine(
        feature_store,
        vector_search,
        RecommendationConfig(
            enable_content_cluster_pools=True,
            content_cluster_metadata_path=str(tmp_path / "missing.json"),
            content_cluster_centroids_path=str(tmp_path / "missing.npz"),
            candidates_per_source=2,
            max_total_candidates=10,
            max_pool_trending_candidates=0,
            max_random_candidates=0,
        ),
    )
    engine.cf_engine = NoopCFEngine()

    assert await engine._try_load_content_cluster_artifacts() is False
    candidates, profile = await engine.generate_candidates(
        "u1",
        content_features=ContentFeatures(
            content_id="content",
            visual_embedding=[1.0, 0.0],
        ),
        user_features=UserFeatures(user_id="u1"),
        user_interactions=[],
        include_profile=True,
    )

    assert candidates == []
    assert profile["source_counts"]["cluster_pool"] == 0


def _manual_cluster_artifact(
    version: str,
    assignments: dict[str, int],
) -> ContentClusterArtifact:
    return ContentClusterArtifact(
        cluster_model_version=version,
        num_clusters=2,
        centroids=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        product_cluster_map=dict(assignments),
        product_count=len(assignments),
        embedding_dim=2,
        source_catalog_context={"catalog_version": version},
    )


def test_serving_version_context_includes_content_cluster_version_and_mtimes(
    monkeypatch,
    tmp_path,
):
    metadata_path = tmp_path / "clusters.json"
    centroids_path = tmp_path / "clusters.npz"
    metadata_path.write_text("{}", encoding="utf-8")
    centroids_path.write_bytes(b"centroids")
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            model_config=SimpleNamespace(
                ranking_model_path=str(tmp_path / "ranking.pt")
            ),
            vector_config=SimpleNamespace(index_path=str(tmp_path / "vector.faiss")),
            recommendation_config=RecommendationConfig(
                enable_content_cluster_pools=True,
                content_cluster_metadata_path=str(metadata_path),
                content_cluster_centroids_path=str(centroids_path),
            ),
        )
    )

    monkeypatch.setattr(
        recommendation_api_module,
        "recommendation_engine",
        SimpleNamespace(
            loaded_two_tower_version="two-tower-1",
            loaded_sasrec_version=None,
            loaded_content_cluster_version="cluster-v1",
            cf_engine=SimpleNamespace(
                model_version="two-tower-1",
                cold_start_overlay_version=None,
                new_item_pool_version=None,
            ),
        ),
    )
    monkeypatch.setattr(
        recommendation_api_module,
        "ranking_model",
        SimpleNamespace(model_version="ranking-1"),
    )
    monkeypatch.setattr(
        recommendation_api_module,
        "vector_search",
        SimpleNamespace(
            get_catalog_version_context=lambda: {
                "catalog_version": 1,
                "last_updated": 1,
                "product_count": 1,
            },
            get_content_cluster_version_context=lambda: {
                "cluster_model_version": "cluster-v1",
                "num_clusters": 2,
                "assignment_count": 1,
                "product_count": 1,
                "created_at": 10,
            },
        ),
    )

    first = recommendation_api_module._build_serving_version_context_uncached(runtime)
    monkeypatch.setattr(
        recommendation_api_module,
        "vector_search",
        SimpleNamespace(
            get_catalog_version_context=lambda: {
                "catalog_version": 1,
                "last_updated": 1,
                "product_count": 1,
            },
            get_content_cluster_version_context=lambda: {
                "cluster_model_version": "cluster-v2",
                "num_clusters": 2,
                "assignment_count": 1,
                "product_count": 1,
                "created_at": 20,
            },
        ),
    )
    second = recommendation_api_module._build_serving_version_context_uncached(runtime)

    assert first["content_cluster_model"] == "cluster-v1"
    assert first["content_cluster"]["cluster_model_version"] == "cluster-v1"
    assert first["content_cluster_metadata_mtime"] is not None
    assert first["content_cluster_centroids_mtime"] is not None
    assert second["content_cluster"]["cluster_model_version"] == "cluster-v2"
