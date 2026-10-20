import json

import faiss
import numpy as np
import pytest

from video_commerce.common.config import VectorConfig
from video_commerce.ml.vector_search import (
    CONTENT_INDEX_METRIC,
    VectorIndexMetricMismatch,
    VectorSearchEngine,
)


async def _engine_with_products(tmp_path, **config_overrides):
    config = VectorConfig(
        embedding_dim=3,
        index_path=str(tmp_path / "catalog.faiss"),
        **config_overrides,
    )
    engine = VectorSearchEngine(config)
    await engine._create_empty_index()
    for product_id, vector in {
        "far": [-1.0, 0.0, 0.0],
        "near": [1.0, 0.05, 0.0],
        "orthogonal": [0.0, 0.0, 1.0],
    }.items():
        await engine.add_product_embedding(
            product_id, np.asarray(vector, dtype=np.float32), {"category": "home"}
        )
    return engine


@pytest.mark.asyncio
async def test_content_index_ranks_nearest_product_first_above_threshold(tmp_path):
    engine = await _engine_with_products(tmp_path)

    assert engine.index.metric_type == faiss.METRIC_INNER_PRODUCT
    candidates = engine.search_similar_products_sync(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32), k=3
    )

    assert [candidate.product_id for candidate in candidates] == ["near"]
    assert candidates[0].content_similarity_score == pytest.approx(
        1.0 / np.sqrt(1.0 + 0.05**2), rel=1e-5
    )
    assert (
        candidates[0].content_similarity_score > engine.config.similarity_threshold
    )


@pytest.mark.asyncio
async def test_content_index_manifest_records_metric_and_reloads(tmp_path):
    engine = await _engine_with_products(tmp_path)
    await engine.save_index()

    index_path, _ = engine._resolve_active_artifact_paths(tmp_path / "catalog.faiss")
    manifest = json.loads(engine._manifest_path(index_path).read_text())
    loaded = VectorSearchEngine(engine.config)
    await loaded.load_index()

    assert manifest["metric"] == CONTENT_INDEX_METRIC
    assert loaded.index.metric_type == faiss.METRIC_INNER_PRODUCT
    assert len(loaded.product_index_map) == 3


def _strip_manifest_metric(engine, tmp_path):
    index_path, _ = engine._resolve_active_artifact_paths(tmp_path / "catalog.faiss")
    manifest_path = engine._manifest_path(index_path)
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("metric")
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.asyncio
async def test_metricless_generation_is_rebuilt_in_empty_bootstrap_mode(tmp_path):
    engine = await _engine_with_products(tmp_path, bootstrap_mode="empty")
    await engine.save_index()
    _strip_manifest_metric(engine, tmp_path)

    reloaded = VectorSearchEngine(engine.config)
    await reloaded.load_index()

    assert reloaded.is_loaded
    assert reloaded.index.ntotal == 0
    assert reloaded.index.metric_type == faiss.METRIC_INNER_PRODUCT
    # The normal ingest path repopulates and publishes a current generation.
    await reloaded.add_product_embedding(
        "near", np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    )
    await reloaded.save_index()
    fresh = VectorSearchEngine(engine.config)
    await fresh.load_index()
    assert list(fresh.product_index_map.values()) == ["near"]


@pytest.mark.asyncio
async def test_metricless_generation_is_rejected_in_required_mode(tmp_path):
    engine = await _engine_with_products(tmp_path, bootstrap_mode="required")
    await engine.save_index()
    _strip_manifest_metric(engine, tmp_path)

    with pytest.raises(VectorIndexMetricMismatch):
        await VectorSearchEngine(engine.config).load_index()


@pytest.mark.asyncio
async def test_legacy_l2_index_without_manifest_is_not_served(tmp_path):
    config = VectorConfig(
        embedding_dim=3,
        index_path=str(tmp_path / "catalog.faiss"),
        bootstrap_mode="empty",
    )
    legacy = faiss.IndexHNSWFlat(3, 32)
    legacy.add(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
    faiss.write_index(legacy, str(tmp_path / "catalog.faiss"))
    (tmp_path / "catalog.metadata.json").write_text(
        json.dumps({"index_map": {"0": "p1"}, "product_metadata": {}})
    )

    engine = VectorSearchEngine(config)
    await engine.load_index()

    assert engine.index.metric_type == faiss.METRIC_INNER_PRODUCT
    assert engine.index.ntotal == 0


def test_cf_index_uses_inner_product():
    engine = VectorSearchEngine(VectorConfig(embedding_dim=2))

    assert engine.create_cf_index(2).metric_type == faiss.METRIC_INNER_PRODUCT
