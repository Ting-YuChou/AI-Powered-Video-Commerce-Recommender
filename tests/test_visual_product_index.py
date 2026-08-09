import json
import hashlib

import numpy as np
import pytest
from PIL import Image

from video_commerce.ml.visual_product_index import (
    build_visual_product_index_from_catalog,
    load_visual_product_index_bundle,
    publish_visual_product_index_bundle,
)
from video_commerce.common.config import VectorConfig
from video_commerce.ml.vector_search import VectorSearchEngine


def test_visual_product_index_bundle_is_atomic_and_checksum_locked(tmp_path):
    manifest_path = publish_visual_product_index_bundle(
        tmp_path,
        product_embeddings={
            "p1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "p2": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        },
        product_metadata={
            "p1": {"category": "shoes"},
            "p2": {"category": "bags"},
        },
        expected_product_count=2,
        clip_model_id="clip-test",
        clip_revision="revision-1",
        catalog_activation_id="catalog-42",
        model_version="visual-products-42",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "visual_product_index_v1"
    assert payload["coverage"] == pytest.approx(1.0)
    assert payload["clip_revision"] == "revision-1"
    assert payload["catalog_activation_id"] == "catalog-42"

    bundle = load_visual_product_index_bundle(
        manifest_path,
        expected_clip_model_id="clip-test",
        expected_clip_revision="revision-1",
        expected_catalog_activation_id="catalog-42",
    )
    scores, indices = bundle.index.search(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        1,
    )
    assert bundle.product_index_map[int(indices[0, 0])] == "p1"
    assert scores[0, 0] == pytest.approx(1.0)


def test_visual_product_index_bundle_rejects_tampered_artifact(tmp_path):
    manifest_path = publish_visual_product_index_bundle(
        tmp_path,
        product_embeddings={"p1": np.array([1.0, 0.0], dtype=np.float32)},
        product_metadata={"p1": {}},
        expected_product_count=1,
        clip_model_id="clip-test",
        clip_revision="revision-1",
        catalog_activation_id="catalog-42",
        model_version="visual-products-42",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_path = tmp_path / payload["artifacts"]["index"]["path"]
    index_path.write_bytes(index_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        load_visual_product_index_bundle(manifest_path)


def test_visual_product_index_bundle_fails_closed_below_coverage_gate(tmp_path):
    with pytest.raises(ValueError, match="coverage"):
        publish_visual_product_index_bundle(
            tmp_path,
            product_embeddings={"p1": np.array([1.0, 0.0], dtype=np.float32)},
            product_metadata={"p1": {}},
            expected_product_count=2,
            minimum_coverage=0.95,
            clip_model_id="clip-test",
            clip_revision="revision-1",
            catalog_activation_id="catalog-42",
            model_version="visual-products-42",
        )


@pytest.mark.asyncio
async def test_catalog_image_builder_verifies_checksums_and_quarantines_bad_images(
    tmp_path,
):
    good = tmp_path / "good.png"
    bad = tmp_path / "bad.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(good)
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(bad)

    class Storage:
        async def materialize_for_processing(self, uri, suggested_suffix=""):
            return str(tmp_path / uri), False

    def encoder(images):
        return np.asarray(
            [[1.0, float(index)] for index, _ in enumerate(images)],
            dtype=np.float32,
        )

    digest = hashlib.sha256(good.read_bytes()).hexdigest()
    result = await build_visual_product_index_from_catalog(
        [
            {
                "product_id": "p1",
                "image_storage_uri": "good.png",
                "image_sha256": digest,
                "category": "shoes",
            },
            {
                "product_id": "p2",
                "image_storage_uri": "bad.png",
                "image_sha256": "0" * 64,
                "category": "bags",
            },
        ],
        object_storage=Storage(),
        encode_images=encoder,
        output_dir=tmp_path / "index",
        clip_model_id="clip-test",
        clip_revision="revision-1",
        catalog_activation_id="catalog-42",
        model_version="visual-products-42",
        catalog_available_at=0.0,
        minimum_coverage=0.5,
    )

    assert result.indexed_product_ids == ("p1",)
    assert result.quarantine == {
        "p2": "image checksum mismatch",
    }
    bundle = load_visual_product_index_bundle(result.manifest_path)
    assert set(bundle.product_embeddings) == {"p1"}


def test_vector_search_atomically_activates_lineage_checked_product_index(tmp_path):
    manifest_path = publish_visual_product_index_bundle(
        tmp_path,
        product_embeddings={"p1": np.array([1.0, 0.0], dtype=np.float32)},
        product_metadata={"p1": {"category": "shoes"}},
        expected_product_count=1,
        clip_model_id="clip-test",
        clip_revision="revision-1",
        catalog_activation_id="catalog-42",
        model_version="visual-products-42",
    )
    engine = VectorSearchEngine(VectorConfig(embedding_dim=2))

    manifest = engine.activate_visual_product_index(
        manifest_path,
        expected_clip_model_id="clip-test",
        expected_clip_revision="revision-1",
        expected_catalog_activation_id="catalog-42",
    )

    assert engine.visual_product_index_version == "visual-products-42"
    assert engine.product_index_map == {0: "p1"}
    assert manifest["catalog_activation_id"] == "catalog-42"

    with pytest.raises(ValueError, match="clip_revision"):
        engine.activate_visual_product_index(
            manifest_path,
            expected_clip_revision="wrong",
        )
    assert engine.visual_product_index_version == "visual-products-42"
