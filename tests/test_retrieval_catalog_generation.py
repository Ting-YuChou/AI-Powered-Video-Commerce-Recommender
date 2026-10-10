import asyncio
import hashlib
import json

from video_commerce.ml.retrieval_catalog import RetrievalCatalogPublisher


class LocalImmutableStorage:
    def __init__(self, root):
        self.root = root

    async def persist_immutable_bytes(self, payload, *, object_name, content_type=None):
        path = self.root / object_name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != payload:
            raise FileExistsError(object_name)
        path.write_bytes(payload)
        return str(path)


def test_catalog_generation_is_deterministic_and_keeps_missing_modalities(tmp_path):
    publisher = RetrievalCatalogPublisher(LocalImmutableStorage(tmp_path))
    products = [
        {
            "product_id": "p2",
            "active": True,
            "in_stock": True,
            "metadata": {"price": 12.0},
            "modalities": {"visual": False, "ocr": False},
        },
        {
            "product_id": "p1",
            "active": True,
            "in_stock": False,
            "metadata": {"price": 8.0},
            "modalities": {"visual": True, "ocr": True},
        },
    ]
    first = asyncio.run(
        publisher.publish(
            products=products,
            embeddings={"p1": [1.0, 0.0]},
            generation_id="catalog-1",
            effective_at=100.0,
            available_at=110.0,
            embedding_model_revision="clip-v1",
            artifact_prefix="retrieval/catalogs",
        )
    )
    second = asyncio.run(
        publisher.publish(
            products=list(reversed(products)),
            embeddings={"p1": [1.0, 0.0]},
            generation_id="catalog-1",
            effective_at=100.0,
            available_at=110.0,
            embedding_model_revision="clip-v1",
            artifact_prefix="retrieval/catalogs",
        )
    )

    assert first == second
    assert first["catalog_generation"]["product_count"] == 2
    catalog_path = first["catalog_generation"]["uri"]
    sidecar_path = first["item_feature_sidecar"]["uri"]
    catalog = json.loads(open(catalog_path, encoding="utf-8").read())
    sidecar = json.loads(open(sidecar_path, encoding="utf-8").read())
    assert [row["product_id"] for row in catalog["products"]] == ["p1", "p2"]
    assert sidecar["items"][1]["clip_embedding"] is None
    assert (
        first["catalog_generation"]["sha256"]
        == hashlib.sha256(open(catalog_path, "rb").read()).hexdigest()
    )


def test_catalog_generation_rejects_mixed_embedding_dimensions(tmp_path):
    publisher = RetrievalCatalogPublisher(LocalImmutableStorage(tmp_path))
    try:
        asyncio.run(
            publisher.publish(
                products=[
                    {"product_id": "p1", "active": True, "in_stock": True},
                    {"product_id": "p2", "active": True, "in_stock": True},
                ],
                embeddings={"p1": [1.0, 0.0], "p2": [1.0]},
                generation_id="catalog-1",
                effective_at=100.0,
                available_at=110.0,
                embedding_model_revision="clip-v1",
                artifact_prefix="retrieval/catalogs",
            )
        )
    except ValueError as exc:
        assert "dimension" in str(exc)
    else:
        raise AssertionError("mixed embedding dimensions were accepted")
