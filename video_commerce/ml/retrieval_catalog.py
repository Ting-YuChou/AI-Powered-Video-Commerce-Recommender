"""Build immutable catalog generations used by retrieval PIT datasets."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from video_commerce.ml.retrieval_pit_dataset import (
    RETRIEVAL_CATALOG_SCHEMA_VERSION,
    RETRIEVAL_ELIGIBILITY_POLICY_VERSION,
    RETRIEVAL_ITEM_FEATURE_SCHEMA_VERSION,
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


class RetrievalCatalogPublisher:
    """Persist a complete membership manifest and aligned item-feature sidecar."""

    def __init__(self, object_storage: Any) -> None:
        self.object_storage = object_storage

    async def publish(
        self,
        *,
        products: Iterable[Mapping[str, Any]],
        embeddings: Mapping[str, Sequence[float]],
        generation_id: str,
        effective_at: float,
        available_at: float,
        embedding_model_revision: str,
        artifact_prefix: str,
    ) -> dict[str, dict[str, Any]]:
        generation = str(generation_id or "").strip()
        revision = str(embedding_model_revision or "").strip()
        prefix = str(artifact_prefix or "").strip("/")
        if not generation or not revision or not prefix:
            raise ValueError("retrieval catalog identity is incomplete")
        effective = self._finite(effective_at, "effective_at")
        available = self._finite(available_at, "available_at")
        if available < effective:
            raise ValueError("retrieval catalog available_at precedes effective_at")

        normalized_products: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in products:
            product_id = str(raw.get("product_id") or "").strip()
            if not product_id or product_id in seen:
                raise ValueError("retrieval catalog product IDs must be unique")
            seen.add(product_id)
            modalities = dict(
                raw.get("modality_presence") or raw.get("modalities") or {}
            )
            normalized_products.append(
                {
                    "product_id": product_id,
                    "active": bool(raw.get("active", True)),
                    "in_stock": bool(raw.get("in_stock", True)),
                    "metadata": dict(raw.get("metadata") or {}),
                    "modality_presence": {
                        str(key): bool(value)
                        for key, value in sorted(modalities.items())
                    },
                }
            )
        normalized_products.sort(key=lambda row: row["product_id"])
        if not normalized_products:
            raise ValueError("retrieval catalog cannot be empty")
        unknown_embeddings = set(embeddings) - seen
        if unknown_embeddings:
            raise ValueError(
                "retrieval embeddings contain products outside the catalog"
            )

        dimension = 0
        items: list[dict[str, Any]] = []
        for product in normalized_products:
            product_id = product["product_id"]
            raw_embedding = embeddings.get(product_id)
            embedding = None
            if raw_embedding is not None:
                embedding = [
                    self._finite(value, "embedding") for value in raw_embedding
                ]
                if not embedding:
                    raise ValueError("retrieval embedding dimension must be positive")
                if dimension == 0:
                    dimension = len(embedding)
                elif len(embedding) != dimension:
                    raise ValueError("retrieval embedding dimension mismatch")
            items.append({"product_id": product_id, "clip_embedding": embedding})
        if dimension <= 0:
            raise ValueError("retrieval catalog requires at least one item embedding")

        catalog = {
            "schema_version": RETRIEVAL_CATALOG_SCHEMA_VERSION,
            "generation_id": generation,
            "effective_at": effective,
            "available_at": available,
            "eligibility_policy_version": RETRIEVAL_ELIGIBILITY_POLICY_VERSION,
            "product_count": len(normalized_products),
            "products": normalized_products,
        }
        sidecar = {
            "schema_version": RETRIEVAL_ITEM_FEATURE_SCHEMA_VERSION,
            "generation_id": generation,
            "available_at": available,
            "embedding_dimension": dimension,
            "embedding_model_revision": revision,
            "item_count": len(items),
            "items": items,
        }
        catalog_bytes = _canonical_json(catalog)
        sidecar_bytes = _canonical_json(sidecar)
        base = f"{prefix}/{generation}"
        catalog_uri = await self.object_storage.persist_immutable_bytes(
            catalog_bytes,
            object_name=f"{base}/catalog.json",
            content_type="application/json",
        )
        sidecar_uri = await self.object_storage.persist_immutable_bytes(
            sidecar_bytes,
            object_name=f"{base}/item-features.json",
            content_type="application/json",
        )
        return {
            "catalog_generation": {
                "generation_id": generation,
                "uri": catalog_uri,
                "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
                "product_count": len(normalized_products),
                "schema_version": RETRIEVAL_CATALOG_SCHEMA_VERSION,
                "effective_at": effective,
                "available_at": available,
            },
            "item_feature_sidecar": {
                "generation_id": generation,
                "uri": sidecar_uri,
                "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
                "schema_version": RETRIEVAL_ITEM_FEATURE_SCHEMA_VERSION,
                "embedding_dimension": dimension,
                "embedding_model_revision": revision,
                "available_at": available,
            },
        }

    @staticmethod
    def _finite(value: Any, name: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"retrieval catalog {name} must be finite")
        return parsed
