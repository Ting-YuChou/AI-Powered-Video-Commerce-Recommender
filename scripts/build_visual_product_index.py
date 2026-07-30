"""Build a checksum-locked product-image CLIP FAISS bundle offline."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from video_commerce.common.config import Config
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.ml.visual_product_index import (
    FrozenCLIPProductImageEncoder,
    build_visual_product_index_from_catalog,
)


async def run(
    *,
    catalog_path: str,
    output_dir: str,
    catalog_activation_id: str,
    model_version: str,
    catalog_available_at: float,
    publish: bool,
) -> int:
    config = Config()
    payload = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    products = payload.get("products") if isinstance(payload, dict) else payload
    if not isinstance(products, list):
        raise ValueError("catalog snapshot must be a product list or {products: [...]}")
    storage = ObjectStorage(config.object_storage_config)
    await storage.initialize()
    device = config.model_config.device
    if device == "auto":
        import torch

        device = (
            "cuda"
            if config.model_config.enable_gpu and torch.cuda.is_available()
            else "cpu"
        )
    encoder = FrozenCLIPProductImageEncoder(
        model_id=config.model_config.clip_model,
        revision=config.model_config.clip_revision,
        device=device,
        cache_dir=config.model_config.cache_dir,
        batch_size=config.model_config.batch_size,
    )
    result = await build_visual_product_index_from_catalog(
        products,
        object_storage=storage,
        encode_images=encoder.encode,
        output_dir=output_dir,
        clip_model_id=config.model_config.clip_model,
        clip_revision=config.model_config.clip_revision,
        catalog_activation_id=catalog_activation_id,
        model_version=model_version,
        catalog_available_at=catalog_available_at,
    )
    persisted = None
    if publish:
        store = SystemStore(config.database_config)
        await store.initialize()
        try:
            manager = ModelArtifactManager(
                system_store=store,
                object_storage=storage,
                model_config=config.model_config,
                recommendation_config=config.recommendation_config,
            )
            persisted = await manager.persist_visual_product_index_bundle(
                manifest_path=str(result.manifest_path)
            )
            if persisted is None:
                raise RuntimeError("visual product index record was not persisted")
        finally:
            await store.close()
    print(
        json.dumps(
            {
                "manifest_path": str(result.manifest_path),
                "indexed_product_count": len(result.indexed_product_ids),
                "quarantine": result.quarantine,
                "published_path": (
                    persisted.checkpoint_path if persisted is not None else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--catalog-activation-id", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--catalog-available-at", required=True, type=float)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Persist the complete bundle and record it for atomic activation",
    )
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            run(
                catalog_path=args.catalog,
                output_dir=args.output_dir,
                catalog_activation_id=args.catalog_activation_id,
                model_version=args.model_version,
                catalog_available_at=args.catalog_available_at,
                publish=args.publish,
            )
        )
    )


if __name__ == "__main__":
    main()
