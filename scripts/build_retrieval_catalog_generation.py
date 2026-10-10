#!/usr/bin/env python3
"""Publish one immutable retrieval catalog generation from a prepared JSON snapshot."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from video_commerce.common.config import Config
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.ml.retrieval_catalog import RetrievalCatalogPublisher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build-retrieval-catalog-generation")
    parser.add_argument(
        "--input", required=True, help="JSON with products and embeddings"
    )
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--effective-at", type=float, required=True)
    parser.add_argument("--available-at", type=float, required=True)
    parser.add_argument("--embedding-model-revision", required=True)
    parser.add_argument("--artifact-prefix", default="retrieval/catalogs")
    parser.add_argument("--output-reference", required=True)
    return parser


async def _run(args: argparse.Namespace) -> dict:
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    config = Config()
    storage = ObjectStorage(config.object_storage_config)
    await storage.initialize()
    return await RetrievalCatalogPublisher(storage).publish(
        products=payload.get("products") or [],
        embeddings=payload.get("embeddings") or {},
        generation_id=args.generation_id,
        effective_at=args.effective_at,
        available_at=args.available_at,
        embedding_model_revision=args.embedding_model_revision,
        artifact_prefix=args.artifact_prefix,
    )


def main() -> int:
    args = build_parser().parse_args()
    result = asyncio.run(_run(args))
    output = Path(args.output_reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
