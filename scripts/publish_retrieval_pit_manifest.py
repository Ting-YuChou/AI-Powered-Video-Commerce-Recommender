#!/usr/bin/env python3
"""Validate a retrieval PIT export and atomically publish its latest pointer."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from video_commerce.common.config import Config
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.ml.retrieval_pit_manifest import RetrievalPitManifestPublisher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="publish-retrieval-pit-manifest")
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--output-prefix")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--attribution-cutoff", type=float, required=True)
    parser.add_argument("--attribution-window-hours", type=int, default=168)
    parser.add_argument("--allowed-lateness-hours", type=int, default=1)
    parser.add_argument("--catalog-reference", required=True)
    return parser


async def _run(args: argparse.Namespace) -> str:
    config = Config()
    output_prefix = str(
        args.output_prefix or config.feature_lake_config.retrieval_pit_export_uri or ""
    )
    if not output_prefix:
        raise RuntimeError("retrieval PIT output prefix is required")
    refs = json.loads(Path(args.catalog_reference).read_text(encoding="utf-8"))
    storage = ObjectStorage(config.object_storage_config)
    await storage.initialize()
    return await RetrievalPitManifestPublisher(storage).publish(
        shard_uris=args.shard,
        output_prefix=output_prefix,
        materialization_run_id=args.run_id,
        dataset_version=args.dataset_version,
        attribution_cutoff=args.attribution_cutoff,
        attribution_window_hours=args.attribution_window_hours,
        allowed_lateness_hours=args.allowed_lateness_hours,
        catalog_generation=refs["catalog_generation"],
        item_feature_sidecar=refs["item_feature_sidecar"],
    )


def main() -> int:
    result = asyncio.run(_run(build_parser().parse_args()))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
