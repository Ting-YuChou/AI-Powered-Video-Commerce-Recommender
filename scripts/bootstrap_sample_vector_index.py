"""Explicitly create the local/demo sample vector index artifact."""

import asyncio
from pathlib import Path

from video_commerce.common.config import VectorConfig
from video_commerce.ml.vector_search import (
    VectorIndexMetricMismatch,
    VectorSearchEngine,
)


async def main() -> None:
    config = VectorConfig()
    if config.bootstrap_mode != "sample":
        return
    engine = VectorSearchEngine(config)
    index_path, metadata_path = engine._resolve_active_artifact_paths(
        Path(config.index_path)
    )
    manifest_path = engine._manifest_path(index_path)
    if index_path.exists() and metadata_path.exists() and manifest_path.exists():
        try:
            engine._validate_index_manifest(index_path, metadata_path)
            return
        except VectorIndexMetricMismatch:
            # Pre-inner-product generations are rebuilt below.
            pass
    await engine._build_new_index()


if __name__ == "__main__":
    asyncio.run(main())
