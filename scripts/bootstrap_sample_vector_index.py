"""Explicitly create the local/demo sample vector index artifact."""

import asyncio
from pathlib import Path

from video_commerce.common.config import VectorConfig
from video_commerce.ml.vector_search import VectorSearchEngine


async def main() -> None:
    config = VectorConfig()
    if config.bootstrap_mode != "sample":
        return
    index_path = Path(config.index_path)
    metadata_path = index_path.with_suffix(".metadata.json")
    manifest_path = index_path.with_suffix(".manifest.json")
    if index_path.exists() and metadata_path.exists() and manifest_path.exists():
        return
    engine = VectorSearchEngine(config)
    await engine._build_new_index()


if __name__ == "__main__":
    asyncio.run(main())
