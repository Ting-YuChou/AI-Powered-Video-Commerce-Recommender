"""CLI for exporting an exact verified ranking checkpoint to ONNX."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from video_commerce.common.config import Config
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.ml.ranking import RankingModel
from video_commerce.ml.ranking_onnx import export_ranking_onnx


async def _export(model_version: str, output: str) -> None:
    config = Config()
    if not config.database_config.enable:
        raise RuntimeError("exact-version ONNX export requires the metadata store")
    store = SystemStore(config.database_config)
    object_storage = ObjectStorage(config.object_storage_config)
    try:
        await store.initialize()
        await object_storage.initialize()
        manager = ModelArtifactManager(
            system_store=store,
            object_storage=object_storage,
            model_config=config.model_config,
            recommendation_config=config.recommendation_config,
        )
        ranking_model = RankingModel(config.ranking_config)
        record = await manager.sync_ranking_checkpoint_version(
            model_version,
            expected_feature_schema_version=ranking_model.feature_schema_version,
            require_onnx=False,
        )
        if record is None:
            raise RuntimeError("required ranking checkpoint is unavailable")
        await ranking_model.load_model(config.model_config.ranking_model_path)
        if not ranking_model.mark_artifact_record_verified(record):
            raise RuntimeError("ranking checkpoint verification failed")
        metadata = export_ranking_onnx(ranking_model, Path(output))
        metadata["source_checkpoint_sha256"] = ranking_model.artifact_sha256
        print(json.dumps(metadata, sort_keys=True))
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    asyncio.run(_export(arguments.model_version, arguments.output))


if __name__ == "__main__":
    main()
