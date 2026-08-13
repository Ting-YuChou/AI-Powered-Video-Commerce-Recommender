"""Materialize one exact verified ranking ONNX artifact for Triton startup."""

from __future__ import annotations

import asyncio
import json

from video_commerce.common.config import Config
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.ml.model_artifacts import ModelArtifactManager
from video_commerce.ml.ranking import RankingModel
from video_commerce.ranking_runtime.ranking_triton import (
    materialize_triton_repository,
)


async def _run() -> None:
    config = Config()
    ranking = config.ranking_config
    if ranking.inference_backend != "triton":
        raise RuntimeError("RANKING_INFERENCE_BACKEND=triton is required")
    if not ranking.required_model_version:
        raise RuntimeError("RANKING_REQUIRED_MODEL_VERSION is required")
    system_store = None
    object_storage = ObjectStorage(config.object_storage_config)
    try:
        if config.database_config.enable:
            system_store = SystemStore(config.database_config)
            await system_store.initialize()
        await object_storage.initialize()
        manager = ModelArtifactManager(
            system_store=system_store,
            object_storage=object_storage,
            model_config=config.model_config,
            recommendation_config=config.recommendation_config,
        )
        expected_schema = RankingModel(config.ranking_config).feature_schema_version
        record = await manager.sync_ranking_checkpoint_version(
            ranking.required_model_version,
            expected_feature_schema_version=expected_schema,
            require_onnx=True,
        )
        if record is None:
            raise RuntimeError("required ranking ONNX artifact is unavailable")
        result = materialize_triton_repository(
            record,
            ranking.triton_model_repository,
            required_model_version=ranking.required_model_version,
            model_name=ranking.triton_model_name,
            max_batch_size=ranking.triton_max_batch_size,
            max_queue_delay_microseconds=ranking.triton_queue_delay_microseconds,
            max_queue_size=ranking.triton_queue_size,
            default_timeout_microseconds=ranking.triton_queue_timeout_microseconds,
            instance_count=ranking.triton_instance_count,
            intra_op_threads=ranking.triton_ort_intra_op_threads,
            inter_op_threads=ranking.triton_ort_inter_op_threads,
        )
        print(json.dumps(result.__dict__, sort_keys=True))
    finally:
        if system_store is not None:
            await system_store.close()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
