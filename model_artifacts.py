"""
Helpers for durable model artifact storage and local cache synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from config import ModelConfig, RecommendationConfig
from object_storage import ObjectStorage
from system_store import SystemStore

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifactRecord:
    model_name: str
    model_version: str
    checkpoint_path: str
    payload: Dict[str, Any]
    created_at: Optional[float] = None


class ModelArtifactManager:
    """Coordinate Postgres metadata with local/object-storage model artifacts."""

    RANKING_MODEL_NAME = "ranking_model"
    TWO_TOWER_MODEL_NAME = "two_tower_retrieval"

    def __init__(
        self,
        *,
        system_store: Optional[SystemStore],
        object_storage: Optional[ObjectStorage],
        model_config: ModelConfig,
        recommendation_config: RecommendationConfig,
    ) -> None:
        self.system_store = system_store
        self.object_storage = object_storage
        self.model_config = model_config
        self.recommendation_config = recommendation_config

    @property
    def ranking_local_path(self) -> str:
        return self.model_config.ranking_model_path

    @property
    def two_tower_local_checkpoint_path(self) -> str:
        return self.recommendation_config.cf_index_path.replace(".faiss", ".pt")

    @property
    def two_tower_local_index_path(self) -> str:
        return self.recommendation_config.cf_index_path

    @property
    def two_tower_local_metadata_path(self) -> str:
        return str(Path(self.recommendation_config.cf_index_path).with_suffix(".cf_meta.json"))

    async def get_latest_model_checkpoint(
        self,
        model_name: str,
    ) -> Optional[ModelArtifactRecord]:
        if not self.system_store:
            return None

        checkpoint = await self.system_store.get_latest_model_checkpoint(model_name)
        if not checkpoint:
            return None

        return ModelArtifactRecord(
            model_name=checkpoint["model_name"],
            model_version=checkpoint["model_version"],
            checkpoint_path=checkpoint["checkpoint_path"],
            payload=checkpoint.get("payload") or {},
            created_at=checkpoint.get("created_at"),
        )

    async def sync_latest_ranking_checkpoint(self) -> Optional[ModelArtifactRecord]:
        record = await self.get_latest_model_checkpoint(self.RANKING_MODEL_NAME)
        if not record:
            return None

        await self._sync_path_to_local(record.checkpoint_path, self.ranking_local_path)
        return record

    async def sync_latest_two_tower_artifacts(self) -> Optional[ModelArtifactRecord]:
        record = await self.get_latest_model_checkpoint(self.TWO_TOWER_MODEL_NAME)
        if not record:
            return None

        payload = record.payload
        await self._sync_path_to_local(record.checkpoint_path, self.two_tower_local_checkpoint_path)
        cf_index_path = payload.get("cf_index_path")
        cf_metadata_path = payload.get("cf_index_metadata_path")
        if cf_index_path:
            await self._sync_path_to_local(cf_index_path, self.two_tower_local_index_path)
        if cf_metadata_path:
            await self._sync_path_to_local(cf_metadata_path, self.two_tower_local_metadata_path)
        return record

    async def persist_ranking_checkpoint(
        self,
        *,
        local_path: str,
        model_version: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[ModelArtifactRecord]:
        if not self.system_store:
            return None

        persisted_path = await self._persist_artifact(
            local_path=local_path,
            model_name=self.RANKING_MODEL_NAME,
            model_version=model_version,
        )
        record_payload = dict(payload or {})
        record_payload["local_cache_path"] = self.ranking_local_path
        await self.system_store.record_model_checkpoint(
            model_name=self.RANKING_MODEL_NAME,
            model_version=model_version,
            checkpoint_path=persisted_path,
            payload=record_payload,
        )
        return ModelArtifactRecord(
            model_name=self.RANKING_MODEL_NAME,
            model_version=model_version,
            checkpoint_path=persisted_path,
            payload=record_payload,
        )

    async def persist_two_tower_artifacts(
        self,
        *,
        checkpoint_path: str,
        index_path: str,
        metadata_path: str,
        model_version: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[ModelArtifactRecord]:
        if not self.system_store:
            return None

        persisted_checkpoint = await self._persist_artifact(
            local_path=checkpoint_path,
            model_name=self.TWO_TOWER_MODEL_NAME,
            model_version=model_version,
        )
        persisted_index = await self._persist_artifact(
            local_path=index_path,
            model_name=self.TWO_TOWER_MODEL_NAME,
            model_version=model_version,
        )
        persisted_metadata = await self._persist_artifact(
            local_path=metadata_path,
            model_name=self.TWO_TOWER_MODEL_NAME,
            model_version=model_version,
            content_type="application/json",
        )

        record_payload = dict(payload or {})
        record_payload.update(
            {
                "cf_index_path": persisted_index,
                "cf_index_metadata_path": persisted_metadata,
                "local_cache_checkpoint_path": self.two_tower_local_checkpoint_path,
                "local_cache_index_path": self.two_tower_local_index_path,
                "local_cache_metadata_path": self.two_tower_local_metadata_path,
            }
        )
        await self.system_store.record_model_checkpoint(
            model_name=self.TWO_TOWER_MODEL_NAME,
            model_version=model_version,
            checkpoint_path=persisted_checkpoint,
            payload=record_payload,
        )
        return ModelArtifactRecord(
            model_name=self.TWO_TOWER_MODEL_NAME,
            model_version=model_version,
            checkpoint_path=persisted_checkpoint,
            payload=record_payload,
        )

    async def _persist_artifact(
        self,
        *,
        local_path: str,
        model_name: str,
        model_version: str,
        content_type: Optional[str] = None,
    ) -> str:
        artifact_path = Path(local_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact does not exist: {local_path}")

        if not self.object_storage:
            return local_path

        object_name = self.object_storage.build_artifact_object_name(
            model_name=model_name,
            model_version=model_version,
            filename=Path(local_path).name,
        )
        return await self.object_storage.persist_staged_file(
            local_path,
            object_name=object_name,
            content_type=content_type,
        )

    async def _sync_path_to_local(self, storage_path: str, local_path: str) -> str:
        if not storage_path:
            raise ValueError("storage_path is required to sync a model artifact")
        if not self.object_storage:
            return storage_path
        try:
            return await self.object_storage.sync_to_local_path(storage_path, local_path)
        except Exception as exc:
            logger.error(
                "Failed to sync model artifact",
                extra={
                    "storage_path": storage_path,
                    "local_path": local_path,
                    "error": str(exc),
                },
            )
            raise
