"""
Helpers for durable model artifact storage and local cache synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from video_commerce.common.config import ModelConfig, RecommendationConfig
from video_commerce.data_plane.object_storage import ObjectStorage
from video_commerce.data_plane.system_store import SystemStore

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
    RANKING_SHADOW_MODEL_NAME = "ranking_model_pit_shadow"
    TWO_TOWER_MODEL_NAME = "two_tower_retrieval"
    SASREC_MODEL_NAME = "sasrec_retrieval"
    SWING_ITEMCF_MODEL_NAME = "swing_itemcf_recall"

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
        return str(
            Path(self.recommendation_config.cf_index_path).with_suffix(".cf_meta.json")
        )

    @property
    def two_tower_local_embedding_sidecar_path(self) -> str:
        return str(
            Path(self.recommendation_config.cf_index_path).with_suffix(
                ".cf_embeddings.npz"
            )
        )

    @property
    def two_tower_local_adapter_path(self) -> str:
        return str(
            Path(self.recommendation_config.cf_index_path).with_suffix(
                ".cf_adapter.npz"
            )
        )

    @property
    def sasrec_local_checkpoint_path(self) -> str:
        return self.recommendation_config.sasrec_checkpoint_path or str(
            Path(self.model_config.cache_dir) / "sasrec_model.pt"
        )

    @property
    def sasrec_local_vocab_path(self) -> str:
        return self.recommendation_config.sasrec_vocab_path or str(
            Path(self.model_config.cache_dir) / "sasrec_vocab.json"
        )

    @property
    def sasrec_local_metadata_path(self) -> str:
        return self.recommendation_config.sasrec_metadata_path or str(
            Path(self.model_config.cache_dir) / "sasrec_metadata.json"
        )

    @property
    def swing_itemcf_local_index_path(self) -> str:
        return self.recommendation_config.swing_itemcf_index_path or str(
            Path(self.model_config.cache_dir) / "swing_itemcf.json.gz"
        )

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

        checkpoint_sha256 = self._extract_artifact_sha256(
            record.payload,
            "checkpoint",
            legacy_key="artifact_sha256",
        )
        await self._sync_path_to_local(
            record.checkpoint_path,
            self.ranking_local_path,
            expected_sha256=checkpoint_sha256,
        )
        return record

    async def sync_latest_two_tower_artifacts(self) -> Optional[ModelArtifactRecord]:
        record = await self.get_latest_model_checkpoint(self.TWO_TOWER_MODEL_NAME)
        if not record:
            return None

        payload = record.payload
        cf_index_path = payload.get("cf_index_path")
        cf_metadata_path = payload.get("cf_index_metadata_path")
        if not cf_index_path or not cf_metadata_path:
            logger.warning(
                "Two-Tower artifact record is incomplete; skipping sync",
                extra={
                    "model_version": record.model_version,
                    "has_checkpoint": bool(record.checkpoint_path),
                    "has_cf_index": bool(cf_index_path),
                    "has_cf_metadata": bool(cf_metadata_path),
                },
            )
            return None
        artifact_specs: List[Tuple[str, str, Optional[str]]] = [
            (
                record.checkpoint_path,
                self.two_tower_local_checkpoint_path,
                self._extract_artifact_sha256(payload, "checkpoint"),
            )
        ]
        if cf_index_path:
            artifact_specs.append(
                (
                    cf_index_path,
                    self.two_tower_local_index_path,
                    self._extract_artifact_sha256(
                        payload, "cf_index", legacy_key="cf_index_sha256"
                    ),
                )
            )
        if cf_metadata_path:
            artifact_specs.append(
                (
                    cf_metadata_path,
                    self.two_tower_local_metadata_path,
                    self._extract_artifact_sha256(
                        payload,
                        "cf_index_metadata",
                        legacy_key="cf_index_metadata_sha256",
                    ),
                )
            )
        embedding_sidecar_path = payload.get("cf_embedding_sidecar_path")
        stale_optional_local_paths: List[str] = []
        if embedding_sidecar_path:
            artifact_specs.append(
                (
                    embedding_sidecar_path,
                    self.two_tower_local_embedding_sidecar_path,
                    self._extract_artifact_sha256(payload, "cf_embedding_sidecar"),
                )
            )
        else:
            stale_optional_local_paths.append(
                self.two_tower_local_embedding_sidecar_path
            )
        adapter_path = payload.get("cf_adapter_path")
        if adapter_path:
            artifact_specs.append(
                (
                    adapter_path,
                    self.two_tower_local_adapter_path,
                    self._extract_artifact_sha256(payload, "cf_adapter"),
                )
            )
        else:
            stale_optional_local_paths.append(self.two_tower_local_adapter_path)
        await self._sync_paths_to_local_atomically(artifact_specs)
        self._remove_local_artifacts(stale_optional_local_paths)
        return record

    async def sync_latest_sasrec_artifacts(self) -> Optional[ModelArtifactRecord]:
        record = await self.get_latest_model_checkpoint(self.SASREC_MODEL_NAME)
        if not record:
            return None

        payload = record.payload
        vocab_path = payload.get("vocab_path")
        metadata_path = payload.get("metadata_path")
        if not vocab_path or not metadata_path:
            logger.warning(
                "SASRec artifact record is incomplete; skipping sync",
                extra={
                    "model_version": record.model_version,
                    "has_checkpoint": bool(record.checkpoint_path),
                    "has_vocab": bool(vocab_path),
                    "has_metadata": bool(metadata_path),
                },
            )
            return None

        artifact_specs: List[Tuple[str, str, Optional[str]]] = [
            (
                record.checkpoint_path,
                self.sasrec_local_checkpoint_path,
                self._extract_artifact_sha256(payload, "checkpoint"),
            ),
            (
                vocab_path,
                self.sasrec_local_vocab_path,
                self._extract_artifact_sha256(
                    payload, "vocab", legacy_key="vocab_sha256"
                ),
            ),
            (
                metadata_path,
                self.sasrec_local_metadata_path,
                self._extract_artifact_sha256(
                    payload, "metadata", legacy_key="metadata_sha256"
                ),
            ),
        ]
        await self._sync_paths_to_local_atomically(artifact_specs)
        return record

    async def sync_latest_swing_itemcf_artifact(self) -> Optional[ModelArtifactRecord]:
        record = await self.get_latest_model_checkpoint(self.SWING_ITEMCF_MODEL_NAME)
        if not record:
            return None

        payload = record.payload
        index_path = payload.get("index_path") or record.checkpoint_path
        if not index_path:
            logger.warning(
                "Swing ItemCF artifact record is incomplete; skipping sync",
                extra={"model_version": record.model_version},
            )
            return None

        await self._sync_paths_to_local_atomically(
            [
                (
                    index_path,
                    self.swing_itemcf_local_index_path,
                    self._extract_artifact_sha256(
                        payload,
                        "index",
                        legacy_key="index_sha256",
                    ),
                )
            ]
        )
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

        artifact_sha256 = ObjectStorage.calculate_sha256(local_path)
        materialization_run_id = str(
            (payload or {}).get("feature_lake_materialization_run_id") or ""
        ).strip()
        storage_version = model_version
        if materialization_run_id:
            # A trainer whose lease expires during an in-flight S3 upload must
            # never overwrite the winner's bytes. Content-address the PIT object
            # while keeping the externally visible model version deterministic.
            storage_version = f"{model_version}-{artifact_sha256[:16]}"
        persisted_path = await self._persist_artifact(
            local_path=local_path,
            model_name=self.RANKING_MODEL_NAME,
            model_version=storage_version,
        )
        record_payload = dict(payload or {})
        record_payload.update(
            {
                "local_cache_path": self.ranking_local_path,
                "artifact_sha256": artifact_sha256,
                "artifact_manifest": {
                    "checkpoint": {
                        "path": persisted_path,
                        "sha256": artifact_sha256,
                        "local_cache_path": self.ranking_local_path,
                    }
                },
            }
        )
        inserted = await self.system_store.record_model_checkpoint(
            model_name=self.RANKING_MODEL_NAME,
            model_version=model_version,
            checkpoint_path=persisted_path,
            payload=record_payload,
        )
        if inserted is False:
            if not materialization_run_id:
                raise RuntimeError("ranking checkpoint record was not persisted")
            existing = (
                await self.system_store.get_model_checkpoint_for_materialization_run(
                    self.RANKING_MODEL_NAME,
                    materialization_run_id,
                )
            )
            if existing is None:
                raise RuntimeError(
                    "PIT ranking checkpoint conflict has no durable winner"
                )
            await self._verify_existing_checkpoint(existing)
            return ModelArtifactRecord(
                model_name=str(existing["model_name"]),
                model_version=str(existing["model_version"]),
                checkpoint_path=str(existing["checkpoint_path"]),
                payload=dict(existing.get("payload") or {}),
            )
        return ModelArtifactRecord(
            model_name=self.RANKING_MODEL_NAME,
            model_version=model_version,
            checkpoint_path=persisted_path,
            payload=record_payload,
        )

    async def _verify_existing_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        path = str(checkpoint.get("checkpoint_path") or "").strip()
        payload = dict(checkpoint.get("payload") or {})
        expected_sha256 = str(payload.get("artifact_sha256") or "").strip()
        if not path or len(expected_sha256) != 64:
            raise RuntimeError("existing PIT ranking checkpoint is incomplete")
        if self.object_storage is None:
            ObjectStorage.verify_sha256(path, expected_sha256)
            return
        (
            local_path,
            should_delete,
        ) = await self.object_storage.materialize_for_processing(
            path, suggested_suffix=Path(path).suffix or ".pt"
        )
        try:
            ObjectStorage.verify_sha256(local_path, expected_sha256)
        finally:
            if should_delete and os.path.exists(local_path):
                os.remove(local_path)

    async def persist_ranking_shadow_checkpoint(
        self,
        *,
        local_path: str,
        model_version: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[ModelArtifactRecord]:
        """Persist a PIT shadow artifact under a namespace serving never syncs."""
        if not self.system_store:
            return None
        artifact_sha256 = ObjectStorage.calculate_sha256(local_path)
        persisted_path = await self._persist_artifact(
            local_path=local_path,
            model_name=self.RANKING_SHADOW_MODEL_NAME,
            model_version=model_version,
        )
        record_payload = dict(payload or {})
        record_payload.update(
            {
                "shadow": True,
                "activation_allowed": False,
                "artifact_sha256": artifact_sha256,
                "artifact_manifest": {
                    "checkpoint": {
                        "path": persisted_path,
                        "sha256": artifact_sha256,
                        "local_cache_path": local_path,
                    }
                },
            }
        )
        await self.system_store.record_model_checkpoint(
            model_name=self.RANKING_SHADOW_MODEL_NAME,
            model_version=model_version,
            checkpoint_path=persisted_path,
            payload=record_payload,
        )
        return ModelArtifactRecord(
            model_name=self.RANKING_SHADOW_MODEL_NAME,
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
        embedding_sidecar_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        catalog_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[ModelArtifactRecord]:
        if not self.system_store:
            return None

        checkpoint_sha256 = ObjectStorage.calculate_sha256(checkpoint_path)
        index_sha256 = ObjectStorage.calculate_sha256(index_path)
        metadata_sha256 = ObjectStorage.calculate_sha256(metadata_path)
        embedding_sidecar_sha256 = (
            ObjectStorage.calculate_sha256(embedding_sidecar_path)
            if embedding_sidecar_path and Path(embedding_sidecar_path).exists()
            else None
        )
        adapter_sha256 = (
            ObjectStorage.calculate_sha256(adapter_path)
            if adapter_path and Path(adapter_path).exists()
            else None
        )

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
        persisted_embedding_sidecar = None
        if embedding_sidecar_path and embedding_sidecar_sha256:
            persisted_embedding_sidecar = await self._persist_artifact(
                local_path=embedding_sidecar_path,
                model_name=self.TWO_TOWER_MODEL_NAME,
                model_version=model_version,
            )
        persisted_adapter = None
        if adapter_path and adapter_sha256:
            persisted_adapter = await self._persist_artifact(
                local_path=adapter_path,
                model_name=self.TWO_TOWER_MODEL_NAME,
                model_version=model_version,
            )

        record_payload = dict(payload or {})
        artifact_manifest = {
            "checkpoint": {
                "path": persisted_checkpoint,
                "sha256": checkpoint_sha256,
                "local_cache_path": self.two_tower_local_checkpoint_path,
            },
            "cf_index": {
                "path": persisted_index,
                "sha256": index_sha256,
                "local_cache_path": self.two_tower_local_index_path,
            },
            "cf_index_metadata": {
                "path": persisted_metadata,
                "sha256": metadata_sha256,
                "local_cache_path": self.two_tower_local_metadata_path,
            },
        }
        optional_payload: Dict[str, Any] = {}
        if persisted_embedding_sidecar and embedding_sidecar_sha256:
            optional_payload.update(
                {
                    "cf_embedding_sidecar_path": persisted_embedding_sidecar,
                    "cf_embedding_sidecar_sha256": embedding_sidecar_sha256,
                    "local_cache_embedding_sidecar_path": self.two_tower_local_embedding_sidecar_path,
                }
            )
            artifact_manifest["cf_embedding_sidecar"] = {
                "path": persisted_embedding_sidecar,
                "sha256": embedding_sidecar_sha256,
                "local_cache_path": self.two_tower_local_embedding_sidecar_path,
            }
        if persisted_adapter and adapter_sha256:
            optional_payload.update(
                {
                    "cf_adapter_path": persisted_adapter,
                    "cf_adapter_sha256": adapter_sha256,
                    "local_cache_adapter_path": self.two_tower_local_adapter_path,
                }
            )
            artifact_manifest["cf_adapter"] = {
                "path": persisted_adapter,
                "sha256": adapter_sha256,
                "local_cache_path": self.two_tower_local_adapter_path,
            }
        record_payload.update(
            {
                "cf_index_path": persisted_index,
                "cf_index_metadata_path": persisted_metadata,
                "cf_index_sha256": index_sha256,
                "cf_index_metadata_sha256": metadata_sha256,
                "local_cache_checkpoint_path": self.two_tower_local_checkpoint_path,
                "local_cache_index_path": self.two_tower_local_index_path,
                "local_cache_metadata_path": self.two_tower_local_metadata_path,
                "artifact_manifest": artifact_manifest,
                **optional_payload,
            }
        )
        catalog_activation_id = None
        if catalog_metadata is not None:
            catalog_activation_id = await self.system_store.activate_product_catalog(
                model_version,
                catalog_metadata,
            )
            record_payload["catalog_activation_id"] = catalog_activation_id
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

    async def persist_sasrec_artifacts(
        self,
        *,
        checkpoint_path: str,
        vocab_path: str,
        metadata_path: str,
        model_version: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[ModelArtifactRecord]:
        if not self.system_store:
            return None

        checkpoint_sha256 = ObjectStorage.calculate_sha256(checkpoint_path)
        vocab_sha256 = ObjectStorage.calculate_sha256(vocab_path)
        metadata_sha256 = ObjectStorage.calculate_sha256(metadata_path)

        persisted_checkpoint = await self._persist_artifact(
            local_path=checkpoint_path,
            model_name=self.SASREC_MODEL_NAME,
            model_version=model_version,
        )
        persisted_vocab = await self._persist_artifact(
            local_path=vocab_path,
            model_name=self.SASREC_MODEL_NAME,
            model_version=model_version,
            content_type="application/json",
        )
        persisted_metadata = await self._persist_artifact(
            local_path=metadata_path,
            model_name=self.SASREC_MODEL_NAME,
            model_version=model_version,
            content_type="application/json",
        )

        record_payload = dict(payload or {})
        record_payload.update(
            {
                "vocab_path": persisted_vocab,
                "metadata_path": persisted_metadata,
                "vocab_sha256": vocab_sha256,
                "metadata_sha256": metadata_sha256,
                "local_cache_checkpoint_path": self.sasrec_local_checkpoint_path,
                "local_cache_vocab_path": self.sasrec_local_vocab_path,
                "local_cache_metadata_path": self.sasrec_local_metadata_path,
                "artifact_manifest": {
                    "checkpoint": {
                        "path": persisted_checkpoint,
                        "sha256": checkpoint_sha256,
                        "local_cache_path": self.sasrec_local_checkpoint_path,
                    },
                    "vocab": {
                        "path": persisted_vocab,
                        "sha256": vocab_sha256,
                        "local_cache_path": self.sasrec_local_vocab_path,
                    },
                    "metadata": {
                        "path": persisted_metadata,
                        "sha256": metadata_sha256,
                        "local_cache_path": self.sasrec_local_metadata_path,
                    },
                },
            }
        )
        await self.system_store.record_model_checkpoint(
            model_name=self.SASREC_MODEL_NAME,
            model_version=model_version,
            checkpoint_path=persisted_checkpoint,
            payload=record_payload,
        )
        return ModelArtifactRecord(
            model_name=self.SASREC_MODEL_NAME,
            model_version=model_version,
            checkpoint_path=persisted_checkpoint,
            payload=record_payload,
        )

    async def persist_swing_itemcf_artifact(
        self,
        *,
        index_path: str,
        model_version: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[ModelArtifactRecord]:
        if not self.system_store:
            return None

        index_sha256 = ObjectStorage.calculate_sha256(index_path)
        persisted_index = await self._persist_artifact(
            local_path=index_path,
            model_name=self.SWING_ITEMCF_MODEL_NAME,
            model_version=model_version,
            content_type="application/gzip",
        )
        record_payload = dict(payload or {})
        record_payload.update(
            {
                "index_path": persisted_index,
                "index_sha256": index_sha256,
                "local_cache_index_path": self.swing_itemcf_local_index_path,
                "artifact_manifest": {
                    "index": {
                        "path": persisted_index,
                        "sha256": index_sha256,
                        "local_cache_path": self.swing_itemcf_local_index_path,
                    }
                },
            }
        )
        await self.system_store.record_model_checkpoint(
            model_name=self.SWING_ITEMCF_MODEL_NAME,
            model_version=model_version,
            checkpoint_path=persisted_index,
            payload=record_payload,
        )
        return ModelArtifactRecord(
            model_name=self.SWING_ITEMCF_MODEL_NAME,
            model_version=model_version,
            checkpoint_path=persisted_index,
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

    async def _sync_path_to_local(
        self,
        storage_path: str,
        local_path: str,
        *,
        expected_sha256: Optional[str] = None,
    ) -> str:
        if not storage_path:
            raise ValueError("storage_path is required to sync a model artifact")
        if not self.object_storage:
            return storage_path
        try:
            return await self.object_storage.sync_to_local_path(
                storage_path,
                local_path,
                expected_sha256=expected_sha256,
            )
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

    async def _sync_paths_to_local_atomically(
        self,
        artifact_specs: List[Tuple[str, str, Optional[str]]],
    ) -> None:
        if not artifact_specs:
            return
        if not self.object_storage:
            return

        staged: List[Tuple[str, str]] = []
        try:
            for storage_path, local_path, expected_sha256 in artifact_specs:
                staged_path = await self.object_storage.stage_to_local_temp(
                    storage_path,
                    local_path,
                    expected_sha256=expected_sha256,
                )
                staged.append((staged_path, local_path))

            for staged_path, local_path in staged:
                target_path = Path(local_path)
                if Path(staged_path).resolve() == target_path.resolve():
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_path, target_path)
        except Exception:
            logger.error("Failed to atomically sync model artifact set")
            raise
        finally:
            for staged_path, local_path in staged:
                if Path(staged_path).resolve() == Path(local_path).resolve():
                    continue
                if os.path.exists(staged_path):
                    os.remove(staged_path)

    @staticmethod
    def _extract_artifact_sha256(
        payload: Dict[str, Any],
        manifest_key: str,
        *,
        legacy_key: Optional[str] = None,
    ) -> Optional[str]:
        manifest = payload.get("artifact_manifest") or {}
        artifact_entry = manifest.get(manifest_key) or {}
        checksum = artifact_entry.get("sha256")
        if checksum:
            return str(checksum)
        if legacy_key and payload.get(legacy_key):
            return str(payload[legacy_key])
        return None

    @staticmethod
    def _remove_local_artifacts(paths: List[str]) -> None:
        for path in paths:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "Failed to remove stale local artifact %s: %s", path, exc
                )
