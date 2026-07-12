"""Fail-closed reader for immutable manifest-guarded PIT Parquet datasets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

from video_commerce.common.feature_history_contracts import payload_sha256
from video_commerce.common.models import CandidateProduct, UserFeatures
from video_commerce.ml.ranking_features import FeatureBundle
from video_commerce.ml.din import parse_din_behavior_sequences
from video_commerce.ml.ranking_training import (
    RANKING_LABEL_DEFINITION_VERSION,
    AttributionFacts,
    RankingTrainingExample,
)


class PitTrainingDatasetError(ValueError):
    """Raised when a PIT export cannot be safely used for model training."""


class PitTrainingDatasetUnavailable(PitTrainingDatasetError):
    """Raised while a greenfield deployment is waiting for its first pointer."""


@dataclass(frozen=True)
class PitTrainingDataset:
    dataset_version: str
    materialization_run_id: str
    feature_definition_version: str
    label_definition_version: str
    manifest_uri: str
    iceberg_snapshot_id: str
    schema_hash: str
    quarantine_rows: int
    examples: List[RankingTrainingExample]


def arrow_schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


class PitTrainingDatasetReader:
    """Resolve latest once, pin one manifest, and validate every shard before loading."""

    def __init__(
        self,
        object_storage,
        *,
        expected_feature_definition_version: str,
        expected_label_definition_version: str = RANKING_LABEL_DEFINITION_VERSION,
        observability=None,
    ):
        self.object_storage = object_storage
        self.expected_feature_definition_version = expected_feature_definition_version
        self.expected_label_definition_version = expected_label_definition_version
        self.observability = observability

    async def read(self, dataset_uri: str) -> PitTrainingDataset:
        try:
            return await self._read_pinned(dataset_uri)
        except PitTrainingDatasetUnavailable:
            raise
        except PitTrainingDatasetError as exc:
            if self.observability is not None:
                self.observability.record_pit_manifest_validation_failure(
                    type(exc).__name__
                )
            raise

    async def _read_pinned(self, dataset_uri: str) -> PitTrainingDataset:
        pointer = await self._read_json_uri(dataset_uri, kind="latest pointer")
        manifest_uri = str(pointer.get("manifest_uri") or "").strip()
        pointer_run_id = str(pointer.get("materialization_run_id") or "").strip()
        if not manifest_uri or not pointer_run_id:
            raise PitTrainingDatasetError(
                "PIT latest pointer is missing manifest_uri or materialization_run_id"
            )

        manifest = await self._read_json_uri(manifest_uri, kind="manifest")
        if manifest.get("status") != "complete":
            raise PitTrainingDatasetError("PIT dataset manifest is not complete")
        run_id = str(manifest.get("materialization_run_id") or "").strip()
        if run_id != pointer_run_id:
            raise PitTrainingDatasetError(
                "PIT latest pointer and manifest materialization_run_id do not match"
            )
        definition_version = str(manifest.get("feature_definition_version") or "")
        if definition_version != self.expected_feature_definition_version:
            raise PitTrainingDatasetError(
                "PIT dataset feature definition version does not match the trainer"
            )
        label_definition_version = str(manifest.get("label_definition_version") or "")
        if label_definition_version != self.expected_label_definition_version:
            raise PitTrainingDatasetError(
                "PIT dataset label definition version does not match the trainer"
            )
        dataset_version = str(manifest.get("dataset_version") or "").strip()
        if not dataset_version:
            raise PitTrainingDatasetError(
                "PIT dataset manifest is missing dataset_version"
            )
        iceberg_table_id = str(manifest.get("iceberg_table_id") or "").strip()
        iceberg_snapshot_id = str(manifest.get("iceberg_snapshot_id") or "").strip()
        if not iceberg_table_id or not iceberg_snapshot_id:
            raise PitTrainingDatasetError(
                "PIT dataset manifest is missing Iceberg table or snapshot ID"
            )
        expected_schema_hash = str(manifest.get("schema_hash") or "").strip()
        if len(expected_schema_hash) != 64:
            raise PitTrainingDatasetError(
                "PIT dataset manifest has invalid schema hash"
            )
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise PitTrainingDatasetError("PIT dataset manifest has no Parquet shards")

        tables: List[pa.Table] = []
        pinned_schema: pa.Schema | None = None
        for shard in shards:
            table = await self._read_verified_shard(
                shard,
                expected_schema_hash=expected_schema_hash,
                expected_definition_version=definition_version,
            )
            if pinned_schema is None:
                pinned_schema = table.schema
            elif table.schema != pinned_schema:
                raise PitTrainingDatasetError("PIT Parquet shard schema mismatch")
            tables.append(table)

        table = pa.concat_tables(tables)
        expected_rows = int(manifest.get("row_count", -1))
        if expected_rows < 0 or table.num_rows != expected_rows:
            raise PitTrainingDatasetError(
                f"PIT dataset row count mismatch: expected {expected_rows}, got {table.num_rows}"
            )
        examples = [
            self._normalize_training_row(
                row, definition_version, label_definition_version
            )
            for row in table.to_pylist()
        ]
        return PitTrainingDataset(
            dataset_version=dataset_version,
            materialization_run_id=run_id,
            feature_definition_version=definition_version,
            label_definition_version=label_definition_version,
            manifest_uri=manifest_uri,
            iceberg_snapshot_id=iceberg_snapshot_id,
            schema_hash=expected_schema_hash,
            quarantine_rows=max(0, int(manifest.get("quarantine_row_count", 0))),
            examples=examples,
        )

    async def _read_json_uri(self, uri: str, *, kind: str) -> Dict[str, Any]:
        try:
            path, should_delete = await self.object_storage.materialize_for_processing(
                uri, suggested_suffix=".json"
            )
        except Exception as exc:
            if kind == "latest pointer" and self._is_not_found(exc):
                raise PitTrainingDatasetUnavailable(
                    f"PIT {kind} is unavailable: {uri}"
                ) from exc
            raise PitTrainingDatasetError(f"unable to read PIT {kind}: {exc}") from exc
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError as exc:
            if kind == "latest pointer":
                raise PitTrainingDatasetUnavailable(
                    f"PIT {kind} is unavailable: {uri}"
                ) from exc
            raise PitTrainingDatasetError(f"unable to read PIT {kind}: {exc}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise PitTrainingDatasetError(f"unable to read PIT {kind}: {exc}") from exc
        finally:
            if should_delete and os.path.exists(path):
                os.remove(path)
        if not isinstance(payload, dict):
            raise PitTrainingDatasetError(f"PIT {kind} must be a JSON object")
        return payload

    @staticmethod
    def _is_not_found(exc: Exception) -> bool:
        if isinstance(exc, FileNotFoundError):
            return True
        response = getattr(exc, "response", None)
        if not isinstance(response, dict):
            return False
        code = str((response.get("Error") or {}).get("Code") or "")
        return code in {"404", "NoSuchKey", "NotFound"}

    async def _read_verified_shard(
        self,
        shard: Any,
        *,
        expected_schema_hash: str,
        expected_definition_version: str,
    ) -> pa.Table:
        if not isinstance(shard, dict):
            raise PitTrainingDatasetError("PIT manifest shard entry must be an object")
        uri = str(shard.get("uri") or "").strip()
        expected_size = int(shard.get("byte_size", -1))
        expected_hash = str(shard.get("sha256") or "").strip()
        if not uri or expected_size < 0 or len(expected_hash) != 64:
            raise PitTrainingDatasetError("PIT manifest shard entry is incomplete")
        try:
            path, should_delete = await self.object_storage.materialize_for_processing(
                uri, suggested_suffix=".parquet"
            )
        except (OSError, FileNotFoundError) as exc:
            raise PitTrainingDatasetError(
                f"PIT Parquet shard is missing: {uri}"
            ) from exc
        try:
            if not os.path.exists(path):
                raise PitTrainingDatasetError(f"PIT Parquet shard is missing: {uri}")
            actual_size = os.path.getsize(path)
            if actual_size != expected_size:
                raise PitTrainingDatasetError(
                    f"PIT Parquet shard size mismatch for {uri}"
                )
            actual_hash = self._sha256(path)
            if actual_hash != expected_hash:
                raise PitTrainingDatasetError(
                    f"PIT Parquet shard checksum mismatch for {uri}"
                )
            try:
                table = pq.read_table(path)
            except Exception as exc:
                raise PitTrainingDatasetError(
                    f"unable to read PIT Parquet shard {uri}: {exc}"
                ) from exc
            if arrow_schema_sha256(table.schema) != expected_schema_hash:
                raise PitTrainingDatasetError(
                    f"PIT Parquet shard schema mismatch for {uri}"
                )
            if "feature_definition_version" not in table.column_names:
                raise PitTrainingDatasetError(
                    "PIT Parquet shard schema is missing feature_definition_version"
                )
            versions = set(table.column("feature_definition_version").to_pylist())
            if versions != {expected_definition_version}:
                raise PitTrainingDatasetError(
                    "PIT Parquet shard feature definition version mismatch"
                )
            return table
        finally:
            if should_delete and os.path.exists(path):
                os.remove(path)

    @staticmethod
    def _normalize_training_row(
        row: Dict[str, Any],
        definition_version: str,
        label_definition_version: str,
    ) -> RankingTrainingExample:
        if row.get("as_of_ts") is None:
            raise PitTrainingDatasetError("PIT training row is missing as_of_ts")
        if len(str(row.get("feature_bundle_hash") or "")) != 64:
            raise PitTrainingDatasetError(
                "PIT training row is missing feature_bundle_hash"
            )
        if len(str(row.get("online_feature_bundle_hash") or "")) != 64:
            raise PitTrainingDatasetError(
                "PIT training row is missing online_feature_bundle_hash"
            )
        if row.get("feature_definition_version") != definition_version:
            raise PitTrainingDatasetError(
                "PIT training row feature definition version does not match manifest"
            )
        if row.get("label_definition_version") != label_definition_version:
            raise PitTrainingDatasetError(
                "PIT training row label definition version does not match manifest"
            )
        normalized = dict(row)
        for source, target in (
            ("user_features_json", "user_features"),
            ("product_metadata_json", "product_metadata"),
            ("context_json", "context"),
            ("candidate_features_json", "candidate_scores"),
            ("behavior_sequences_json", "behavior_sequences"),
        ):
            raw = normalized.pop(source, None)
            if raw is None:
                continue
            if (
                source == "behavior_sequences_json"
                and len(str(raw).encode("utf-8")) > 512_000
            ):
                raise PitTrainingDatasetError(
                    "PIT training row has oversized behavior_sequences_json"
                )
            try:
                normalized[target] = json.loads(raw)
            except (TypeError, json.JSONDecodeError) as exc:
                raise PitTrainingDatasetError(
                    f"PIT training row has invalid {source}: {exc}"
                ) from exc
        bundle_payload = {
            "as_of_ts": float(normalized["as_of_ts"]),
            "candidate_features": normalized.get("candidate_scores") or {},
            "context": normalized.get("context") or {},
            "feature_definition_version": definition_version,
            "product_id": str(normalized.get("product_id") or ""),
            "product_metadata": normalized.get("product_metadata") or {},
            "user_features": normalized.get("user_features") or {},
            "user_id": str(normalized.get("user_id") or ""),
        }
        if normalized.get("behavior_sequences") is not None:
            bundle_payload["behavior_sequences"] = normalized["behavior_sequences"]
        expected_bundle_hash = payload_sha256(bundle_payload)
        if normalized["feature_bundle_hash"] != expected_bundle_hash:
            raise PitTrainingDatasetError(
                "PIT training row feature_bundle_hash does not match final bundle"
            )
        impression_id = str(normalized.get("impression_id") or "").strip()
        observation_id = str(normalized.get("observation_id") or "").strip()
        user_id = str(normalized.get("user_id") or "").strip()
        product_id = str(normalized.get("product_id") or "").strip()
        if not all((impression_id, observation_id, user_id, product_id)):
            raise PitTrainingDatasetError(
                "PIT training row is missing impression, observation, user, or product ID"
            )
        user_payload = normalized.get("user_features")
        product_metadata = normalized.get("product_metadata")
        context = normalized.get("context")
        scores = normalized.get("candidate_scores")
        if not isinstance(user_payload, dict) or not user_payload:
            raise PitTrainingDatasetError("PIT training row is missing user features")
        if not isinstance(product_metadata, dict) or not product_metadata:
            raise PitTrainingDatasetError(
                "PIT training row is missing product metadata"
            )
        if not isinstance(context, dict):
            raise PitTrainingDatasetError("PIT training row is missing context")
        required_scores = {
            "collaborative_score",
            "content_similarity_score",
            "popularity_score",
            "combined_score",
        }
        if not isinstance(scores, dict) or not required_scores.issubset(scores):
            raise PitTrainingDatasetError(
                "PIT training row is missing complete candidate scores"
            )
        user_payload = dict(user_payload)
        user_payload.setdefault("user_id", user_id)
        try:
            user_features = UserFeatures(**user_payload)
            candidate = CandidateProduct(
                product_id=product_id,
                collaborative_score=float(scores["collaborative_score"]),
                content_similarity_score=float(scores["content_similarity_score"]),
                popularity_score=float(scores["popularity_score"]),
                combined_score=float(scores["combined_score"]),
                source="pit_observation",
            )
            attribution = AttributionFacts(
                attributed_action=str(normalized.get("attributed_action") or ""),
                attributed_click=bool(normalized.get("attributed_click")),
                attributed_purchase=bool(normalized.get("attributed_purchase")),
                attributed_value=normalized.get("attributed_value"),
                attributed_value_source=normalized.get("attributed_value_source"),
            )
            bundle = FeatureBundle(
                as_of_ts=float(normalized["as_of_ts"]),
                feature_definition_version=definition_version,
                user_features=user_features,
                product_metadata=dict(product_metadata),
                context=dict(context),
                candidate=candidate,
                behavior_sequences=(
                    parse_din_behavior_sequences(
                        normalized["behavior_sequences"],
                        expected_as_of_ts=float(normalized["as_of_ts"]),
                    )
                    if normalized.get("behavior_sequences") is not None
                    else None
                ),
            )
            return RankingTrainingExample(
                observation_id=observation_id,
                impression_id=impression_id,
                bundle=bundle,
                attribution=attribution,
            )
        except (TypeError, ValueError) as exc:
            raise PitTrainingDatasetError(
                f"PIT training row typed contract validation failed: {exc}"
            ) from exc

    @staticmethod
    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
