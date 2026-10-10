"""Publish immutable retrieval PIT manifests after Flink Parquet export."""

from __future__ import annotations

import asyncio
import hashlib
import os
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq

from video_commerce.ml.pit_manifest import PitManifestPublisher
from video_commerce.ml.pit_training_dataset import arrow_schema_sha256
from video_commerce.ml.retrieval_pit_dataset import (
    RETRIEVAL_ELIGIBILITY_POLICY_VERSION,
    RETRIEVAL_LABEL_POLICY_VERSION,
    RETRIEVAL_PIT_SCHEMA_VERSION,
    RetrievalPitDatasetError,
)


REQUIRED_COLUMNS = frozenset(
    {
        "query_id",
        "user_id",
        "product_id",
        "label_source",
        "label_type",
        "label_weight",
        "as_of_ts",
        "label_event_time",
        "label_available_at",
        "feature_event_time",
        "feature_available_at",
        "catalog_generation_id",
        "user_features_json",
        "seen_product_ids_json",
        "ranker_score",
    }
)


class RetrievalPitManifestPublisher(PitManifestPublisher):
    async def publish(
        self,
        *,
        shard_uris: Iterable[str],
        output_prefix: str,
        materialization_run_id: str,
        dataset_version: str,
        attribution_cutoff: float,
        attribution_window_hours: int,
        allowed_lateness_hours: int,
        catalog_generation: Mapping[str, Any],
        item_feature_sidecar: Mapping[str, Any],
    ) -> str:
        run_id = str(materialization_run_id or "").strip()
        if not run_id:
            raise RetrievalPitDatasetError("retrieval materialization run ID is blank")
        if attribution_window_hours <= 0 or allowed_lateness_hours < 0:
            raise RetrievalPitDatasetError("retrieval attribution windows are invalid")
        catalog_ref = dict(catalog_generation)
        sidecar_ref = dict(item_feature_sidecar)
        if catalog_ref.get("generation_id") != sidecar_ref.get("generation_id"):
            raise RetrievalPitDatasetError(
                "retrieval catalog and item sidecar generations differ"
            )
        await self._verify_reference(catalog_ref, "catalog")
        await self._verify_reference(sidecar_ref, "item feature")

        shards = []
        row_count = 0
        schema_hash = None
        min_as_of = None
        max_as_of = None
        catalog_ids = set()
        for uri in sorted(set(str(uri) for uri in shard_uris)):
            path, cleanup = await self.object_storage.materialize_for_processing(
                uri, suggested_suffix=".parquet"
            )
            try:
                table = pq.read_table(path)
                missing = REQUIRED_COLUMNS - set(table.column_names)
                if missing:
                    raise RetrievalPitDatasetError(
                        "retrieval shard is missing columns: "
                        + ",".join(sorted(missing))
                    )
                current_schema_hash = arrow_schema_sha256(table.schema)
                if schema_hash is None:
                    schema_hash = current_schema_hash
                elif schema_hash != current_schema_hash:
                    raise RetrievalPitDatasetError("retrieval shard schema mismatch")
                as_of_values = [
                    float(value)
                    for value in table.column("as_of_ts").to_pylist()
                    if value is not None
                ]
                if as_of_values:
                    min_as_of = (
                        min(as_of_values)
                        if min_as_of is None
                        else min(min_as_of, min(as_of_values))
                    )
                    max_as_of = (
                        max(as_of_values)
                        if max_as_of is None
                        else max(max_as_of, max(as_of_values))
                    )
                catalog_ids.update(
                    str(value)
                    for value in table.column("catalog_generation_id").to_pylist()
                    if value
                )
                shards.append(
                    {
                        "uri": uri,
                        "byte_size": os.path.getsize(path),
                        "sha256": await asyncio.to_thread(self._sha256, path),
                    }
                )
                row_count += table.num_rows
            finally:
                if cleanup and os.path.exists(path):
                    os.remove(path)
        if not shards or row_count <= 0 or schema_hash is None:
            raise RetrievalPitDatasetError("retrieval export contains no rows")
        if catalog_ids != {str(catalog_ref.get("generation_id") or "")}:
            raise RetrievalPitDatasetError(
                "retrieval shard catalog generation does not match manifest"
            )
        manifest = {
            "status": "complete",
            "schema_version": RETRIEVAL_PIT_SCHEMA_VERSION,
            "dataset_version": str(dataset_version),
            "materialization_run_id": run_id,
            "schema_hash": schema_hash,
            "attribution_cutoff": float(attribution_cutoff),
            "attribution_window_hours": int(attribution_window_hours),
            "allowed_lateness_hours": int(allowed_lateness_hours),
            "min_as_of_ts": min_as_of,
            "max_as_of_ts": max_as_of,
            "row_count": row_count,
            "label_policy_version": RETRIEVAL_LABEL_POLICY_VERSION,
            "eligibility_policy_version": RETRIEVAL_ELIGIBILITY_POLICY_VERSION,
            "catalog_generation": catalog_ref,
            "item_feature_sidecar": sidecar_ref,
            "shards": shards,
        }
        manifest_uri = self._join_uri(output_prefix, f"runs/{run_id}/manifest.json")
        latest_uri = self._join_uri(output_prefix, "latest.json")
        existing = await self._read_json_if_exists(manifest_uri)
        if existing is not None and existing != manifest:
            raise RetrievalPitDatasetError(
                "existing immutable retrieval manifest conflicts with retry"
            )
        if existing is None:
            await self._write_json(manifest_uri, manifest, create_only=True)
        await self._write_latest_pointer(
            latest_uri,
            {
                "manifest_uri": manifest_uri,
                "materialization_run_id": run_id,
                "attribution_cutoff": float(attribution_cutoff),
            },
        )
        return latest_uri

    async def _verify_reference(self, reference: Mapping[str, Any], kind: str) -> None:
        uri = str(reference.get("uri") or "")
        expected = str(reference.get("sha256") or "")
        if not uri or len(expected) != 64:
            raise RetrievalPitDatasetError(f"retrieval {kind} reference is incomplete")
        path, cleanup = await self.object_storage.materialize_for_processing(uri)
        try:
            actual = await asyncio.to_thread(self._sha256, path)
            if actual != expected:
                raise RetrievalPitDatasetError(
                    f"retrieval {kind} reference checksum mismatch"
                )
        finally:
            if cleanup and os.path.exists(path):
                os.remove(path)
