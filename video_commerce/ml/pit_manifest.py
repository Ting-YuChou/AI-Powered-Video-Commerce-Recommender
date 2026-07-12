"""Validate immutable PIT Parquet shards and publish manifest/latest atomically."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

import pyarrow.parquet as pq

from video_commerce.ml.pit_training_dataset import (
    PitTrainingDatasetError,
    arrow_schema_sha256,
)
from video_commerce.ml.ranking_training import RANKING_LABEL_DEFINITION_VERSION


class PitManifestPublisher:
    def __init__(self, object_storage: Any) -> None:
        self.object_storage = object_storage

    async def publish(
        self,
        *,
        shard_uris: Iterable[str],
        output_prefix: str,
        materialization_run_id: str,
        iceberg_table_id: str,
        iceberg_snapshot_id: str,
        feature_definition_version: str,
        label_definition_version: str,
        attribution_cutoff: float,
        quarantine_row_count: int = 0,
        expected_iceberg_row_count: int | None = None,
    ) -> str:
        run_id = str(materialization_run_id or "").strip()
        if not run_id:
            raise PitTrainingDatasetError("materialization_run_id must not be blank")
        shards: List[Dict[str, Any]] = []
        schema_hash = None
        row_count = 0
        parity_matches = 0
        min_as_of = None
        max_as_of = None
        for uri in sorted(set(str(uri) for uri in shard_uris)):
            path, should_delete = await self.object_storage.materialize_for_processing(
                uri, suggested_suffix=".parquet"
            )
            try:
                if not os.path.exists(path):
                    raise PitTrainingDatasetError(
                        f"PIT Parquet shard is missing: {uri}"
                    )
                table = pq.read_table(path)
                current_schema_hash = arrow_schema_sha256(table.schema)
                if schema_hash is None:
                    schema_hash = current_schema_hash
                elif current_schema_hash != schema_hash:
                    raise PitTrainingDatasetError("PIT Parquet shard schema mismatch")
                if "as_of_ts" not in table.column_names:
                    raise PitTrainingDatasetError(
                        "PIT Parquet shard is missing as_of_ts"
                    )
                if "feature_definition_version" not in table.column_names:
                    raise PitTrainingDatasetError(
                        "PIT Parquet shard is missing feature_definition_version"
                    )
                if "label_definition_version" not in table.column_names:
                    raise PitTrainingDatasetError(
                        "PIT Parquet shard is missing label_definition_version"
                    )
                for required_column in (
                    "observation_id",
                    "impression_id",
                    "user_features_json",
                    "product_metadata_json",
                    "context_json",
                    "candidate_features_json",
                    "attributed_action",
                    "attributed_click",
                    "attributed_purchase",
                    "attributed_value",
                    "attributed_value_source",
                ):
                    if required_column not in table.column_names:
                        raise PitTrainingDatasetError(
                            f"PIT Parquet shard is missing {required_column}"
                        )
                for required_hash in (
                    "online_feature_bundle_hash",
                    "feature_bundle_hash",
                ):
                    if required_hash not in table.column_names:
                        raise PitTrainingDatasetError(
                            f"PIT Parquet shard is missing {required_hash}"
                        )
                parity_matches += sum(
                    1
                    for online, offline in zip(
                        table.column("online_feature_bundle_hash").to_pylist(),
                        table.column("feature_bundle_hash").to_pylist(),
                    )
                    if online == offline
                )
                versions = set(table.column("feature_definition_version").to_pylist())
                if versions != {feature_definition_version}:
                    raise PitTrainingDatasetError(
                        "PIT Parquet shard feature definition version mismatch"
                    )
                label_versions = set(
                    table.column("label_definition_version").to_pylist()
                )
                if label_versions != {label_definition_version}:
                    raise PitTrainingDatasetError(
                        "PIT Parquet shard label definition version mismatch"
                    )
                as_of_values = [
                    float(value)
                    for value in table.column("as_of_ts").to_pylist()
                    if value is not None
                ]
                if as_of_values:
                    shard_min = min(as_of_values)
                    shard_max = max(as_of_values)
                    min_as_of = (
                        shard_min if min_as_of is None else min(min_as_of, shard_min)
                    )
                    max_as_of = (
                        shard_max if max_as_of is None else max(max_as_of, shard_max)
                    )
                size = os.path.getsize(path)
                shards.append(
                    {
                        "uri": uri,
                        "byte_size": size,
                        "sha256": await asyncio.to_thread(self._sha256, path),
                    }
                )
                row_count += table.num_rows
            finally:
                if should_delete and os.path.exists(path):
                    os.remove(path)
        if not shards or schema_hash is None or row_count <= 0:
            raise PitTrainingDatasetError("PIT export contains no Parquet shards")
        if (
            expected_iceberg_row_count is not None
            and row_count != expected_iceberg_row_count
        ):
            raise PitTrainingDatasetError(
                "PIT Parquet row count does not match the pinned Iceberg run"
            )

        manifest_uri = self._join_uri(output_prefix, f"runs/{run_id}/manifest.json")
        latest_uri = self._join_uri(output_prefix, "latest.json")
        manifest = {
            "status": "complete",
            "dataset_version": f"iceberg-snapshot-{iceberg_snapshot_id}",
            "materialization_run_id": run_id,
            "iceberg_table_id": iceberg_table_id,
            "iceberg_snapshot_id": str(iceberg_snapshot_id),
            "feature_definition_version": feature_definition_version,
            "label_definition_version": label_definition_version,
            "schema_hash": schema_hash,
            "attribution_cutoff": float(attribution_cutoff),
            "row_count": row_count,
            "quarantine_row_count": max(0, int(quarantine_row_count)),
            "online_offline_parity_ratio": parity_matches / row_count,
            "min_as_of_ts": min_as_of,
            "max_as_of_ts": max_as_of,
            "shards": shards,
        }
        existing_manifest = await self._read_json_if_exists(manifest_uri)
        latest_pointer = {
            "manifest_uri": manifest_uri,
            "materialization_run_id": run_id,
            "attribution_cutoff": float(attribution_cutoff),
        }
        if existing_manifest is not None:
            if existing_manifest != manifest:
                raise PitTrainingDatasetError(
                    "existing immutable PIT manifest conflicts with retry payload"
                )
            await self._write_latest_pointer(latest_uri, latest_pointer)
            return latest_uri
        await self._write_json(manifest_uri, manifest, create_only=True)
        await self._write_latest_pointer(latest_uri, latest_pointer)
        return latest_uri

    async def _write_latest_pointer(
        self, latest_uri: str, pointer: Dict[str, Any]
    ) -> None:
        existing = await self._read_json_if_exists(latest_uri)
        if existing is not None:
            existing_cutoff = float(existing.get("attribution_cutoff") or 0.0)
            new_cutoff = float(pointer["attribution_cutoff"])
            if existing_cutoff > new_cutoff:
                return
            if (
                existing_cutoff == new_cutoff
                and existing.get("materialization_run_id")
                != pointer["materialization_run_id"]
            ):
                raise PitTrainingDatasetError(
                    "PIT latest pointer cutoff conflicts with another run"
                )
        await self._write_json(latest_uri, pointer)

    async def _read_json_if_exists(self, uri: str) -> Dict[str, Any] | None:
        if not uri.startswith("s3://"):
            path = Path(uri)
            if not path.exists():
                return None
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PitTrainingDatasetError(
                    f"existing PIT manifest cannot be read: {uri}"
                ) from exc
        try:
            path, should_delete = await self.object_storage.materialize_for_processing(
                uri, suggested_suffix=".json"
            )
        except Exception as exc:
            if self._is_not_found(exc):
                return None
            raise PitTrainingDatasetError(
                f"existing PIT manifest cannot be read: {uri}"
            ) from exc
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PitTrainingDatasetError(
                f"existing PIT manifest cannot be read: {uri}"
            ) from exc
        finally:
            if should_delete and os.path.exists(path):
                os.remove(path)

    @staticmethod
    def _is_not_found(exc: Exception) -> bool:
        if isinstance(exc, FileNotFoundError):
            return True
        response = getattr(exc, "response", None)
        if not isinstance(response, dict):
            return False
        code = str((response.get("Error") or {}).get("Code") or "")
        return code in {"404", "NoSuchKey", "NotFound"}

    async def _write_json(
        self, uri: str, payload: Dict[str, Any], *, create_only: bool = False
    ) -> None:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if uri.startswith("s3://"):
            parsed = urlparse(uri)
            configured_bucket = getattr(
                getattr(self.object_storage, "config", None), "bucket", None
            )
            if parsed.netloc != configured_bucket:
                raise PitTrainingDatasetError(
                    "PIT manifest output bucket does not match object storage configuration"
                )
            fd, temp_path = tempfile.mkstemp(suffix=".json")
            os.close(fd)
            try:
                Path(temp_path).write_bytes(encoded)
                persist = (
                    self.object_storage.persist_staged_file_create_only
                    if create_only
                    else self.object_storage.persist_staged_file
                )
                try:
                    await persist(
                        temp_path,
                        object_name=parsed.path.lstrip("/"),
                        content_type="application/json",
                    )
                except Exception as exc:
                    if create_only:
                        raise PitTrainingDatasetError(
                            f"PIT manifest already exists or cannot be created: {uri}"
                        ) from exc
                    raise
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            return
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        if create_only:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError as exc:
                raise PitTrainingDatasetError(
                    f"PIT manifest already exists and is immutable: {uri}"
                ) from exc
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            return
        fd, temp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def _join_uri(prefix: str, suffix: str) -> str:
        return f"{prefix.rstrip('/')}/{suffix.lstrip('/')}"

    @staticmethod
    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def _is_parquet_shard_uri(uri: str) -> bool:
    """Accept Flink part files without trusting filename extensions."""
    return Path(urlparse(str(uri)).path).name.startswith("part-")


async def _main() -> None:
    from video_commerce.common.config import Config
    from video_commerce.data_plane.object_storage import ObjectStorage

    config = Config()
    feature_lake = config.feature_lake_config
    run_id = os.environ.get("FEATURE_LAKE_MATERIALIZATION_RUN_ID", "").strip()
    export_prefix = os.environ.get("FEATURE_LAKE_PIT_EXPORT_URI", "").strip()
    expected_snapshot_id = os.environ.get(
        "FEATURE_LAKE_PIT_ICEBERG_SNAPSHOT_ID", ""
    ).strip()
    cutoff = os.environ.get("FEATURE_LAKE_MATERIALIZATION_CUTOFF", "").strip()
    if not run_id or not export_prefix or not cutoff:
        raise PitTrainingDatasetError("run ID, export URI, and cutoff are required")
    storage = ObjectStorage(config.object_storage_config)
    await storage.initialize()
    export_attempt = max(1, int(os.environ.get("FEATURE_LAKE_EXPORT_ATTEMPT", "1")))
    shard_prefix = (
        f"{export_prefix.rstrip('/')}/runs/{run_id}/attempts/"
        f"{export_attempt}/shards"
    )
    shard_objects = await storage.list_storage_uris(shard_prefix)
    shards = [uri for uri in shard_objects if _is_parquet_shard_uri(uri)]
    snapshot_id, iceberg_row_count, quarantine_row_count = await asyncio.to_thread(
        load_pinned_iceberg_run,
        catalog_uri=feature_lake.catalog_uri,
        warehouse_uri=feature_lake.warehouse_uri,
        namespace=feature_lake.namespace,
        storage=storage,
        run_id=run_id,
    )
    if expected_snapshot_id and expected_snapshot_id != snapshot_id:
        raise PitTrainingDatasetError(
            "configured Iceberg snapshot ID does not match the committed PIT table"
        )
    await PitManifestPublisher(storage).publish(
        shard_uris=shards,
        output_prefix=export_prefix,
        materialization_run_id=run_id,
        iceberg_table_id=f"{feature_lake.namespace}.ranking_training_pit",
        iceberg_snapshot_id=snapshot_id,
        feature_definition_version=feature_lake.feature_definition_version,
        label_definition_version=RANKING_LABEL_DEFINITION_VERSION,
        attribution_cutoff=float(cutoff),
        quarantine_row_count=quarantine_row_count,
        expected_iceberg_row_count=iceberg_row_count,
    )


def load_pinned_iceberg_run(
    *, catalog_uri: str, warehouse_uri: str, namespace: str, storage: Any, run_id: str
) -> tuple[str, int, int]:
    from pyiceberg.catalog import load_catalog
    from pyiceberg.expressions import EqualTo

    config = storage.config
    catalog = load_catalog(
        "pit_manifest",
        type="rest",
        uri=catalog_uri,
        warehouse=warehouse_uri,
        **{
            "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
            "s3.endpoint": config.endpoint_url,
            "s3.access-key-id": config.access_key_id,
            "s3.secret-access-key": config.secret_access_key,
            "s3.region": config.region,
        },
    )
    table = catalog.load_table(f"{namespace}.ranking_training_pit")
    snapshot = table.current_snapshot()
    if snapshot is None:
        return "", 0, 0
    rows = table.scan(
        row_filter=EqualTo("materialization_run_id", run_id),
        selected_fields=("observation_id",),
        snapshot_id=snapshot.snapshot_id,
    ).to_arrow()
    quarantine = catalog.load_table(f"{namespace}.ranking_training_pit_quarantine")
    quarantine_snapshot = quarantine.current_snapshot()
    quarantine_rows = 0
    if quarantine_snapshot is not None:
        quarantine_rows = (
            quarantine.scan(
                row_filter=EqualTo("materialization_run_id", run_id),
                selected_fields=("observation_id",),
                snapshot_id=quarantine_snapshot.snapshot_id,
            )
            .to_arrow()
            .num_rows
        )
    return str(snapshot.snapshot_id), rows.num_rows, quarantine_rows


_load_pinned_iceberg_run = load_pinned_iceberg_run


if __name__ == "__main__":
    asyncio.run(_main())
