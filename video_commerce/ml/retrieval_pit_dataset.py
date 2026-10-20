"""Immutable point-in-time dataset contract for collaborative retrieval.

Catalog semantics: a dataset pins ONE catalog generation, which is a static
item snapshot (membership, eligibility, metadata, CLIP embeddings). It is not
an as-of-query catalog, so the reader does not require it to predate every
row. Doing so rejected every query older than the pinned generation. The
reader requires the snapshot to be available by ``max_as_of_ts``. The
training adapters (``retrieval_training``) additionally require it to be
available by the holdout start. Holdout queries can then never see item
state from after they happened, while training rows share the
serving-like snapshot.

``ranker_score_coverage`` is the share of ``ranker_rejected`` rows that
carry a real ``ranker_score``. The publisher records it in the manifest and
the reader verifies it. ``teacher_soft`` training is refused below a
configured threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from video_commerce.ml.pit_streaming import (
    PitSpoolBuilder,
    VerifiedPitSource,
    iter_arrow_batch_rows,
)


RETRIEVAL_PIT_SCHEMA_VERSION = "retrieval_training_pit_v1"
RETRIEVAL_CATALOG_SCHEMA_VERSION = "retrieval_catalog_generation_v1"
RETRIEVAL_ITEM_FEATURE_SCHEMA_VERSION = "retrieval_item_feature_sidecar_v1"
RETRIEVAL_LABEL_POLICY_VERSION = "retrieval_label_v1"
RETRIEVAL_ELIGIBILITY_POLICY_VERSION = "retrieval_eligibility_v1"
RANKER_REJECTED = "ranker_rejected"


class RetrievalPitDatasetError(ValueError):
    """Raised when a retrieval dataset cannot be trusted for training."""


class RetrievalPitDatasetUnavailable(RetrievalPitDatasetError):
    """Raised when a complete published retrieval manifest is unavailable."""


@dataclass(frozen=True)
class RetrievalCatalogProduct:
    product_id: str
    active: bool
    in_stock: bool
    metadata: Mapping[str, Any]
    modality_presence: Mapping[str, bool]

    @property
    def eligible(self) -> bool:
        return bool(self.product_id and self.active and self.in_stock)


@dataclass(frozen=True)
class RetrievalItemFeature:
    product_id: str
    clip_embedding: tuple[float, ...] | None


@dataclass(frozen=True)
class RetrievalCatalog:
    generation_id: str
    effective_at: float
    available_at: float
    eligibility_policy_version: str
    products: Mapping[str, RetrievalCatalogProduct]
    item_features: Mapping[str, RetrievalItemFeature]
    item_features_available_at: float
    embedding_dimension: int
    embedding_model_revision: str

    @property
    def eligible_products(self) -> tuple[RetrievalCatalogProduct, ...]:
        return tuple(
            product for _, product in sorted(self.products.items()) if product.eligible
        )


@dataclass(frozen=True)
class RetrievalPitManifest:
    materialization_run_id: str
    dataset_version: str
    attribution_cutoff: float
    attribution_window_hours: int
    allowed_lateness_hours: int
    min_as_of_ts: float
    max_as_of_ts: float
    row_count: int
    label_policy_version: str
    eligibility_policy_version: str
    catalog_generation_id: str
    catalog_uri: str
    catalog_sha256: str
    manifest_uri: str
    manifest_sha256: str
    ranker_rejected_rows: int | None = None
    ranker_scored_rows: int | None = None

    @property
    def ranker_score_coverage(self) -> float | None:
        """Non-null ``ranker_score`` share of ranker-rejected rows."""
        return _coverage(self.ranker_rejected_rows, self.ranker_scored_rows)


@dataclass(frozen=True)
class RetrievalPitRow:
    query_id: str
    user_id: str
    product_id: str
    label_source: str
    label_type: str
    label_weight: float
    as_of_ts: float
    label_event_time: float
    label_available_at: float
    feature_event_time: float
    feature_available_at: float
    catalog_generation_id: str
    user_features: Mapping[str, Any]
    seen_product_ids: tuple[str, ...]
    ranker_score: float | None = None

    @classmethod
    def for_test(
        cls, *, query_id: str, user_id: str, as_of_ts: float
    ) -> "RetrievalPitRow":
        return cls(
            query_id=query_id,
            user_id=user_id,
            product_id="p1",
            label_source="organic_positive",
            label_type="click",
            label_weight=2.0,
            as_of_ts=float(as_of_ts),
            label_event_time=float(as_of_ts),
            label_available_at=float(as_of_ts),
            feature_event_time=float(as_of_ts),
            feature_available_at=float(as_of_ts),
            catalog_generation_id="catalog-test",
            user_features={},
            seen_product_ids=(),
        )


@dataclass(frozen=True)
class RetrievalPitDataset:
    manifest: RetrievalPitManifest
    catalog: RetrievalCatalog
    rows: tuple[RetrievalPitRow, ...]


@dataclass(frozen=True)
class RetrievalHoldoutWindow:
    start_ts: float
    end_ts: float


class RetrievalTrainingSource:
    """Re-iterable retrieval rows backed by a verified disk spool."""

    def __init__(
        self,
        source: VerifiedPitSource,
        *,
        manifest: RetrievalPitManifest,
        catalog: RetrievalCatalog,
        raw_manifest: Mapping[str, Any],
    ) -> None:
        self.source = source
        self.manifest = manifest
        self.catalog = catalog
        self.raw_manifest = dict(raw_manifest)

    @property
    def path(self) -> Path:
        return self.source.path

    @property
    def work_dir(self) -> Path:
        return self.source.path.parent.parent / "work"

    @property
    def row_count(self) -> int:
        return self.source.row_count

    @property
    def group_count(self) -> int:
        return self.source.group_count

    @property
    def lineage(self) -> Mapping[str, Any]:
        return dict(self.source.metadata.get("lineage") or {})

    def assign_holdout(self, *, holdout_days: int) -> RetrievalHoldoutWindow:
        self.source.assign_ranking_splits(
            holdout_days=holdout_days,
            attribution_cutoff=self.manifest.attribution_cutoff,
            validation_fraction=0.0,
        )
        end_ts = self.manifest.max_as_of_ts
        return RetrievalHoldoutWindow(
            start_ts=end_ts - int(holdout_days) * 86400.0,
            end_ts=end_ts,
        )

    def iter_row_batches(
        self,
        *,
        split: str,
        batch_rows: int,
        seed: int,
        epoch: int = 0,
    ):
        for rows in self.source.iter_group_batches(
            split=split,
            batch_rows=batch_rows,
            seed=seed,
            epoch=epoch,
        ):
            yield [
                RetrievalPitDatasetReader._parse_row(
                    row, self.raw_manifest, self.catalog
                )
                for row in rows
            ]


class RetrievalPitDatasetReader:
    """Read latest -> manifest -> checksummed shards and catalog generation."""

    def __init__(self, object_storage: Any) -> None:
        self.object_storage = object_storage

    async def read(self, latest_pointer_uri: str) -> RetrievalPitDataset:
        latest = await self._read_json(latest_pointer_uri, "latest pointer")
        manifest_uri = str(latest.get("manifest_uri") or "")
        if not manifest_uri:
            raise RetrievalPitDatasetUnavailable(
                "retrieval latest pointer is missing manifest_uri"
            )
        manifest_bytes = await self._read_bytes(manifest_uri)
        try:
            raw_manifest = json.loads(manifest_bytes)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RetrievalPitDatasetError(
                "retrieval manifest is invalid JSON"
            ) from exc
        self._validate_manifest(raw_manifest, latest)

        catalog_ref = dict(raw_manifest["catalog_generation"])
        catalog_bytes = await self._read_bytes(str(catalog_ref["uri"]))
        if _sha256(catalog_bytes) != str(catalog_ref["sha256"]):
            raise RetrievalPitDatasetError("retrieval catalog checksum mismatch")
        catalog = self._parse_catalog(catalog_bytes, raw_manifest, catalog_ref)
        sidecar_ref = dict(raw_manifest["item_feature_sidecar"])
        sidecar_bytes = await self._read_bytes(str(sidecar_ref["uri"]))
        if _sha256(sidecar_bytes) != str(sidecar_ref["sha256"]):
            raise RetrievalPitDatasetError("retrieval item feature checksum mismatch")
        catalog = self._attach_item_features(catalog, sidecar_bytes, sidecar_ref)

        rows: list[RetrievalPitRow] = []
        coverage = _RankerScoreCoverage()
        for shard_ref in raw_manifest["shards"]:
            shard_uri = str(shard_ref.get("uri") or "")
            local_path, cleanup = await self.object_storage.materialize_for_processing(
                shard_uri, suggested_suffix=".parquet"
            )
            try:
                shard_bytes = Path(local_path).read_bytes()
                if len(shard_bytes) != int(shard_ref.get("byte_size", -1)):
                    raise RetrievalPitDatasetError(
                        "retrieval shard byte size does not match manifest"
                    )
                if _sha256(shard_bytes) != str(shard_ref.get("sha256") or ""):
                    raise RetrievalPitDatasetError("retrieval shard checksum mismatch")
                for raw in pq.read_table(local_path).to_pylist():
                    row = self._parse_row(raw, raw_manifest, catalog)
                    coverage.add(row)
                    rows.append(row)
            finally:
                if cleanup:
                    Path(local_path).unlink(missing_ok=True)

        if len(rows) != int(raw_manifest["row_count"]):
            raise RetrievalPitDatasetError(
                "retrieval row count does not match manifest"
            )
        coverage.verify(raw_manifest)
        parsed_manifest = self._parsed_manifest(
            raw_manifest,
            catalog_ref,
            manifest_uri,
            _sha256(manifest_bytes),
            coverage=coverage,
        )
        return RetrievalPitDataset(parsed_manifest, catalog, tuple(rows))

    async def validate_and_stage(
        self,
        latest_pointer_uri: str,
        *,
        staging_dir: str | Path,
        batch_rows: int,
        max_group_rows: int,
        max_bytes: int,
    ) -> RetrievalTrainingSource:
        latest = await self._read_json(latest_pointer_uri, "latest pointer")
        manifest_uri = str(latest.get("manifest_uri") or "")
        if not manifest_uri:
            raise RetrievalPitDatasetUnavailable(
                "retrieval latest pointer is missing manifest_uri"
            )
        manifest_bytes = await self._read_bytes(manifest_uri)
        try:
            raw_manifest = json.loads(manifest_bytes)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RetrievalPitDatasetError(
                "retrieval manifest is invalid JSON"
            ) from exc
        self._validate_manifest(raw_manifest, latest)
        catalog_ref = dict(raw_manifest["catalog_generation"])
        catalog_bytes = await self._read_bytes(str(catalog_ref["uri"]))
        if _sha256(catalog_bytes) != str(catalog_ref["sha256"]):
            raise RetrievalPitDatasetError("retrieval catalog checksum mismatch")
        catalog = self._parse_catalog(catalog_bytes, raw_manifest, catalog_ref)
        sidecar_ref = dict(raw_manifest["item_feature_sidecar"])
        sidecar_bytes = await self._read_bytes(str(sidecar_ref["uri"]))
        if _sha256(sidecar_bytes) != str(sidecar_ref["sha256"]):
            raise RetrievalPitDatasetError("retrieval item feature checksum mismatch")
        catalog = self._attach_item_features(catalog, sidecar_bytes, sidecar_ref)
        manifest_sha = _sha256(manifest_bytes)
        parsed_manifest = self._parsed_manifest(
            raw_manifest, catalog_ref, manifest_uri, manifest_sha
        )
        coverage = _RankerScoreCoverage()
        lineage = {
            "manifest_uri": manifest_uri,
            "manifest_sha256": manifest_sha,
            "materialization_run_id": parsed_manifest.materialization_run_id,
            "dataset_version": parsed_manifest.dataset_version,
            "catalog_generation_id": parsed_manifest.catalog_generation_id,
            "catalog_manifest_sha256": parsed_manifest.catalog_sha256,
            "attribution_cutoff": parsed_manifest.attribution_cutoff,
            "min_as_of_ts": parsed_manifest.min_as_of_ts,
            "max_as_of_ts": parsed_manifest.max_as_of_ts,
            "pit_reader_mode": "streaming",
            "pit_stream_contract_version": "pit_stream_v1",
        }
        builder = PitSpoolBuilder(
            staging_dir,
            generation_id=f"retrieval-{manifest_sha}",
            kind="retrieval",
            max_group_rows=max_group_rows,
            max_bytes=max_bytes,
            lineage=lineage,
        )
        seen_rows = 0
        pinned_schema = None
        try:
            for shard_ref in raw_manifest["shards"]:
                uri = str(shard_ref.get("uri") or "")
                (
                    local_path,
                    cleanup,
                ) = await self.object_storage.materialize_for_processing(
                    uri, suggested_suffix=".parquet"
                )
                try:
                    if os.path.getsize(local_path) != int(
                        shard_ref.get("byte_size", -1)
                    ):
                        raise RetrievalPitDatasetError(
                            "retrieval shard byte size does not match manifest"
                        )
                    if _sha256_path(local_path) != str(shard_ref.get("sha256") or ""):
                        raise RetrievalPitDatasetError(
                            "retrieval shard checksum mismatch"
                        )
                    parquet = pq.ParquetFile(local_path)
                    schema = parquet.schema_arrow
                    expected_schema = str(raw_manifest.get("schema_hash") or "")
                    if (
                        expected_schema
                        and _arrow_schema_sha256(schema) != expected_schema
                    ):
                        raise RetrievalPitDatasetError(
                            "retrieval shard schema mismatch"
                        )
                    if pinned_schema is None:
                        pinned_schema = schema
                    elif pinned_schema != schema:
                        raise RetrievalPitDatasetError(
                            "retrieval shard schema mismatch"
                        )
                    for batch in parquet.iter_batches(
                        batch_size=max(1, int(batch_rows))
                    ):
                        spool_rows = []
                        for raw in iter_arrow_batch_rows(batch):
                            row = self._parse_row(raw, raw_manifest, catalog)
                            coverage.add(row)
                            if not row.query_id or not row.user_id:
                                raise RetrievalPitDatasetError(
                                    "retrieval row query and user identity are required"
                                )
                            spool_rows.append(
                                {
                                    "group_id": row.query_id,
                                    "row_id": (
                                        f"{row.product_id}:{row.label_source}:"
                                        f"{row.label_type}:{seen_rows + len(spool_rows)}"
                                    ),
                                    "as_of_ts": row.as_of_ts,
                                    "payload": raw,
                                }
                            )
                        builder.add_rows(spool_rows)
                        seen_rows += len(spool_rows)
                finally:
                    if cleanup:
                        Path(local_path).unlink(missing_ok=True)
            coverage.verify(raw_manifest)
            parsed_manifest = replace(
                parsed_manifest,
                ranker_rejected_rows=coverage.rejected_rows,
                ranker_scored_rows=coverage.scored_rows,
            )
            builder.lineage["ranker_score_coverage"] = coverage.as_manifest()
            verified = builder.finalize(expected_rows=int(raw_manifest["row_count"]))
            return RetrievalTrainingSource(
                verified,
                manifest=parsed_manifest,
                catalog=catalog,
                raw_manifest=raw_manifest,
            )
        except Exception:
            builder.abort()
            raise

    @staticmethod
    def _parsed_manifest(
        raw_manifest, catalog_ref, manifest_uri, manifest_sha, *, coverage=None
    ):
        return RetrievalPitManifest(
            materialization_run_id=str(raw_manifest["materialization_run_id"]),
            dataset_version=str(raw_manifest["dataset_version"]),
            attribution_cutoff=float(raw_manifest["attribution_cutoff"]),
            attribution_window_hours=int(raw_manifest["attribution_window_hours"]),
            allowed_lateness_hours=int(raw_manifest["allowed_lateness_hours"]),
            min_as_of_ts=float(raw_manifest["min_as_of_ts"]),
            max_as_of_ts=float(raw_manifest["max_as_of_ts"]),
            row_count=int(raw_manifest["row_count"]),
            label_policy_version=str(raw_manifest["label_policy_version"]),
            eligibility_policy_version=str(raw_manifest["eligibility_policy_version"]),
            catalog_generation_id=str(catalog_ref["generation_id"]),
            catalog_uri=str(catalog_ref["uri"]),
            catalog_sha256=str(catalog_ref["sha256"]),
            manifest_uri=manifest_uri,
            manifest_sha256=manifest_sha,
            ranker_rejected_rows=(
                coverage.rejected_rows if coverage is not None else None
            ),
            ranker_scored_rows=coverage.scored_rows if coverage is not None else None,
        )

    async def _read_json(self, uri: str, kind: str) -> Mapping[str, Any]:
        try:
            raw = await self._read_bytes(uri)
        except (FileNotFoundError, OSError) as exc:
            raise RetrievalPitDatasetUnavailable(
                f"retrieval {kind} is unavailable"
            ) from exc
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RetrievalPitDatasetError(f"retrieval {kind} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise RetrievalPitDatasetError(f"retrieval {kind} must be an object")
        return value

    async def _read_bytes(self, uri: str) -> bytes:
        local_path, cleanup = await self.object_storage.materialize_for_processing(uri)
        try:
            return Path(local_path).read_bytes()
        finally:
            if cleanup:
                Path(local_path).unlink(missing_ok=True)

    @staticmethod
    def _validate_manifest(raw: Mapping[str, Any], latest: Mapping[str, Any]) -> None:
        if raw.get("status") != "complete":
            raise RetrievalPitDatasetUnavailable("retrieval manifest is not complete")
        if raw.get("schema_version") != RETRIEVAL_PIT_SCHEMA_VERSION:
            raise RetrievalPitDatasetError("retrieval manifest schema is incompatible")
        if raw.get("label_policy_version") != RETRIEVAL_LABEL_POLICY_VERSION:
            raise RetrievalPitDatasetError("retrieval label policy is incompatible")
        if (
            raw.get("eligibility_policy_version")
            != RETRIEVAL_ELIGIBILITY_POLICY_VERSION
        ):
            raise RetrievalPitDatasetError(
                "retrieval eligibility policy is incompatible"
            )
        if latest.get("materialization_run_id") != raw.get("materialization_run_id"):
            raise RetrievalPitDatasetError("retrieval latest pointer run mismatch")
        if not raw.get("shards") or not isinstance(raw.get("shards"), list):
            raise RetrievalPitDatasetError("retrieval manifest has no shards")
        if not isinstance(raw.get("catalog_generation"), dict):
            raise RetrievalPitDatasetError(
                "retrieval manifest has no catalog generation"
            )
        if not isinstance(raw.get("item_feature_sidecar"), dict):
            raise RetrievalPitDatasetError(
                "retrieval manifest has no item feature sidecar"
            )

    @staticmethod
    def _parse_catalog(
        raw_bytes: bytes,
        manifest: Mapping[str, Any],
        catalog_ref: Mapping[str, Any],
    ) -> RetrievalCatalog:
        try:
            raw = json.loads(raw_bytes)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RetrievalPitDatasetError("retrieval catalog is invalid JSON") from exc
        if raw.get("schema_version") != RETRIEVAL_CATALOG_SCHEMA_VERSION:
            raise RetrievalPitDatasetError("retrieval catalog schema is incompatible")
        generation_id = str(raw.get("generation_id") or "")
        if generation_id != str(catalog_ref.get("generation_id") or ""):
            raise RetrievalPitDatasetError("retrieval catalog generation mismatch")
        if raw.get("eligibility_policy_version") != manifest.get(
            "eligibility_policy_version"
        ):
            raise RetrievalPitDatasetError("retrieval catalog eligibility mismatch")
        products: dict[str, RetrievalCatalogProduct] = {}
        for item in raw.get("products") or []:
            product_id = str(item.get("product_id") or "")
            if not product_id or product_id in products:
                raise RetrievalPitDatasetError(
                    "retrieval catalog has blank or duplicate product ID"
                )
            products[product_id] = RetrievalCatalogProduct(
                product_id=product_id,
                active=bool(item.get("active", True)),
                in_stock=bool(item.get("in_stock", True)),
                metadata=dict(item.get("metadata") or {}),
                modality_presence={
                    str(key): bool(value)
                    for key, value in dict(item.get("modality_presence") or {}).items()
                },
            )
        if len(products) != int(catalog_ref.get("product_count", -1)):
            raise RetrievalPitDatasetError(
                "retrieval catalog product count does not match manifest"
            )
        available_at = _finite_float(raw.get("available_at"), "catalog available_at")
        # Dataset-level, not per-row: see the module docstring.
        if available_at > float(manifest["max_as_of_ts"]):
            raise RetrievalPitDatasetError("retrieval catalog is future data")
        return RetrievalCatalog(
            generation_id=generation_id,
            effective_at=_finite_float(raw.get("effective_at"), "catalog effective_at"),
            available_at=available_at,
            eligibility_policy_version=str(raw["eligibility_policy_version"]),
            products=products,
            item_features={},
            item_features_available_at=0.0,
            embedding_dimension=0,
            embedding_model_revision="",
        )

    @staticmethod
    def _attach_item_features(
        catalog: RetrievalCatalog,
        raw_bytes: bytes,
        reference: Mapping[str, Any],
    ) -> RetrievalCatalog:
        try:
            raw = json.loads(raw_bytes)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RetrievalPitDatasetError(
                "retrieval item feature sidecar is invalid JSON"
            ) from exc
        if raw.get("schema_version") != RETRIEVAL_ITEM_FEATURE_SCHEMA_VERSION:
            raise RetrievalPitDatasetError(
                "retrieval item feature sidecar schema is incompatible"
            )
        if raw.get("schema_version") != reference.get("schema_version"):
            raise RetrievalPitDatasetError(
                "retrieval item feature sidecar schema does not match manifest"
            )
        if str(raw.get("generation_id") or "") != catalog.generation_id:
            raise RetrievalPitDatasetError(
                "retrieval item feature catalog generation mismatch"
            )
        embedding_dimension = int(raw.get("embedding_dimension") or 0)
        if embedding_dimension <= 0 or embedding_dimension != int(
            reference.get("embedding_dimension") or 0
        ):
            raise RetrievalPitDatasetError(
                "retrieval item feature embedding dimension mismatch"
            )
        revision = str(raw.get("embedding_model_revision") or "")
        if not revision or revision != str(
            reference.get("embedding_model_revision") or ""
        ):
            raise RetrievalPitDatasetError(
                "retrieval item feature model revision mismatch"
            )
        available_at = _finite_float(
            raw.get("available_at"), "item feature available_at"
        )
        if available_at != _finite_float(
            reference.get("available_at"), "item feature manifest available_at"
        ):
            raise RetrievalPitDatasetError(
                "retrieval item feature available time mismatch"
            )
        features: dict[str, RetrievalItemFeature] = {}
        for item in raw.get("items") or []:
            product_id = str(item.get("product_id") or "")
            if product_id not in catalog.products or product_id in features:
                raise RetrievalPitDatasetError(
                    "retrieval item feature product membership mismatch"
                )
            raw_embedding = item.get("clip_embedding")
            embedding = None
            if raw_embedding is not None:
                if (
                    not isinstance(raw_embedding, list)
                    or len(raw_embedding) != embedding_dimension
                ):
                    raise RetrievalPitDatasetError(
                        "retrieval item feature embedding dimension mismatch"
                    )
                embedding = tuple(
                    _finite_float(value, "item feature embedding")
                    for value in raw_embedding
                )
            features[product_id] = RetrievalItemFeature(product_id, embedding)
        if set(features) != set(catalog.products):
            raise RetrievalPitDatasetError(
                "retrieval item feature sidecar does not cover catalog membership"
            )
        return RetrievalCatalog(
            generation_id=catalog.generation_id,
            effective_at=catalog.effective_at,
            available_at=catalog.available_at,
            eligibility_policy_version=catalog.eligibility_policy_version,
            products=catalog.products,
            item_features=features,
            item_features_available_at=available_at,
            embedding_dimension=embedding_dimension,
            embedding_model_revision=revision,
        )

    @staticmethod
    def _parse_row(
        raw: Mapping[str, Any],
        manifest: Mapping[str, Any],
        catalog: RetrievalCatalog,
    ) -> RetrievalPitRow:
        as_of_ts = _finite_float(raw.get("as_of_ts"), "as_of_ts")
        feature_event_time = _finite_float(
            raw.get("feature_event_time"), "feature_event_time"
        )
        feature_available_at = _finite_float(
            raw.get("feature_available_at"), "feature_available_at"
        )
        label_event_time = _finite_float(
            raw.get("label_event_time"), "label_event_time"
        )
        label_available_at = _finite_float(
            raw.get("label_available_at"), "label_available_at"
        )
        if feature_event_time > as_of_ts:
            raise RetrievalPitDatasetError(
                "retrieval feature event time leaks future data"
            )
        if feature_available_at > as_of_ts:
            raise RetrievalPitDatasetError(
                "retrieval feature available time leaks future data"
            )
        if label_event_time < as_of_ts:
            raise RetrievalPitDatasetError("retrieval label precedes query")
        if label_available_at > float(manifest["attribution_cutoff"]):
            raise RetrievalPitDatasetError(
                "retrieval label availability exceeds attribution cutoff"
            )
        if (
            label_event_time
            > as_of_ts + int(manifest["attribution_window_hours"]) * 3600.0
        ):
            raise RetrievalPitDatasetError("retrieval label exceeds attribution window")
        maturity_time = (
            as_of_ts
            + (
                int(manifest["attribution_window_hours"])
                + int(manifest["allowed_lateness_hours"])
            )
            * 3600.0
        )
        if maturity_time > float(manifest["attribution_cutoff"]):
            raise RetrievalPitDatasetError("retrieval query is not mature")
        generation_id = str(raw.get("catalog_generation_id") or "")
        if generation_id != catalog.generation_id:
            raise RetrievalPitDatasetError("retrieval catalog generation mismatch")
        product_id = str(raw.get("product_id") or "")
        if product_id not in catalog.products:
            raise RetrievalPitDatasetError(
                "retrieval row product is absent from catalog"
            )
        try:
            user_features = json.loads(str(raw.get("user_features_json") or "{}"))
            seen = json.loads(str(raw.get("seen_product_ids_json") or "[]"))
        except json.JSONDecodeError as exc:
            raise RetrievalPitDatasetError(
                "retrieval row JSON fields are invalid"
            ) from exc
        if not isinstance(user_features, dict) or not isinstance(seen, list):
            raise RetrievalPitDatasetError(
                "retrieval row JSON fields have invalid types"
            )
        ranker_score = raw.get("ranker_score")
        return RetrievalPitRow(
            query_id=str(raw.get("query_id") or ""),
            user_id=str(raw.get("user_id") or ""),
            product_id=product_id,
            label_source=str(raw.get("label_source") or ""),
            label_type=str(raw.get("label_type") or ""),
            label_weight=_finite_float(raw.get("label_weight"), "label_weight"),
            as_of_ts=as_of_ts,
            label_event_time=label_event_time,
            label_available_at=label_available_at,
            feature_event_time=feature_event_time,
            feature_available_at=feature_available_at,
            catalog_generation_id=generation_id,
            user_features=user_features,
            seen_product_ids=tuple(str(item) for item in seen),
            ranker_score=(
                None
                if ranker_score is None
                else _finite_float(ranker_score, "ranker_score")
            ),
        )


def split_retrieval_holdout(
    rows: Sequence[RetrievalPitRow],
    *,
    holdout_days: int,
    attribution_cutoff: float,
) -> tuple[list[RetrievalPitRow], list[RetrievalPitRow], RetrievalHoldoutWindow]:
    if not rows:
        raise ValueError("retrieval PIT dataset is empty")
    if holdout_days <= 0:
        raise ValueError("retrieval holdout days must be positive")
    end_ts = max(float(row.as_of_ts) for row in rows)
    if end_ts > float(attribution_cutoff):
        raise ValueError("retrieval holdout exceeds attribution cutoff")
    start_ts = end_ts - int(holdout_days) * 86400.0
    training = [row for row in rows if float(row.as_of_ts) < start_ts]
    holdout = [row for row in rows if start_ts <= float(row.as_of_ts) <= end_ts]
    return training, holdout, RetrievalHoldoutWindow(start_ts, end_ts)


def is_ranker_rejected(row: RetrievalPitRow) -> bool:
    return RANKER_REJECTED in (row.label_source, row.label_type)


class _RankerScoreCoverage:
    """Count ranker-rejected rows and how many carry a real ranker score."""

    def __init__(self) -> None:
        self.rejected_rows = 0
        self.scored_rows = 0

    def add(self, row: RetrievalPitRow) -> None:
        if is_ranker_rejected(row):
            self.rejected_rows += 1
            if row.ranker_score is not None:
                self.scored_rows += 1

    def as_manifest(self) -> dict[str, Any]:
        return ranker_score_coverage_manifest(self.rejected_rows, self.scored_rows)

    def verify(self, raw_manifest: Mapping[str, Any]) -> None:
        declared = raw_manifest.get("ranker_score_coverage")
        if declared is None:
            return
        if not isinstance(declared, Mapping) or (
            int(declared.get("ranker_rejected_rows", -1)) != self.rejected_rows
            or int(declared.get("ranker_scored_rows", -1)) != self.scored_rows
        ):
            raise RetrievalPitDatasetError(
                "retrieval ranker score coverage does not match manifest"
            )


def ranker_score_coverage_manifest(rejected_rows: int, scored_rows: int) -> dict:
    return {
        "ranker_rejected_rows": int(rejected_rows),
        "ranker_scored_rows": int(scored_rows),
        "coverage": _coverage(rejected_rows, scored_rows),
    }


def _coverage(rejected_rows: int | None, scored_rows: int | None) -> float | None:
    if not rejected_rows or scored_rows is None:
        return None
    return float(scored_rows) / float(rejected_rows)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrow_schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _finite_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RetrievalPitDatasetError(f"retrieval {field} is invalid") from exc
    if not math.isfinite(parsed):
        raise RetrievalPitDatasetError(f"retrieval {field} is not finite")
    return parsed
