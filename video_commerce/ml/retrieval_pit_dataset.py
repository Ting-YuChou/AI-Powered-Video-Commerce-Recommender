"""Immutable point-in-time dataset contract for collaborative retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq


RETRIEVAL_PIT_SCHEMA_VERSION = "retrieval_training_pit_v1"
RETRIEVAL_CATALOG_SCHEMA_VERSION = "retrieval_catalog_generation_v1"
RETRIEVAL_ITEM_FEATURE_SCHEMA_VERSION = "retrieval_item_feature_sidecar_v1"
RETRIEVAL_LABEL_POLICY_VERSION = "retrieval_label_v1"
RETRIEVAL_ELIGIBILITY_POLICY_VERSION = "retrieval_eligibility_v1"


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
                rows.extend(
                    self._parse_row(row, raw_manifest, catalog)
                    for row in pq.read_table(local_path).to_pylist()
                )
            finally:
                if cleanup:
                    Path(local_path).unlink(missing_ok=True)

        if len(rows) != int(raw_manifest["row_count"]):
            raise RetrievalPitDatasetError(
                "retrieval row count does not match manifest"
            )
        parsed_manifest = RetrievalPitManifest(
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
            manifest_sha256=_sha256(manifest_bytes),
        )
        return RetrievalPitDataset(parsed_manifest, catalog, tuple(rows))

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
        if catalog.available_at > as_of_ts:
            raise RetrievalPitDatasetError("retrieval catalog is future data")
        if catalog.item_features_available_at > as_of_ts:
            raise RetrievalPitDatasetError("retrieval item features are future data")
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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RetrievalPitDatasetError(f"retrieval {field} is invalid") from exc
    if not math.isfinite(parsed):
        raise RetrievalPitDatasetError(f"retrieval {field} is not finite")
    return parsed
