"""Immutable, checksummed content-feature artifacts for PIT-safe training."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from video_commerce.common.models import ContentFeatures
from video_commerce.data_plane.object_storage import ObjectStorage


CONTENT_ARTIFACT_SCHEMA_VERSION = "content_feature_artifact_v2"


def content_artifact_reference(features: ContentFeatures) -> dict[str, Any] | None:
    """Return the immutable lineage fields safe to persist in PIT examples."""
    if not all(
        (
            features.artifact_uri,
            features.artifact_sha256,
            features.artifact_schema_version,
            features.artifact_created_at is not None,
        )
    ):
        return None
    return {
        "content_id": features.content_id,
        "uri": features.artifact_uri,
        "sha256": features.artifact_sha256,
        "schema_version": features.artifact_schema_version,
        "created_at": float(features.artifact_created_at),
    }


async def load_content_feature_artifact(
    storage: ObjectStorage,
    reference: dict[str, Any],
) -> ContentFeatures:
    """Load exactly the artifact named by a PIT reference and verify its lineage."""
    uri = str(reference.get("uri") or "").strip()
    expected_sha256 = str(reference.get("sha256") or "").strip()
    expected_schema = str(reference.get("schema_version") or "").strip()
    expected_content_id = str(reference.get("content_id") or "").strip()
    if (
        not uri
        or len(expected_sha256) != 64
        or expected_schema != CONTENT_ARTIFACT_SCHEMA_VERSION
        or not expected_content_id
    ):
        raise ValueError("content artifact reference is incomplete or incompatible")
    path, should_delete = await storage.materialize_for_processing(
        uri, suggested_suffix=".json"
    )
    try:
        with open(path, "rb") as handle:
            payload_bytes = handle.read()
    finally:
        if should_delete and os.path.exists(path):
            os.remove(path)
    if hashlib.sha256(payload_bytes).hexdigest() != expected_sha256:
        raise ValueError("content artifact checksum mismatch")
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("content artifact is not valid JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != expected_schema
        or not isinstance(payload.get("features"), dict)
    ):
        raise ValueError("content artifact schema mismatch")
    features = ContentFeatures(**payload["features"])
    if features.content_id != expected_content_id:
        raise ValueError("content artifact content_id mismatch")
    features.artifact_uri = uri
    features.artifact_sha256 = expected_sha256
    features.artifact_schema_version = expected_schema
    features.artifact_created_at = float(
        reference.get("created_at")
        if reference.get("created_at") is not None
        else payload.get("created_at")
    )
    return features


def canonical_content_feature_bytes(features: ContentFeatures) -> bytes:
    payload_features = features.dict()
    for field in (
        "artifact_uri",
        "artifact_sha256",
        "artifact_schema_version",
        "artifact_created_at",
    ):
        payload_features[field] = None
    payload: dict[str, Any] = {
        "schema_version": CONTENT_ARTIFACT_SCHEMA_VERSION,
        "created_at": float(features.created_at),
        "features": payload_features,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


async def publish_content_feature_artifact(
    storage: ObjectStorage,
    features: ContentFeatures,
) -> ContentFeatures:
    payload = canonical_content_feature_bytes(features)
    digest = hashlib.sha256(payload).hexdigest()
    object_name = storage.build_content_feature_object_name(
        content_id=features.content_id,
        sha256=digest,
    )
    uri = await storage.persist_immutable_bytes(
        payload,
        object_name=object_name,
        content_type="application/json",
    )
    published = features.copy(deep=True)
    published.artifact_uri = uri
    published.artifact_sha256 = digest
    published.artifact_schema_version = CONTENT_ARTIFACT_SCHEMA_VERSION
    published.artifact_created_at = float(features.created_at)
    return published


async def persist_content_features(
    features: ContentFeatures,
    *,
    object_storage: ObjectStorage | None,
    system_store: Any,
    feature_store: Any,
) -> ContentFeatures:
    """Publish immutable bytes before advancing durable and online pointers."""
    published = (
        await publish_content_feature_artifact(object_storage, features)
        if object_storage is not None
        else features
    )
    if system_store is not None:
        await system_store.upsert_content_feature_artifact(
            published.content_id,
            published.dict(),
            schema_version=published.multimodal_schema_version,
        )
    await feature_store.store_content_features(published.content_id, published)
    return published
