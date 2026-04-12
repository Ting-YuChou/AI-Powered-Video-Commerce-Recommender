"""
Versioned cache serialization helpers.
"""

from __future__ import annotations

from typing import Any

import msgpack
import orjson


CACHE_SCHEMA_VERSION = 1


class CacheDecodeError(ValueError):
    """Raised when a cached payload cannot be decoded safely."""


def pack_cache_payload(kind: str, payload: Any) -> bytes:
    """Serialize a cache payload with an explicit schema envelope."""
    return msgpack.packb(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "kind": kind,
            "payload": payload,
        },
        use_bin_type=True,
    )


def unpack_cache_payload(raw: bytes, expected_kind: str | None = None) -> Any:
    """Deserialize a cache payload and validate the schema envelope."""
    try:
        decoded = msgpack.unpackb(raw, raw=False)
    except Exception as exc:
        raise CacheDecodeError(f"Invalid cache payload: {exc}") from exc

    if not isinstance(decoded, dict):
        raise CacheDecodeError("Cache payload must decode to an object")

    schema_version = decoded.get("schema_version")
    if schema_version != CACHE_SCHEMA_VERSION:
        raise CacheDecodeError(
            f"Unsupported cache schema version: {schema_version}"
        )

    kind = decoded.get("kind")
    if expected_kind and kind != expected_kind:
        raise CacheDecodeError(
            f"Unexpected cache payload kind: expected {expected_kind}, got {kind}"
        )

    return decoded.get("payload")


def json_dumps(payload: Any) -> bytes:
    """Serialize JSON payloads as UTF-8 bytes."""
    return orjson.dumps(payload)


def json_loads(raw: bytes | str) -> Any:
    """Deserialize JSON payloads from bytes or strings."""
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return orjson.loads(raw)
