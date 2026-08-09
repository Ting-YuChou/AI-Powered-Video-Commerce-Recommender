"""Atomic, checksum-aware product CLIP index artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import faiss
import numpy as np
from PIL import Image, ImageEnhance


VISUAL_PRODUCT_INDEX_SCHEMA_VERSION = "visual_product_index_v1"


@dataclass
class VisualProductIndexBundle:
    index: Any
    product_index_map: dict[int, str]
    product_embeddings: dict[str, np.ndarray]
    product_metadata: dict[str, dict[str, Any]]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class VisualProductIndexBuildResult:
    manifest_path: Path
    indexed_product_ids: tuple[str, ...]
    quarantine: dict[str, str]


class FrozenCLIPProductImageEncoder:
    """Offline-only product image encoder pinned to the video CLIP lineage."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str = "cpu",
        cache_dir: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if not model_id or not revision:
            raise ValueError("frozen CLIP encoder requires model and revision")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.cache_dir = cache_dir
        self.batch_size = max(1, int(batch_size))
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._model = CLIPModel.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
        ).to(self.device)
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._processor = CLIPProcessor.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
        )

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        import torch

        self.load()
        batches = []
        for offset in range(0, len(images), self.batch_size):
            batch_images = [
                ImageEnhance.Sharpness(image).enhance(1.2)
                for image in images[offset : offset + self.batch_size]
            ]
            inputs = self._processor(
                images=batch_images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                values = self._model.get_image_features(**inputs)
                values = torch.nn.functional.normalize(values, dim=-1)
            batches.append(values.cpu().numpy().astype(np.float32))
        if not batches:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(batches)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_version(value: str) -> str:
    normalized = str(value or "").strip()
    if (
        not normalized
        or Path(normalized).name != normalized
        or normalized in {".", ".."}
    ):
        raise ValueError("visual product index model version is unsafe")
    return normalized


def _normalized_embedding(value: Any, *, dimension: int | None) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    if dimension is not None and embedding.shape != (dimension,):
        raise ValueError("visual product embeddings must share one dimension")
    if not np.isfinite(embedding).all():
        raise ValueError("visual product embedding must be finite")
    norm = float(np.linalg.norm(embedding))
    if norm <= 0.0:
        raise ValueError("visual product embedding must have non-zero norm")
    return (embedding / norm).astype(np.float32)


def publish_visual_product_index_bundle(
    output_dir: str | Path,
    *,
    product_embeddings: Mapping[str, Any],
    product_metadata: Mapping[str, Mapping[str, Any]],
    expected_product_count: int,
    clip_model_id: str,
    clip_revision: str,
    catalog_activation_id: str,
    model_version: str,
    catalog_available_at: float = 0.0,
    minimum_coverage: float = 0.95,
) -> Path:
    """Publish index artifacts first and the checksum manifest last."""
    if expected_product_count <= 0:
        raise ValueError("expected product count must be positive")
    if not 0.0 <= float(minimum_coverage) <= 1.0:
        raise ValueError("minimum coverage must be between zero and one")
    if not np.isfinite(float(catalog_available_at)) or catalog_available_at < 0:
        raise ValueError("catalog availability timestamp is invalid")
    product_ids = sorted(str(product_id) for product_id in product_embeddings)
    coverage = len(product_ids) / int(expected_product_count)
    if coverage < float(minimum_coverage):
        raise ValueError(
            f"visual product image coverage {coverage:.6f} is below "
            f"the required coverage {float(minimum_coverage):.6f}"
        )
    if not product_ids:
        raise ValueError("visual product index requires embeddings")
    if not all(
        str(value or "").strip()
        for value in (clip_model_id, clip_revision, catalog_activation_id)
    ):
        raise ValueError("visual product index lineage is incomplete")

    dimension = None
    embeddings = []
    for product_id in product_ids:
        value = np.asarray(product_embeddings[product_id]).reshape(-1)
        if dimension is None:
            dimension = int(value.shape[0])
        normalized = _normalized_embedding(value, dimension=dimension)
        embeddings.append(normalized)
    matrix = np.vstack(embeddings).astype(np.float32)
    index = faiss.IndexHNSWFlat(int(dimension), 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 50
    index.add(matrix)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    version = _safe_version(model_version)
    names = {
        "index": f"{version}.faiss",
        "embeddings": f"{version}.embeddings.npz",
        "metadata": f"{version}.metadata.json",
    }
    metadata_payload = {
        "schema_version": VISUAL_PRODUCT_INDEX_SCHEMA_VERSION,
        "model_version": version,
        "embedding_dim": int(dimension),
        "product_index_map": {
            str(index_value): product_id
            for index_value, product_id in enumerate(product_ids)
        },
        "product_metadata": {
            product_id: dict(product_metadata.get(product_id) or {})
            for product_id in product_ids
        },
    }

    temporary_dir = Path(tempfile.mkdtemp(prefix=".visual-index-", dir=target_dir))
    try:
        temporary_paths = {
            name: temporary_dir / filename for name, filename in names.items()
        }
        faiss.write_index(index, str(temporary_paths["index"]))
        with open(temporary_paths["embeddings"], "wb") as handle:
            np.savez_compressed(
                handle,
                product_ids=np.asarray(product_ids, dtype=np.str_),
                embeddings=matrix,
            )
            handle.flush()
            os.fsync(handle.fileno())
        with open(temporary_paths["metadata"], "w", encoding="utf-8") as handle:
            json.dump(
                metadata_payload,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())

        artifacts = {}
        for name, filename in names.items():
            source = temporary_paths[name]
            target = target_dir / filename
            os.replace(source, target)
            artifacts[name] = {"path": filename, "sha256": _sha256(target)}
        manifest = {
            "schema_version": VISUAL_PRODUCT_INDEX_SCHEMA_VERSION,
            "model_version": version,
            "clip_model_id": str(clip_model_id),
            "clip_revision": str(clip_revision),
            "catalog_activation_id": str(catalog_activation_id),
            "catalog_available_at": float(catalog_available_at),
            "expected_product_count": int(expected_product_count),
            "indexed_product_count": len(product_ids),
            "coverage": coverage,
            "embedding_dim": int(dimension),
            "index_type": "hnsw_inner_product",
            "artifacts": artifacts,
        }
        manifest_path = target_dir / f"{version}.manifest.json"
        descriptor, temporary_manifest = tempfile.mkstemp(
            prefix=f".{version}.manifest.",
            suffix=".tmp",
            dir=target_dir,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    manifest,
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_manifest, manifest_path)
        finally:
            if os.path.exists(temporary_manifest):
                os.remove(temporary_manifest)
        return manifest_path
    finally:
        for path in temporary_dir.iterdir():
            path.unlink(missing_ok=True)
        temporary_dir.rmdir()


def load_visual_product_index_bundle(
    manifest_path: str | Path,
    *,
    expected_clip_model_id: str | None = None,
    expected_clip_revision: str | None = None,
    expected_catalog_activation_id: str | None = None,
) -> VisualProductIndexBundle:
    source = Path(manifest_path)
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("visual product index manifest is unreadable") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != VISUAL_PRODUCT_INDEX_SCHEMA_VERSION
    ):
        raise ValueError("visual product index manifest schema mismatch")
    try:
        catalog_available_at = float(manifest["catalog_available_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "visual product index catalog availability is missing"
        ) from exc
    if not np.isfinite(catalog_available_at) or catalog_available_at < 0:
        raise ValueError("visual product index catalog availability is invalid")
    expected_values = {
        "clip_model_id": expected_clip_model_id,
        "clip_revision": expected_clip_revision,
        "catalog_activation_id": expected_catalog_activation_id,
    }
    for key, expected in expected_values.items():
        if expected is not None and manifest.get(key) != expected:
            raise ValueError(f"visual product index {key} mismatch")

    resolved = {}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("visual product index artifact manifest is incomplete")
    for name in ("index", "embeddings", "metadata"):
        reference = artifacts.get(name)
        if not isinstance(reference, dict):
            raise ValueError("visual product index artifact manifest is incomplete")
        path_value = str(reference.get("path") or "")
        expected_sha256 = str(reference.get("sha256") or "")
        path = source.parent / path_value
        if (
            not path_value
            or Path(path_value).name != path_value
            or len(expected_sha256) != 64
            or not path.exists()
        ):
            raise ValueError("visual product index artifact reference is incomplete")
        if _sha256(path) != expected_sha256:
            raise ValueError(f"visual product index {name} checksum mismatch")
        resolved[name] = path

    try:
        metadata = json.loads(resolved["metadata"].read_text(encoding="utf-8"))
        with np.load(resolved["embeddings"], allow_pickle=False) as payload:
            product_ids = [str(value) for value in payload["product_ids"].tolist()]
            embedding_matrix = payload["embeddings"].astype(np.float32, copy=True)
        index = faiss.read_index(str(resolved["index"]))
    except Exception as exc:
        raise ValueError("visual product index artifact payload is invalid") from exc
    product_index_map = {
        int(index_value): str(product_id)
        for index_value, product_id in (metadata.get("product_index_map") or {}).items()
    }
    if (
        embedding_matrix.ndim != 2
        or len(product_ids) != embedding_matrix.shape[0]
        or index.ntotal != len(product_ids)
        or len(product_index_map) != len(product_ids)
        or int(index.d) != embedding_matrix.shape[1]
        or int(manifest.get("embedding_dim", -1)) != embedding_matrix.shape[1]
    ):
        raise ValueError("visual product index artifact dimensions are inconsistent")
    return VisualProductIndexBundle(
        index=index,
        product_index_map=product_index_map,
        product_embeddings={
            product_id: embedding_matrix[row]
            for row, product_id in enumerate(product_ids)
        },
        product_metadata={
            str(product_id): dict(value or {})
            for product_id, value in (metadata.get("product_metadata") or {}).items()
        },
        manifest=manifest,
    )


async def build_visual_product_index_from_catalog(
    catalog_snapshot: Sequence[Mapping[str, Any]],
    *,
    object_storage: Any,
    encode_images: Callable[[Sequence[Image.Image]], Any],
    output_dir: str | Path,
    clip_model_id: str,
    clip_revision: str,
    catalog_activation_id: str,
    model_version: str,
    catalog_available_at: float,
    minimum_coverage: float = 0.95,
    batch_size: int = 32,
) -> VisualProductIndexBuildResult:
    """Build a real product-image index without synthetic or random fallbacks."""
    if not catalog_snapshot:
        raise ValueError("visual product index catalog snapshot is empty")
    if batch_size <= 0:
        raise ValueError("visual product image batch size must be positive")
    seen: set[str] = set()
    valid_records: list[tuple[str, dict[str, Any], Image.Image]] = []
    quarantine: dict[str, str] = {}
    for raw in catalog_snapshot:
        record = dict(raw)
        product_id = str(record.get("product_id") or "").strip()
        if not product_id or product_id in seen:
            raise ValueError("catalog product IDs must be unique and non-empty")
        seen.add(product_id)
        uri = str(record.get("image_storage_uri") or "").strip()
        expected_sha256 = str(record.get("image_sha256") or "").strip()
        if not uri or len(expected_sha256) != 64:
            quarantine[product_id] = "image reference missing"
            continue
        local_path = ""
        should_delete = False
        try:
            local_path, should_delete = await object_storage.materialize_for_processing(
                uri, suggested_suffix=Path(uri).suffix or ".image"
            )
            path = Path(local_path)
            if _sha256(path) != expected_sha256:
                quarantine[product_id] = "image checksum mismatch"
                continue
            with Image.open(path) as source:
                source.verify()
            with Image.open(path) as source:
                image = source.convert("RGB").copy()
            valid_records.append((product_id, record, image))
        except Exception:
            quarantine[product_id] = "image unreadable"
        finally:
            if should_delete and local_path and os.path.exists(local_path):
                os.remove(local_path)

    embeddings: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    try:
        for offset in range(0, len(valid_records), batch_size):
            batch = valid_records[offset : offset + batch_size]
            encoded = encode_images([record[2] for record in batch])
            if inspect.isawaitable(encoded):
                encoded = await encoded
            matrix = np.asarray(encoded, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != len(batch):
                raise ValueError("product image encoder returned an invalid batch")
            for row, (product_id, record, _) in enumerate(batch):
                embeddings[product_id] = matrix[row]
                metadata[product_id] = record
    finally:
        for _, _, image in valid_records:
            image.close()

    manifest_path = publish_visual_product_index_bundle(
        output_dir,
        product_embeddings=embeddings,
        product_metadata=metadata,
        expected_product_count=len(catalog_snapshot),
        clip_model_id=clip_model_id,
        clip_revision=clip_revision,
        catalog_activation_id=catalog_activation_id,
        model_version=model_version,
        catalog_available_at=catalog_available_at,
        minimum_coverage=minimum_coverage,
    )
    return VisualProductIndexBuildResult(
        manifest_path=manifest_path,
        indexed_product_ids=tuple(sorted(embeddings)),
        quarantine=quarantine,
    )
