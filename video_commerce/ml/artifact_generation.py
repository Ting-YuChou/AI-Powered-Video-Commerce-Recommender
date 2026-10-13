"""Immutable local model generations and process-local reference swaps."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Generic, Iterator, Mapping, Sequence, TypeVar
import uuid

from video_commerce.data_plane.object_storage import ObjectStorage


_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")
T = TypeVar("T")


def _segment(value: str) -> str:
    normalized = _SAFE_SEGMENT.sub("-", str(value).strip()).strip(".-")
    if not normalized:
        raise ValueError("artifact generation path segment is empty")
    return normalized


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    storage_path: str
    sha256: str
    filename: str | None = None

    def validated(self) -> "ArtifactSpec":
        if not self.name or not self.storage_path:
            raise ValueError("artifact name and storage path are required")
        digest = str(self.sha256 or "").lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"artifact {self.name!r} checksum is invalid")
        return ArtifactSpec(
            name=_segment(self.name),
            storage_path=str(self.storage_path),
            sha256=digest,
            filename=_segment(
                self.filename or Path(self.storage_path).name or self.name
            ),
        )


@dataclass(frozen=True)
class ResolvedArtifactGeneration:
    model_name: str
    release_id: str
    model_version: str
    active_generation: int
    bundle_digest: str
    root: Path
    artifact_paths: Mapping[str, Path]
    metadata: Mapping[str, Any]

    @property
    def identity(self) -> tuple[str, int, str]:
        return self.release_id, self.active_generation, self.bundle_digest


class ArtifactGenerationInstaller:
    """Stage, verify, and publish a complete bundle with one directory rename."""

    MANIFEST_NAME = "generation.json"

    def __init__(self, *, root: str | Path, object_storage: ObjectStorage) -> None:
        self.root = Path(root)
        self.object_storage = object_storage

    async def install(
        self,
        *,
        model_name: str,
        release_id: str,
        model_version: str,
        active_generation: int,
        artifacts: Sequence[ArtifactSpec],
        metadata: Mapping[str, Any],
    ) -> ResolvedArtifactGeneration:
        if int(active_generation) < 1:
            raise ValueError("active generation must be positive")
        verified_specs = tuple(spec.validated() for spec in artifacts)
        if not verified_specs:
            raise ValueError("artifact generation must contain at least one artifact")
        names = [spec.name for spec in verified_specs]
        filenames = [str(spec.filename) for spec in verified_specs]
        if len(names) != len(set(names)) or len(filenames) != len(set(filenames)):
            raise ValueError("artifact generation names and filenames must be unique")

        family_root = self.root / _segment(model_name)
        generation_name = f"{int(active_generation)}-{_segment(release_id)}"
        final_root = family_root / generation_name
        manifest_payload = {
            "contract_version": "model_artifact_generation_v1",
            "model_name": str(model_name),
            "release_id": str(release_id),
            "model_version": str(model_version),
            "active_generation": int(active_generation),
            "artifacts": [
                {
                    "name": spec.name,
                    "filename": spec.filename,
                    "sha256": spec.sha256,
                    "storage_path": spec.storage_path,
                }
                for spec in sorted(verified_specs, key=lambda item: item.name)
            ],
            "metadata": dict(metadata),
        }
        bundle_digest = hashlib.sha256(
            json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        manifest_payload["bundle_digest"] = bundle_digest

        if final_root.exists():
            return self._load_existing(final_root, manifest_payload)

        family_root.mkdir(parents=True, exist_ok=True)
        staging_root = family_root / f".staging-{uuid.uuid4().hex}"
        staging_root.mkdir(mode=0o750)
        try:
            for spec in verified_specs:
                destination = staging_root / str(spec.filename)
                staged = await self.object_storage.stage_to_local_temp(
                    spec.storage_path,
                    str(destination),
                    expected_sha256=spec.sha256,
                )
                if Path(staged).resolve() != destination.resolve():
                    os.replace(staged, destination)
                self._fsync_file(destination)

            manifest_path = staging_root / self.MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._fsync_file(manifest_path)
            self._fsync_directory(staging_root)
            os.rename(staging_root, final_root)
            self._fsync_directory(family_root)
        except Exception:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise

        return self._resolved(final_root, manifest_payload)

    def _load_existing(
        self, final_root: Path, expected: Mapping[str, Any]
    ) -> ResolvedArtifactGeneration:
        manifest_path = final_root / self.MANIFEST_NAME
        try:
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                "existing artifact generation manifest is unreadable"
            ) from exc
        if actual.get("bundle_digest") != expected.get("bundle_digest"):
            raise ValueError(
                "existing artifact generation has a different bundle digest"
            )
        for artifact in actual.get("artifacts", []):
            path = final_root / str(artifact["filename"])
            digest = ObjectStorage.calculate_sha256(str(path))
            if digest != artifact["sha256"]:
                raise ValueError(
                    f"existing artifact {artifact['name']!r} checksum mismatch"
                )
        return self._resolved(final_root, actual)

    def cleanup(
        self,
        *,
        model_name: str,
        current_identity: tuple[str, int, str],
        protected_identities: set[tuple[str, int]],
        inflight_identities: set[tuple[str, int]],
        keep: int,
        min_age_hours: float,
        now_epoch_seconds: float | None = None,
    ) -> tuple[Path, ...]:
        """Remove old immutable generations without touching referenced bundles."""
        family_root = self.root / _segment(model_name)
        if not family_root.exists():
            return ()
        now = time.time() if now_epoch_seconds is None else float(now_epoch_seconds)
        entries: list[tuple[int, str, Path]] = []
        for directory in family_root.iterdir():
            if not directory.is_dir() or directory.name.startswith(".staging-"):
                continue
            try:
                manifest = json.loads(
                    (directory / self.MANIFEST_NAME).read_text(encoding="utf-8")
                )
                entries.append(
                    (
                        int(manifest["active_generation"]),
                        str(manifest["release_id"]),
                        directory,
                    )
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                # An unreadable directory is not safe to delete automatically.
                continue
        entries.sort(key=lambda item: item[0], reverse=True)
        retained_latest = {
            (release, generation)
            for generation, release, _ in entries[: max(0, int(keep))]
        }
        protected = (
            set(protected_identities) | set(inflight_identities) | retained_latest
        )
        protected.add((str(current_identity[0]), int(current_identity[1])))
        deleted: list[Path] = []
        for generation, release, directory in entries:
            if (release, generation) in protected:
                continue
            age_hours = (now - directory.stat().st_mtime) / 3600.0
            if age_hours < float(min_age_hours):
                continue
            shutil.rmtree(directory)
            deleted.append(directory)
        if deleted:
            self._fsync_directory(family_root)
        return tuple(sorted(deleted, key=lambda path: path.name))

    @staticmethod
    def _resolved(
        root: Path, manifest: Mapping[str, Any]
    ) -> ResolvedArtifactGeneration:
        return ResolvedArtifactGeneration(
            model_name=str(manifest["model_name"]),
            release_id=str(manifest["release_id"]),
            model_version=str(manifest["model_version"]),
            active_generation=int(manifest["active_generation"]),
            bundle_digest=str(manifest["bundle_digest"]),
            root=root,
            artifact_paths={
                str(item["name"]): root / str(item["filename"])
                for item in manifest["artifacts"]
            },
            metadata=dict(manifest.get("metadata") or {}),
        )

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class AtomicModelReference(Generic[T]):
    """Swap a model reference while tracking identities used by in-flight work."""

    def __init__(self, value: T, *, identity: tuple[str, int, str]) -> None:
        self._value = value
        self._identity = identity
        self._inflight: Counter[tuple[str, int, str]] = Counter()
        self._lock = threading.RLock()

    def current(self) -> T:
        with self._lock:
            return self._value

    @property
    def identity(self) -> tuple[str, int, str]:
        with self._lock:
            return self._identity

    @contextmanager
    def snapshot(self) -> Iterator[T]:
        with self._lock:
            value = self._value
            identity = self._identity
            self._inflight[identity] += 1
        try:
            yield value
        finally:
            with self._lock:
                self._inflight[identity] -= 1
                if self._inflight[identity] <= 0:
                    del self._inflight[identity]

    def swap(self, value: T, *, identity: tuple[str, int, str]) -> T:
        with self._lock:
            previous = self._value
            self._value = value
            self._identity = identity
            return previous

    def inflight_count(self, identity: tuple[str, int, str]) -> int:
        with self._lock:
            return int(self._inflight.get(identity, 0))

    def inflight_identities(self) -> set[tuple[str, int]]:
        with self._lock:
            return {(release, generation) for release, generation, _ in self._inflight}
