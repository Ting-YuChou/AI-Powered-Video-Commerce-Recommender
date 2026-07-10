"""Read an immutable, manifest-guarded export of the Iceberg PIT dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, List


class PitTrainingDatasetError(ValueError):
    """Raised when a PIT export cannot be safely used for model training."""


@dataclass(frozen=True)
class PitTrainingDataset:
    dataset_version: str
    feature_definition_version: str
    rows: List[dict[str, Any]]


class PitTrainingDatasetReader:
    """Load a materialized Iceberg snapshot export without a legacy fallback.

    The batch Flink job writes this JSONL manifest/export for the Python model
    trainer. The manifest pins the Iceberg snapshot identifier and shared
    feature-definition version before a single training row is accepted.
    """

    def __init__(self, object_storage, *, expected_feature_definition_version: str):
        self.object_storage = object_storage
        self.expected_feature_definition_version = expected_feature_definition_version

    async def read(self, dataset_uri: str) -> PitTrainingDataset:
        local_path, should_delete = await self.object_storage.materialize_for_processing(
            dataset_uri,
            suggested_suffix=".jsonl",
        )
        try:
            return self._read_local_path(local_path)
        finally:
            if should_delete and os.path.exists(local_path):
                os.remove(local_path)

    def _read_local_path(self, path: str) -> PitTrainingDataset:
        try:
            with open(path, "r", encoding="utf-8") as dataset_file:
                records = [
                    json.loads(line)
                    for line in dataset_file
                    if line.strip()
                ]
        except (OSError, json.JSONDecodeError) as exc:
            raise PitTrainingDatasetError(f"unable to read PIT dataset: {exc}") from exc

        if not records or records[0].get("record_type") != "manifest":
            raise PitTrainingDatasetError("PIT dataset must begin with a manifest record")
        manifest = records[0]
        if manifest.get("status") != "complete":
            raise PitTrainingDatasetError("PIT dataset manifest is not complete")
        definition_version = str(manifest.get("feature_definition_version") or "")
        if definition_version != self.expected_feature_definition_version:
            raise PitTrainingDatasetError(
                "PIT dataset feature definition version does not match the trainer"
            )
        dataset_version = str(manifest.get("dataset_version") or "").strip()
        if not dataset_version:
            raise PitTrainingDatasetError("PIT dataset manifest is missing dataset_version")

        rows: List[dict[str, Any]] = []
        for record in records[1:]:
            if record.get("record_type") != "training_row":
                continue
            if record.get("feature_definition_version") != definition_version:
                raise PitTrainingDatasetError(
                    "PIT training row feature definition version does not match manifest"
                )
            if record.get("as_of_ts") is None:
                raise PitTrainingDatasetError("PIT training row is missing as_of_ts")
            if not record.get("feature_bundle_hash"):
                raise PitTrainingDatasetError("PIT training row is missing feature_bundle_hash")
            rows.append(dict(record))
        return PitTrainingDataset(
            dataset_version=dataset_version,
            feature_definition_version=definition_version,
            rows=rows,
        )
