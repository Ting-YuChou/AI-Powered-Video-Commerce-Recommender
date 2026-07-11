import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from video_commerce.common.feature_history_contracts import payload_sha256
from video_commerce.ml.pit_training_dataset import (
    PitTrainingDatasetError,
    PitTrainingDatasetReader,
    arrow_schema_sha256,
)
from video_commerce.ml.pit_manifest import PitManifestPublisher, _is_parquet_shard_uri
from video_commerce.ml.ranking_features import RANKING_LTR_FEATURE_DEFINITION_VERSION


class LocalObjectStorage:
    async def materialize_for_processing(self, storage_path, *, suggested_suffix=""):
        return storage_path, False


def _write_dataset(tmp_path, *, status="complete", definition_version=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    definition_version = definition_version or RANKING_LTR_FEATURE_DEFINITION_VERSION
    shard = tmp_path / "part-00000.parquet"
    row = {
        "observation_id": "imp-1:p1",
        "user_id": "u1",
        "product_id": "p1",
        "action": "view",
        "as_of_ts": 50.0,
        "feature_definition_version": definition_version,
        "user_features_json": '{"total_interactions":3}',
        "product_metadata_json": '{"price":9.0}',
        "context_json": "{}",
        "candidate_features_json": (
            '{"collaborative_score":0.8,"content_similarity_score":0.7,'
            '"popularity_score":0.6,"combined_score":0.5}'
        ),
        "online_feature_bundle_hash": "a" * 64,
    }
    row["feature_bundle_hash"] = payload_sha256(
        {
            "as_of_ts": 50.0,
            "candidate_features": {
                "collaborative_score": 0.8,
                "content_similarity_score": 0.7,
                "popularity_score": 0.6,
                "combined_score": 0.5,
            },
            "context": {},
            "feature_definition_version": definition_version,
            "product_id": "p1",
            "product_metadata": {"price": 9.0},
            "user_features": {"total_interactions": 3},
            "user_id": "u1",
        }
    )
    table = pa.Table.from_pylist([row])
    pq.write_table(table, shard)
    shard_bytes = shard.read_bytes()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": status,
                "dataset_version": "iceberg-snapshot-42",
                "materialization_run_id": "run-42",
                "iceberg_table_id": "video_commerce.ranking_training_pit",
                "iceberg_snapshot_id": "42",
                "feature_definition_version": definition_version,
                "schema_hash": arrow_schema_sha256(table.schema),
                "attribution_cutoff": 1_700_000_000.0,
                "row_count": 1,
                "min_as_of_ts": 50.0,
                "max_as_of_ts": 50.0,
                "shards": [
                    {
                        "uri": str(shard),
                        "byte_size": len(shard_bytes),
                        "sha256": hashlib.sha256(shard_bytes).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    latest = tmp_path / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "manifest_uri": str(manifest),
                "materialization_run_id": "run-42",
            }
        ),
        encoding="utf-8",
    )
    return latest, manifest, shard


@pytest.mark.asyncio
async def test_pit_training_dataset_reader_pins_pointer_and_validates_parquet(tmp_path):
    latest, _, _ = _write_dataset(tmp_path)

    loaded = await PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
    ).read(str(latest))

    assert loaded.dataset_version == "iceberg-snapshot-42"
    assert loaded.materialization_run_id == "run-42"
    assert len(loaded.rows) == 1
    assert loaded.rows[0]["user_features"] == {"total_interactions": 3}
    assert loaded.rows[0]["product_metadata"] == {"price": 9.0}
    assert loaded.rows[0]["candidate_scores"] == {
        "collaborative_score": 0.8,
        "content_similarity_score": 0.7,
        "popularity_score": 0.6,
        "combined_score": 0.5,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation,error",
    [
        ("incomplete", "not complete"),
        ("missing_shard", "missing"),
        ("hash_mismatch", "checksum"),
        ("schema_mismatch", "schema"),
    ],
)
async def test_pit_training_dataset_reader_fails_closed(tmp_path, mutation, error):
    latest, manifest_path, shard = _write_dataset(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "incomplete":
        manifest["status"] = "writing"
    elif mutation == "missing_shard":
        shard.unlink()
    elif mutation == "hash_mismatch":
        manifest["shards"][0]["sha256"] = "0" * 64
    elif mutation == "schema_mismatch":
        manifest["schema_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reader = PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
    )
    with pytest.raises(PitTrainingDatasetError, match=error):
        await reader.read(str(latest))


@pytest.mark.asyncio
async def test_pit_training_dataset_reader_rejects_feature_definition_mismatch(
    tmp_path,
):
    latest, _, _ = _write_dataset(tmp_path, definition_version="ranking_ltr_v0")
    reader = PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
    )
    with pytest.raises(PitTrainingDatasetError, match="feature definition version"):
        await reader.read(str(latest))


@pytest.mark.asyncio
async def test_manifest_publisher_writes_manifest_before_latest_pointer(tmp_path):
    _, _, shard = _write_dataset(tmp_path / "source")
    output = tmp_path / "published"
    publisher = PitManifestPublisher(LocalObjectStorage())

    latest_uri = await publisher.publish(
        shard_uris=[str(shard)],
        output_prefix=str(output),
        materialization_run_id="run-99",
        iceberg_table_id="video_commerce.ranking_training_pit",
        iceberg_snapshot_id="99",
        feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
        attribution_cutoff=1_700_000_000.0,
    )

    pointer = json.loads((output / "latest.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output / "runs/run-99/manifest.json").read_text(encoding="utf-8")
    )
    assert latest_uri == str(output / "latest.json")
    assert pointer["manifest_uri"] == str(output / "runs/run-99/manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["row_count"] == 1
    assert manifest["shards"][0]["sha256"]

    with pytest.raises(PitTrainingDatasetError, match="immutable"):
        await publisher.publish(
            shard_uris=[str(shard)],
            output_prefix=str(output),
            materialization_run_id="run-99",
            iceberg_table_id="video_commerce.ranking_training_pit",
            iceberg_snapshot_id="99",
            feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
            attribution_cutoff=1_700_000_000.0,
        )


def test_manifest_discovers_flink_parquet_part_files_without_extension():
    assert _is_parquet_shard_uri("s3://features/run/shards/part-abc-task-0-file-0")
    assert _is_parquet_shard_uri("s3://features/run/shards/part-abc.parquet")
    assert not _is_parquet_shard_uri("s3://features/run/shards/_SUCCESS")
    assert not _is_parquet_shard_uri("s3://features/run/shards/.staging/file")
