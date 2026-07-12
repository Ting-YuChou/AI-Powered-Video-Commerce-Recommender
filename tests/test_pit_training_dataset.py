import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from video_commerce.common.feature_history_contracts import payload_sha256
from video_commerce.ml.pit_training_dataset import (
    PitTrainingDatasetError,
    PitTrainingDatasetUnavailable,
    PitTrainingDatasetReader,
    arrow_schema_sha256,
)
from video_commerce.ml.pit_manifest import PitManifestPublisher, _is_parquet_shard_uri
from video_commerce.ml.ranking_features import RANKING_LTR_FEATURE_DEFINITION_VERSION
from video_commerce.common.feature_history_contracts import RANKING_LTR_DIN_FEATURE_DEFINITION_VERSION
from video_commerce.ml.ranking_training import RANKING_LABEL_DEFINITION_VERSION


class LocalObjectStorage:
    async def materialize_for_processing(self, storage_path, *, suggested_suffix=""):
        return storage_path, False


class DownloadedObjectStorage:
    def __init__(self, local_path):
        self.local_path = str(local_path)

    async def materialize_for_processing(self, storage_path, *, suggested_suffix=""):
        return self.local_path, True


@pytest.mark.asyncio
async def test_missing_latest_pointer_is_bootstrap_unavailable(tmp_path):
    reader = PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
    )

    with pytest.raises(PitTrainingDatasetUnavailable, match="latest pointer"):
        await reader.read(str(tmp_path / "missing-latest.json"))


def _write_dataset(
    tmp_path, *, status="complete", definition_version=None, behavior_sequences=None
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    definition_version = definition_version or (
        RANKING_LTR_DIN_FEATURE_DEFINITION_VERSION
        if behavior_sequences is not None
        else RANKING_LTR_FEATURE_DEFINITION_VERSION
    )
    shard = tmp_path / "part-00000.parquet"
    row = {
        "observation_id": "imp-1:p1",
        "impression_id": "imp-1",
        "user_id": "u1",
        "product_id": "p1",
        "action": "view",
        "as_of_ts": 50.0,
        "feature_definition_version": definition_version,
        "label_definition_version": RANKING_LABEL_DEFINITION_VERSION,
        "attributed_action": "click",
        "attributed_click": 1,
        "attributed_purchase": 0,
        "attributed_value": None,
        "attributed_value_source": None,
        "user_features_json": '{"total_interactions":3}',
        "product_metadata_json": '{"price":9.0}',
        "context_json": "{}",
        "candidate_features_json": (
            '{"collaborative_score":0.8,"content_similarity_score":0.7,'
            '"popularity_score":0.6,"combined_score":0.5}'
        ),
        "online_feature_bundle_hash": "a" * 64,
    }
    if behavior_sequences is not None:
        row["behavior_sequences_json"] = json.dumps(behavior_sequences)
    bundle_payload = {
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
    if behavior_sequences is not None:
        bundle_payload["behavior_sequences"] = behavior_sequences
    row["feature_bundle_hash"] = payload_sha256(bundle_payload)
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
                "label_definition_version": RANKING_LABEL_DEFINITION_VERSION,
                "schema_hash": arrow_schema_sha256(table.schema),
                "attribution_cutoff": 1_700_000_000.0,
                "row_count": 1,
                "quarantine_row_count": 0,
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


def _din_sequences(*, leaking=False):
    actions = {}
    for action in ("click", "cart", "purchase"):
        product_ids = [""] * 60
        event_times = [0.0] * 60
        event_ids = [None] * 60
        mask = [False] * 60
        if action == "click":
            product_ids[-1] = "p-history"
            event_times[-1] = 51.0 if leaking else 49.0
            event_ids[-1] = "event-1"
            mask[-1] = True
        actions[action] = {
            "product_ids": product_ids,
            "event_times": event_times,
            "event_ids": event_ids,
            "mask": mask,
        }
    return {
        "contract_version": "din_sequence_v1",
        "as_of_ts": 50.0,
        "actions": actions,
    }


@pytest.mark.asyncio
async def test_pit_training_dataset_reader_pins_pointer_and_validates_parquet(tmp_path):
    latest, _, _ = _write_dataset(tmp_path)

    loaded = await PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
    ).read(str(latest))

    assert loaded.dataset_version == "iceberg-snapshot-42"
    assert loaded.materialization_run_id == "run-42"
    assert loaded.label_definition_version == RANKING_LABEL_DEFINITION_VERSION
    assert loaded.quarantine_rows == 0
    assert len(loaded.examples) == 1
    example = loaded.examples[0]
    assert example.impression_id == "imp-1"
    assert example.bundle.user_features.total_interactions == 3
    assert example.bundle.product_metadata == {"price": 9.0}
    assert example.bundle.candidate.collaborative_score == 0.8
    assert example.attribution.attributed_action == "click"


@pytest.mark.asyncio
async def test_pit_reader_loads_canonical_din_sequences(tmp_path):
    latest, _, _ = _write_dataset(tmp_path, behavior_sequences=_din_sequences())

    loaded = await PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_DIN_FEATURE_DEFINITION_VERSION,
    ).read(str(latest))

    sequences = loaded.examples[0].bundle.behavior_sequences
    assert sequences is not None
    assert sequences.actions["click"].length == 1
    assert sequences.actions["click"].product_ids[-1] == "p-history"


@pytest.mark.asyncio
async def test_pit_reader_rejects_leaking_din_sequence(tmp_path):
    latest, _, _ = _write_dataset(
        tmp_path, behavior_sequences=_din_sequences(leaking=True)
    )

    with pytest.raises(PitTrainingDatasetError, match="DIN click sequence leaks"):
        await PitTrainingDatasetReader(
            LocalObjectStorage(),
            expected_feature_definition_version=RANKING_LTR_DIN_FEATURE_DEFINITION_VERSION,
        ).read(str(latest))


@pytest.mark.asyncio
async def test_pit_training_dataset_reader_rejects_missing_label_contract(tmp_path):
    latest, manifest_path, _ = _write_dataset(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("label_definition_version")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reader = PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
    )
    with pytest.raises(PitTrainingDatasetError, match="label definition version"):
        await reader.read(str(latest))


@pytest.mark.asyncio
async def test_pit_training_dataset_reader_rejects_unpinned_iceberg_manifest(tmp_path):
    latest, manifest_path, _ = _write_dataset(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("iceberg_snapshot_id")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reader = PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
    )
    with pytest.raises(PitTrainingDatasetError, match="Iceberg table or snapshot"):
        await reader.read(str(latest))


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
        label_definition_version=RANKING_LABEL_DEFINITION_VERSION,
        attribution_cutoff=1_700_000_000.0,
        quarantine_row_count=0,
    )

    pointer = json.loads((output / "latest.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output / "runs/run-99/manifest.json").read_text(encoding="utf-8")
    )
    assert latest_uri == str(output / "latest.json")
    assert pointer["manifest_uri"] == str(output / "runs/run-99/manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["label_definition_version"] == RANKING_LABEL_DEFINITION_VERSION
    assert manifest["row_count"] == 1
    assert manifest["shards"][0]["sha256"]

    retry_uri = await publisher.publish(
        shard_uris=[str(shard)],
        output_prefix=str(output),
        materialization_run_id="run-99",
        iceberg_table_id="video_commerce.ranking_training_pit",
        iceberg_snapshot_id="99",
        feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
        label_definition_version=RANKING_LABEL_DEFINITION_VERSION,
        attribution_cutoff=1_700_000_000.0,
        quarantine_row_count=0,
    )
    assert retry_uri == latest_uri

    with pytest.raises(PitTrainingDatasetError, match="conflicts"):
        await publisher.publish(
            shard_uris=[str(shard)],
            output_prefix=str(output),
            materialization_run_id="run-99",
            iceberg_table_id="video_commerce.ranking_training_pit",
            iceberg_snapshot_id="100",
            feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
            label_definition_version=RANKING_LABEL_DEFINITION_VERSION,
            attribution_cutoff=1_700_000_000.0,
            quarantine_row_count=0,
        )

    await publisher.publish(
        shard_uris=[str(shard)],
        output_prefix=str(output),
        materialization_run_id="run-98",
        iceberg_table_id="video_commerce.ranking_training_pit",
        iceberg_snapshot_id="98",
        feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
        label_definition_version=RANKING_LABEL_DEFINITION_VERSION,
        attribution_cutoff=1_699_999_999.0,
        quarantine_row_count=0,
    )
    pointer_after_older_run = json.loads(
        (output / "latest.json").read_text(encoding="utf-8")
    )
    assert pointer_after_older_run["materialization_run_id"] == "run-99"


def test_manifest_discovers_flink_parquet_part_files_without_extension():
    assert _is_parquet_shard_uri("s3://features/run/shards/part-abc-task-0-file-0")
    assert _is_parquet_shard_uri("s3://features/run/shards/part-abc.parquet")
    assert not _is_parquet_shard_uri("s3://features/run/shards/_SUCCESS")
    assert not _is_parquet_shard_uri("s3://features/run/shards/.staging/file")


@pytest.mark.asyncio
async def test_manifest_retry_reads_and_cleans_downloaded_s3_manifest_once(tmp_path):
    downloaded = tmp_path / "downloaded-manifest.json"
    downloaded.write_text('{"status":"complete"}', encoding="utf-8")
    publisher = PitManifestPublisher(DownloadedObjectStorage(downloaded))

    payload = await publisher._read_json_if_exists(
        "s3://video-commerce-features/runs/run-1/manifest.json"
    )

    assert payload == {"status": "complete"}
    assert not downloaded.exists()
