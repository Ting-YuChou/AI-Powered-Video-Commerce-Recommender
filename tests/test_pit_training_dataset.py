import asyncio
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from video_commerce.common.feature_history_contracts import payload_sha256
from video_commerce.common.models import ContentFeatures
from video_commerce.ml.content_artifacts import canonical_content_feature_bytes
from video_commerce.ml.candidate_embedding_sidecar import (
    CANDIDATE_SIDECAR_SCHEMA_VERSION,
    write_candidate_embedding_sidecar,
)
from video_commerce.ml.pit_training_dataset import (
    PitDatasetQualityError,
    PitTrainingDatasetError,
    PitTrainingDatasetUnavailable,
    PitTrainingDatasetReader,
    arrow_schema_sha256,
    validate_pit_manifest_quality,
)
from video_commerce.ml.pit_manifest import (
    PitManifestPublisher,
    _candidate_sidecar_reference_from_environment,
    _is_parquet_shard_uri,
)
from video_commerce.ml.ranking_features import RANKING_LTR_FEATURE_DEFINITION_VERSION
from video_commerce.common.feature_history_contracts import (
    RANKING_LTR_DIN_FEATURE_DEFINITION_VERSION,
)
from video_commerce.ml.ranking_training import RANKING_LABEL_DEFINITION_VERSION


class LocalObjectStorage:
    async def materialize_for_processing(self, storage_path, *, suggested_suffix=""):
        return storage_path, False


def test_pit_row_preserves_retrieval_signals_and_masks_conflicting_gross_values(
    tmp_path,
):
    from video_commerce.ml.ranking_training import RankingLabelBuilder

    _, _, shard = _write_dataset(tmp_path)
    row = pq.read_table(shard).to_pylist()[0]
    scores = json.loads(row["candidate_features_json"])
    scores["retrieval_signals"] = {
        "two_tower": {
            "raw_source": "cf",
            "raw_score": 0.8,
            "source_rank": 1,
            "source_count": 2,
            "normalized_rank": 1.0,
            "reciprocal_rank": 1 / 61,
        }
    }
    scores["fusion_policy_version"] = "candidate_rrf_v1"
    scores["candidate_pool_position"] = 1
    row["candidate_features_json"] = json.dumps(scores)
    row["feature_bundle_hash"] = payload_sha256(
        {
            "as_of_ts": row["as_of_ts"],
            "candidate_features": scores,
            "context": {},
            "feature_definition_version": "ranking_ltr_v1",
            "product_id": "p1",
            "product_metadata": {"price": 9.0},
            "user_features": {"total_interactions": 3},
            "user_id": "u1",
        }
    )
    row.update(
        attributed_action="purchase",
        attributed_click=1,
        attributed_purchase=1,
        attributed_value=99.0,
        attributed_value_source="margin",
        attributed_purchase_value=12.0,
        attributed_gmv=10.0,
        attributed_value_currency="USD",
    )
    example = PitTrainingDatasetReader._normalize_training_row(
        row, "ranking_ltr_v1", RANKING_LABEL_DEFINITION_VERSION
    )
    assert example.bundle.candidate.retrieval_signals["two_tower"].source_rank == 1
    assert example.bundle.candidate.fusion_policy_version == "candidate_rrf_v1"
    labels = RankingLabelBuilder(target_policy="gross_revenue_v1").build(
        example.attribution
    )
    assert labels.ctcvr == 1.0
    assert labels.value_mask == 0.0
    assert labels.value_rejection_reason == "conflicting_gross_revenue_values"


def test_missing_pit_recency_uses_observation_time_instead_of_current_clock(tmp_path):
    _, _, shard = _write_dataset(tmp_path)
    row = pq.read_table(shard).to_pylist()[0]
    example = PitTrainingDatasetReader._normalize_training_row(
        row, "ranking_ltr_v1", RANKING_LABEL_DEFINITION_VERSION
    )
    assert example.bundle.user_features.last_active == example.bundle.as_of_ts


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
    tmp_path,
    *,
    status="complete",
    definition_version=None,
    context=None,
    behavior_sequences=None,
    candidate_embedding_sidecar=None,
    parity_ratio=1.0,
    quarantine_row_count=0,
    attributed_click=1,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    definition_version = definition_version or (
        RANKING_LTR_DIN_FEATURE_DEFINITION_VERSION
        if behavior_sequences is not None
        else RANKING_LTR_FEATURE_DEFINITION_VERSION
    )
    shard = tmp_path / "part-00000.parquet"
    context = context or {}
    row = {
        "observation_id": "imp-1:p1",
        "impression_id": "imp-1",
        "user_id": "u1",
        "product_id": "p1",
        "action": "view",
        "as_of_ts": 50.0,
        "feature_definition_version": definition_version,
        "label_definition_version": RANKING_LABEL_DEFINITION_VERSION,
        "attributed_action": "click" if attributed_click else "view",
        "attributed_click": attributed_click,
        "attributed_purchase": 0,
        "attributed_value": None,
        "attributed_value_source": None,
        "user_features_json": '{"total_interactions":3}',
        "product_metadata_json": '{"price":9.0}',
        "context_json": json.dumps(context),
        "candidate_features_json": (
            '{"collaborative_score":0.8,"content_similarity_score":0.7,'
            '"popularity_score":0.6,"combined_score":0.5}'
        ),
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
        "context": context,
        "feature_definition_version": definition_version,
        "product_id": "p1",
        "product_metadata": {"price": 9.0},
        "user_features": {"total_interactions": 3},
        "user_id": "u1",
    }
    if behavior_sequences is not None:
        bundle_payload["behavior_sequences"] = behavior_sequences
    row["feature_bundle_hash"] = payload_sha256(bundle_payload)
    # Serving computed the same bundle, so the row is online/offline parity-matched.
    row["online_feature_bundle_hash"] = row["feature_bundle_hash"]
    table = pa.Table.from_pylist([row])
    pq.write_table(table, shard)
    shard_bytes = shard.read_bytes()
    manifest = tmp_path / "manifest.json"
    manifest_payload = {
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
        "quarantine_row_count": quarantine_row_count,
        "online_offline_parity_ratio": parity_ratio,
        "positive_rate": float(attributed_click),
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
    if candidate_embedding_sidecar is not None:
        manifest_payload["candidate_embedding_sidecar"] = candidate_embedding_sidecar
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
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
async def test_pit_reader_resolves_checksum_pinned_content_artifact(tmp_path):
    features = ContentFeatures(content_id="video-1", visual_embedding=[1.0])
    artifact_bytes = canonical_content_feature_bytes(features)
    artifact = tmp_path / "content.json"
    artifact.write_bytes(artifact_bytes)
    reference = {
        "content_id": "video-1",
        "uri": str(artifact),
        "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "schema_version": "content_feature_artifact_v2",
        "created_at": features.created_at,
    }
    latest, _, _ = _write_dataset(
        tmp_path / "pit", context={"content_feature_ref": reference}
    )

    dataset = await PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
    ).read(str(latest))
    assert dataset.examples[0].multimodal_content.content_id == "video-1"

    reference["sha256"] = "0" * 64
    latest, _, _ = _write_dataset(
        tmp_path / "bad", context={"content_feature_ref": reference}
    )
    with pytest.raises(PitTrainingDatasetError, match="checksum"):
        await PitTrainingDatasetReader(
            LocalObjectStorage(),
            expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
        ).read(str(latest))


@pytest.mark.asyncio
async def test_pit_reader_requires_and_loads_pinned_candidate_sidecar(tmp_path):
    sidecar = tmp_path / "candidate-source.npz"
    sidecar_sha256 = write_candidate_embedding_sidecar(
        sidecar,
        {"p1": {"text": [1.0] * 384}},
        model_version="candidate-source-42",
        training_cutoff=0.0,
    )
    reference = {
        "uri": str(sidecar),
        "sha256": sidecar_sha256,
        "schema_version": CANDIDATE_SIDECAR_SCHEMA_VERSION,
        "model_version": "candidate-source-42",
        "training_cutoff": 0.0,
    }
    latest, manifest_path, _ = _write_dataset(
        tmp_path / "pit",
        candidate_embedding_sidecar=reference,
    )

    dataset = await PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
        require_candidate_sidecar=True,
    ).read(str(latest))

    assert dataset.candidate_embedding_sidecar == reference
    assert set(dataset.examples[0].candidate_embeddings) == {
        "text",
        "available_at",
    }
    assert len(dataset.examples[0].candidate_embeddings["text"]) == 384

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate_embedding_sidecar"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PitTrainingDatasetError, match="candidate sidecar checksum"):
        await PitTrainingDatasetReader(
            LocalObjectStorage(),
            expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
            require_candidate_sidecar=True,
        ).read(str(latest))


@pytest.mark.asyncio
async def test_pit_reader_rejects_missing_required_candidate_sidecar(tmp_path):
    latest, _, _ = _write_dataset(tmp_path)

    with pytest.raises(PitTrainingDatasetError, match="candidate sidecar"):
        await PitTrainingDatasetReader(
            LocalObjectStorage(),
            expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
            require_candidate_sidecar=True,
        ).read(str(latest))


@pytest.mark.asyncio
async def test_pit_reader_resolves_candidate_version_without_future_leakage(tmp_path):
    early = [0.0] * 384
    early[0] = 1.0
    late = [0.0] * 384
    late[1] = 1.0
    sidecar = tmp_path / "candidate-versions.npz"
    sidecar_sha256 = write_candidate_embedding_sidecar(
        sidecar,
        {
            "p1": [
                {"available_at": 40.0, "text": early},
                {"available_at": 60.0, "text": late},
            ]
        },
        model_version="candidate-source-pit",
        training_cutoff=30.0,
    )
    reference = {
        "uri": str(sidecar),
        "sha256": sidecar_sha256,
        "schema_version": CANDIDATE_SIDECAR_SCHEMA_VERSION,
        "model_version": "candidate-source-pit",
        "training_cutoff": 30.0,
    }
    latest, _, _ = _write_dataset(
        tmp_path / "pit-versioned",
        candidate_embedding_sidecar=reference,
    )

    dataset = await PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
        require_candidate_sidecar=True,
    ).read(str(latest))

    assert dataset.examples[0].candidate_embeddings["text"][0] == pytest.approx(1.0)
    assert dataset.examples[0].candidate_embeddings["text"][1] == pytest.approx(0.0)

    leaking_sidecar = tmp_path / "candidate-leaking.npz"
    leaking_sha256 = write_candidate_embedding_sidecar(
        leaking_sidecar,
        {"p1": {"available_at": 40.0, "text": early}},
        model_version="candidate-source-leaking",
        training_cutoff=51.0,
    )
    leaking_reference = {
        "uri": str(leaking_sidecar),
        "sha256": leaking_sha256,
        "schema_version": CANDIDATE_SIDECAR_SCHEMA_VERSION,
        "model_version": "candidate-source-leaking",
        "training_cutoff": 51.0,
    }
    leaking_latest, _, _ = _write_dataset(
        tmp_path / "pit-leaking",
        candidate_embedding_sidecar=leaking_reference,
    )
    with pytest.raises(PitTrainingDatasetError, match="training cutoff"):
        await PitTrainingDatasetReader(
            LocalObjectStorage(),
            expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
            require_candidate_sidecar=True,
        ).read(str(leaking_latest))


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


def test_pit_streaming_reader_stages_batches_without_reading_full_table(
    tmp_path, monkeypatch
):
    latest, _, _ = _write_dataset(tmp_path / "dataset")
    reader = PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
    )

    monkeypatch.setattr(
        pq,
        "read_table",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("streaming reader must not call read_table")
        ),
    )
    source = asyncio.run(
        reader.validate_and_stage(
            str(latest),
            staging_dir=tmp_path / "staging",
            batch_rows=1,
            max_group_rows=16,
            max_bytes=10_000_000,
        )
    )

    assert source.row_count == 1
    assert source.group_count == 1
    batches = list(source.iter_example_batches(split="all", batch_rows=1, seed=42))
    assert batches[0][0].impression_id == "imp-1"
    assert batches[0][0].bundle.user_features.total_interactions == 3
    assert source.lineage["pit_stream_contract_version"] == "pit_stream_v1"


def test_pit_streaming_reader_does_not_publish_corrupt_generation(tmp_path):
    latest, manifest_path, _ = _write_dataset(tmp_path / "dataset")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    reader = PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
    )

    with pytest.raises(PitTrainingDatasetError, match="checksum"):
        asyncio.run(
            reader.validate_and_stage(
                str(latest),
                staging_dir=tmp_path / "staging",
                batch_rows=1,
                max_group_rows=16,
                max_bytes=10_000_000,
            )
        )

    assert not list((tmp_path / "staging").glob("verified/*"))


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


@pytest.mark.asyncio
async def test_manifest_publisher_pins_verified_candidate_sidecar(tmp_path):
    _, _, shard = _write_dataset(tmp_path / "source")
    sidecar = tmp_path / "candidate-source.npz"
    sidecar_sha256 = write_candidate_embedding_sidecar(
        sidecar,
        {"p1": {"text": [1.0] * 384}},
        model_version="candidate-source-42",
        training_cutoff=0.0,
    )
    reference = {
        "uri": str(sidecar),
        "sha256": sidecar_sha256,
        "schema_version": CANDIDATE_SIDECAR_SCHEMA_VERSION,
        "model_version": "candidate-source-42",
        "training_cutoff": 0.0,
    }
    output = tmp_path / "published"

    await PitManifestPublisher(LocalObjectStorage()).publish(
        shard_uris=[str(shard)],
        output_prefix=str(output),
        materialization_run_id="run-candidate-1",
        iceberg_table_id="video_commerce.ranking_training_pit",
        iceberg_snapshot_id="42",
        feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
        label_definition_version=RANKING_LABEL_DEFINITION_VERSION,
        attribution_cutoff=1_700_000_000.0,
        candidate_embedding_sidecar=reference,
    )

    manifest = json.loads(
        (output / "runs/run-candidate-1/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["candidate_embedding_sidecar"] == reference

    future_trained_reference = dict(reference, training_cutoff=51.0)
    with pytest.raises(PitTrainingDatasetError, match="exceeds dataset history"):
        await PitManifestPublisher(LocalObjectStorage()).publish(
            shard_uris=[str(shard)],
            output_prefix=str(tmp_path / "future-trained"),
            materialization_run_id="run-candidate-future",
            iceberg_table_id="video_commerce.ranking_training_pit",
            iceberg_snapshot_id="42",
            feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
            label_definition_version=RANKING_LABEL_DEFINITION_VERSION,
            attribution_cutoff=1_700_000_000.0,
            candidate_embedding_sidecar=future_trained_reference,
        )

    bad_reference = dict(reference, sha256="0" * 64)
    with pytest.raises(PitTrainingDatasetError, match="candidate sidecar checksum"):
        await PitManifestPublisher(LocalObjectStorage()).publish(
            shard_uris=[str(shard)],
            output_prefix=str(tmp_path / "bad"),
            materialization_run_id="run-candidate-2",
            iceberg_table_id="video_commerce.ranking_training_pit",
            iceberg_snapshot_id="42",
            feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
            label_definition_version=RANKING_LABEL_DEFINITION_VERSION,
            attribution_cutoff=1_700_000_000.0,
            candidate_embedding_sidecar=bad_reference,
        )


def test_manifest_discovers_flink_parquet_part_files_without_extension():
    assert _is_parquet_shard_uri("s3://features/run/shards/part-abc-task-0-file-0")
    assert _is_parquet_shard_uri("s3://features/run/shards/part-abc.parquet")
    assert not _is_parquet_shard_uri("s3://features/run/shards/_SUCCESS")
    assert not _is_parquet_shard_uri("s3://features/run/shards/.staging/file")


def test_candidate_sidecar_reference_from_environment_is_all_or_nothing(
    monkeypatch,
):
    assert _candidate_sidecar_reference_from_environment() is None

    monkeypatch.setenv(
        "FEATURE_LAKE_CANDIDATE_SIDECAR_URI",
        "s3://features/candidates/source-42.npz",
    )
    with pytest.raises(PitTrainingDatasetError, match="candidate sidecar"):
        _candidate_sidecar_reference_from_environment()

    monkeypatch.setenv("FEATURE_LAKE_CANDIDATE_SIDECAR_SHA256", "a" * 64)
    monkeypatch.setenv(
        "FEATURE_LAKE_CANDIDATE_SIDECAR_MODEL_VERSION",
        "candidate-source-42",
    )
    monkeypatch.setenv("FEATURE_LAKE_CANDIDATE_SIDECAR_TRAINING_CUTOFF", "42.0")
    assert _candidate_sidecar_reference_from_environment() == {
        "uri": "s3://features/candidates/source-42.npz",
        "sha256": "a" * 64,
        "schema_version": CANDIDATE_SIDECAR_SCHEMA_VERSION,
        "model_version": "candidate-source-42",
        "training_cutoff": 42.0,
    }


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


class RecordingObservability:
    def __init__(self):
        self.failures = []

    def record_pit_manifest_validation_failure(self, reason):
        self.failures.append(reason)

    def record_pit_stream_failure(self, stage):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dataset_kwargs", "error"),
    [
        ({"parity_ratio": 0.5}, "parity 0.500000 is below 0.990000"),
        ({"quarantine_row_count": 9}, "quarantine ratio 0.900000 exceeds 0.200000"),
    ],
)
async def test_pit_reader_fails_closed_on_low_parity_or_high_quarantine(
    tmp_path, dataset_kwargs, error
):
    latest, _, _ = _write_dataset(tmp_path / "dataset", **dataset_kwargs)
    observability = RecordingObservability()
    reader = PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
        observability=observability,
        min_online_offline_parity_ratio=0.99,
        max_quarantine_ratio=0.2,
    )

    with pytest.raises(PitDatasetQualityError, match=error):
        await reader.read(str(latest))
    with pytest.raises(PitDatasetQualityError, match=error):
        await reader.validate_and_stage(
            str(latest),
            staging_dir=tmp_path / "staging",
            batch_rows=1,
            max_group_rows=16,
            max_bytes=10_000_000,
        )

    assert observability.failures == ["PitDatasetQualityError"] * 2
    assert not list((tmp_path / "staging").glob("ranking-*"))


@pytest.mark.asyncio
async def test_pit_reader_rejects_manifest_without_parity(tmp_path):
    latest, manifest_path, _ = _write_dataset(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["online_offline_parity_ratio"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PitDatasetQualityError, match="missing parity"):
        await PitTrainingDatasetReader(
            LocalObjectStorage(),
            expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
        ).read(str(latest))


def test_pit_reader_quality_thresholds_default_to_feature_lake_config(monkeypatch):
    monkeypatch.setenv("FEATURE_LAKE_PIT_MIN_ONLINE_OFFLINE_PARITY_RATIO", "0.95")
    monkeypatch.setenv("FEATURE_LAKE_PIT_MAX_QUARANTINE_RATIO", "0.1")

    reader = PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
    )

    assert reader.min_online_offline_parity_ratio == 0.95
    assert reader.max_quarantine_ratio == 0.1


def test_pit_quality_gate_accepts_boundary_values():
    quality = validate_pit_manifest_quality(
        {
            "online_offline_parity_ratio": 0.99,
            "row_count": 80,
            "quarantine_row_count": 20,
        },
        min_online_offline_parity_ratio=0.99,
        max_quarantine_ratio=0.2,
    )

    assert quality == {"online_offline_parity_ratio": 0.99, "quarantine_ratio": 0.2}


def _publish_kwargs(shard, output, run_id, cutoff):
    return {
        "shard_uris": [str(shard)],
        "output_prefix": str(output),
        "materialization_run_id": run_id,
        "iceberg_table_id": "video_commerce.ranking_training_pit",
        "iceberg_snapshot_id": run_id,
        "feature_definition_version": RANKING_LTR_FEATURE_DEFINITION_VERSION,
        "label_definition_version": RANKING_LABEL_DEFINITION_VERSION,
        "attribution_cutoff": cutoff,
    }


@pytest.mark.asyncio
async def test_manifest_records_parity_positive_rate_and_feature_max_age(tmp_path):
    _, _, shard = _write_dataset(tmp_path / "source")
    output = tmp_path / "published"
    publisher = PitManifestPublisher(LocalObjectStorage())

    await publisher.publish(
        **_publish_kwargs(shard, output, "run-1", 1_700_000_000.0),
        feature_max_age_seconds=604800,
    )

    manifest = json.loads(
        (output / "runs/run-1/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["online_offline_parity_ratio"] == 1.0
    assert manifest["positive_rate"] == 1.0
    assert manifest["feature_max_age_seconds"] == 604800
    assert publisher.last_manifest == manifest


@pytest.mark.asyncio
async def test_manifest_publisher_rejects_positive_rate_shift_against_previous_run(
    tmp_path,
):
    _, _, clicked = _write_dataset(tmp_path / "clicked", attributed_click=1)
    _, _, viewed = _write_dataset(tmp_path / "viewed", attributed_click=0)
    output = tmp_path / "published"
    publisher = PitManifestPublisher(
        LocalObjectStorage(),
        max_positive_rate_relative_shift=0.5,
        positive_rate_min_rows=1,
    )
    await publisher.publish(**_publish_kwargs(clicked, output, "run-1", 1_000.0))

    with pytest.raises(PitTrainingDatasetError, match="positive rate 0.000000 shifted"):
        await publisher.publish(**_publish_kwargs(viewed, output, "run-2", 2_000.0))
    assert not (output / "runs/run-2/manifest.json").exists()
    pointer = json.loads((output / "latest.json").read_text(encoding="utf-8"))
    assert pointer["materialization_run_id"] == "run-1"

    # The same rate passes, and too few rows skip the comparison entirely.
    await publisher.publish(**_publish_kwargs(clicked, output, "run-3", 3_000.0))
    await PitManifestPublisher(
        LocalObjectStorage(), positive_rate_min_rows=2
    ).publish(**_publish_kwargs(viewed, output, "run-4", 4_000.0))


@pytest.mark.asyncio
async def test_manifest_publisher_skips_shift_check_without_previous_positive_rate(
    tmp_path,
):
    _, _, viewed = _write_dataset(tmp_path / "viewed", attributed_click=0)
    output = tmp_path / "published"
    legacy_manifest = output / "runs/run-legacy/manifest.json"
    legacy_manifest.parent.mkdir(parents=True)
    legacy_manifest.write_text(
        json.dumps({"row_count": 10_000, "materialization_run_id": "run-legacy"}),
        encoding="utf-8",
    )
    (output / "latest.json").write_text(
        json.dumps(
            {
                "manifest_uri": str(legacy_manifest),
                "materialization_run_id": "run-legacy",
                "attribution_cutoff": 1_000.0,
            }
        ),
        encoding="utf-8",
    )

    await PitManifestPublisher(
        LocalObjectStorage(), positive_rate_min_rows=1
    ).publish(**_publish_kwargs(viewed, output, "run-new", 2_000.0))

    pointer = json.loads((output / "latest.json").read_text(encoding="utf-8"))
    assert pointer["materialization_run_id"] == "run-new"
