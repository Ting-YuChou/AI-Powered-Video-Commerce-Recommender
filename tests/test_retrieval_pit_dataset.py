import asyncio
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from video_commerce.ml.retrieval_pit_dataset import (
    RETRIEVAL_ELIGIBILITY_POLICY_VERSION,
    RETRIEVAL_LABEL_POLICY_VERSION,
    RetrievalPitDatasetError,
    RetrievalPitDatasetReader,
    split_retrieval_holdout,
)


class LocalObjectStorage:
    async def materialize_for_processing(self, storage_path, *, suggested_suffix=""):
        return str(storage_path), False


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_retrieval_dataset(
    tmp_path,
    *,
    row_overrides=None,
    catalog_overrides=None,
    manifest_overrides=None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    catalog = tmp_path / "catalog.json"
    catalog_payload = {
        "schema_version": "retrieval_catalog_generation_v1",
        "generation_id": "catalog-7",
        "effective_at": 100.0,
        "available_at": 110.0,
        "eligibility_policy_version": RETRIEVAL_ELIGIBILITY_POLICY_VERSION,
        "products": [
            {
                "product_id": "p1",
                "active": True,
                "in_stock": True,
                "metadata": {"price": 12.0},
                "modality_presence": {"visual": True, "ocr": False},
            },
            {
                "product_id": "p2",
                "active": True,
                "in_stock": True,
                "metadata": {"price": 4.0},
                "modality_presence": {"visual": False, "ocr": False},
            },
        ],
    }
    catalog_payload.update(catalog_overrides or {})
    catalog.write_text(json.dumps(catalog_payload), encoding="utf-8")
    item_features = tmp_path / "item-features.json"
    item_features_payload = {
        "schema_version": "retrieval_item_feature_sidecar_v1",
        "generation_id": "catalog-7",
        "available_at": 110.0,
        "embedding_dimension": 2,
        "embedding_model_revision": "clip-test-v1",
        "items": [
            {"product_id": "p1", "clip_embedding": [1.0, 0.0]},
            {"product_id": "p2", "clip_embedding": None},
        ],
    }
    item_features.write_text(json.dumps(item_features_payload), encoding="utf-8")

    row = {
        "query_id": "q1",
        "user_id": "u1",
        "product_id": "p1",
        "label_source": "viewed_impression",
        "label_type": "click",
        "label_weight": 2.0,
        "as_of_ts": 120.0,
        "label_event_time": 125.0,
        "label_available_at": 126.0,
        "feature_event_time": 119.0,
        "feature_available_at": 119.5,
        "catalog_generation_id": "catalog-7",
        "user_features_json": '{"total_interactions":4}',
        "seen_product_ids_json": "[]",
        "ranker_score": None,
    }
    row.update(row_overrides or {})
    shard = tmp_path / "part-00000.parquet"
    table = pa.Table.from_pylist([row])
    pq.write_table(table, shard)

    manifest = tmp_path / "manifest.json"
    manifest_payload = {
        "status": "complete",
        "schema_version": "retrieval_training_pit_v1",
        "materialization_run_id": "retrieval-pit-7",
        "dataset_version": "snapshot-7",
        "attribution_cutoff": 700_000.0,
        "attribution_window_hours": 168,
        "allowed_lateness_hours": 1,
        "min_as_of_ts": 120.0,
        "max_as_of_ts": 120.0,
        "row_count": 1,
        "label_policy_version": RETRIEVAL_LABEL_POLICY_VERSION,
        "eligibility_policy_version": RETRIEVAL_ELIGIBILITY_POLICY_VERSION,
        "catalog_generation": {
            "generation_id": "catalog-7",
            "uri": str(catalog),
            "sha256": _sha256(catalog),
            "product_count": 2,
        },
        "item_feature_sidecar": {
            "uri": str(item_features),
            "sha256": _sha256(item_features),
            "generation_id": "catalog-7",
            "schema_version": "retrieval_item_feature_sidecar_v1",
            "embedding_dimension": 2,
            "embedding_model_revision": "clip-test-v1",
            "available_at": 110.0,
        },
        "shards": [
            {
                "uri": str(shard),
                "byte_size": shard.stat().st_size,
                "sha256": _sha256(shard),
            }
        ],
    }
    manifest_payload.update(manifest_overrides or {})
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    latest = tmp_path / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "manifest_uri": str(manifest),
                "materialization_run_id": "retrieval-pit-7",
            }
        ),
        encoding="utf-8",
    )
    return latest


def test_retrieval_reader_pins_catalog_and_preserves_missing_modalities(tmp_path):
    latest = _write_retrieval_dataset(tmp_path)

    dataset = asyncio.run(
        RetrievalPitDatasetReader(LocalObjectStorage()).read(str(latest))
    )

    assert dataset.manifest.catalog_generation_id == "catalog-7"
    assert [product.product_id for product in dataset.catalog.eligible_products] == [
        "p1",
        "p2",
    ]
    assert dataset.catalog.products["p2"].modality_presence["visual"] is False
    assert dataset.catalog.item_features["p1"].clip_embedding == (1.0, 0.0)
    assert dataset.catalog.item_features["p2"].clip_embedding is None
    assert dataset.catalog.embedding_model_revision == "clip-test-v1"
    assert dataset.rows[0].user_features == {"total_interactions": 4}


def test_retrieval_streaming_reader_stages_rows_without_full_table(
    tmp_path, monkeypatch
):
    latest = _write_retrieval_dataset(tmp_path / "dataset")
    reader = RetrievalPitDatasetReader(LocalObjectStorage())
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
    rows = [
        row
        for batch in source.iter_row_batches(split="all", batch_rows=1, seed=42)
        for row in batch
    ]
    assert rows[0].query_id == "q1"
    assert rows[0].user_features == {"total_interactions": 4}
    assert source.lineage["pit_stream_contract_version"] == "pit_stream_v1"


@pytest.mark.parametrize(
    ("row_overrides", "message"),
    [
        ({"feature_event_time": 121.0}, "feature event time"),
        ({"feature_available_at": 121.0}, "feature available time"),
        ({"label_event_time": 119.0}, "label precedes query"),
        ({"label_available_at": 700_001.0}, "label availability"),
        ({"catalog_generation_id": "future"}, "catalog generation"),
    ],
)
def test_retrieval_reader_rejects_temporal_leakage(tmp_path, row_overrides, message):
    latest = _write_retrieval_dataset(tmp_path, row_overrides=row_overrides)

    with pytest.raises(RetrievalPitDatasetError, match=message):
        asyncio.run(RetrievalPitDatasetReader(LocalObjectStorage()).read(str(latest)))


def test_retrieval_reader_rejects_catalog_checksum_mismatch(tmp_path):
    latest = _write_retrieval_dataset(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["catalog_generation"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RetrievalPitDatasetError, match="catalog checksum"):
        asyncio.run(RetrievalPitDatasetReader(LocalObjectStorage()).read(str(latest)))


def test_retrieval_reader_rejects_item_feature_sidecar_checksum_mismatch(tmp_path):
    latest = _write_retrieval_dataset(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["item_feature_sidecar"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RetrievalPitDatasetError, match="item feature checksum"):
        asyncio.run(RetrievalPitDatasetReader(LocalObjectStorage()).read(str(latest)))


def test_retrieval_holdout_uses_latest_mature_days_without_leaking_frequency():
    from video_commerce.ml.retrieval_pit_dataset import RetrievalPitRow

    rows = [
        RetrievalPitRow.for_test(query_id="train", user_id="u1", as_of_ts=100.0),
        RetrievalPitRow.for_test(query_id="holdout", user_id="u2", as_of_ts=900_000.0),
    ]

    training, holdout, window = split_retrieval_holdout(
        rows, holdout_days=7, attribution_cutoff=1_000_000.0
    )

    assert [row.query_id for row in training] == ["train"]
    assert [row.query_id for row in holdout] == ["holdout"]
    assert window.end_ts == 900_000.0


def test_pinned_catalog_does_not_reject_queries_older_than_its_generation(tmp_path):
    # The catalog generation (available_at=110) is a static snapshot: a query
    # at as_of=105 that predates it is valid training data.
    latest = _write_retrieval_dataset(
        tmp_path,
        row_overrides={
            "as_of_ts": 105.0,
            "feature_event_time": 104.0,
            "feature_available_at": 104.5,
        },
    )

    dataset = asyncio.run(
        RetrievalPitDatasetReader(LocalObjectStorage()).read(str(latest))
    )

    assert dataset.rows[0].as_of_ts == 105.0
    assert dataset.catalog.available_at == 110.0


def test_catalog_newer_than_every_query_is_still_future_data(tmp_path):
    latest = _write_retrieval_dataset(tmp_path, catalog_overrides={"available_at": 121.0})

    with pytest.raises(RetrievalPitDatasetError, match="catalog is future data"):
        asyncio.run(RetrievalPitDatasetReader(LocalObjectStorage()).read(str(latest)))


def test_reader_records_and_verifies_ranker_score_coverage(tmp_path):
    rejected = {
        "label_source": "ranker_rejected",
        "label_type": "ranker_rejected",
        "label_weight": 0.15,
        "label_event_time": 120.0,
        "ranker_score": 1.25,
        "seen_product_ids_json": '["p2"]',
    }
    latest = _write_retrieval_dataset(
        tmp_path / "ok",
        row_overrides=rejected,
        manifest_overrides={
            "ranker_score_coverage": {
                "ranker_rejected_rows": 1,
                "ranker_scored_rows": 1,
                "coverage": 1.0,
            }
        },
    )
    reader = RetrievalPitDatasetReader(LocalObjectStorage())

    dataset = asyncio.run(reader.read(str(latest)))
    source = asyncio.run(
        reader.validate_and_stage(
            str(latest),
            staging_dir=tmp_path / "staging",
            batch_rows=1,
            max_group_rows=16,
            max_bytes=10_000_000,
        )
    )

    assert dataset.manifest.ranker_score_coverage == 1.0
    assert dataset.rows[0].ranker_score == 1.25
    assert dataset.rows[0].seen_product_ids == ("p2",)
    assert source.manifest.ranker_score_coverage == 1.0
    assert source.lineage["ranker_score_coverage"]["ranker_scored_rows"] == 1

    tampered = _write_retrieval_dataset(
        tmp_path / "tampered",
        row_overrides={**rejected, "ranker_score": None},
        manifest_overrides={
            "ranker_score_coverage": {
                "ranker_rejected_rows": 1,
                "ranker_scored_rows": 1,
                "coverage": 1.0,
            }
        },
    )
    with pytest.raises(RetrievalPitDatasetError, match="ranker score coverage"):
        asyncio.run(reader.read(str(tampered)))
