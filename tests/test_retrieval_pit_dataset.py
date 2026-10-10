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


def _write_retrieval_dataset(tmp_path, *, row_overrides=None, catalog_overrides=None):
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
