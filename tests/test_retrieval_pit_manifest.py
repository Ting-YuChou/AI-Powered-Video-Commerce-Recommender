import asyncio
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from video_commerce.ml.retrieval_pit_manifest import RetrievalPitManifestPublisher


class LocalObjectStorage:
    async def materialize_for_processing(self, storage_path, *, suggested_suffix=""):
        return str(storage_path), False


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_retrieval_manifest_pins_shards_catalog_and_item_sidecar(tmp_path):
    shard = tmp_path / "part-00000.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "query_id": "q1",
                    "user_id": "u1",
                    "product_id": "p1",
                    "label_source": "viewed_impression",
                    "label_type": "click",
                    "label_weight": 2.0,
                    "as_of_ts": 100.0,
                    "label_event_time": 101.0,
                    "label_available_at": 102.0,
                    "feature_event_time": 99.0,
                    "feature_available_at": 99.5,
                    "catalog_generation_id": "catalog-1",
                    "user_features_json": "{}",
                    "seen_product_ids_json": "[]",
                    "ranker_score": None,
                }
            ]
        ),
        shard,
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text('{"generation_id":"catalog-1"}', encoding="utf-8")
    sidecar = tmp_path / "items.json"
    sidecar.write_text('{"generation_id":"catalog-1"}', encoding="utf-8")

    latest = asyncio.run(
        RetrievalPitManifestPublisher(LocalObjectStorage()).publish(
            shard_uris=[str(shard)],
            output_prefix=str(tmp_path / "published"),
            materialization_run_id="retrieval-pit-1",
            dataset_version="snapshot-1",
            attribution_cutoff=200.0,
            attribution_window_hours=168,
            allowed_lateness_hours=1,
            catalog_generation={
                "generation_id": "catalog-1",
                "uri": str(catalog),
                "sha256": _sha(catalog),
                "product_count": 1,
            },
            item_feature_sidecar={
                "generation_id": "catalog-1",
                "uri": str(sidecar),
                "sha256": _sha(sidecar),
                "schema_version": "retrieval_item_feature_sidecar_v1",
                "embedding_dimension": 2,
                "embedding_model_revision": "clip-v1",
                "available_at": 90.0,
            },
        )
    )

    pointer = json.loads((tmp_path / "published" / "latest.json").read_text())
    manifest = json.loads(open(pointer["manifest_uri"], encoding="utf-8").read())
    assert latest == str(tmp_path / "published" / "latest.json")
    assert manifest["schema_version"] == "retrieval_training_pit_v1"
    assert manifest["catalog_generation"]["sha256"] == _sha(catalog)
    assert manifest["item_feature_sidecar"]["sha256"] == _sha(sidecar)
    assert manifest["row_count"] == 1
