import json

import pytest

from video_commerce.ml.pit_training_dataset import (
    PitTrainingDatasetError,
    PitTrainingDatasetReader,
)
from video_commerce.ml.ranking_features import RANKING_LTR_FEATURE_DEFINITION_VERSION


class LocalObjectStorage:
    async def materialize_for_processing(self, storage_path, *, suggested_suffix=""):
        return storage_path, False


@pytest.mark.asyncio
async def test_pit_training_dataset_reader_requires_completed_matching_manifest(tmp_path):
    dataset = tmp_path / "ranking-pit.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_type": "manifest",
                        "status": "complete",
                        "dataset_version": "iceberg-snapshot-42",
                        "feature_definition_version": RANKING_LTR_FEATURE_DEFINITION_VERSION,
                    }
                ),
                json.dumps(
                    {
                        "record_type": "training_row",
                        "user_id": "user-1",
                        "product_id": "product-1",
                        "action": "view",
                        "as_of_ts": 50.0,
                        "feature_definition_version": RANKING_LTR_FEATURE_DEFINITION_VERSION,
                        "feature_bundle_hash": "a" * 64,
                        "user_features": {"total_interactions": 3},
                        "product_metadata": {"price": 9.0},
                        "context": {},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    loaded = await PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
    ).read(str(dataset))

    assert loaded.dataset_version == "iceberg-snapshot-42"
    assert len(loaded.rows) == 1
    assert loaded.rows[0]["as_of_ts"] == 50.0


@pytest.mark.asyncio
async def test_pit_training_dataset_reader_rejects_feature_definition_mismatch(tmp_path):
    dataset = tmp_path / "ranking-pit.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "record_type": "manifest",
                "status": "complete",
                "dataset_version": "iceberg-snapshot-42",
                "feature_definition_version": "ranking_ltr_v0",
            }
        ),
        encoding="utf-8",
    )

    reader = PitTrainingDatasetReader(
        LocalObjectStorage(),
        expected_feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
    )
    with pytest.raises(PitTrainingDatasetError, match="feature definition version"):
        await reader.read(str(dataset))
