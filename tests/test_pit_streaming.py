import json

import pytest

from video_commerce.ml.pit_streaming import (
    PitSpoolError,
    PitSpoolBuilder,
    VerifiedPitSource,
)


def _row(group_id: str, row_id: str, as_of_ts: float, value: float = 0.0):
    return {
        "group_id": group_id,
        "row_id": row_id,
        "as_of_ts": as_of_ts,
        "business_value": value,
        "value_bucket": 1,
        "is_purchase": value > 0.0,
        "payload": {"group_id": group_id, "row_id": row_id},
    }


def test_verified_spool_keeps_cross_shard_group_in_one_batch(tmp_path):
    builder = PitSpoolBuilder(
        tmp_path,
        generation_id="manifest-a",
        kind="ranking",
        max_group_rows=4,
        max_bytes=10_000_000,
    )
    builder.add_rows([_row("imp-1", "a", 1.0), _row("imp-2", "c", 2.0)])
    builder.add_rows([_row("imp-1", "b", 1.0)])

    source = builder.finalize(expected_rows=3)
    batches = list(source.iter_group_batches(split="all", batch_rows=1, seed=42))

    assert source.row_count == 3
    assert source.group_count == 2
    assert sorted(len(batch) for batch in batches) == [1, 2]
    imp_one = next(batch for batch in batches if batch[0]["group_id"] == "imp-1")
    assert [row["row_id"] for row in imp_one] == ["a", "b"]


def test_spool_rejects_large_group_without_publishing_generation(tmp_path):
    builder = PitSpoolBuilder(
        tmp_path,
        generation_id="manifest-b",
        kind="ranking",
        max_group_rows=1,
        max_bytes=10_000_000,
    )
    builder.add_rows([_row("imp-1", "a", 1.0), _row("imp-1", "b", 1.0)])

    with pytest.raises(PitSpoolError, match="maximum row count"):
        builder.finalize(expected_rows=2)

    assert not (tmp_path / "verified" / "manifest-b").exists()


def test_spool_split_and_order_are_reproducible(tmp_path):
    builder = PitSpoolBuilder(
        tmp_path,
        generation_id="manifest-c",
        kind="ranking",
        max_group_rows=10,
        max_bytes=10_000_000,
    )
    builder.add_rows(
        [
            _row("train-1", "a", 1.0, 10.0),
            _row("train-2", "b", 2.0, 20.0),
            _row("validation", "c", 3.0, 30.0),
            _row("holdout", "d", 900_000.0, 40.0),
        ]
    )
    source = builder.finalize(expected_rows=4)
    source.assign_ranking_splits(
        holdout_days=7,
        attribution_cutoff=1_000_000.0,
        validation_fraction=1 / 3,
    )

    assert source.group_ids("training") == ("train-1", "train-2")
    assert source.group_ids("validation") == ("validation",)
    assert source.group_ids("holdout") == ("holdout",)
    first = [
        row["row_id"]
        for batch in source.iter_group_batches(
            split="training", batch_rows=8, seed=7, epoch=2
        )
        for row in batch
    ]
    reopened = VerifiedPitSource.open(source.path)
    second = [
        row["row_id"]
        for batch in reopened.iter_group_batches(
            split="training", batch_rows=8, seed=7, epoch=2
        )
        for row in batch
    ]
    assert first == second


def test_ranking_split_reserves_calibration_after_validation_and_before_holdout(
    tmp_path,
):
    builder = PitSpoolBuilder(
        tmp_path,
        generation_id="manifest-calibration",
        kind="ranking",
        max_group_rows=10,
        max_bytes=10_000_000,
    )
    builder.add_rows(
        [
            _row("train-1", "a", 1.0),
            _row("train-2", "b", 2.0),
            _row("train-3", "c", 3.0),
            _row("validation", "d", 4.0),
            _row("calibration", "e", 5.0),
            _row("holdout", "f", 900_000.0),
        ]
    )
    source = builder.finalize(expected_rows=6)

    source.assign_ranking_splits(
        holdout_days=7,
        attribution_cutoff=1_000_000.0,
        validation_fraction=0.25,
        calibration_fraction=0.20,
    )

    assert source.group_ids("training") == ("train-1", "train-2", "train-3")
    assert source.group_ids("validation") == ("validation",)
    assert source.group_ids("calibration") == ("calibration",)
    assert source.group_ids("holdout") == ("holdout",)


def test_ranking_split_purges_groups_labelled_inside_holdout_reach(tmp_path):
    builder = PitSpoolBuilder(
        tmp_path,
        generation_id="manifest-purge",
        kind="ranking",
        max_group_rows=10,
        max_bytes=10_000_000,
    )
    day = 86400.0
    builder.add_rows(
        [
            _row("old", "a", 1.0),
            _row("near-holdout", "b", 10 * day),
            _row("holdout", "c", 20 * day),
        ]
    )
    source = builder.finalize(expected_rows=3)

    source.assign_ranking_splits(
        holdout_days=7,
        attribution_cutoff=30 * day,
        validation_fraction=0.0,
        purge_seconds=7 * day,
    )

    assert source.group_ids("training") == ("old",)
    assert source.group_ids("purged") == ("near-holdout",)
    assert source.group_ids("holdout") == ("holdout",)


def test_spool_value_stats_match_numpy_linear_quantile(tmp_path):
    builder = PitSpoolBuilder(
        tmp_path,
        generation_id="manifest-d",
        kind="ranking",
        max_group_rows=10,
        max_bytes=10_000_000,
    )
    builder.add_rows(
        [
            _row("a", "a", 1.0, 1.0),
            _row("b", "b", 2.0, 2.0),
            _row("c", "c", 3.0, 3.0),
            _row("d", "d", 4.0, 100.0),
        ]
    )
    source = builder.finalize(expected_rows=4)

    stats = source.compute_value_transform_stats(split="all", quantile=0.75)

    assert stats["global"]["count"] == 4
    assert stats["global"]["clip"] == pytest.approx(27.25)
    assert json.loads((source.path / "metadata.json").read_text())["row_count"] == 4
