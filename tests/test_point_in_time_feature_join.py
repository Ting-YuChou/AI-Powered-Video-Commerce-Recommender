import pytest

from video_commerce.ml.point_in_time import (
    HistoricalFeatureRecord,
    PointInTimeFeatureJoiner,
)
from video_commerce.ml.ranking_features import RANKING_LTR_FEATURE_DEFINITION_VERSION


def _record(entity_type, entity_id, event_time, available_at, values, *, version=None):
    return HistoricalFeatureRecord(
        entity_type=entity_type,
        entity_id=entity_id,
        event_time=event_time,
        available_at=available_at,
        feature_definition_version=version or RANKING_LTR_FEATURE_DEFINITION_VERSION,
        values=values,
    )


def test_point_in_time_join_excludes_features_not_available_at_observation_time():
    joiner = PointInTimeFeatureJoiner(RANKING_LTR_FEATURE_DEFINITION_VERSION)
    records = [
        _record("user", "user-1", 10.0, 10.0, {"total_interactions": 1}),
        # Its event time is earlier, but the computation completed after the impression.
        _record("user", "user-1", 20.0, 100.0, {"total_interactions": 999}),
    ]

    selected = joiner.latest_available(
        records,
        entity_type="user",
        entity_id="user-1",
        as_of_ts=50.0,
    )

    assert selected is not None
    assert selected.values["total_interactions"] == 1


def test_point_in_time_join_requires_the_shared_feature_definition_version():
    joiner = PointInTimeFeatureJoiner(RANKING_LTR_FEATURE_DEFINITION_VERSION)
    records = [
        _record(
            "item",
            "product-1",
            10.0,
            10.0,
            {"price": 9.0},
            version="ranking_ltr_v0",
        )
    ]

    assert (
        joiner.latest_available(
            records,
            entity_type="item",
            entity_id="product-1",
            as_of_ts=50.0,
        )
        is None
    )


def test_pit_observation_captures_versioned_bundle_and_stable_hash():
    joiner = PointInTimeFeatureJoiner(RANKING_LTR_FEATURE_DEFINITION_VERSION)
    records = [
        _record("user", "user-1", 10.0, 10.0, {"total_interactions": 3}),
        _record("item", "product-1", 11.0, 11.0, {"price": 9.0, "category": "shoes"}),
    ]

    observation = joiner.build_ranking_observation(
        observation_id="impression-1:product-1",
        user_id="user-1",
        product_id="product-1",
        as_of_ts=50.0,
        records=records,
        context={"impression_id": "impression-1"},
    )
    training_row = observation.as_training_row(action="view")

    assert training_row["as_of_ts"] == 50.0
    assert training_row["feature_definition_version"] == RANKING_LTR_FEATURE_DEFINITION_VERSION
    assert training_row["user_features"]["total_interactions"] == 3
    assert training_row["product_metadata"]["price"] == 9.0
    assert len(training_row["feature_bundle_hash"]) == 64
    assert training_row["feature_bundle_hash"] == observation.as_training_row(
        action="view"
    )["feature_bundle_hash"]
