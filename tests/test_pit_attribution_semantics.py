"""Ranking label attribution must agree between the legacy path and the Flink PIT job.

The Flink SQL itself is executed against the same scenario in
``flink-jobs/.../PointInTimeAttributionSqlTest``; this module pins the Python side
and the cross-language version contract.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

from video_commerce.common.config import FeatureLakeConfig
from video_commerce.data_plane.system_store import (
    build_ltr_training_samples_from_impression_records,
)
from video_commerce.ml.ranking_training import RANKING_LABEL_DEFINITION_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
PIT_JOB = (
    REPO_ROOT
    / "flink-jobs/interaction-features/src/main/java/com/videocommerce/flink"
    / "PointInTimeFeatureJoinJob.java"
)
BASE = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _impression(impression_id: str, offset_seconds: int) -> dict:
    return {
        "impression_id": impression_id,
        "user_id": "u1",
        "product_id": "p1",
        "position": 1,
        "context": {},
        "created_at": BASE + timedelta(seconds=offset_seconds),
    }


def _interaction(action: str, impression_id: str | None, offset_seconds: int) -> dict:
    context = {"price": 20.0}
    if impression_id is not None:
        context["impression_id"] = impression_id
    return {
        "event_id": f"{action}-{impression_id}-{offset_seconds}",
        "user_id": "u1",
        "product_id": "p1",
        "action": action,
        "context": context,
        "occurred_at": BASE + timedelta(seconds=offset_seconds),
    }


def test_three_impressions_and_one_purchase_yield_exactly_one_positive():
    impressions = [
        _impression("imp-1", 0),
        _impression("imp-2", 1_000),
        _impression("imp-3", 2_000),
    ]
    interactions = [
        _interaction("click", "imp-3", 2_100),
        _interaction("purchase", "imp-3", 2_500),
        # Organic: no impression context, so no slate gets the credit.
        _interaction("purchase", None, 2_600),
    ]
    views = [
        {"impression_id": item["impression_id"], "product_id": "p1"}
        for item in impressions
    ]

    samples = build_ltr_training_samples_from_impression_records(
        impressions,
        interactions,
        impression_views=views,
        attribution_window_hours=168,
    )

    outcomes = {
        sample["context"]["impression_id"]: sample["action"] for sample in samples
    }
    assert outcomes == {"imp-1": "view", "imp-2": "view", "imp-3": "purchase"}
    assert sum(sample["context"]["attributed_purchase"] for sample in samples) == 1
    assert sum(sample["context"]["attributed_click"] for sample in samples) == 1


def test_pit_label_version_and_feature_max_age_match_the_flink_job():
    source = PIT_JOB.read_text(encoding="utf-8")
    label_version = re.search(r'LABEL_DEFINITION_VERSION = "([^"]+)"', source).group(1)
    max_age = re.search(r"DEFAULT_FEATURE_MAX_AGE_SECONDS = ([0-9_]+)L", source).group(
        1
    )

    assert label_version == RANKING_LABEL_DEFINITION_VERSION
    assert RANKING_LABEL_DEFINITION_VERSION != "ranking_labels_v1"
    assert int(max_age.replace("_", "")) == FeatureLakeConfig().feature_max_age_seconds
