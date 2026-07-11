import pytest

from video_commerce.ml.feature_history_reconciliation import (
    BackfillReconciliationEvidence,
)


def _evidence(**overrides):
    values = {
        "kafka_source_event_ids": frozenset({"e1", "e2"}),
        "iceberg_source_event_ids": frozenset({"e1"}),
        "dlq_source_event_ids": frozenset({"e2"}),
        "iceberg_row_count": 1,
        "kafka_end_offsets": {"topic:0": 2},
        "iceberg_snapshot_ids": {"interaction_history": 42},
    }
    values.update(overrides)
    return BackfillReconciliationEvidence(**values)


def test_reconciliation_requires_exact_run_scoped_identity_sets():
    assert _evidence().validate(2)["duplicate_count"] == 0


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"kafka_source_event_ids": frozenset({"e1"})}, "published count"),
        ({"iceberg_row_count": 2}, "duplicate"),
        ({"dlq_source_event_ids": frozenset()}, "exactly explain"),
    ],
)
def test_reconciliation_fails_closed(overrides, error):
    with pytest.raises(RuntimeError, match=error):
        _evidence(**overrides).validate(2)
