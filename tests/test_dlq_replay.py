import hashlib
import json
import time

from scripts.replay_kafka_dlq import (
    ReplayPolicy,
    build_audit_entry,
    inspect_envelope,
)


def _envelope(**overrides):
    payload = {
        "schema_version": 2,
        "event_id": "dlq-1",
        "source_topic": "user-interactions",
        "source_partition": 0,
        "source_offset": 7,
        "source_key": "u1",
        "source_value": {"event_id": "event-1", "user_id": "u1"},
        "source_headers": [["x-request-id", "req-1"]],
        "occurred_at": time.time(),
    }
    payload.update(overrides)
    return payload


def test_valid_dlq_event_is_replayable_and_hash_is_stable():
    decision = inspect_envelope(
        _envelope(), ReplayPolicy({"user-interactions"}, max_age_days=7)
    )

    assert decision.replayable is True
    assert decision.event_id == "event-1"
    expected = hashlib.sha256(
        json.dumps(
            {"event_id": "event-1", "user_id": "u1"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert decision.payload_hash == expected
    assert decision.source_partition == 0
    assert decision.source_offset == 7


def test_audit_entry_excludes_replayed_content_and_preserves_lineage():
    decision = inspect_envelope(
        _envelope(), ReplayPolicy({"user-interactions"}, max_age_days=7)
    )

    audit = build_audit_entry(
        decision, dlq_identity="dead-letter-events:0:12"
    )

    assert audit["payload_hash"] == decision.payload_hash
    assert audit["source_partition"] == 0
    assert audit["source_offset"] == 7
    assert "source_value" not in audit
    assert "source_key" not in audit
    assert "source_headers" not in audit


def test_dlq_policy_rejects_old_disallowed_and_unstable_events():
    policy = ReplayPolicy({"user-interactions"}, max_age_days=7)

    assert inspect_envelope(
        _envelope(occurred_at=time.time() - 8 * 86400), policy
    ).reason == "outside_replay_horizon"
    assert inspect_envelope(
        _envelope(source_topic="video-processing-tasks"), policy
    ).reason == "source_topic_not_allowed"
    assert inspect_envelope(
        _envelope(source_value={"user_id": "u1"}), policy
    ).reason == "missing_stable_event_id"


def test_v1_string_payload_remains_compatible():
    raw = json.dumps({"event_id": "event-1", "user_id": "u1"})
    decision = inspect_envelope(
        _envelope(schema_version=1, source_value=raw),
        ReplayPolicy({"user-interactions"}, max_age_days=7),
    )

    assert decision.replayable is True
    assert decision.source_value == raw
