from datetime import datetime, timezone

from video_commerce.data_plane.system_store import build_chronological_user_sequences


def _event(user_id, product_id, action, event_id, timestamp):
    return {
        "event_id": event_id,
        "schema_version": 1,
        "request_id": f"req-{event_id}",
        "user_id": user_id,
        "product_id": product_id,
        "action": action,
        "context": {"page": "test"},
        "occurred_at": datetime.fromtimestamp(timestamp, tz=timezone.utc),
    }


def test_build_chronological_user_sequences_groups_and_orders_per_user():
    sequences = build_chronological_user_sequences(
        [
            _event("u2", "p4", "purchase", "e4", 4),
            _event("u1", "p2", "click", "e2", 2),
            _event("u1", "p1", "view", "e1", 1),
            _event("u2", "p3", "view", "e3", 3),
        ],
        max_events_per_user=10,
        min_sequence_length=2,
    )

    assert [event["product_id"] for event in sequences["u1"]] == ["p1", "p2"]
    assert [event["product_id"] for event in sequences["u2"]] == ["p3", "p4"]


def test_build_chronological_user_sequences_uses_event_id_for_timestamp_ties():
    sequences = build_chronological_user_sequences(
        [
            _event("u1", "p2", "click", "e2", 10),
            _event("u1", "p1", "view", "e1", 10),
        ],
        max_events_per_user=10,
        min_sequence_length=2,
    )

    assert [event["event_id"] for event in sequences["u1"]] == ["e1", "e2"]


def test_build_chronological_user_sequences_filters_noisy_actions():
    sequences = build_chronological_user_sequences(
        [
            _event("u1", "p1", "view", "e1", 1),
            _event("u1", "p2", "remove_from_cart", "e2", 2),
            _event("u1", "p3", "share", "e3", 3),
            _event("u1", "p4", "purchase", "e4", 4),
        ],
        max_events_per_user=10,
        min_sequence_length=2,
    )

    assert [event["product_id"] for event in sequences["u1"]] == ["p1", "p4"]


def test_build_chronological_user_sequences_keeps_recent_window_in_order():
    sequences = build_chronological_user_sequences(
        [
            _event("u1", "p1", "view", "e1", 1),
            _event("u1", "p2", "click", "e2", 2),
            _event("u1", "p3", "add_to_cart", "e3", 3),
            _event("u1", "p4", "purchase", "e4", 4),
        ],
        max_events_per_user=2,
        min_sequence_length=2,
    )

    assert [event["product_id"] for event in sequences["u1"]] == ["p3", "p4"]
