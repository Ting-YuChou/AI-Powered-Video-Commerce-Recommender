"""Event-time invariants shared by public ingestion and offline replay."""

from __future__ import annotations

import math

PUBLIC_EVENT_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
PUBLIC_EVENT_MAX_FUTURE_SKEW_SECONDS = 5 * 60


def validate_public_event_time(event_time: float, server_received_at: float) -> float:
    """Validate an untrusted client event timestamp against receive time.

    Replay/backfill jobs intentionally do not call this function: their source
    records are governed separately and preserve their original availability
    time. Public clients may only submit events in the bounded acceptance
    window, which prevents an accidental or malicious timestamp from changing
    online state or future point-in-time training joins.
    """
    resolved_event_time = float(event_time)
    received_at = float(server_received_at)
    if not math.isfinite(resolved_event_time):
        raise ValueError("event_time must be a finite Unix timestamp")
    if not math.isfinite(received_at):
        raise ValueError("server_received_at must be a finite Unix timestamp")
    if resolved_event_time < received_at - PUBLIC_EVENT_MAX_AGE_SECONDS:
        raise ValueError("event_time is older than the 30-day public acceptance window")
    if resolved_event_time > received_at + PUBLIC_EVENT_MAX_FUTURE_SKEW_SECONDS:
        raise ValueError("event_time exceeds the 5-minute future acceptance window")
    return resolved_event_time
