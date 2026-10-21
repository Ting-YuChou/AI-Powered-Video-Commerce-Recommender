"""Served-impression outbox micro-batcher: coalescing, bounds, outcomes."""

from __future__ import annotations

import asyncio
import time

import pytest

from video_commerce.services.recommendation.impression_outbox import (
    EXPIRED,
    ImpressionOutboxBatcher,
)


class RecordingWriter:
    def __init__(self, delay_seconds=0.0, outcomes=None, error=None):
        self.delay_seconds = delay_seconds
        self.outcomes = outcomes
        self.error = error
        self.batches = []
        self.holds = []
        self.active = 0
        self.max_active = 0

    async def __call__(self, entries, publish_hold_seconds):
        self.batches.append([event["event_id"] for event, _deadline in entries])
        self.holds.append(publish_hold_seconds)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay_seconds)
        finally:
            self.active -= 1
        if self.error is not None:
            raise self.error
        if self.outcomes is not None:
            return [self.outcomes(event) for event, _deadline in entries]
        return ["inserted"] * len(entries)


def _batcher(writer, *, max_inflight=1, max_batch_size=32, max_pending=64):
    return ImpressionOutboxBatcher(
        writer,
        max_inflight=max_inflight,
        max_batch_size=max_batch_size,
        max_pending=max_pending,
        publish_hold_seconds=1.1,
    )


def _event(index):
    return {"event_id": f"e{index}"}


def _deadline(seconds=5.0):
    return time.monotonic() + seconds


@pytest.mark.asyncio
async def test_burst_in_one_loop_turn_is_written_in_one_transaction():
    writer = RecordingWriter()
    batcher = _batcher(writer, max_inflight=2)

    futures = [batcher.submit(_event(index), _deadline()) for index in range(10)]
    outcomes = await asyncio.gather(*futures)

    assert outcomes == ["inserted"] * 10
    assert writer.batches == [[f"e{index}" for index in range(10)]]
    assert writer.holds == [1.1]
    assert batcher.inflight == 0 and batcher.pending == 0


@pytest.mark.asyncio
async def test_rows_arriving_during_a_write_join_the_next_batch_within_the_cap():
    writer = RecordingWriter(delay_seconds=0.05)
    batcher = _batcher(writer, max_inflight=2)

    first = batcher.submit(_event(0), _deadline())
    await asyncio.sleep(0.01)
    second = batcher.submit(_event(1), _deadline())
    await asyncio.sleep(0.01)
    # Both transactions are busy now: these wait and go out together.
    later = [batcher.submit(_event(index), _deadline()) for index in range(2, 6)]
    await asyncio.gather(first, second, *later)

    assert writer.batches == [["e0"], ["e1"], ["e2", "e3", "e4", "e5"]]
    assert writer.max_active == 2


@pytest.mark.asyncio
async def test_batches_are_split_at_max_batch_size():
    writer = RecordingWriter()
    batcher = _batcher(writer, max_batch_size=3)

    await asyncio.gather(
        *(batcher.submit(_event(index), _deadline()) for index in range(7))
    )

    assert [len(batch) for batch in writer.batches] == [3, 3, 1]


@pytest.mark.asyncio
async def test_full_queue_refuses_instead_of_queueing():
    writer = RecordingWriter(delay_seconds=0.05)
    batcher = _batcher(writer, max_pending=2)

    accepted = [batcher.submit(_event(index), _deadline()) for index in range(2)]
    assert batcher.submit(_event(2), _deadline()) is None
    await asyncio.gather(*accepted)
    # Space frees up once the queue drains.
    assert await batcher.submit(_event(3), _deadline()) == "inserted"


@pytest.mark.asyncio
async def test_rows_past_their_deadline_are_never_written():
    writer = RecordingWriter(delay_seconds=0.1)
    batcher = _batcher(writer)

    busy = batcher.submit(_event(0), _deadline())
    await asyncio.sleep(0.01)
    stale = batcher.submit(_event(1), _deadline(0.02))
    fresh = batcher.submit(_event(2), _deadline())

    assert await stale == EXPIRED
    await asyncio.gather(busy, fresh)
    assert writer.batches == [["e0"], ["e2"]]


@pytest.mark.asyncio
async def test_per_row_outcomes_and_batch_errors_reach_each_caller():
    rejected = ValueError("collision")
    writer = RecordingWriter(
        outcomes=lambda event: rejected if event["event_id"] == "e1" else "duplicate"
    )
    batcher = _batcher(writer)

    results = await asyncio.gather(
        *(batcher.submit(_event(index), _deadline()) for index in range(3)),
        return_exceptions=True,
    )
    assert results == ["duplicate", rejected, "duplicate"]

    writer.error = ConnectionResetError("lost")
    results = await asyncio.gather(
        *(batcher.submit(_event(index), _deadline()) for index in range(3, 5)),
        return_exceptions=True,
    )
    assert all(isinstance(result, ConnectionResetError) for result in results)
    # A failed batch does not wedge the batcher.
    writer.error = None
    assert await batcher.submit(_event(5), _deadline()) == "duplicate"


@pytest.mark.asyncio
async def test_writer_outcome_count_mismatch_fails_the_batch():
    async def short_writer(entries, publish_hold_seconds):
        return []

    batcher = _batcher(short_writer)
    with pytest.raises(RuntimeError, match="outcome count"):
        await batcher.submit(_event(0), _deadline())


@pytest.mark.asyncio
async def test_close_lets_inflight_write_finish_and_expires_queued_rows():
    writer = RecordingWriter(delay_seconds=0.05)
    batcher = _batcher(writer)

    inflight = batcher.submit(_event(0), _deadline())
    await asyncio.sleep(0.01)
    queued = batcher.submit(_event(1), _deadline())
    await batcher.close()

    assert await inflight == "inserted"
    assert await queued == EXPIRED
    assert writer.batches == [["e0"]]
