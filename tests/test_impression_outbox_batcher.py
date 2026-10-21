"""Served-impression outbox micro-batcher: coalescing, bounds, outcomes."""

from __future__ import annotations

import asyncio
import time

import pytest

from video_commerce.services.recommendation.impression_outbox import (
    EXPIRED,
    REFUSED_ADMISSION,
    REFUSED_CLOSED,
    REFUSED_QUEUE_FULL,
    ImpressionOutboxBatcher,
    LateImpressionRetirer,
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
    assert batcher.submit(_event(2), _deadline()) == REFUSED_QUEUE_FULL
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


@pytest.mark.asyncio
async def test_write_age_tracks_the_oldest_inflight_write_and_close_refuses():
    writer = RecordingWriter(delay_seconds=0.1)
    batcher = _batcher(writer)

    assert batcher.oldest_write_age_seconds() == 0.0
    future = batcher.submit(_event(0), _deadline())
    await asyncio.sleep(0.05)
    assert 0.03 <= batcher.oldest_write_age_seconds() < 0.1
    await future
    assert batcher.oldest_write_age_seconds() == 0.0

    await batcher.close()
    assert batcher.closed
    assert batcher.submit(_event(1), _deadline()) == REFUSED_CLOSED


class RecordingRetire:
    def __init__(self, failures=0):
        self.calls = []
        self.failures = failures

    async def __call__(self, event_ids):
        self.calls.append(list(event_ids))
        await asyncio.sleep(0.01)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("db slow")
        return {event_id: "retired" for event_id in event_ids}


def _retirer(retire, results, errors, delays=(0.01,)):
    return LateImpressionRetirer(
        retire,
        on_result=lambda entry, status: results.append((entry.event_id, status)),
        on_error=lambda batch, attempt, exc: errors.append((len(batch), attempt)),
        retry_delays_seconds=delays,
    )


@pytest.mark.asyncio
async def test_retirer_batches_rows_and_runs_one_update_at_a_time():
    retire = RecordingRetire()
    results, errors = [], []
    retirer = _retirer(retire, results, errors)

    for index in range(40):
        retirer.submit(f"e{index}", f"i{index}", time.monotonic() + 5)
    await asyncio.sleep(0.005)
    # Rows arriving while an UPDATE runs wait for the next one.
    for index in range(40, 45):
        retirer.submit(f"e{index}", f"i{index}", time.monotonic() + 5)
    await asyncio.sleep(0.1)

    assert [len(call) for call in retire.calls] == [40, 5]
    assert len(results) == 45 and {status for _, status in results} == {"retired"}


@pytest.mark.asyncio
async def test_retirer_retries_until_retire_by_then_gives_up():
    retire = RecordingRetire(failures=2)
    results, errors = [], []
    retirer = _retirer(retire, results, errors)

    retirer.submit("e0", "i0", time.monotonic() + 5)
    await asyncio.sleep(0.15)
    assert results == [("e0", "retired")]
    assert [attempt for _, attempt in errors] == [1, 2]

    retire.failures = 1000
    retirer.submit("e1", "i1", time.monotonic() + 0.05)
    await asyncio.sleep(0.2)
    assert results[-1] == ("e1", None)
    assert retirer.pending == 0
    await retirer.close()


# ------------------------------------------------------------ admission


async def _warm(batcher, writer, seconds):
    """Run one write of ``seconds`` so the duration estimate is ``seconds``."""
    writer.delay_seconds = seconds
    await batcher.submit(_event("warm"), _deadline())
    assert batcher.write_seconds_estimate == pytest.approx(seconds, abs=0.02)


@pytest.mark.asyncio
async def test_cold_estimator_admits_even_with_every_slot_busy():
    writer = RecordingWriter(delay_seconds=0.05)
    batcher = _batcher(writer, max_inflight=1)

    busy = batcher.submit(_event(0), _deadline())
    await asyncio.sleep(0.01)
    # No write has finished yet: nothing to predict from, so admit.
    assert batcher.write_seconds_estimate is None
    queued = batcher.submit(_event(1), _deadline(0.01))
    assert isinstance(queued, asyncio.Future)
    await asyncio.gather(busy, queued)


@pytest.mark.asyncio
async def test_saturated_writer_refuses_rows_that_cannot_make_their_budget():
    writer = RecordingWriter()
    batcher = _batcher(writer, max_inflight=1, max_batch_size=2)
    await _warm(batcher, writer, 0.08)

    busy = batcher.submit(_event(0), _deadline())
    await asyncio.sleep(0.01)
    # Behind a write with ~70 ms left plus its own ~80 ms write.
    assert batcher.predicted_wait_seconds() == pytest.approx(0.15, abs=0.03)
    assert batcher.submit(_event(1), _deadline(0.1)) == REFUSED_ADMISSION
    # A larger budget still fits.
    admitted = batcher.submit(_event(2), _deadline(1.0))
    assert isinstance(admitted, asyncio.Future)
    # Two more queued rows fill the next batch; the row after them needs a
    # third round.
    batcher.submit(_event(3), _deadline(1.0))
    assert batcher.predicted_wait_seconds() == pytest.approx(0.23, abs=0.03)
    await asyncio.gather(busy, admitted)
    await asyncio.sleep(0.2)
    assert all("e1" not in batch for batch in writer.batches)


@pytest.mark.asyncio
async def test_expired_budget_is_refused_without_queueing():
    writer = RecordingWriter()
    batcher = _batcher(writer)

    assert batcher.submit(_event(0), time.monotonic() - 0.001) == REFUSED_ADMISSION
    assert batcher.pending == 0 and writer.batches == []


@pytest.mark.asyncio
async def test_idle_batcher_admits_despite_a_slow_estimate_and_relearns():
    writer = RecordingWriter()
    batcher = _batcher(writer)
    await _warm(batcher, writer, 0.2)

    # Nothing in flight: the row is a probe even though the estimate (200 ms)
    # is over its 100 ms budget, so a recovered DB is noticed.
    writer.delay_seconds = 0.0
    probe = batcher.submit(_event(0), _deadline(0.1))
    assert isinstance(probe, asyncio.Future)
    assert await probe == "inserted"
    assert batcher.write_seconds_estimate < 0.2
