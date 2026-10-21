"""Micro-batched served-impression outbox writes and late-commit retires.

Recommendation responses tend to finish in bursts (ranking runs in batches),
and one transaction per impression made each burst compete for a few pooled
connections: per process, at most ``max_inflight`` writes ran and the rest of
the burst was served untracked. The batcher instead coalesces rows that arrive
while a write is in flight into one multi-row transaction, so a burst costs
one or two transactions and the same bounded number of connections.

Rows whose COMMIT outcome the client never saw (a late COMMIT, or an error
raised by COMMIT) are retired as ``not_exposed`` by one retirer per process,
which also batches: one UPDATE at a time, however many rows a slow batch
left behind.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
)

# Receives (event, commit_deadline) pairs; returns one outcome or exception per
# pair, in order, or raises to fail the whole batch.
OutboxBatchWriter = Callable[[Sequence[tuple], float], Awaitable[List[Any]]]
# Receives event_ids; returns {event_id: status} or raises to retry them all.
RetireManyWriter = Callable[[List[str]], Awaitable[Dict[str, str]]]

EXPIRED = "expired"


@dataclass
class _PendingImpression:
    event: Dict[str, Any]
    commit_deadline: float
    future: asyncio.Future = field(repr=False)


class ImpressionOutboxBatcher:
    """Coalesce served-impression outbox inserts into bounded transactions.

    ``max_inflight`` caps concurrent write transactions (pooled connections)
    per process, ``max_batch_size`` caps rows per transaction, and at most
    ``max_pending`` rows wait; beyond that ``submit`` refuses instead of
    queueing. A queued row whose deadline passes before its batch starts is
    resolved ``"expired"`` without touching the database.

    Writes are never cancelled while serving: the store bounds them server
    side (statement timeout) and rolls back rows that missed their deadline.
    Only ``close`` cancels, after a bounded wait at shutdown.
    """

    def __init__(
        self,
        writer: OutboxBatchWriter,
        *,
        max_inflight: int,
        max_batch_size: int,
        max_pending: int,
        publish_hold_seconds: float,
    ) -> None:
        self._writer = writer
        self._max_inflight = max(1, int(max_inflight))
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_pending = max(1, int(max_pending))
        self._publish_hold_seconds = float(publish_hold_seconds)
        self._pending: Deque[_PendingImpression] = deque()
        self._inflight = 0
        self._write_started_at: Dict[int, float] = {}
        self._tasks: Set[asyncio.Task] = set()
        self._closed = False

    @property
    def max_inflight(self) -> int:
        return self._max_inflight

    @property
    def publish_hold_seconds(self) -> float:
        return self._publish_hold_seconds

    @property
    def pending(self) -> int:
        return len(self._pending)

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def closed(self) -> bool:
        return self._closed

    def oldest_write_age_seconds(self) -> float:
        """Age of the longest-running write transaction, 0 when idle."""
        if not self._write_started_at:
            return 0.0
        return max(0.0, time.monotonic() - min(self._write_started_at.values()))

    def submit(
        self, event: Dict[str, Any], commit_deadline: float
    ) -> Optional[asyncio.Future]:
        """Queue one row; return None when the queue is full or closed."""
        if self._closed or len(self._pending) >= self._max_pending:
            return None
        future = asyncio.get_running_loop().create_future()
        self._pending.append(_PendingImpression(event, commit_deadline, future))
        self._maybe_start_flush()
        return future

    def _maybe_start_flush(self) -> None:
        if self._pending and self._inflight < self._max_inflight:
            self._inflight += 1
            task = asyncio.create_task(self._flush(), name="impression-outbox-flush")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def _take_batch(self) -> List[_PendingImpression]:
        batch: List[_PendingImpression] = []
        now = time.monotonic()
        while self._pending and len(batch) < self._max_batch_size:
            entry = self._pending.popleft()
            if entry.future.done():
                continue
            if now >= entry.commit_deadline:
                entry.future.set_result(EXPIRED)
                continue
            batch.append(entry)
        return batch

    async def _flush(self) -> None:
        try:
            # Let the other responses of a burst, resumed in this same loop
            # turn, join the batch.
            await asyncio.sleep(0)
            while True:
                batch = self._take_batch()
                if not batch:
                    return
                await self._write(batch)
        finally:
            self._inflight -= 1
            self._maybe_start_flush()

    async def _write(self, batch: List[_PendingImpression]) -> None:
        key = id(batch)
        self._write_started_at[key] = time.monotonic()
        try:
            outcomes = await self._writer(
                [(entry.event, entry.commit_deadline) for entry in batch],
                self._publish_hold_seconds,
            )
            if len(outcomes) != len(batch):
                raise RuntimeError("outbox batch writer returned a wrong outcome count")
        except asyncio.CancelledError as exc:
            # Cancelled mid-write (shutdown): whether it committed is unknown.
            outcomes = [RuntimeError(f"outbox batch write cancelled: {exc!r}")] * len(
                batch
            )
            self._resolve(batch, outcomes)
            raise
        except Exception as exc:
            outcomes = [exc] * len(batch)
        finally:
            self._write_started_at.pop(key, None)
        self._resolve(batch, outcomes)

    @staticmethod
    def _resolve(batch: List[_PendingImpression], outcomes: List[Any]) -> None:
        for entry, outcome in zip(batch, outcomes):
            if entry.future.done():
                continue
            if isinstance(outcome, BaseException):
                entry.future.set_exception(outcome)
            else:
                entry.future.set_result(outcome)

    async def close(self, timeout_seconds: float = 2.0) -> None:
        """Refuse new rows, let in-flight writes finish, then cancel the rest.

        A write cancelled here can leave its COMMIT outcome unknown; its rows
        resolve with an error so callers still schedule their retire.
        """
        self._closed = True
        while self._pending:
            entry = self._pending.popleft()
            if not entry.future.done():
                entry.future.set_result(EXPIRED)
        tasks = list(self._tasks)
        if not tasks:
            return
        _, still_running = await asyncio.wait(tasks, timeout=timeout_seconds)
        for task in still_running:
            task.cancel()
        if still_running:
            await asyncio.gather(*still_running, return_exceptions=True)


@dataclass
class _PendingRetire:
    event_id: str
    impression_id: str
    retire_by: float


class LateImpressionRetirer:
    """Retire late-committed impressions, one batched UPDATE at a time.

    A slow COMMIT can leave a whole batch of rows whose responses went out
    without their impression_id. Retiring them one transaction per row would
    take more connections exactly when the database is slow, so rows are
    queued here and retired together by a single worker. Failed attempts are
    retried with backoff until each row's ``retire_by`` (the end of its
    publish hold, less a margin); after that a dispatcher may already claim
    it, so the row is reported through ``on_result(entry, None)``.
    """

    def __init__(
        self,
        retire_many: RetireManyWriter,
        *,
        on_result: Callable[[_PendingRetire, Optional[str]], None],
        on_error: Callable[[List[_PendingRetire], int, BaseException], None],
        retry_delays_seconds: Sequence[float] = (0.2, 0.5, 1.0),
        max_batch_size: int = 256,
    ) -> None:
        self._retire_many = retire_many
        self._on_result = on_result
        self._on_error = on_error
        self._retry_delays = tuple(retry_delays_seconds) or (0.5,)
        self._max_batch_size = max(1, int(max_batch_size))
        self._pending: Deque[_PendingRetire] = deque()
        self._task: Optional[asyncio.Task] = None

    @property
    def pending(self) -> int:
        return len(self._pending)

    def submit(self, event_id: str, impression_id: str, retire_by: float) -> None:
        self._pending.append(_PendingRetire(event_id, impression_id, retire_by))
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(
                self._run(), name="retire-unexposed-impressions"
            )

    async def _run(self) -> None:
        attempt = 0
        while self._pending:
            now = time.monotonic()
            batch: List[_PendingRetire] = []
            while self._pending and len(batch) < self._max_batch_size:
                entry = self._pending.popleft()
                if attempt and now >= entry.retire_by:
                    # Every attempt failed within its hold; give up on it.
                    self._on_result(entry, None)
                    continue
                batch.append(entry)
            if not batch:
                attempt = 0
                continue
            try:
                statuses = await self._retire_many([entry.event_id for entry in batch])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempt += 1
                self._on_error(batch, attempt, exc)
                # Retry these first; rows queued meanwhile join the retry.
                self._pending.extendleft(reversed(batch))
                await asyncio.sleep(
                    self._retry_delays[min(attempt, len(self._retry_delays)) - 1]
                )
                continue
            attempt = 0
            for entry in batch:
                self._on_result(entry, statuses.get(entry.event_id, "missing"))

    async def close(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
