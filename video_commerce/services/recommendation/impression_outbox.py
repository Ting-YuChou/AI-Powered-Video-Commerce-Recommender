"""Micro-batched served-impression outbox writes.

Recommendation responses tend to finish in bursts (ranking runs in batches),
and one transaction per impression made each burst compete for a few pooled
connections: per process, at most ``max_inflight`` writes ran and the rest of
the burst was served untracked. The batcher instead coalesces rows that arrive
while a write is in flight into one multi-row transaction, so a burst costs
one or two transactions and the same bounded number of connections.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Sequence, Set

# Receives (event, commit_deadline) pairs; returns one outcome or exception per
# pair, in order, or raises to fail the whole batch.
OutboxBatchWriter = Callable[[Sequence[tuple], float], Awaitable[List[Any]]]

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
        self._tasks: Set[asyncio.Task] = set()

    @property
    def pending(self) -> int:
        return len(self._pending)

    @property
    def inflight(self) -> int:
        return self._inflight

    def submit(
        self, event: Dict[str, Any], commit_deadline: float
    ) -> Optional[asyncio.Future]:
        """Queue one row; return None when the queue is full."""
        if len(self._pending) >= self._max_pending:
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
        """Let in-flight writes finish (bounded), then cancel the rest."""
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
