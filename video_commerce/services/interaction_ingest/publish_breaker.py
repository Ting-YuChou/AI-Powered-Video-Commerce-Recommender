"""Consecutive-failure breaker for interaction ingest's Kafka publishes."""

from __future__ import annotations

import time
from typing import Callable


class PublishBreaker:
    """Fail fast after repeated publish timeouts until one probe succeeds.

    aiokafka keeps a started producer looking connected through a broker
    outage, so without this every request would wait out the full publish
    timeout. Once ``failure_threshold`` publishes in a row time out or fail,
    callers are refused for ``cooldown_seconds``; after that one probe publish
    is let through, and its success closes the breaker while its failure starts
    another cooldown.
    """

    def __init__(
        self,
        failure_threshold: int,
        cooldown_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._failures = 0
        self._open_until = 0.0
        self._probing = False

    @property
    def tripped(self) -> bool:
        return self._failures >= self.failure_threshold

    def blocking(self) -> bool:
        """True while callers must be refused without starting a publish."""
        return self.tripped and (self._clock() < self._open_until or self._probing)

    def admit(self) -> bool:
        """Claim the right to publish; a tripped breaker admits one probe."""
        if not self.tripped:
            return True
        if self.blocking():
            return False
        self._probing = True
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._probing = False

    def record_failure(self) -> None:
        self._failures += 1
        self._probing = False
        if self.tripped:
            self._open_until = self._clock() + self.cooldown_seconds

    def release_probe(self) -> None:
        """Forget an admitted publish whose outcome nobody is waiting for."""
        self._probing = False
