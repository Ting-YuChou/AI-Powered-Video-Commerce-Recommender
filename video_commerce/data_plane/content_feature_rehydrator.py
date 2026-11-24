"""Re-project durable content features into Redis after the projection expires.

Redis holds content features with ``CACHE_CONTENT_FEATURES_TTL``, while Postgres
``content_feature_artifacts`` is authoritative. Nothing else rewrites the Redis
key after the content worker completes, so without this a video loses its
content features on the serving path one TTL after it was processed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from video_commerce.common.models import ContentFeatures

logger = logging.getLogger(__name__)


class ContentFeatureRehydrator:
    """Load one content's durable features on a Redis miss and re-project them.

    Concurrent misses for the same content share one load. Callers await the
    load under their own timeout; ``asyncio.shield`` keeps it running past that
    timeout, so a slow load still repairs Redis for the next request. Content
    IDs without a durable artifact are remembered for ``negative_ttl_seconds``
    so unknown IDs cannot turn every request into a Postgres read. Read errors
    are not remembered and retry on the next miss.
    """

    def __init__(
        self,
        *,
        feature_store: Any,
        system_store: Any,
        max_concurrency: int = 4,
        max_inflight: int = 256,
        negative_ttl_seconds: float = 60.0,
        max_negative_entries: int = 10000,
        observability: Any = None,
    ):
        self._feature_store = feature_store
        self._system_store = system_store
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._max_inflight = max(1, int(max_inflight))
        self._negative_ttl_seconds = max(0.0, float(negative_ttl_seconds))
        self._max_negative_entries = max(0, int(max_negative_entries))
        self._observability = observability
        self._inflight: Dict[str, asyncio.Task] = {}
        self._not_found_until: "OrderedDict[str, float]" = OrderedDict()

    async def load(self, content_id: str) -> Optional[ContentFeatures]:
        if not content_id:
            return None
        if self._recently_not_found(content_id):
            self._record("negative_cached")
            return None
        task = self._inflight.get(content_id)
        if task is None:
            if len(self._inflight) >= self._max_inflight:
                self._record("shed")
                return None
            task = asyncio.create_task(
                self._rehydrate(content_id),
                name=f"content-feature-rehydrate-{content_id}",
            )
            self._inflight[content_id] = task
            task.add_done_callback(
                lambda done, key=content_id: self._forget_inflight(key, done)
            )
        return await asyncio.shield(task)

    async def close(self) -> None:
        tasks = list(self._inflight.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._inflight.clear()

    async def _rehydrate(self, content_id: str) -> Optional[ContentFeatures]:
        async with self._semaphore:
            try:
                payload = await self._system_store.get_content_feature_artifact(
                    content_id, hot_path=True
                )
            except Exception as exc:
                self._record("error")
                logger.warning(
                    "content_feature_rehydrate_read_failed: content_id=%s error=%s",
                    content_id,
                    exc,
                )
                return None
            if payload is None:
                self._remember_not_found(content_id)
                self._record("not_found")
                return None
            try:
                features = ContentFeatures.parse_obj(payload)
            except Exception as exc:
                # A row that cannot be parsed will not parse on the next miss
                # either; remember it instead of re-reading it per request.
                self._remember_not_found(content_id)
                self._record("invalid")
                logger.error(
                    "content_feature_rehydrate_invalid_artifact: content_id=%s error=%s",
                    content_id,
                    exc,
                )
                return None
            await self._feature_store.store_content_features(content_id, features)
            self._record("rehydrated")
            return features

    def _recently_not_found(self, content_id: str) -> bool:
        expires_at = self._not_found_until.get(content_id)
        if expires_at is None:
            return False
        if expires_at > time.monotonic():
            return True
        self._not_found_until.pop(content_id, None)
        return False

    def _remember_not_found(self, content_id: str) -> None:
        if self._negative_ttl_seconds <= 0 or self._max_negative_entries <= 0:
            return
        self._not_found_until[content_id] = (
            time.monotonic() + self._negative_ttl_seconds
        )
        self._not_found_until.move_to_end(content_id)
        while len(self._not_found_until) > self._max_negative_entries:
            self._not_found_until.popitem(last=False)

    def _forget_inflight(self, content_id: str, task: asyncio.Task) -> None:
        if self._inflight.get(content_id) is task:
            self._inflight.pop(content_id, None)

    def _record(self, outcome: str) -> None:
        recorder = getattr(self._observability, "record_content_feature_rehydration", None)
        if recorder is not None:
            recorder(outcome)
