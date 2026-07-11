"""Resumable one-time backfill of operational records into isolated Kafka topics."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import time
from typing import Any, Dict, Mapping, Optional
import uuid

from video_commerce.common.config import Config
from video_commerce.common.feature_history_contracts import (
    FEATURE_HISTORY_PAYLOAD_SCHEMA_VERSION,
    RANKING_LTR_FEATURE_DEFINITION_VERSION,
    build_catalog_feature_event,
    payload_sha256,
)
from video_commerce.data_plane.kafka_client import close_kafka, init_kafka
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.ml.feature_history_reconciliation import (
    BackfillReconciliationEvidence,
    collect_iceberg_evidence,
    collect_kafka_evidence,
)


def _timestamp(value: Any) -> float:
    if isinstance(value, datetime):
        aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return aware.timestamp()
    return float(value)


def build_interaction_backfill_event(
    record: Mapping[str, Any], *, run_id: Optional[str] = None
) -> Dict[str, Any]:
    event_id = str(record["event_id"])
    event_time = _timestamp(record["occurred_at"])
    available_at = _timestamp(record.get("created_at") or record["occurred_at"])
    payload = {
        "user_id": str(record["user_id"]),
        "product_id": str(record["product_id"]),
        "action": str(record["action"]),
        "context": dict(record.get("context") or {}),
    }
    event = {
        "schema_version": 1,
        "event_id": event_id,
        "event_type": "user_interaction",
        "request_id": record.get("request_id"),
        **payload,
        "occurred_at": event_time,
        "timestamp": event_time,
        "event_time": event_time,
        "server_received_at": available_at,
        "available_at": available_at,
        "source_event_id": event_id,
        "source_version": "interaction-v1",
        "feature_definition_version": RANKING_LTR_FEATURE_DEFINITION_VERSION,
        "payload_schema_version": FEATURE_HISTORY_PAYLOAD_SCHEMA_VERSION,
        "payload_hash": payload_sha256(payload),
    }
    if run_id:
        event["backfill_run_id"] = run_id
    return event


def build_legacy_observation_quarantine(
    run_id: str, record: Mapping[str, Any]
) -> Dict[str, Any]:
    impression_id = str(record["impression_id"])
    return {
        "schema_version": 1,
        "event_id": f"{run_id}:{impression_id}",
        "event_type": "feature_history_backfill_quarantine",
        "source_event_id": impression_id,
        "source_version": str(record.get("ranking_model_version") or "legacy"),
        "reason": "missing_complete_ranking_ltr_v1_observation_bundle",
        "run_id": run_id,
        "backfill_run_id": run_id,
        "impression_id": impression_id,
        "user_id": record.get("user_id"),
        "event_time": _timestamp(record["created_at"]),
        "available_at": time.time(),
        "displayed_item_count": len(record.get("displayed_items") or []),
    }


class FeatureHistoryBackfillRunner:
    def __init__(
        self,
        *,
        system_store: Any,
        kafka_manager: Any,
        topics: Mapping[str, str],
        page_size: int = 1000,
    ) -> None:
        self.system_store = system_store
        self.kafka_manager = kafka_manager
        self.topics = dict(topics)
        self.page_size = max(1, int(page_size))

    async def run_interactions(
        self,
        run_id: str,
        *,
        range_end: float,
        range_start: Optional[float] = None,
    ) -> None:
        run = await self.system_store.get_feature_history_backfill_run(run_id)
        if run is None:
            raise RuntimeError(f"unknown backfill run {run_id}")
        cursor_time = run.get("cursor_time")
        cursor_id = run.get("cursor_id")
        counts = dict(run.get("counts") or {})
        while True:
            page = await self.system_store.get_backfill_interactions_page(
                range_start=range_start,
                range_end=range_end,
                cursor_time=cursor_time,
                cursor_id=cursor_id,
                limit=self.page_size,
            )
            if not page:
                await self.system_store.checkpoint_feature_history_backfill(
                    run_id, phase="observations", counts=counts
                )
                return
            for record in page:
                event = build_interaction_backfill_event(record, run_id=run_id)
                acknowledged = (
                    await self.kafka_manager.send_feature_history_backfill_event(
                        topic=self.topics["interactions"],
                        event=event,
                        key=str(record["user_id"]),
                    )
                )
                if not acknowledged:
                    raise RuntimeError(
                        "Kafka did not acknowledge interaction backfill event"
                    )
            last = page[-1]
            cursor_time = _timestamp(last["occurred_at"])
            cursor_id = str(last["event_id"])
            counts["interactions_published"] = int(
                counts.get("interactions_published", 0)
            ) + len(page)
            await self.system_store.checkpoint_feature_history_backfill(
                run_id,
                cursor_time=cursor_time,
                cursor_id=cursor_id,
                counts=counts,
            )

    async def run_catalog(self, run_id: str) -> None:
        rows = await self.system_store.get_backfill_catalog_snapshot()
        counts: Dict[str, int] = {}
        for row in rows:
            source_version = str(
                row.get("source_version") or f"initial-backfill-{run_id}"
            )
            event_time = _timestamp(row["updated_at"])
            event = build_catalog_feature_event(
                product_id=str(row["product_id"]),
                source_version=source_version,
                event_time=event_time,
                available_at=event_time,
                payload=dict(row.get("snapshot") or {}),
            )
            event["backfill_run_id"] = run_id
            acknowledged = await self.kafka_manager.send_feature_history_backfill_event(
                topic=self.topics["catalog"], event=event, key=str(row["product_id"])
            )
            if not acknowledged:
                raise RuntimeError("Kafka did not acknowledge catalog backfill event")
            counts["catalog_published"] = counts.get("catalog_published", 0) + 1
        await self.system_store.checkpoint_feature_history_backfill(
            run_id, phase="interactions", counts=counts
        )

    async def run_observations(
        self,
        run_id: str,
        *,
        range_start: float,
        range_end: float,
    ) -> None:
        run = await self.system_store.get_feature_history_backfill_run(run_id)
        cursor_time = run.get("cursor_time")
        cursor_id = run.get("cursor_id")
        counts = dict(run.get("counts") or {})
        while True:
            page = await self.system_store.get_backfill_impressions_page(
                range_start=range_start,
                range_end=range_end,
                cursor_time=cursor_time,
                cursor_id=cursor_id,
                limit=self.page_size,
            )
            if not page:
                await self.system_store.checkpoint_feature_history_backfill(
                    run_id, phase="reconcile", counts=counts
                )
                return
            for record in page:
                event = build_legacy_observation_quarantine(run_id, record)
                acknowledged = (
                    await self.kafka_manager.send_feature_history_backfill_event(
                        topic=self.topics["quarantine"],
                        event=event,
                        key=str(record["impression_id"]),
                    )
                )
                if not acknowledged:
                    raise RuntimeError("Kafka did not acknowledge quarantine event")
            last = page[-1]
            cursor_time = _timestamp(last["created_at"])
            cursor_id = str(last["impression_id"])
            counts["observations_quarantined"] = int(
                counts.get("observations_quarantined", 0)
            ) + len(page)
            await self.system_store.checkpoint_feature_history_backfill(
                run_id,
                cursor_time=cursor_time,
                cursor_id=cursor_id,
                counts=counts,
            )

    async def run(self, run_id: str, *, range_start: float, range_end: float) -> None:
        run = await self.system_store.start_feature_history_backfill(
            run_id=run_id, range_start=range_start, range_end=range_end
        )
        if run["status"] == "complete":
            return
        if run["phase"] == "catalog":
            await self.run_catalog(run_id)
            run = await self.system_store.get_feature_history_backfill_run(run_id)
        if run["phase"] == "interactions":
            await self.run_interactions(
                run_id, range_start=range_start, range_end=range_end
            )
            run = await self.system_store.get_feature_history_backfill_run(run_id)
        if run["phase"] == "observations":
            await self.run_observations(
                run_id, range_start=range_start, range_end=range_end
            )

    async def reconcile(
        self,
        run_id: str,
        *,
        evidence: BackfillReconciliationEvidence,
    ) -> None:
        run = await self.system_store.get_feature_history_backfill_run(run_id)
        if run is None:
            raise RuntimeError(f"unknown backfill run {run_id}")
        if run.get("status") == "complete":
            if dict(run.get("reconciliation") or {}) == reconciliation:
                return
            raise RuntimeError("completed backfill reconciliation is immutable")
        if run.get("phase") != "reconcile":
            raise RuntimeError(
                "backfill run must reach reconcile phase before completion"
            )
        counts = dict(run.get("counts") or {})
        expected_kafka = int(counts.get("interactions_published", 0)) + int(
            counts.get("catalog_published", 0)
        )
        reconciliation = evidence.validate(expected_kafka)
        await self.system_store.checkpoint_feature_history_backfill(
            run_id,
            phase="complete",
            status="complete",
            reconciliation=reconciliation,
        )


async def _run(args: argparse.Namespace) -> None:
    config = Config()
    store = SystemStore(config.database_config)
    await store.initialize()
    if args.reconcile:
        runner = FeatureHistoryBackfillRunner(
            system_store=store,
            kafka_manager=None,
            topics={},
        )
        try:
            kafka_ids, dlq_ids, end_offsets = await asyncio.to_thread(
                collect_kafka_evidence,
                bootstrap_servers=config.kafka_config.bootstrap_servers,
                source_topics=(
                    config.kafka_config.user_interactions_backfill_topic,
                    config.kafka_config.recommendation_events_backfill_topic,
                    config.kafka_config.feature_updates_backfill_topic,
                    config.kafka_config.catalog_feature_events_backfill_topic,
                ),
                dlq_topic=config.kafka_config.dead_letter_topic,
                run_id=args.run_id,
            )
            iceberg_ids, iceberg_rows, snapshot_ids = await asyncio.to_thread(
                collect_iceberg_evidence,
                catalog_uri=config.feature_lake_config.catalog_uri,
                warehouse_uri=config.feature_lake_config.warehouse_uri,
                namespace=config.feature_lake_config.namespace,
                s3_endpoint=config.object_storage_config.endpoint_url,
                s3_access_key=config.object_storage_config.access_key_id,
                s3_secret_key=config.object_storage_config.secret_access_key,
                run_id=args.run_id,
                table_names=(
                    "interaction_history",
                    "ranking_observations",
                    "user_feature_history",
                    "item_feature_history",
                    "window_feature_history",
                ),
            )
            await runner.reconcile(
                args.run_id,
                evidence=BackfillReconciliationEvidence(
                    kafka_source_event_ids=kafka_ids,
                    iceberg_source_event_ids=iceberg_ids,
                    dlq_source_event_ids=dlq_ids,
                    iceberg_row_count=iceberg_rows,
                    kafka_end_offsets=end_offsets,
                    iceberg_snapshot_ids=snapshot_ids,
                ),
            )
        finally:
            await store.close()
        return
    kafka = await init_kafka(config.kafka_config)
    runner = FeatureHistoryBackfillRunner(
        system_store=store,
        kafka_manager=kafka,
        topics={
            "interactions": config.kafka_config.user_interactions_backfill_topic,
            "catalog": config.kafka_config.catalog_feature_events_backfill_topic,
            "quarantine": config.kafka_config.feature_history_backfill_quarantine_topic,
        },
        page_size=args.page_size,
    )
    try:
        await runner.run(
            args.run_id, range_start=args.range_start, range_end=args.range_end
        )
    finally:
        await close_kafka()
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"backfill-{uuid.uuid4().hex[:12]}")
    parser.add_argument("--range-start", type=float)
    parser.add_argument("--range-end", type=float)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()
    if not args.reconcile and (args.range_start is None or args.range_end is None):
        parser.error("backfill publication requires --range-start and --range-end")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
