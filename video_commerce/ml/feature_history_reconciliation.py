"""Run-scoped Kafka and Iceberg evidence for one-time feature-history backfill."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class BackfillReconciliationEvidence:
    kafka_source_event_ids: frozenset[str]
    iceberg_source_event_ids: frozenset[str]
    dlq_source_event_ids: frozenset[str]
    iceberg_row_count: int
    kafka_end_offsets: Mapping[str, int]
    iceberg_snapshot_ids: Mapping[str, int]

    def validate(self, expected_count: int) -> Dict[str, Any]:
        if len(self.kafka_source_event_ids) != expected_count:
            raise RuntimeError(
                "backfill reconciliation mismatch: Kafka distinct source IDs do not "
                "match the operational published count"
            )
        if self.iceberg_row_count != len(self.iceberg_source_event_ids):
            raise RuntimeError(
                "backfill reconciliation mismatch: duplicate source_event_id in Iceberg"
            )
        explained = self.iceberg_source_event_ids | self.dlq_source_event_ids
        if explained != self.kafka_source_event_ids:
            raise RuntimeError(
                "backfill reconciliation mismatch: Iceberg plus DLQ source IDs do not "
                "exactly explain Kafka source IDs"
            )
        return {
            "kafka_accepted": len(self.kafka_source_event_ids),
            "iceberg_accepted": self.iceberg_row_count,
            "dlq_count": len(self.dlq_source_event_ids),
            "duplicate_count": 0,
            "kafka_end_offsets": dict(sorted(self.kafka_end_offsets.items())),
            "iceberg_snapshot_ids": dict(sorted(self.iceberg_snapshot_ids.items())),
        }


def collect_kafka_evidence(
    *,
    bootstrap_servers: str,
    source_topics: Sequence[str],
    dlq_topic: str,
    run_id: str,
) -> tuple[frozenset[str], frozenset[str], Dict[str, int]]:
    from kafka import KafkaConsumer
    from kafka.structs import TopicPartition

    topics = list(dict.fromkeys([*source_topics, dlq_topic]))
    consumer = KafkaConsumer(
        bootstrap_servers=bootstrap_servers,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        consumer_timeout_ms=1000,
        value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
    )
    try:
        partitions = []
        for topic in topics:
            discovered = consumer.partitions_for_topic(topic)
            if discovered is None:
                raise RuntimeError(
                    f"backfill reconciliation topic does not exist: {topic}"
                )
            partitions.extend(
                TopicPartition(topic, partition) for partition in discovered
            )
        consumer.assign(partitions)
        starts = consumer.beginning_offsets(partitions)
        ends = consumer.end_offsets(partitions)
        for partition in partitions:
            consumer.seek(partition, starts[partition])

        kafka_ids: set[str] = set()
        dlq_ids: set[str] = set()
        while any(
            consumer.position(partition) < ends[partition] for partition in partitions
        ):
            records = consumer.poll(timeout_ms=1000, max_records=2000)
            if not records and any(
                consumer.position(partition) < ends[partition]
                for partition in partitions
            ):
                raise RuntimeError(
                    "Kafka reconciliation scan stalled before bounded end offsets"
                )
            for partition, batch in records.items():
                for record in batch:
                    event = record.value if isinstance(record.value, dict) else {}
                    if event.get("backfill_run_id") != run_id:
                        continue
                    if partition.topic == dlq_topic:
                        source_id = str(
                            event.get("failed_source_event_id") or ""
                        ).strip()
                        if source_id:
                            dlq_ids.add(source_id)
                    else:
                        source_id = str(event.get("source_event_id") or "").strip()
                        if source_id:
                            kafka_ids.add(source_id)
        end_offsets = {
            f"{partition.topic}:{partition.partition}": int(offset)
            for partition, offset in ends.items()
        }
        return frozenset(kafka_ids), frozenset(dlq_ids), end_offsets
    finally:
        consumer.close()


def collect_iceberg_evidence(
    *,
    catalog_uri: str,
    warehouse_uri: str,
    namespace: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    run_id: str,
    table_names: Iterable[str],
) -> tuple[frozenset[str], int, Dict[str, int]]:
    from pyiceberg.catalog import load_catalog
    from pyiceberg.expressions import EqualTo

    catalog = load_catalog(
        "feature_history_reconciliation",
        type="rest",
        uri=catalog_uri,
        warehouse=warehouse_uri,
        **{
            "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
            "s3.endpoint": s3_endpoint,
            "s3.access-key-id": s3_access_key,
            "s3.secret-access-key": s3_secret_key,
            "s3.region": "us-east-1",
        },
    )
    source_ids: set[str] = set()
    row_count = 0
    snapshots: Dict[str, int] = {}
    for table_name in table_names:
        table = catalog.load_table(f"{namespace}.{table_name}")
        snapshot = table.current_snapshot()
        if snapshot is None:
            continue
        snapshots[table_name] = int(snapshot.snapshot_id)
        rows = table.scan(
            row_filter=EqualTo("backfill_run_id", run_id),
            selected_fields=("source_event_id",),
        ).to_arrow()
        values = [str(value) for value in rows.column("source_event_id").to_pylist()]
        row_count += len(values)
        source_ids.update(values)
    return frozenset(source_ids), row_count, snapshots
