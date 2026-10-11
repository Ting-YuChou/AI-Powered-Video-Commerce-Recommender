#!/usr/bin/env python3
"""Bounded, auditable replay for Kafka dead-letter envelopes."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
import time
import uuid
from typing import Any, Iterable


@dataclass(frozen=True)
class ReplayPolicy:
    allowed_source_topics: set[str]
    max_age_days: int = 7


@dataclass(frozen=True)
class ReplayDecision:
    replayable: bool
    reason: str | None
    source_topic: str | None
    source_partition: int | None
    source_offset: int | None
    event_id: str | None
    payload_hash: str | None
    source_value: Any
    source_key: Any
    source_headers: list[Any]


def _canonical_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode()
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _payload_event_id(value: Any) -> str | None:
    candidate = value
    if isinstance(value, (str, bytes)):
        try:
            candidate = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(candidate, dict):
        return None
    event_id = candidate.get("event_id") or candidate.get("source_event_id")
    return str(event_id) if event_id else None


def inspect_envelope(
    envelope: dict[str, Any], policy: ReplayPolicy, *, now: float | None = None
) -> ReplayDecision:
    now = time.time() if now is None else now
    topic = envelope.get("source_topic")
    value = envelope.get("source_value")
    event_id = _payload_event_id(value)
    reason = None
    if topic not in policy.allowed_source_topics:
        reason = "source_topic_not_allowed"
    elif not event_id:
        reason = "missing_stable_event_id"
    else:
        occurred_at = envelope.get("occurred_at")
        try:
            age = now - float(occurred_at)
        except (TypeError, ValueError):
            reason = "missing_occurred_at"
        else:
            if age < 0 or age > policy.max_age_days * 86400:
                reason = "outside_replay_horizon"
    digest = hashlib.sha256(_canonical_bytes(value)).hexdigest() if value is not None else None
    return ReplayDecision(
        replayable=reason is None,
        reason=reason,
        source_topic=str(topic) if topic else None,
        source_partition=envelope.get("source_partition"),
        source_offset=envelope.get("source_offset"),
        event_id=event_id,
        payload_hash=digest,
        source_value=value,
        source_key=envelope.get("source_key"),
        source_headers=list(envelope.get("source_headers") or []),
    )


def build_audit_entry(
    decision: ReplayDecision, *, dlq_identity: str
) -> dict[str, Any]:
    """Build a secret-safe audit row without copying payloads, keys, or headers."""
    return {
        "dlq_identity": dlq_identity,
        "replayable": decision.replayable,
        "reason": decision.reason,
        "event_id": decision.event_id,
        "payload_hash": decision.payload_hash,
        "source_topic": decision.source_topic,
        "source_partition": decision.source_partition,
        "source_offset": decision.source_offset,
        "executed": False,
    }


def _headers(raw: Iterable[Any], replay_id: str, identity: str) -> list[tuple[str, bytes]]:
    result: list[tuple[str, bytes]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            result.append((str(item[0]), str(item[1]).encode()))
    result.extend(
        [
            ("x-dlq-replay-id", replay_id.encode()),
            ("x-original-dlq-identity", identity.encode()),
        ]
    )
    return result


async def replay(args: argparse.Namespace) -> list[dict[str, Any]]:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from aiokafka.structs import TopicPartition

    consumer = AIOKafkaConsumer(
        args.dlq_topic,
        bootstrap_servers=args.bootstrap_servers,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        group_id=None,
        value_deserializer=lambda raw: json.loads(raw.decode()),
    )
    producer = None
    audit: list[dict[str, Any]] = []
    await consumer.start()
    try:
        await consumer.topics()
        partitions = consumer.partitions_for_topic(args.dlq_topic) or set()
        if not partitions:
            raise RuntimeError(f"DLQ topic has no partitions: {args.dlq_topic}")
        topic_partitions = [TopicPartition(args.dlq_topic, p) for p in sorted(partitions)]
        consumer.unsubscribe()
        consumer.assign(topic_partitions)
        starts = await consumer.beginning_offsets(topic_partitions)
        ends = await consumer.end_offsets(topic_partitions)
        for tp in topic_partitions:
            consumer.seek(tp, starts[tp])
        if args.execute:
            producer = AIOKafkaProducer(
                bootstrap_servers=args.bootstrap_servers,
                acks="all",
            )
            await producer.start()
        policy = ReplayPolicy(set(args.allowed_source_topic), args.max_age_days)
        remaining = dict(ends)
        last_progress = time.monotonic()
        while True:
            positions = {
                tp: await consumer.position(tp) for tp in topic_partitions
            }
            if not any(positions[tp] < remaining[tp] for tp in topic_partitions):
                break
            batches = await consumer.getmany(timeout_ms=1000, max_records=500)
            if not batches:
                if time.monotonic() - last_progress >= args.read_timeout_seconds:
                    pending = {
                        f"{tp.topic}:{tp.partition}": {
                            "position": positions[tp],
                            "end_offset": remaining[tp],
                        }
                        for tp in topic_partitions
                        if positions[tp] < remaining[tp]
                    }
                    raise RuntimeError(
                        "timed out before reaching fixed DLQ end offsets: "
                        + json.dumps(pending, sort_keys=True)
                    )
                continue
            last_progress = time.monotonic()
            for tp, records in batches.items():
                for record in records:
                    if record.offset >= ends[tp]:
                        continue
                    decision = inspect_envelope(record.value, policy)
                    identity = f"{tp.topic}:{tp.partition}:{record.offset}"
                    item = build_audit_entry(decision, dlq_identity=identity)
                    if decision.replayable and producer is not None:
                        replay_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
                        metadata = await producer.send_and_wait(
                            decision.source_topic,
                            _canonical_bytes(decision.source_value),
                            key=(
                                _canonical_bytes(decision.source_key)
                                if decision.source_key is not None
                                else None
                            ),
                            headers=_headers(decision.source_headers, replay_id, identity),
                        )
                        item.update(
                            executed=True,
                            destination_topic=metadata.topic,
                            destination_partition=metadata.partition,
                            destination_offset=metadata.offset,
                        )
                    audit.append(item)
        return audit
    finally:
        if producer is not None:
            await producer.stop()
        try:
            await consumer.stop()
        except asyncio.CancelledError:
            # aiokafka may cancel its coordinator reset task for manually
            # assigned, group-less bounded consumers during shutdown.
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    )
    parser.add_argument(
        "--dlq-topic",
        default=os.getenv("KAFKA_DEAD_LETTER_TOPIC", "dead-letter-events"),
    )
    parser.add_argument("--allowed-source-topic", action="append", required=True)
    parser.add_argument("--max-age-days", type=int, default=7)
    parser.add_argument(
        "--read-timeout-seconds",
        type=float,
        default=60.0,
        help="Fail if fixed end offsets make no progress for this duration",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Publish replayable messages; default is dry-run",
    )
    parser.add_argument("--audit-output")
    args = parser.parse_args()
    report = asyncio.run(replay(args))
    rendered = json.dumps(report, sort_keys=True, indent=2, default=str)
    if args.audit_output:
        with open(args.audit_output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
