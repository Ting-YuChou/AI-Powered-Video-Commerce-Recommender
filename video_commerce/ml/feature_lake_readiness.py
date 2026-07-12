"""Fail-closed readiness barrier between streaming materialization and PIT batch export."""

from __future__ import annotations

import json
import os
import time
import asyncio
from typing import Any, Dict, Mapping
from urllib.request import urlopen


class FeatureLakeReadinessError(RuntimeError):
    pass


def readiness_reasons(
    *,
    job_state: str,
    completed_checkpoint_at_ms: int,
    gate_started_at_ms: int,
    committed_offsets: Mapping[str, int],
    end_offsets: Mapping[str, int],
) -> list[str]:
    reasons = []
    if job_state != "RUNNING":
        reasons.append("materializer_not_running")
    if completed_checkpoint_at_ms < gate_started_at_ms:
        reasons.append("no_completed_checkpoint_after_gate_start")
    partitions = set(committed_offsets) | set(end_offsets)
    if not partitions:
        reasons.append("consumer_group_has_no_offsets")
    for partition in sorted(partitions):
        committed = int(committed_offsets.get(partition, -1))
        end = int(end_offsets.get(partition, 0))
        if committed < 0:
            if end > 0:
                reasons.append(f"consumer_lag:{partition}:{end}")
            continue
        if committed < end:
            reasons.append(f"consumer_lag:{partition}:{end - committed}")
    return reasons


def _read_json(url: str) -> Dict[str, Any]:
    with urlopen(
        url, timeout=5
    ) as response:  # nosec: local/external Flink control plane
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise FeatureLakeReadinessError(f"unexpected Flink response from {url}")
    return payload


def _flink_status(rest_url: str, job_name: str) -> tuple[str, int]:
    overview = _read_json(f"{rest_url.rstrip('/')}/jobs/overview")
    terminal_states = {"FINISHED", "CANCELED", "FAILED", "SUSPENDED"}
    matches = [
        job
        for job in overview.get("jobs", [])
        if job.get("name") == job_name
        and str(job.get("state") or "UNKNOWN").upper() not in terminal_states
    ]
    if len(matches) != 1:
        return "MISSING", 0
    job = matches[0]
    checkpoints = _read_json(f"{rest_url.rstrip('/')}/jobs/{job['jid']}/checkpoints")
    completed = (checkpoints.get("latest") or {}).get("completed") or {}
    completed_at = int(
        completed.get("latest_ack_timestamp") or completed.get("trigger_timestamp") or 0
    )
    return str(job.get("state") or "UNKNOWN"), completed_at


async def _read_kafka_offsets(
    bootstrap_servers: str, group_id: str, topics: list[str]
) -> tuple[dict, dict]:
    from aiokafka import AIOKafkaConsumer
    from aiokafka.admin import AIOKafkaAdminClient
    from aiokafka.structs import TopicPartition

    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap_servers)
    consumer = AIOKafkaConsumer(bootstrap_servers=bootstrap_servers)
    await admin.start()
    await consumer.start()
    try:
        offsets = await admin.list_consumer_group_offsets(group_id)
        topic_partitions = set(offsets)
        for topic in topics:
            for partition in consumer.partitions_for_topic(topic) or set():
                topic_partitions.add(TopicPartition(topic, partition))
        end_offsets_raw = (
            await consumer.end_offsets(list(topic_partitions))
            if topic_partitions
            else {}
        )
        committed = {
            f"{tp.topic}:{tp.partition}": int(metadata.offset)
            for tp, metadata in offsets.items()
        }
        end_offsets = {
            f"{tp.topic}:{tp.partition}": int(offset)
            for tp, offset in end_offsets_raw.items()
        }
        return committed, end_offsets
    finally:
        await consumer.stop()
        await admin.close()


def _kafka_offsets(
    bootstrap_servers: str, group_id: str, topics: list[str]
) -> tuple[dict, dict]:
    return asyncio.run(_read_kafka_offsets(bootstrap_servers, group_id, topics))


def _materializer_topics() -> list[str]:
    topics = [
        os.environ.get("KAFKA_USER_INTERACTIONS_TOPIC", "user-interactions"),
        os.environ.get("KAFKA_RECOMMENDATION_EVENTS_TOPIC", "recommendation-events"),
        os.environ.get("KAFKA_FEATURE_UPDATES_TOPIC", "feature-updates"),
        os.environ.get("KAFKA_CATALOG_FEATURE_EVENTS_TOPIC", "catalog-feature-events"),
    ]
    if (
        os.environ.get("FEATURE_HISTORY_INCLUDE_BACKFILL_TOPICS", "false").lower()
        == "true"
    ):
        topics.extend(
            [
                os.environ.get(
                    "KAFKA_USER_INTERACTIONS_BACKFILL_TOPIC",
                    "user-interactions-backfill",
                ),
                os.environ.get(
                    "KAFKA_RECOMMENDATION_EVENTS_BACKFILL_TOPIC",
                    "recommendation-events-backfill",
                ),
                os.environ.get(
                    "KAFKA_FEATURE_UPDATES_BACKFILL_TOPIC",
                    "feature-updates-backfill",
                ),
                os.environ.get(
                    "KAFKA_CATALOG_FEATURE_EVENTS_BACKFILL_TOPIC",
                    "catalog-feature-events-backfill",
                ),
            ]
        )
    return topics


def wait_until_ready() -> None:
    rest_url = os.environ.get("FLINK_REST_URL", "http://flink-jobmanager:8081")
    job_name = os.environ.get(
        "FEATURE_HISTORY_MATERIALIZER_JOB_NAME",
        "video-commerce-feature-history-materializer",
    )
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    group_id = os.environ.get(
        "FEATURE_HISTORY_CONSUMER_GROUP", "feature-history-materializer-v1"
    )
    topics = _materializer_topics()
    timeout = max(
        1, int(os.environ.get("FEATURE_LAKE_READINESS_TIMEOUT_SECONDS", "600"))
    )
    poll = max(1, int(os.environ.get("FEATURE_LAKE_READINESS_POLL_SECONDS", "5")))
    started_ms = int(time.time() * 1000)
    deadline = time.monotonic() + timeout
    last_reasons: list[str] = ["not_checked"]
    while time.monotonic() < deadline:
        try:
            state, checkpoint_at = _flink_status(rest_url, job_name)
            committed, ends = _kafka_offsets(bootstrap, group_id, topics)
            last_reasons = readiness_reasons(
                job_state=state,
                completed_checkpoint_at_ms=checkpoint_at,
                gate_started_at_ms=started_ms,
                committed_offsets=committed,
                end_offsets=ends,
            )
            if not last_reasons:
                return
        except Exception as exc:
            last_reasons = [f"readiness_probe_error:{type(exc).__name__}:{exc}"]
        time.sleep(poll)
    raise FeatureLakeReadinessError(
        "feature lake did not reach a checkpointed zero-lag boundary: "
        + ",".join(last_reasons)
    )


if __name__ == "__main__":
    wait_until_ready()
