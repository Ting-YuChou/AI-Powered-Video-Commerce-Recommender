"""List or re-enqueue durable videos missing temporal multimodal artifacts."""

import argparse
import asyncio
import json

from video_commerce.common.config import Config
from video_commerce.data_plane.kafka_client import KafkaManager
from video_commerce.data_plane.system_store import SystemStore


async def run(*, limit: int, enqueue: bool) -> int:
    config = Config()
    store = SystemStore(config.database_config)
    await store.initialize()
    kafka = KafkaManager(config.kafka_config) if enqueue else None
    try:
        jobs = await store.list_content_jobs_missing_feature_artifact(
            limit=limit,
            expected_schema_version="temporal_multimodal_v2",
        )
        if not enqueue:
            print(json.dumps({"count": len(jobs), "jobs": jobs}, indent=2))
            return 0
        await kafka.start()
        failures = []
        for job in jobs:
            success = await kafka.send_video_processing_task(
                content_id=job["content_id"],
                file_path=job["storage_path"],
                filename=job["filename"],
                user_id=job["user_id"],
                priority=job["priority"] or "normal",
                request_id=f"temporal-backfill:{job['content_id']}",
            )
            if not success:
                failures.append(job["content_id"])
        print(
            json.dumps(
                {
                    "selected": len(jobs),
                    "enqueued": len(jobs) - len(failures),
                    "failures": failures,
                },
                indent=2,
            )
        )
        return 1 if failures else 0
    finally:
        if kafka is not None:
            await kafka.stop()
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--enqueue",
        action="store_true",
        help="Publish selected durable jobs to Kafka; default is dry-run JSON",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(limit=args.limit, enqueue=args.enqueue)))


if __name__ == "__main__":
    main()
