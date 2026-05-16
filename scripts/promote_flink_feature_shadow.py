#!/usr/bin/env python3
"""Promote Flink shadow Redis feature keys into the official namespace.

The script is intentionally dry-run by default. Use --execute only after the
shadow comparison checks pass and the Python feature-worker is stopped.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import redis


def _client() -> redis.Redis:
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD") or None,
        decode_responses=False,
    )


def promote(*, execute: bool, batch_size: int, client: Optional[redis.Redis] = None) -> int:
    client = client or _client()
    promoted = 0
    for raw_key in client.scan_iter(match=b"flink:shadow:*", count=batch_size):
        official_key = raw_key[len(b"flink:shadow:") :]
        ttl_ms = client.pttl(raw_key)
        dumped = client.dump(raw_key)
        if dumped is None:
            continue
        if execute:
            restore_ttl = 0 if ttl_ms is None or ttl_ms < 0 else ttl_ms
            client.restore(official_key, restore_ttl, dumped, replace=True)
        promoted += 1
    return promoted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="write official keys")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    count = promote(execute=args.execute, batch_size=args.batch_size)
    mode = "promoted" if args.execute else "would_promote"
    print(f"{mode}={count}")


if __name__ == "__main__":
    main()
