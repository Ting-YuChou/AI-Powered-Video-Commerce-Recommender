#!/usr/bin/env python3
"""
Run a simple HTTP load baseline against the recommendation endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import time
from typing import Dict, List

import httpx


@dataclass
class RequestResult:
    status_code: int
    duration_ms: float
    ok: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommendation API load baseline")
    parser.add_argument("--base-url", required=True, help="Base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=1000, help="Total request count")
    parser.add_argument("--concurrency", type=int, default=50, help="Concurrency level")
    parser.add_argument("--mode", choices=["hot", "unique"], default="hot", help="Request distribution mode")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout seconds")
    parser.add_argument(
        "--output",
        help="Optional output JSON path. Defaults to loadtest/results/httpx-baseline-<mode>.json",
    )
    parser.add_argument("--api-key", help="Optional X-API-Key header")
    return parser.parse_args()


def build_payload(index: int, mode: str) -> Dict[str, object]:
    user_id = "loadtest-hot-user" if mode == "hot" else f"loadtest-user-{index}"
    return {
        "user_id": user_id,
        "k": 10,
        "context": {
            "source": "loadtest-api-baseline",
            "request_index": index,
            "mode": mode,
        },
    }


async def run_request(
    client: httpx.AsyncClient,
    base_url: str,
    index: int,
    mode: str,
    headers: Dict[str, str],
) -> RequestResult:
    started_at = time.perf_counter()
    try:
        response = await client.post(
            f"{base_url.rstrip('/')}/api/recommendations",
            json=build_payload(index, mode),
            headers=headers,
        )
        ok = response.status_code == 200
        return RequestResult(
            status_code=response.status_code,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            ok=ok,
        )
    except Exception:
        return RequestResult(
            status_code=0,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            ok=False,
        )


async def run_load(args: argparse.Namespace) -> List[RequestResult]:
    headers = {}
    api_key = args.api_key or os.environ.get("API_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=max(1, min(args.concurrency, 100)),
    )

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def guarded(index: int) -> RequestResult:
            async with semaphore:
                return await run_request(client, args.base_url, index, args.mode, headers)

        return await asyncio.gather(*(guarded(index) for index in range(args.requests)))


def summarize(results: List[RequestResult], args: argparse.Namespace) -> Dict[str, object]:
    durations = [result.duration_ms for result in results]
    status_counts = Counter(str(result.status_code) for result in results)
    ok_results = [result for result in results if result.ok]
    success_rate = len(ok_results) / len(results) if results else 0.0

    def percentile(p: float) -> float:
        if not durations:
            return 0.0
        ordered = sorted(durations)
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
        return ordered[index]

    return {
        "base_url": args.base_url,
        "mode": args.mode,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        "success_rate": round(success_rate, 4),
        "average_ms": round(statistics.fmean(durations), 2) if durations else 0.0,
        "p50_ms": round(percentile(0.50), 2),
        "p95_ms": round(percentile(0.95), 2),
        "p99_ms": round(percentile(0.99), 2),
        "max_ms": round(max(durations), 2) if durations else 0.0,
        "status_counts": dict(status_counts),
    }


def main() -> None:
    args = parse_args()
    results = asyncio.run(run_load(args))
    summary = summarize(results, args)

    output_path = Path(
        args.output
        or f"loadtest/results/httpx-baseline-{args.mode}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
