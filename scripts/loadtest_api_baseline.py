#!/usr/bin/env python3
"""
Run a simple HTTP load baseline against the recommendation endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
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
    cache_hit: bool | None = None
    impression_tracking: str | None = None
    impression_id: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommendation API load baseline")
    parser.add_argument(
        "--base-url", required=True, help="Base URL, e.g. http://127.0.0.1:8000"
    )
    parser.add_argument(
        "--requests", type=int, default=1000, help="Total request count"
    )
    parser.add_argument("--concurrency", type=int, default=50, help="Concurrency level")
    parser.add_argument(
        "--mode",
        choices=["hot", "unique"],
        default="hot",
        help="Request distribution mode",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Per-request timeout seconds"
    )
    parser.add_argument(
        "--run-id",
        help="Optional identifier used to isolate users and correlate result rows",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="Warm-up request count excluded from the measured summary",
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON path. Defaults to loadtest/results/httpx-baseline-<mode>.json",
    )
    parser.add_argument("--api-key", help="Optional X-API-Key header")
    parser.add_argument("--internal-key", help="Optional X-Internal-Service-Key header")
    return parser.parse_args()


def build_payload(
    index: int,
    mode: str,
    *,
    run_id: str | None = None,
    user_run_id: str | None = None,
) -> Dict[str, object]:
    effective_user_run_id = user_run_id or run_id
    user_prefix = (
        "loadtest" if not effective_user_run_id else f"loadtest-{effective_user_run_id}"
    )
    user_id = (
        f"{user_prefix}-hot-user" if mode == "hot" else f"{user_prefix}-user-{index}"
    )
    return {
        "user_id": user_id,
        "k": 10,
        "context": {
            "source": "loadtest-api-baseline",
            "request_index": index,
            "mode": mode,
            **({"run_id": run_id} if run_id else {}),
            **({"session_id": run_id} if run_id else {}),
        },
    }


async def run_request(
    client: httpx.AsyncClient,
    base_url: str,
    index: int,
    mode: str,
    headers: Dict[str, str],
    run_id: str | None,
    user_run_id: str | None,
) -> RequestResult:
    started_at = time.perf_counter()
    try:
        response = await client.post(
            f"{base_url.rstrip('/')}/api/recommendations",
            json=build_payload(
                index,
                mode,
                run_id=run_id,
                user_run_id=user_run_id,
            ),
            headers=headers,
        )
        ok = response.status_code == 200
        cache_hit = None
        impression_tracking = None
        impression_id = None
        if ok:
            try:
                metadata = (response.json() or {}).get("metadata") or {}
                cache_hit = metadata.get("cache_hit")
                impression_tracking = metadata.get("impression_tracking")
                impression_id = metadata.get("impression_id")
            except (TypeError, ValueError):
                pass
        return RequestResult(
            status_code=response.status_code,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            ok=ok,
            cache_hit=cache_hit if isinstance(cache_hit, bool) else None,
            impression_tracking=(
                str(impression_tracking) if impression_tracking is not None else None
            ),
            impression_id=str(impression_id) if impression_id is not None else None,
        )
    except Exception:
        return RequestResult(
            status_code=0,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            ok=False,
        )


async def run_load(args: argparse.Namespace) -> tuple[List[RequestResult], float]:
    headers = {}
    api_key = args.api_key or os.environ.get("API_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    internal_key = args.internal_key or os.environ.get("SECURITY_INTERNAL_SERVICE_KEY")
    if internal_key:
        headers["x-internal-service-key"] = internal_key

    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=max(1, min(args.concurrency, 100)),
    )

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

        async def guarded(index: int) -> RequestResult:
            async with semaphore:
                return await run_request(
                    client,
                    args.base_url,
                    index,
                    args.mode,
                    headers,
                    args.run_id,
                    getattr(args, "user_run_id", None),
                )

        started_at = time.perf_counter()
        results = await asyncio.gather(
            *(guarded(index) for index in range(args.requests))
        )
        return results, time.perf_counter() - started_at


def summarize(
    results: List[RequestResult],
    args: argparse.Namespace,
    elapsed_seconds: float,
) -> Dict[str, object]:
    durations = [result.duration_ms for result in results]
    status_counts = Counter(str(result.status_code) for result in results)
    ok_results = [result for result in results if result.ok]
    successful_durations = [result.duration_ms for result in ok_results]
    server_error_count = sum(
        1
        for result in results
        if result.status_code == 0 or 500 <= result.status_code <= 599
    )
    five_xx_count = sum(1 for result in results if 500 <= result.status_code <= 599)
    transport_error_count = sum(1 for result in results if result.status_code == 0)
    success_rate = len(ok_results) / len(results) if results else 0.0
    error_rate = 1.0 - success_rate if results else 0.0
    server_error_rate = server_error_count / len(results) if results else 0.0
    five_xx_rate = five_xx_count / len(results) if results else 0.0
    transport_error_rate = transport_error_count / len(results) if results else 0.0
    elapsed_seconds = max(elapsed_seconds, 0.000001)

    def percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
        return ordered[index]

    cache_observations = [
        result.cache_hit for result in ok_results if result.cache_hit is not None
    ]
    tracking_observations = [
        result.impression_tracking
        for result in ok_results
        if result.impression_tracking is not None
    ]
    durable_impression_ids = [
        result.impression_id
        for result in ok_results
        if result.impression_tracking == "durable" and result.impression_id
    ]

    return {
        "base_url": args.base_url,
        "mode": args.mode,
        "run_id": getattr(args, "run_id", None),
        "requests": args.requests,
        "warmup_requests": getattr(args, "warmup_requests", 0),
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "qps_total": round(len(results) / elapsed_seconds, 2),
        "qps_2xx": round(len(ok_results) / elapsed_seconds, 2),
        "success_rate": round(success_rate, 4),
        "error_rate": round(error_rate, 4),
        "server_error_count": server_error_count,
        "server_error_rate": round(server_error_rate, 4),
        "five_xx_count": five_xx_count,
        "five_xx_rate": round(five_xx_rate, 4),
        "transport_error_count": transport_error_count,
        "transport_error_rate": round(transport_error_rate, 4),
        "average_ms": round(statistics.fmean(durations), 2) if durations else 0.0,
        "p50_ms": round(percentile(durations, 0.50), 2),
        "p95_ms": round(percentile(durations, 0.95), 2),
        "p99_ms": round(percentile(durations, 0.99), 2),
        "max_ms": round(max(durations), 2) if durations else 0.0,
        "successful_average_ms": (
            round(statistics.fmean(successful_durations), 2)
            if successful_durations
            else 0.0
        ),
        "successful_p50_ms": round(percentile(successful_durations, 0.50), 2),
        "successful_p95_ms": round(percentile(successful_durations, 0.95), 2),
        "successful_p99_ms": round(percentile(successful_durations, 0.99), 2),
        "cache_observation_count": len(cache_observations),
        "cache_hit_rate": round(
            sum(1 for value in cache_observations if value) / len(cache_observations),
            4,
        )
        if cache_observations
        else 0.0,
        "tracking_observation_count": len(tracking_observations),
        "durable_tracking_rate": round(
            tracking_observations.count("durable") / len(tracking_observations), 4
        )
        if tracking_observations
        else 0.0,
        "unavailable_tracking_rate": round(
            tracking_observations.count("unavailable") / len(tracking_observations),
            4,
        )
        if tracking_observations
        else 0.0,
        "durable_impression_count": len(durable_impression_ids),
        "unique_durable_impression_count": len(set(durable_impression_ids)),
        "status_counts": dict(status_counts),
    }


def main() -> None:
    args = parse_args()
    if args.warmup_requests < 0:
        raise SystemExit("--warmup-requests must be non-negative")
    if args.warmup_requests:
        warmup_args = copy.copy(args)
        warmup_args.requests = args.warmup_requests
        warmup_args.run_id = f"{args.run_id or 'default'}-warmup"
        warmup_args.user_run_id = args.run_id if args.mode == "hot" else None
        asyncio.run(run_load(warmup_args))
    results, elapsed_seconds = asyncio.run(run_load(args))
    summary = summarize(results, args, elapsed_seconds)

    output_path = Path(
        args.output or f"loadtest/results/httpx-baseline-{args.mode}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
