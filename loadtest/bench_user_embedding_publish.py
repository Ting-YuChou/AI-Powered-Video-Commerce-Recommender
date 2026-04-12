#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List


def build_payloads(count: int, user_pool: int, content_pool: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    payloads: List[Dict[str, Any]] = []
    for i in range(count):
        payloads.append(
            {
                "user_id": f"user_{rng.randrange(user_pool):06d}",
                "content_id": f"content_bench_publish_{seed}_{i % content_pool:06d}",
                "context": {
                    "device": "mobile" if rng.random() < 0.8 else "desktop",
                    "page": "home" if rng.random() < 0.6 else "detail",
                    "session_position": rng.randrange(10) + 1,
                    "time_on_page": rng.randrange(180),
                },
                "k": 20,
            }
        )
    return payloads


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return {
                "ok": resp.status == 200,
                "status": resp.status,
                "elapsed_ms": elapsed_ms,
                "body": json.loads(raw),
            }
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "ok": False,
            "status": exc.code,
            "elapsed_ms": elapsed_ms,
            "error": f"http_error:{exc.code}",
        }
    except Exception as exc:  # pragma: no cover
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "ok": False,
            "status": None,
            "elapsed_ms": elapsed_ms,
            "error": repr(exc),
        }


def run_benchmark(
    base_url: str,
    payloads: List[Dict[str, Any]],
    concurrency: int,
    timeout: float,
) -> Dict[str, Any]:
    target_url = f"{base_url.rstrip('/')}/api/recommendations"
    lock = threading.Lock()
    latencies: List[float] = []
    server_total_ms: List[float] = []
    embedding_lookup_ms: List[float] = []
    embedding_cache_hits = 0
    errors = 0
    status_counts: Dict[str, int] = {}

    started = time.perf_counter()

    def task(payload: Dict[str, Any]) -> None:
        nonlocal embedding_cache_hits, errors
        result = post_json(target_url, payload, timeout=timeout)
        with lock:
            latencies.append(result["elapsed_ms"])
            status_key = str(result.get("status"))
            status_counts[status_key] = status_counts.get(status_key, 0) + 1
            if not result["ok"]:
                errors += 1
                return
            body = result["body"]
            profile = body.get("metadata", {}).get("profile", {})
            candidate_profile = profile.get("candidate_profile", {})
            if profile.get("total_ms") is not None:
                server_total_ms.append(float(profile["total_ms"]))
            if candidate_profile.get("user_embedding_lookup_ms") is not None:
                embedding_lookup_ms.append(float(candidate_profile["user_embedding_lookup_ms"]))
            if candidate_profile.get("user_embedding_cache_hit") is True:
                embedding_cache_hits += 1

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(task, payload) for payload in payloads]
        for future in as_completed(futures):
            future.result()

    elapsed_s = time.perf_counter() - started
    requests = len(payloads)
    return {
        "requests": requests,
        "concurrency": concurrency,
        "elapsed_s": round(elapsed_s, 3),
        "rps": round(requests / elapsed_s, 3) if elapsed_s else 0.0,
        "error_rate": round(errors / requests, 6) if requests else 0.0,
        "status_counts": status_counts,
        "client_latency_ms": {
            "avg": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "server_total_ms": {
            "avg": round(statistics.fmean(server_total_ms), 3) if server_total_ms else 0.0,
            "p95": round(percentile(server_total_ms, 0.95), 3),
        },
        "candidate_profile": {
            "user_embedding_cache_hit_rate": round(embedding_cache_hits / requests, 6) if requests else 0.0,
            "user_embedding_lookup_ms_avg": round(statistics.fmean(embedding_lookup_ms), 3) if embedding_lookup_ms else 0.0,
            "user_embedding_lookup_ms_p95": round(percentile(embedding_lookup_ms, 0.95), 3),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--user-pool", type=int, default=2000)
    parser.add_argument("--content-pool", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    payloads = build_payloads(args.requests + args.warmup, args.user_pool, args.content_pool, args.seed)

    if args.warmup:
        run_benchmark(args.base_url, payloads[: args.warmup], min(args.concurrency, args.warmup), args.timeout)
    result = run_benchmark(args.base_url, payloads[args.warmup :], args.concurrency, args.timeout)
    result.update(
        {
            "label": args.label,
            "base_url": args.base_url,
            "seed": args.seed,
            "warmup": args.warmup,
            "generated_at": time.time(),
        }
    )
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
