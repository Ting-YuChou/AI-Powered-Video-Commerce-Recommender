"""Lightweight protocol-aware health and drain client for ranking runners."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Dict

from video_commerce.ranking_runtime.ranking_coordinator_client import (
    DRAIN_OPERATION,
    HEALTH_OPERATION,
    decode_response,
    encode_request,
    read_frame,
)


async def probe_runner(
    *,
    host: str,
    port: int,
    operation: str = "health",
    timeout_seconds: float = 2.0,
    allow_degraded: bool = False,
) -> Dict[str, Any]:
    operation_byte = HEALTH_OPERATION if operation == "health" else DRAIN_OPERATION
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host, int(port)),
        timeout=max(0.1, float(timeout_seconds)),
    )
    try:
        writer.write(encode_request(operation_byte))
        await asyncio.wait_for(writer.drain(), timeout=timeout_seconds)
        response = decode_response(
            await asyncio.wait_for(read_frame(reader), timeout=timeout_seconds)
        )
        payload = json.loads(response.body)
        if not isinstance(payload, dict):
            raise RuntimeError("ranking runner returned a non-object health payload")
        expected_statuses = (
            {"ready", "degraded"}
            if operation == "health" and allow_degraded
            else {"ready"}
            if operation == "health"
            else {"drained"}
        )
        acceptable_status_code = response.status_code == 200 or (
            allow_degraded and operation == "health" and response.status_code == 503
        )
        if not acceptable_status_code or payload.get("status") not in expected_statuses:
            raise RuntimeError(
                f"ranking runner {operation} failed: status={response.status_code}, payload={payload}"
            )
        return payload
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--operation", choices=("health", "drain"), default="health")
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--allow-degraded", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(
            probe_runner(
                host=args.host,
                port=args.port,
                operation=args.operation,
                timeout_seconds=args.timeout_seconds,
                allow_degraded=args.allow_degraded,
            )
        )
    except Exception as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
