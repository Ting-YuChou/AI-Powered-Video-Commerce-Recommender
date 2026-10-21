"""Lightweight protocol-aware health, drain and undrain client for ranking runners.

Also used as the coordinator readiness probe (health only). Drain and undrain
are accepted by runners only from loopback peers, so run them inside the pod.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Dict, Optional

from video_commerce.ranking_runtime.ranking_auth import (
    ROLE_RUNNER,
    SERVER_ROLES,
    RankingRuntimeAuthSettings,
)
from video_commerce.ranking_runtime.ranking_coordinator_client import (
    DRAIN_OPERATION,
    HEALTH_OPERATION,
    UNDRAIN_OPERATION,
    decode_response,
    encode_request,
    perform_client_handshake,
    read_frame,
)

OPERATIONS = {
    "health": HEALTH_OPERATION,
    "drain": DRAIN_OPERATION,
    "undrain": UNDRAIN_OPERATION,
}
EXPECTED_STATUSES = {"drain": {"drained"}, "undrain": {"undrained"}}


async def probe_runner(
    *,
    host: str,
    port: int,
    operation: str = "health",
    timeout_seconds: float = 2.0,
    allow_degraded: bool = False,
    auth_settings: Optional[RankingRuntimeAuthSettings] = None,
    role: str = ROLE_RUNNER,
) -> Dict[str, Any]:
    if operation not in OPERATIONS:
        raise ValueError(f"unsupported ranking runner operation: {operation}")
    operation_byte = OPERATIONS[operation]
    if auth_settings is None:
        auth_settings = RankingRuntimeAuthSettings.from_env()
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host, int(port)),
        timeout=max(0.1, float(timeout_seconds)),
    )
    try:
        if auth_settings.client_sends_handshake:
            await perform_client_handshake(
                reader,
                writer,
                auth_settings,
                server_role=role,
                timeout_seconds=timeout_seconds,
                peer_label=f"{host}:{port}",
            )
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
            else EXPECTED_STATUSES[operation]
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
    parser.add_argument("--operation", choices=tuple(OPERATIONS), default="health")
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--allow-degraded", action="store_true")
    parser.add_argument(
        "--role",
        choices=SERVER_ROLES,
        default=ROLE_RUNNER,
        help="Role of the probed server; bound into the auth handshake",
    )
    args = parser.parse_args()
    try:
        asyncio.run(
            probe_runner(
                host=args.host,
                port=args.port,
                operation=args.operation,
                timeout_seconds=args.timeout_seconds,
                allow_degraded=args.allow_degraded,
                role=args.role,
            )
        )
    except Exception as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
