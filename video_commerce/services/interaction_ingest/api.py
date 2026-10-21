"""
Dedicated high-throughput interaction ingest API.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
import hashlib
import json
from typing import Awaitable, Callable, Optional, Set

from fastapi import Body, HTTPException, Request
from fastapi.responses import JSONResponse

from video_commerce.common.config import Config
from video_commerce.common.event_time import (
    clamp_public_event_time,
    validate_public_event_time,
)
from video_commerce.data_plane.feature_store import FeatureStore
from video_commerce.data_plane.kafka_client import KafkaProducerSupervisor
from video_commerce.common.models import UserInteractionRequest, ViewedImpressionRequest
from video_commerce.data_plane.system_store import SystemStore
from video_commerce.common.service_common import (
    build_health_response,
    build_liveness_payload,
    build_metrics_response,
    build_readiness_response,
    component_health,
    configure_service_logging,
    create_service_app,
    internal_service_auth_response,
)

logger = logging.getLogger(__name__)

IDEMPOTENCY_PENDING_TTL_SECONDS = 30
IDEMPOTENCY_PUBLISHED_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_PUBLISH_TIMEOUT_SECONDS = 3.0
LATE_PUBLISH_DRAIN_SECONDS = 5.0

app = create_service_app(
    title="Interaction Ingest Service",
    description="Dedicated interaction ingestion service for async event capture",
    service_name="interaction-ingest-service",
)

feature_store: Optional[FeatureStore] = None
kafka_manager = None
kafka_supervisor: Optional[KafkaProducerSupervisor] = None
system_store: Optional[SystemStore] = None
# Startup replaces both from config.
publish_timeout_seconds = DEFAULT_PUBLISH_TIMEOUT_SECONDS
pending_lease_seconds = IDEMPOTENCY_PENDING_TTL_SECONDS
# Sends that outlived the publish budget; aiokafka still delivers or fails them.
_late_publishes: Set[asyncio.Task] = set()


@app.middleware("http")
async def internal_auth(request: Request, call_next):
    denied = internal_service_auth_response(request, app.state.runtime)
    if denied is not None:
        return denied
    return await call_next(request)


@app.on_event("startup")
async def startup_event():
    global feature_store, kafka_manager, kafka_supervisor, system_store
    global publish_timeout_seconds, pending_lease_seconds

    runtime = app.state.runtime
    runtime.config = Config()
    configure_service_logging(runtime)
    publish_timeout_seconds = (
        runtime.config.service_topology_config.interaction_ingest_publish_timeout_ms
        / 1000
    )
    pending_lease_seconds = _pending_lease_seconds(
        runtime.config.kafka_config.request_timeout_ms
    )

    feature_store = FeatureStore(runtime.config.redis_config, runtime.config.cache_config)
    await feature_store.initialize()

    if runtime.config.database_config.enable:
        system_store = SystemStore(
            runtime.config.database_config, observability=runtime.observability
        )
        await system_store.initialize()

    if runtime.config.kafka_config.enable:
        kafka_supervisor = KafkaProducerSupervisor(
            runtime.config.kafka_config,
            service_name="interaction-ingest-service",
            observability=runtime.observability,
            on_connected=_set_kafka_manager,
        )
        kafka_manager = await kafka_supervisor.start()


def _set_kafka_manager(manager) -> None:
    global kafka_manager
    kafka_manager = manager


def _pending_lease_seconds(kafka_request_timeout_ms: int) -> int:
    """Lease long enough to outlive a send that missed the publish budget.

    aiokafka can hold a send for up to twice its request timeout, and the claim
    stays pending until that send resolves. A shorter lease would let a retry
    reclaim the event_id and publish it a second time.
    """
    return max(
        IDEMPOTENCY_PENDING_TTL_SECONDS,
        math.ceil(2 * kafka_request_timeout_ms / 1000) + 5,
    )


@app.on_event("shutdown")
async def shutdown_event():
    if _late_publishes:
        # Let late sends settle their ledger claims while the stores are open.
        await asyncio.wait(set(_late_publishes), timeout=LATE_PUBLISH_DRAIN_SECONDS)
    if feature_store:
        await feature_store.close()
    if system_store:
        await system_store.close()
    if kafka_supervisor:
        await kafka_supervisor.stop()


@app.get("/")
async def root():
    return {
        "service": "interaction-ingest-service",
        "version": "1.0.0",
        "health": "/health",
        "livez": "/livez",
        "readyz": "/readyz",
    }


@app.get("/livez")
async def livez():
    return build_liveness_payload(app.state.runtime)


@app.get("/readyz")
async def readyz():
    runtime = app.state.runtime
    feature_store_health = await feature_store.health_check()
    kafka_health = {"status": "healthy", "response_time_ms": 0.0}
    if kafka_manager:
        kafka_status = await kafka_manager.health_check()
        producer_health = kafka_status.get("producer", {})
        kafka_health = {
            "status": producer_health.get("status", "healthy"),
            "response_time_ms": 0.0,
            "error": None if producer_health.get("connected") else "Kafka producer unavailable",
        }
    elif runtime.config.kafka_config.enable:
        kafka_health = {"status": "unhealthy", "error": "Kafka producer reconnecting"}

    database_health = {"status": "healthy", "response_time_ms": 0.0}
    if runtime.config.database_config.enable:
        if system_store is None:
            database_health = {
                "status": "unhealthy",
                "error": "Postgres idempotency ledger unavailable",
            }
        else:
            database_status = await system_store.health_check()
            database_health = {
                "status": database_status.status,
                "response_time_ms": database_status.response_time_ms,
                "error": database_status.error,
            }

    # Ingest acknowledges only after Kafka accepts the event and dedups through
    # the Postgres ledger, so those two decide readiness. Redis only backs
    # best-effort cache invalidation.
    checks = {
        "redis": feature_store_health,
        "kafka": kafka_health,
        "database": database_health,
    }
    for name, check in checks.items():
        runtime.observability.set_dependency_status(
            "interaction-ingest-service", name, str(check.get("status", "unhealthy"))
        )
    return build_readiness_response(runtime, checks, critical=("kafka", "database"))


@app.post("/api/interactions")
async def ingest_interaction(http_request: Request, request: UserInteractionRequest = Body(...)):
    action = request.action.value if hasattr(request.action, "value") else str(request.action)
    server_received_at = time.time()
    try:
        client_event_time = validate_public_event_time(
            request.event_time, server_received_at
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # The model fills a missing event_time with time.time() at parse; only a
    # client-supplied value is a client clock reading worth keeping.
    client_supplied_time = bool({"event_time", "timestamp"} & request.__fields_set__)
    # No served-slate lookup on this hot path: the PIT jobs bound feedback from
    # below with the server available_at, so only the future side is clamped.
    event_time = clamp_public_event_time(
        client_event_time, upper_bound=server_received_at
    )

    request_id = http_request.state.request_id
    unavailable = _kafka_unavailable_reason()
    if unavailable is not None:
        app.state.runtime.observability.record_interaction_ingest(action, "kafka_unavailable")
        raise _kafka_error("KAFKA_UNAVAILABLE", unavailable, request_id)

    event_id = str(request.event_id or uuid.uuid4())
    hashed_fields = {
        "user_id": request.user_id,
        "product_id": request.product_id,
        "action": action,
        "context": request.context,
    }
    # Hash the client-supplied time, never the clamped one: the clamp depends on
    # each retry's receive time, so hashing it would turn retries into collisions.
    if client_supplied_time:
        hashed_fields["event_time"] = request.event_time
    payload_hash = hashlib.sha256(
        json.dumps(
            hashed_fields,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    duplicate = False
    owns_claim = False
    claim_token: Optional[str] = None
    uses_postgres_ledger = system_store is not None
    event_payload = {
        "user_id": request.user_id,
        "product_id": request.product_id,
        "action": action,
        "context": request.context,
        "event_time": event_time,
        "client_event_time": client_event_time if client_supplied_time else None,
        "server_received_at": server_received_at,
        "request_id": request_id,
    }
    ledger_key = f"interaction:idempotency:{event_id}"
    if uses_postgres_ledger:
        claim = await system_store.claim_interaction_event(
            event_id=event_id,
            payload_hash=payload_hash,
            event_payload=event_payload,
            lease_seconds=pending_lease_seconds,
        )
        if claim["status"] == "collision":
            raise HTTPException(
                status_code=409,
                detail="event_id was already used for a different interaction",
            )
        if claim["status"] == "pending":
            raise HTTPException(
                status_code=503,
                detail="Interaction with this event_id is still publishing; retry",
                headers={"Retry-After": "1"},
            )
        duplicate = claim["status"] == "duplicate"
        owns_claim = claim["status"] == "claimed"
        claim_token = claim.get("claim_token")
        if owns_claim and not claim_token:
            raise RuntimeError("interaction ledger claim did not return an owner token")
        event_payload = claim.get("event_payload", event_payload)
    elif feature_store and feature_store.redis_client:
        pending = json.dumps({"payload_hash": payload_hash, "status": "pending"})
        owns_claim = bool(
            await feature_store.redis_client.set(
                ledger_key,
                pending,
                nx=True,
                ex=pending_lease_seconds,
            )
        )
        if not owns_claim:
            for _attempt in range(10):
                existing = await feature_store.redis_client.get(ledger_key)
                if not existing:
                    owns_claim = bool(
                        await feature_store.redis_client.set(
                            ledger_key,
                            pending,
                            nx=True,
                            ex=pending_lease_seconds,
                        )
                    )
                    if owns_claim:
                        break
                else:
                    record = json.loads(existing)
                    if record.get("payload_hash") != payload_hash:
                        raise HTTPException(
                            status_code=409,
                            detail="event_id was already used for a different interaction",
                        )
                    if record.get("status") == "published":
                        duplicate = True
                        break
                await asyncio.sleep(0.02)
            if not owns_claim and not duplicate:
                raise HTTPException(
                    status_code=503,
                    detail="Interaction with this event_id is still publishing; retry",
                    headers={"Retry-After": "1"},
                )

    async def release_claim() -> None:
        if owns_claim and uses_postgres_ledger:
            await system_store.release_interaction_event_claim(
                event_id, claim_token=claim_token
            )
        elif owns_claim and feature_store and feature_store.redis_client:
            await feature_store.redis_client.delete(ledger_key)

    async def mark_published() -> bool:
        if owns_claim and uses_postgres_ledger:
            return await system_store.mark_interaction_event_published(
                event_id, claim_token=claim_token
            )
        if owns_claim and feature_store and feature_store.redis_client:
            await feature_store.redis_client.set(
                ledger_key,
                json.dumps({"payload_hash": payload_hash, "status": "published"}),
                ex=IDEMPOTENCY_PUBLISHED_TTL_SECONDS,
            )
        return True

    success = True
    if not duplicate:
        publish = asyncio.ensure_future(
            kafka_manager.send_user_interaction(
                event_id=event_id,
                user_id=event_payload["user_id"],
                product_id=event_payload["product_id"],
                action=event_payload["action"],
                context=event_payload["context"],
                event_time=event_payload["event_time"],
                server_received_at=event_payload["server_received_at"],
                request_id=event_payload["request_id"],
                client_event_time=event_payload.get("client_event_time"),
            )
        )
        try:
            success = await asyncio.wait_for(
                asyncio.shield(publish), timeout=publish_timeout_seconds
            )
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            # aiokafka keeps delivering a send it has buffered, so the claim
            # stays pending and follows the send's real outcome. Releasing it
            # now would let a retry publish the same event_id twice.
            _track_late_publish(
                _settle_late_interaction(
                    publish,
                    action=action,
                    user_id=request.user_id,
                    mark_published=mark_published,
                    release_claim=release_claim,
                )
            )
            if isinstance(exc, asyncio.CancelledError):
                raise
            app.state.runtime.observability.record_interaction_ingest(
                action, "publish_timeout"
            )
            raise _kafka_error(
                "KAFKA_PUBLISH_TIMEOUT",
                "Kafka did not acknowledge the interaction in time; retry with "
                "the same event_id",
                request_id,
            )
    if not success:
        await release_claim()
        app.state.runtime.observability.record_interaction_ingest(action, "publish_failed")
        raise _kafka_error(
            "KAFKA_PUBLISH_FAILED",
            "Failed to publish interaction event to Kafka",
            request_id,
        )
    if not await mark_published():
        raise HTTPException(
            status_code=503,
            detail="Interaction publish lease expired before acknowledgement",
        )
    app.state.runtime.observability.record_interaction_ingest(action, "accepted")

    if not duplicate and client_supplied_time:
        app.state.runtime.observability.record_public_event_time(
            "interaction",
            client_time=client_event_time,
            server_received_at=server_received_at,
            clamped_bound=_clamped_bound(client_event_time, event_time),
        )

    if not duplicate:
        asyncio.create_task(
            _invalidate_user_serving_cache(request.user_id),
            name=f"invalidate-serving-cache-{request.user_id}",
        )

    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "queue": "kafka",
            "processing_mode": "async",
            "event_id": event_id,
            "duplicate": duplicate,
        },
    )


def _kafka_unavailable_reason() -> Optional[str]:
    """Why a publish cannot start now, or None when the producer is usable."""
    if not app.state.runtime.config.kafka_config.enable or kafka_manager is None:
        return "Kafka producer is not started"
    if not kafka_manager.producer.is_connected:
        # send() would reconnect inline and hold the request for up to the
        # Kafka request timeout; answer now and reconnect in the background.
        if kafka_supervisor is not None:
            kafka_supervisor.reconnect_in_background()
        return "Kafka producer is disconnected"
    return None


def _kafka_error(code: str, message: str, request_id: Optional[str]) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "retryable": True,
            }
        },
        headers={"Retry-After": "1"},
    )


def _track_late_publish(coro: Awaitable[None]) -> None:
    task = asyncio.ensure_future(coro)
    _late_publishes.add(task)
    task.add_done_callback(_late_publishes.discard)


async def _settle_late_interaction(
    publish: Awaitable[bool],
    *,
    action: str,
    user_id: str,
    mark_published: Callable[[], Awaitable[bool]],
    release_claim: Callable[[], Awaitable[None]],
) -> None:
    """Finish the ledger claim of a send that outlived the publish budget."""
    observability = app.state.runtime.observability
    try:
        published = await publish
    except Exception:
        logger.exception("late_interaction_publish_failed")
        published = False
    try:
        if not published:
            await release_claim()
            observability.record_interaction_ingest(action, "late_publish_failed")
            return
        if not await mark_published():
            # The lease ran out first, so a retry may have published again.
            logger.warning("late_interaction_publish_lost_lease", extra={"action": action})
        observability.record_interaction_ingest(action, "published_late")
    except Exception:
        # The lease still expires, so a retry recovers the event_id.
        logger.exception("late_interaction_ledger_update_failed")
        return
    await _invalidate_user_serving_cache(user_id)


def _clamped_bound(client_time: float, server_time: float) -> Optional[str]:
    if server_time < client_time:
        return "upper"
    if server_time > client_time:
        return "lower"
    return None


async def _invalidate_user_serving_cache(user_id: str) -> None:
    try:
        await asyncio.wait_for(
            feature_store.invalidate_user_serving_cache(user_id),
            timeout=0.25,
        )
    except asyncio.TimeoutError:
        logger.warning("serving_cache_invalidation_timed_out", extra={"user_id": user_id})
    except Exception as exc:
        logger.warning(
            "serving_cache_invalidation_failed",
            extra={"user_id": user_id, "error": str(exc)},
        )


@app.post("/api/impressions/viewed")
async def ingest_viewed_impression(
    http_request: Request, request: ViewedImpressionRequest = Body(...)
):
    received_at = time.time()
    try:
        client_viewed_at = validate_public_event_time(request.viewed_at, received_at)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    request_id = http_request.state.request_id
    unavailable = _kafka_unavailable_reason()
    if unavailable is not None:
        raise _kafka_error("KAFKA_UNAVAILABLE", unavailable, request_id)

    slate = None
    if system_store is not None:
        slate = await system_store.get_recommendation_impression_slate(
            str(request.impression_id)
        )
        requested_items = {item.product_id: item.position for item in request.items}
        if slate is None or any(
            slate["items"].get(product_id) != position
            for product_id, position in requested_items.items()
        ):
            app.state.runtime.observability.record_recommendation_impression(
                "viewed", "orphan"
            )
            return JSONResponse(
                status_code=202,
                content={"status": "isolated", "reason": "unknown_slate_item"},
            )

    user_id = str(
        (slate or {}).get("user_id")
        or request.context.get("user_id")
        or request.impression_id
    )
    # A view cannot precede its serve or follow its receipt; the PIT jobs anchor
    # on this value, so the client clock only places it inside that window.
    served_at = (slate or {}).get("served_at")
    viewed_at = clamp_public_event_time(
        client_viewed_at, upper_bound=received_at, lower_bound=served_at
    )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + publish_timeout_seconds
    for item in request.items:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise _view_publish_timeout(request_id)
        publish = asyncio.ensure_future(
            kafka_manager.send_recommendation_view(
                event_id=str(item.event_id),
                impression_id=str(request.impression_id),
                user_id=user_id,
                product_id=item.product_id,
                position=item.position,
                viewed_at=viewed_at,
                context=request.context,
                request_id=request_id,
                client_viewed_at=client_viewed_at,
                served_at=served_at,
                server_received_at=received_at,
            )
        )
        try:
            success = await asyncio.wait_for(asyncio.shield(publish), timeout=remaining)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            # Views carry per-item event_ids and have no ledger, so a retry
            # only re-sends them; keep the buffered send alive until it ends.
            _track_late_publish(publish)
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise _view_publish_timeout(request_id)
        if not success:
            raise _kafka_error(
                "KAFKA_PUBLISH_FAILED", "Failed to publish viewed impression", request_id
            )

    if system_store is not None:
        await system_store.record_recommendation_impression_views(
            impression_id=str(request.impression_id),
            viewed_at=viewed_at,
            items=[item.dict() for item in request.items],
            context=request.context,
        )
    app.state.runtime.observability.record_recommendation_impression(
        "viewed", "accepted"
    )
    app.state.runtime.observability.record_public_event_time(
        "view",
        client_time=client_viewed_at,
        server_received_at=received_at,
        clamped_bound=_clamped_bound(client_viewed_at, viewed_at),
    )
    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "impression_id": str(request.impression_id),
            "accepted_items": len(request.items),
        },
    )


def _view_publish_timeout(request_id: Optional[str]) -> HTTPException:
    app.state.runtime.observability.record_recommendation_impression(
        "viewed", "publish_timeout"
    )
    return _kafka_error(
        "KAFKA_PUBLISH_TIMEOUT",
        "Kafka did not acknowledge the viewed impression in time; retry",
        request_id,
    )


@app.get("/health")
async def health_check():
    feature_store_health = await feature_store.health_check()
    kafka_health = {"status": "healthy", "response_time_ms": 0.0}
    if kafka_manager:
        kafka_status = await kafka_manager.health_check()
        producer_health = kafka_status.get("producer", {})
        kafka_health = {
            "status": producer_health.get("status", "healthy"),
            "response_time_ms": 0.0,
            "error": None if producer_health.get("connected") else "Kafka producer unavailable",
        }
    elif app.state.runtime.config.kafka_config.enable:
        kafka_health = {"status": "degraded", "error": "Kafka producer unavailable"}

    return build_health_response(
        {
            "feature_store": component_health(
                feature_store_health.get("status", "unhealthy"),
                feature_store_health.get("response_time_ms"),
                feature_store_health.get("error"),
            ),
            "kafka": component_health(
                kafka_health.get("status", "healthy"),
                kafka_health.get("response_time_ms"),
                kafka_health.get("error"),
            ),
        },
        app.state.runtime.started_at,
    )


@app.get("/metrics")
async def metrics():
    return await build_metrics_response(
        app.state.runtime,
        feature_store=feature_store,
        kafka_manager=kafka_manager,
    )
