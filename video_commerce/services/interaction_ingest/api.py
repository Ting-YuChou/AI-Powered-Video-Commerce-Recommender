"""
Dedicated high-throughput interaction ingest API.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
import hashlib
import json
from typing import Awaitable, Callable, Dict, Optional, Set, Tuple

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
from video_commerce.services.interaction_ingest.publish_breaker import PublishBreaker
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
# Claims held by sends that outlived the publish timeout are renewed this
# often, well inside the lease, until each send resolves.
CLAIM_RENEW_INTERVAL_SECONDS = IDEMPOTENCY_PENDING_TTL_SECONDS / 3
LATE_PUBLISH_DRAIN_SECONDS = 5.0
# Extends a Redis claim only while it still holds this request's pending marker,
# so a renewal never shortens the TTL of a marker that just turned published.
_RENEW_PENDING_REDIS_CLAIM = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('expire', KEYS[1], ARGV[2]) end return 0"
)

app = create_service_app(
    title="Interaction Ingest Service",
    description="Dedicated interaction ingestion service for async event capture",
    service_name="interaction-ingest-service",
)

feature_store: Optional[FeatureStore] = None
kafka_manager = None
kafka_supervisor: Optional[KafkaProducerSupervisor] = None
system_store: Optional[SystemStore] = None
# Startup replaces these from ServiceTopologyConfig.
publish_timeout_seconds = 3.0
max_outstanding_publishes = 1000
publish_breaker = PublishBreaker(failure_threshold=5, cooldown_seconds=5.0)
# Every Kafka send not yet resolved, including ones past the publish timeout.
_outstanding_publishes: Set[asyncio.Future] = set()
# Tasks settling sends that outlived the publish timeout; aiokafka still
# delivers or fails a send it has buffered.
_late_publishes: Set[asyncio.Task] = set()
# event_id -> (claim_token, ledger_key, redis pending marker) for claims those
# sends keep pending.
_late_claims: Dict[str, Tuple[Optional[str], str, str]] = {}
_claim_renewer: Optional[asyncio.Task] = None


@app.middleware("http")
async def internal_auth(request: Request, call_next):
    denied = internal_service_auth_response(request, app.state.runtime)
    if denied is not None:
        return denied
    return await call_next(request)


@app.on_event("startup")
async def startup_event():
    global feature_store, kafka_manager, kafka_supervisor, system_store
    global publish_timeout_seconds, max_outstanding_publishes, publish_breaker

    runtime = app.state.runtime
    runtime.config = Config()
    configure_service_logging(runtime)
    topology = runtime.config.service_topology_config
    publish_timeout_seconds = topology.interaction_ingest_publish_timeout_ms / 1000
    max_outstanding_publishes = topology.interaction_ingest_max_outstanding_publishes
    publish_breaker = PublishBreaker(
        failure_threshold=topology.interaction_ingest_breaker_failure_threshold,
        cooldown_seconds=topology.interaction_ingest_breaker_cooldown_seconds,
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


@app.on_event("shutdown")
async def shutdown_event():
    if _late_publishes:
        # Let late sends settle their ledger claims while the stores are open.
        await asyncio.wait(set(_late_publishes), timeout=LATE_PUBLISH_DRAIN_SECONDS)
    if _claim_renewer is not None:
        # Claims still pending lapse within one lease, so a retry recovers them.
        _claim_renewer.cancel()
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

    observability = app.state.runtime.observability
    request_id = http_request.state.request_id
    event_id = str(request.event_id or uuid.uuid4())
    # Checked before the ledger so an outage costs no ledger writes. A retry of
    # an already published event_id therefore also waits for Kafka to return.
    unavailable = _kafka_unavailable_reason()
    if unavailable is not None:
        observability.record_interaction_ingest(action, "kafka_unavailable")
        raise _retryable_503("KAFKA_UNAVAILABLE", unavailable, request_id, event_id=event_id)

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
    pending = json.dumps({"payload_hash": payload_hash, "status": "pending"})
    if uses_postgres_ledger:
        claim = await system_store.claim_interaction_event(
            event_id=event_id,
            payload_hash=payload_hash,
            event_payload=event_payload,
            lease_seconds=IDEMPOTENCY_PENDING_TTL_SECONDS,
        )
        if claim["status"] == "collision":
            raise HTTPException(
                status_code=409,
                detail="event_id was already used for a different interaction",
            )
        if claim["status"] == "pending":
            raise _publish_pending(action, request_id, event_id)
        duplicate = claim["status"] == "duplicate"
        owns_claim = claim["status"] == "claimed"
        claim_token = claim.get("claim_token")
        if owns_claim and not claim_token:
            raise RuntimeError("interaction ledger claim did not return an owner token")
        event_payload = claim.get("event_payload", event_payload)
    elif feature_store and feature_store.redis_client:
        owns_claim = bool(
            await feature_store.redis_client.set(
                ledger_key,
                pending,
                nx=True,
                ex=IDEMPOTENCY_PENDING_TTL_SECONDS,
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
                            ex=IDEMPOTENCY_PENDING_TTL_SECONDS,
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
                raise _publish_pending(action, request_id, event_id)

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

    def settle_later(publish: asyncio.Future, *, count_failure: bool) -> None:
        # aiokafka keeps delivering a send it has buffered, so the claim stays
        # pending (and renewed) until the send resolves. Releasing it now would
        # let a retry publish the same event_id twice.
        if owns_claim:
            _hold_claim(event_id, claim_token, ledger_key, pending)
        _settle_later(
            _settle_late_interaction(
                publish,
                event_id=event_id,
                action=action,
                user_id=request.user_id,
                mark_published=mark_published,
                release_claim=release_claim,
                count_failure=count_failure,
            )
        )

    success = True
    if not duplicate:
        if not publish_breaker.admit():
            await release_claim()
            observability.record_interaction_ingest(action, "kafka_unavailable")
            raise _retryable_503(
                "KAFKA_UNAVAILABLE", _BREAKER_OPEN, request_id, event_id=event_id
            )
        publish = _start_publish(
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
        except asyncio.TimeoutError:
            publish_breaker.record_failure()
            settle_later(publish, count_failure=False)
            observability.record_interaction_ingest(action, "publish_timeout")
            raise _retryable_503(
                "KAFKA_PUBLISH_TIMEOUT",
                "Kafka did not acknowledge the interaction in time; retry with "
                "the same event_id",
                request_id,
                event_id=event_id,
            )
        except asyncio.CancelledError:
            # The client went away; the send still runs, so settle it later.
            publish_breaker.release_probe()
            settle_later(publish, count_failure=True)
            raise
        except Exception:
            # Raised before aiokafka buffered anything (send() itself never
            # raises), so the claim can be released right away.
            logger.exception("interaction_publish_raised")
            success = False
        if success:
            publish_breaker.record_success()
        else:
            publish_breaker.record_failure()
    if not success:
        await release_claim()
        observability.record_interaction_ingest(action, "publish_failed")
        raise _retryable_503(
            "KAFKA_PUBLISH_FAILED",
            "Failed to publish interaction event to Kafka",
            request_id,
            event_id=event_id,
        )
    if not await mark_published():
        observability.record_interaction_ingest(action, "lease_lost")
        raise _retryable_503(
            "INTERACTION_LEASE_LOST",
            "Interaction publish lease expired before acknowledgement; retry "
            "with the same event_id",
            request_id,
            event_id=event_id,
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


_BREAKER_OPEN = "Kafka publishes keep failing; retry shortly"


def _kafka_unavailable_reason() -> Optional[str]:
    """Why a publish must not start now, or None when it may."""
    if not app.state.runtime.config.kafka_config.enable or kafka_manager is None:
        return "Kafka producer is not started"
    if not kafka_manager.producer.is_connected:
        # send() would reconnect inline and hold the request for up to the
        # Kafka request timeout; answer now and reconnect in the background.
        if kafka_supervisor is not None:
            kafka_supervisor.reconnect_in_background()
        return "Kafka producer is disconnected"
    if publish_breaker.blocking():
        return _BREAKER_OPEN
    if len(_outstanding_publishes) >= max_outstanding_publishes:
        return "Too many Kafka publishes are unresolved; retry shortly"
    return None


def _retryable_503(
    code: str, message: str, request_id: Optional[str], **fields: str
) -> HTTPException:
    """503 in the shared error envelope, plus ids the client retries with."""
    return HTTPException(
        status_code=503,
        detail={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "retryable": True,
            },
            **fields,
        },
        headers={"Retry-After": "1"},
    )


def _publish_pending(action: str, request_id: Optional[str], event_id: str) -> HTTPException:
    app.state.runtime.observability.record_interaction_ingest(action, "publish_pending")
    return _retryable_503(
        "INTERACTION_PUBLISH_PENDING",
        "Interaction with this event_id is still publishing; retry",
        request_id,
        event_id=event_id,
    )


def _start_publish(coro: Awaitable[bool]) -> asyncio.Future:
    publish = asyncio.ensure_future(coro)
    _outstanding_publishes.add(publish)
    publish.add_done_callback(_outstanding_publishes.discard)
    return publish


def _settle_later(coro: Awaitable[None]) -> None:
    task = asyncio.ensure_future(coro)
    _late_publishes.add(task)
    task.add_done_callback(_late_publishes.discard)


async def _publish_outcome(publish: asyncio.Future, log_event: str) -> bool:
    try:
        # Shielded so cancelling a settle task never cancels the send itself.
        return bool(await asyncio.shield(publish))
    except Exception:
        logger.exception(log_event)
        return False


def _hold_claim(
    event_id: str, claim_token: Optional[str], ledger_key: str, pending_marker: str
) -> None:
    global _claim_renewer
    _late_claims[event_id] = (claim_token, ledger_key, pending_marker)
    if _claim_renewer is None or _claim_renewer.done():
        _claim_renewer = asyncio.ensure_future(_renew_late_claims())


async def _renew_late_claims() -> None:
    """Keep claims of late sends pending until each send resolves.

    aiokafka has no fixed bound on how long it holds a buffered send (a
    metadata refresh walks every broker, each connect up to its request
    timeout), so a fixed lease could lapse first and let a retry publish the
    event_id again. Renewal stops when the process dies, so a crashed worker's
    claims still lapse within one lease.
    """
    while _late_claims:
        await asyncio.sleep(CLAIM_RENEW_INTERVAL_SECONDS)
        claims = dict(_late_claims)
        if not claims:
            break
        try:
            if system_store is not None:
                await system_store.renew_interaction_event_claims(
                    [(event_id, token) for event_id, (token, _, _) in claims.items() if token],
                    lease_seconds=IDEMPOTENCY_PENDING_TTL_SECONDS,
                )
            elif feature_store and feature_store.redis_client:
                for _token, ledger_key, pending_marker in claims.values():
                    await feature_store.redis_client.eval(
                        _RENEW_PENDING_REDIS_CLAIM,
                        1,
                        ledger_key,
                        pending_marker,
                        IDEMPOTENCY_PENDING_TTL_SECONDS,
                    )
        except Exception:
            logger.exception("late_interaction_claim_renewal_failed")


async def _settle_late_interaction(
    publish: asyncio.Future,
    *,
    event_id: str,
    action: str,
    user_id: str,
    mark_published: Callable[[], Awaitable[bool]],
    release_claim: Callable[[], Awaitable[None]],
    count_failure: bool,
) -> None:
    """Finish the ledger claim of a send that outlived the publish timeout."""
    observability = app.state.runtime.observability
    try:
        published = await _publish_outcome(publish, "late_interaction_publish_raised")
    finally:
        _late_claims.pop(event_id, None)
    if published:
        publish_breaker.record_success()
    elif count_failure:
        publish_breaker.record_failure()
    try:
        if not published:
            await release_claim()
            observability.record_interaction_ingest(action, "late_publish_failed")
            return
        if await mark_published():
            observability.record_interaction_ingest(action, "published_late")
        else:
            # Only a renewal that failed for a whole lease gets here. A retry
            # may have published the event_id again; Flink's
            # EventIdDeduplicator drops that second copy.
            logger.warning(
                "late_interaction_publish_lost_lease", extra={"event_id": event_id}
            )
            observability.record_interaction_ingest(action, "late_publish_lost_lease")
    except Exception:
        # Renewal has stopped, so the lease lapses and a retry recovers it.
        logger.exception("late_interaction_ledger_update_failed")
        return
    await _invalidate_user_serving_cache(user_id)


async def _settle_late_view(publish: asyncio.Future, *, count_failure: bool) -> None:
    published = await _publish_outcome(publish, "late_view_publish_raised")
    if published:
        publish_breaker.record_success()
    elif count_failure:
        publish_breaker.record_failure()
    app.state.runtime.observability.record_recommendation_impression(
        "viewed", "published_late" if published else "late_publish_failed"
    )


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
    observability = app.state.runtime.observability
    request_id = http_request.state.request_id
    impression_id = str(request.impression_id)
    unavailable = _kafka_unavailable_reason()
    if unavailable is not None:
        observability.record_recommendation_impression("viewed", "kafka_unavailable")
        raise _retryable_503(
            "KAFKA_UNAVAILABLE", unavailable, request_id, impression_id=impression_id
        )

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
    if not publish_breaker.admit():
        observability.record_recommendation_impression("viewed", "kafka_unavailable")
        raise _retryable_503(
            "KAFKA_UNAVAILABLE", _BREAKER_OPEN, request_id, impression_id=impression_id
        )
    # Views have no ledger: a retry re-sends every item under the same
    # event_ids, and the Flink view sink (ON CONFLICT DO NOTHING on event_id and
    # on impression/product) and SourceEventDeduplicator drop the copies.
    publishes = [
        _start_publish(
            kafka_manager.send_recommendation_view(
                event_id=str(item.event_id),
                impression_id=impression_id,
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
        for item in request.items
    ]
    try:
        _done, unresolved = await asyncio.wait(publishes, timeout=publish_timeout_seconds)
    except asyncio.CancelledError:
        publish_breaker.release_probe()
        for publish in publishes:
            if not publish.done():
                _settle_later(_settle_late_view(publish, count_failure=True))
        raise
    if unresolved:
        publish_breaker.record_failure()
        for publish in unresolved:
            _settle_later(_settle_late_view(publish, count_failure=False))
        observability.record_recommendation_impression("viewed", "publish_timeout")
        raise _retryable_503(
            "KAFKA_PUBLISH_TIMEOUT",
            "Kafka did not acknowledge the viewed impression in time; retry",
            request_id,
            impression_id=impression_id,
        )
    outcomes = [await _publish_outcome(publish, "view_publish_raised") for publish in publishes]
    if not all(outcomes):
        publish_breaker.record_failure()
        observability.record_recommendation_impression("viewed", "publish_failed")
        raise _retryable_503(
            "KAFKA_PUBLISH_FAILED",
            "Failed to publish viewed impression",
            request_id,
            impression_id=impression_id,
        )
    publish_breaker.record_success()

    if system_store is not None:
        await system_store.record_recommendation_impression_views(
            impression_id=impression_id,
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
