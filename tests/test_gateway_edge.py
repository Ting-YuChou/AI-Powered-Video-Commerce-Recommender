"""Stage B edge contract: readiness tiers, rate limiting, body limits, identity."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from video_commerce.common.config import Config
from video_commerce.common.models import (
    RecommendationRequest,
    UserInteractionRequest,
    validate_bounded_context,
)
from video_commerce.common.service_common import (
    RedisRateLimiter,
    ServiceRuntime,
    build_readiness_response,
    internal_service_auth_response,
    parse_trusted_proxy_networks,
    resolve_client_ip,
)
from video_commerce.data_plane import kafka_client
from video_commerce.services.gateway import api as gateway_api


def _request(path="/api/recommendations", headers=None, client=("10.0.0.1", 1234), method="POST"):
    raw_headers = [
        (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": raw_headers,
            "client": client,
            "server": ("gateway", 8000),
            "scheme": "http",
        }
    )


# --- readiness ---------------------------------------------------------------


def _runtime():
    runtime = ServiceRuntime("gateway-api")
    runtime.config = SimpleNamespace(
        monitoring_config=SimpleNamespace(request_id_header="X-Request-ID")
    )
    return runtime


def test_readiness_only_critical_checks_decide():
    checks = {
        "redis": {"status": "healthy"},
        "ranking_service": {"status": "unhealthy", "error": "ConnectError: 10.0.0.9"},
    }

    response = build_readiness_response(_runtime(), checks, critical=["redis"])
    body = json.loads(response.body)

    assert response.status_code == 200
    assert body["status"] == "degraded"
    assert body["checks"]["redis"]["critical"] is True
    assert body["checks"]["ranking_service"]["critical"] is False


def test_readiness_fails_on_critical_unhealthy_and_redacts_errors():
    checks = {"redis": {"status": "unhealthy", "error": "Timeout to 10.1.2.3:6379"}}

    response = build_readiness_response(
        _runtime(), checks, critical=["redis"], redact_errors=True
    )
    body = json.loads(response.body)

    assert response.status_code == 503
    assert body["status"] == "not_ready"
    assert "error" not in body["checks"]["redis"]


def test_readiness_without_critical_keeps_legacy_all_healthy_rule():
    response = build_readiness_response(
        _runtime(), {"a": {"status": "healthy"}, "b": {"status": "degraded"}}
    )

    assert response.status_code == 503


class _HealthyFeatureStore:
    async def health_check(self):
        return {"status": "healthy", "response_time_ms": 0.1}

    async def get_service_heartbeat_status(self, service_name):
        return {"status": "healthy", "live_instances": 1, "service": service_name}


def _gateway_config(**overrides):
    config = Config()
    config.security_config.auth_mode = overrides.get("auth_mode", "disabled")
    config.api_config.api_key = overrides.get("api_key")
    config.feature_pipeline_config.mode = overrides.get("pipeline_mode", "flink")
    config.feature_pipeline_config.flink_rest_url = ""
    config.kafka_config.enable = overrides.get("kafka_enable", True)
    config.api_config.readiness_cache_seconds = 0.0
    return config


def _install_gateway(monkeypatch, config):
    monkeypatch.setattr(gateway_api.app.state.runtime, "config", config)
    monkeypatch.setattr(gateway_api, "feature_store", _HealthyFeatureStore())
    monkeypatch.setattr(gateway_api, "proxy_client", object())
    monkeypatch.setattr(gateway_api, "recommendation_proxy_pool", object())
    monkeypatch.setattr(gateway_api, "interaction_proxy_pool", object())
    monkeypatch.setattr(gateway_api, "bearer_token_validator", None)
    monkeypatch.setattr(gateway_api, "rate_limiter", None)
    monkeypatch.setattr(gateway_api, "failed_auth_limiter", None)
    monkeypatch.setattr(gateway_api, "system_store", None)
    monkeypatch.setattr(gateway_api, "kafka_manager", None)
    monkeypatch.setattr(gateway_api, "_probe_cache", {})


@pytest.mark.asyncio
async def test_gateway_readyz_does_not_probe_shared_downstreams(monkeypatch):
    _install_gateway(monkeypatch, _gateway_config())

    async def fail_probe(_url):
        raise AssertionError("public /readyz must not fan out to downstream services")

    monkeypatch.setattr(gateway_api, "_probe_health", fail_probe)

    response = await gateway_api.readyz()

    assert response.status_code == 200
    assert set(json.loads(response.body)["checks"]) == {"startup", "auth", "redis"}


@pytest.mark.asyncio
async def test_gateway_readyz_not_ready_when_auth_is_not_configured(monkeypatch):
    _install_gateway(monkeypatch, _gateway_config(auth_mode="api_key"))

    response = await gateway_api.readyz()

    assert response.status_code == 503
    assert json.loads(response.body)["checks"]["auth"]["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_gateway_full_readiness_reports_downstreams_and_skips_feature_worker(
    monkeypatch,
):
    _install_gateway(monkeypatch, _gateway_config(kafka_enable=False))
    probed = []

    ranking_url = gateway_api.app.state.runtime.config.service_topology_config.ranking_service_url

    async def probe(url):
        probed.append(url)
        status = "degraded" if url.startswith(ranking_url) else "healthy"
        return {"status": status, "response_time_ms": 1.0}

    monkeypatch.setattr(gateway_api, "_probe_health", probe)
    gateway_api.app.state.runtime.config.kafka_config.enable = True
    monkeypatch.setattr(
        gateway_api,
        "kafka_manager",
        SimpleNamespace(
            health_check=_async_value(
                {"producer": {"status": "healthy", "connected": True}}
            )
        ),
    )
    monkeypatch.setattr(
        gateway_api,
        "object_storage",
        SimpleNamespace(health_check=_async_value({"status": "healthy"}, kwargs=True)),
    )

    response = await gateway_api.readyz_full()
    body = json.loads(response.body)

    assert len(probed) == 3
    assert response.status_code == 200
    assert body["status"] == "degraded"
    assert "feature-worker" not in body["checks"]
    assert {"content-worker", "model-trainer"} <= set(body["checks"])


def _async_value(value, kwargs=False):
    async def call(*_args, **_kwargs):
        return value

    return call


# --- rate limiting -----------------------------------------------------------


class _HangingPipeline:
    def incr(self, _key):
        return self

    def expire(self, _key, _seconds):
        return self

    async def execute(self):
        await asyncio.sleep(5)


class _HangingRedis:
    def pipeline(self, transaction=False):
        return _HangingPipeline()

    async def get(self, _key):
        await asyncio.sleep(5)


@pytest.mark.asyncio
async def test_rate_limiter_fails_open_fast_and_cools_down():
    errors = []
    limiter = RedisRateLimiter(
        _HangingRedis(), limit=1, timeout_seconds=0.01, on_error=errors.append
    )

    started = asyncio.get_running_loop().time()
    assert await limiter.allow("ip:1.2.3.4") is True
    assert await limiter.allow("ip:1.2.3.4") is True
    assert await limiter.is_blocked("ip:1.2.3.4") is False
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.5
    assert errors[0] == "timeout"
    assert errors[1:] == ["cooldown", "cooldown"]


def test_client_ip_uses_forwarded_for_only_from_trusted_proxies():
    trusted = parse_trusted_proxy_networks("10.0.0.0/8")
    spoofed = {"x-forwarded-for": "1.1.1.1, 203.0.113.7, 10.0.0.5"}

    assert resolve_client_ip(_request(headers=spoofed), trusted) == "203.0.113.7"
    assert (
        resolve_client_ip(
            _request(headers=spoofed, client=("198.51.100.2", 1)), trusted
        )
        == "198.51.100.2"
    )
    assert resolve_client_ip(_request(), trusted) == "10.0.0.1"


def test_rate_limit_key_prefers_authenticated_principal():
    request = _request(headers={"x-api-key": "secret"})
    assert gateway_api._rate_limit_key(request, "1.2.3.4") == "ip:1.2.3.4"

    request.state.api_key_authenticated = True
    assert gateway_api._rate_limit_key(request, "1.2.3.4").startswith("key:")

    request.state.auth_claims = {"sub": "user-1"}
    key = gateway_api._rate_limit_key(request, "1.2.3.4")
    assert key.startswith("sub:") and "user-1" not in key


class _CountingLimiter:
    def __init__(self, limit):
        self.limit = limit
        self.counts = {}

    async def allow(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key] <= self.limit


@pytest.mark.asyncio
async def test_repeated_failed_auth_from_one_ip_is_rate_limited(monkeypatch):
    _install_gateway(monkeypatch, _gateway_config(auth_mode="api_key", api_key="secret"))
    monkeypatch.setattr(gateway_api, "failed_auth_limiter", _CountingLimiter(limit=2))

    async def call_next(_request):
        return JSONResponse({"ok": True})

    statuses = [
        (
            await gateway_api.auth_and_rate_limit(
                _request(headers={"x-api-key": "wrong"}), call_next
            )
        ).status_code
        for _ in range(3)
    ]

    assert statuses == [401, 401, 429]


def test_auth_rejections_pass_through_observability_and_cors(monkeypatch):
    _install_gateway(monkeypatch, _gateway_config(auth_mode="api_key", api_key="secret"))
    client = TestClient(gateway_api.app)

    response = client.post(
        "/api/recommendations",
        json={"user_id": "u1"},
        headers={"Origin": "http://localhost", "X-Request-ID": "req-123"},
    )

    assert response.status_code == 401
    # Inside the shared middleware, a 401 gets the validated request id and CORS.
    assert response.headers["x-request-id"] == "req-123"
    assert response.headers["access-control-allow-origin"] in {"*", "http://localhost"}
    assert response.json()["error"]["request_id"] == "req-123"


# --- body limits -------------------------------------------------------------


def test_body_limit_rejects_oversized_json_and_unsized_uploads(monkeypatch):
    config = _gateway_config()
    config.api_config.max_request_size = 1024
    _install_gateway(monkeypatch, config)
    client = TestClient(gateway_api.app)

    too_big = client.post(
        "/api/recommendations",
        content=b"{" + b" " * 2048 + b"}",
        headers={"Content-Type": "application/json"},
    )
    assert too_big.status_code == 413
    assert too_big.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"

    def chunks():
        yield b"--x\r\n"

    unsized = client.post(
        "/api/content/upload",
        content=chunks(),
        headers={"Content-Type": "multipart/form-data; boundary=x"},
    )
    assert unsized.status_code == 411


# --- identity ----------------------------------------------------------------


@pytest.mark.parametrize("user_id", ["*", "a:b", "user 1", "u[1]", "x\n"])
def test_user_id_rejects_glob_separator_and_whitespace(user_id):
    with pytest.raises(ValueError):
        RecommendationRequest(user_id=user_id)


def test_catalog_ids_allow_namespaced_values_but_not_globs():
    UserInteractionRequest(user_id="user_1@example.com", product_id="sku:123/a", action="click")
    with pytest.raises(ValueError):
        UserInteractionRequest(user_id="u1", product_id="sku*", action="click")


def test_context_is_bounded_in_size_keys_and_depth():
    validate_bounded_context({"device": "mobile", "nested": {"a": {"b": 1}}})
    with pytest.raises(ValueError):
        validate_bounded_context({"blob": "x" * 9000})
    with pytest.raises(ValueError):
        validate_bounded_context({f"k{i}": i for i in range(65)})
    with pytest.raises(ValueError):
        validate_bounded_context({"a": {"b": {"c": {"d": 1}}}})


def test_bearer_user_must_match_subject(monkeypatch):
    _install_gateway(monkeypatch, _gateway_config())
    request = _request()
    request.state.auth_claims = {"sub": "user-1"}

    gateway_api._enforce_bearer_user(request, "user-1")
    with pytest.raises(HTTPException) as excinfo:
        gateway_api._enforce_bearer_user(request, "user-2")
    assert excinfo.value.status_code == 403
    assert excinfo.value.detail["error"]["code"] == "USER_MISMATCH"


@pytest.mark.asyncio
async def test_content_status_hides_other_users_jobs(monkeypatch):
    _install_gateway(monkeypatch, _gateway_config())

    class Store:
        async def get_content_job(self, content_id):
            return {"content_id": content_id, "status": "completed", "user_id": "owner", "updated_at": 1.0}

    monkeypatch.setattr(gateway_api, "system_store", Store())
    request = _request(method="GET")
    request.state.auth_claims = {"sub": "intruder"}

    with pytest.raises(HTTPException) as excinfo:
        await gateway_api.content_status("c1", request)
    assert excinfo.value.status_code == 404

    request.state.auth_claims = {"sub": "owner"}
    assert (await gateway_api.content_status("c1", request))["status"] == "completed"


@pytest.mark.asyncio
async def test_analytics_requires_admin_scope_for_bearer_callers(monkeypatch):
    _install_gateway(monkeypatch, _gateway_config())

    class Store:
        async def get_analytics_summary(self):
            return {"ok": True}

    monkeypatch.setattr(gateway_api, "system_store", Store())
    request = _request(method="GET")
    request.state.auth_claims = {"sub": "u1", "scope": "openid profile"}

    with pytest.raises(HTTPException) as excinfo:
        await gateway_api.analytics(request)
    assert excinfo.value.status_code == 403

    request.state.auth_claims = {"sub": "u1", "scope": "openid video-commerce:admin"}
    assert await gateway_api.analytics(request) == {"ok": True}


def test_internal_service_auth_returns_401_instead_of_raising():
    runtime = ServiceRuntime("ranking-service")
    runtime.config = SimpleNamespace(
        security_config=SimpleNamespace(
            internal_service_key="k1", internal_service_header="X-Internal-Service-Key"
        ),
        monitoring_config=SimpleNamespace(request_id_header="X-Request-ID"),
    )

    denied = internal_service_auth_response(_request(path="/api/rank"), runtime)
    allowed = internal_service_auth_response(
        _request(path="/api/rank", headers={"X-Internal-Service-Key": "k1"}), runtime
    )

    assert denied.status_code == 401
    assert allowed is None


# --- Kafka reconnect ---------------------------------------------------------


@pytest.mark.asyncio
async def test_kafka_supervisor_retries_until_connected(monkeypatch):
    attempts = []
    manager = SimpleNamespace(producer=SimpleNamespace(on_unreachable=None))

    async def fake_init(_config, observability=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("broker unavailable")
        return manager

    monkeypatch.setattr(kafka_client, "init_kafka", fake_init)
    connected = []
    supervisor = kafka_client.KafkaProducerSupervisor(
        SimpleNamespace(enable=True),
        service_name="gateway-api",
        on_connected=connected.append,
        initial_backoff_seconds=0.01,
        max_backoff_seconds=0.02,
    )

    assert await supervisor.start() is None
    for _ in range(100):
        if connected:
            break
        await asyncio.sleep(0.01)

    assert connected == [manager]
    assert supervisor.connected
    assert len(attempts) == 3
    supervisor.manager = None
    await supervisor.stop()
