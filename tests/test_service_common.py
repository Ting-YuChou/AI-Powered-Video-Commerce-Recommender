from fastapi import FastAPI
import httpx
import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace

from video_commerce.common.service_common import (
    RoundRobinAsyncClientPool,
    ServiceRuntime,
    _should_log_request,
    build_error_response,
    build_readiness_response,
    create_service_app,
)


def test_build_error_response_uses_standard_envelope():
    response = build_error_response(
        status_code=503,
        code="UPSTREAM_UNAVAILABLE",
        message="dependency unavailable",
        request_id="req-123",
    )

    assert response.status_code == 503
    assert response.body == (
        b'{"error":{"code":"UPSTREAM_UNAVAILABLE","message":"dependency unavailable","request_id":"req-123","retryable":true}}'
    )


def test_build_readiness_response_reflects_unhealthy_dependency():
    runtime = ServiceRuntime("gateway-api")

    ready = build_readiness_response(runtime, {"redis": {"status": "healthy"}})
    not_ready = build_readiness_response(runtime, {"redis": {"status": "unhealthy"}})

    assert ready.status_code == 200
    assert not_ready.status_code == 503


def test_create_service_app_sets_request_id_and_runtime_counters():
    app: FastAPI = create_service_app(
        title="Test App",
        description="test",
        service_name="test-service",
    )

    @app.get("/echo")
    async def echo():
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get("/echo")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert app.state.runtime.handled_requests == 1
    assert app.state.runtime.active_requests == 0


def test_request_logging_samples_completion_logs_by_status_class():
    runtime = ServiceRuntime("test-service")
    runtime.config = SimpleNamespace(
        monitoring_config=SimpleNamespace(
            enable_request_logging=True,
            request_log_sample_rate=0.0,
            error_request_log_sample_rate=1.0,
            slow_request_threshold=1.0,
        )
    )

    assert _should_log_request(runtime, 500, 0.01) is True
    assert _should_log_request(runtime, 200, 0.01) is False
    assert _should_log_request(runtime, 200, 1.5) is False

    runtime.config.monitoring_config.error_request_log_sample_rate = 0.0
    assert _should_log_request(runtime, 500, 0.01) is False

    runtime.config.monitoring_config.request_log_sample_rate = 1.0
    assert _should_log_request(runtime, 200, 0.01) is True
    assert _should_log_request(runtime, 200, 1.5) is True


def test_round_robin_pool_url_parsing_and_selection():
    urls = RoundRobinAsyncClientPool.parse_urls(
        "http://a:8000, http://b:8000",
        fallback="http://fallback:8000",
    )
    assert urls == ["http://a:8000", "http://b:8000"]
    assert RoundRobinAsyncClientPool.parse_urls("", fallback="http://fallback:8000") == [
        "http://fallback:8000"
    ]


@pytest.mark.asyncio
async def test_round_robin_pool_skips_temporarily_unhealthy_upstream():
    pool = RoundRobinAsyncClientPool(
        ["http://a:8000", "http://b:8000"],
        timeout=httpx.Timeout(1.0),
        limits=httpx.Limits(max_connections=2),
    )
    try:
        pool._mark_unhealthy(0)
        index, client = pool._choose_client()

        assert index == 1
        assert str(client.base_url).rstrip("/") == "http://b:8000"
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_round_robin_pool_ignores_environment_proxy_config(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "http://proxyproxy.orb.internal:8305")
    monkeypatch.setenv("HTTP_PROXY", "http://proxyproxy.orb.internal:8305")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxyproxy.orb.internal:8305")
    monkeypatch.setenv(
        "NO_PROXY",
        "localhost,127.0.0.1,::1,fd07:b51a:cc66:f0::/64,*.orb.internal",
    )

    pool = RoundRobinAsyncClientPool(
        ["http://a:8000", "http://b:8000"],
        timeout=httpx.Timeout(1.0),
        limits=httpx.Limits(max_connections=2),
    )
    try:
        index, client = pool._choose_client()

        assert index == 0
        assert str(client.base_url).rstrip("/") == "http://a:8000"
    finally:
        await pool.aclose()
