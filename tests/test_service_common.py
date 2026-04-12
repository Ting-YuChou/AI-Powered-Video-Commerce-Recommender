from fastapi import FastAPI
from fastapi.testclient import TestClient

from service_common import ServiceRuntime, build_error_response, build_readiness_response, create_service_app


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
