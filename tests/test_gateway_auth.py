import json
from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

from video_commerce.services.gateway import api as gateway_api


def _request(path="/api/recommendations", headers=None):
    raw_headers = [
        (key.lower().encode(), value.encode())
        for key, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": raw_headers,
            "client": ("10.0.0.1", 1234),
            "server": ("gateway", 8000),
            "scheme": "http",
        }
    )


def _configure(monkeypatch, *, auth_mode, api_key=None):
    config = SimpleNamespace(
        monitoring_config=SimpleNamespace(request_id_header="X-Request-ID"),
        security_config=SimpleNamespace(auth_mode=auth_mode),
        api_config=SimpleNamespace(api_key=api_key),
    )
    monkeypatch.setattr(gateway_api.app.state, "runtime", SimpleNamespace(config=config))
    monkeypatch.setattr(gateway_api, "bearer_token_validator", None)
    monkeypatch.setattr(gateway_api, "rate_limiter", None)


async def _call(request):
    async def call_next(_request):
        return JSONResponse({"ok": True})

    return await gateway_api.auth_and_rate_limit(request, call_next)


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_mode", ["api_key", "api_key_or_bearer"])
async def test_gateway_fails_closed_when_no_credential_is_configured(
    monkeypatch, auth_mode
):
    _configure(monkeypatch, auth_mode=auth_mode, api_key=None)

    response = await _call(_request())

    assert response.status_code == 503
    assert json.loads(response.body)["error"]["code"] == "AUTH_NOT_CONFIGURED"


@pytest.mark.asyncio
async def test_gateway_disabled_mode_allows_requests_without_key(monkeypatch):
    _configure(monkeypatch, auth_mode="disabled", api_key=None)

    response = await _call(_request())

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_gateway_api_key_mode_checks_the_configured_key(monkeypatch):
    _configure(monkeypatch, auth_mode="api_key", api_key="secret")

    assert (await _call(_request())).status_code == 401
    assert (await _call(_request(headers={"x-api-key": "wrong"}))).status_code == 401
    assert (await _call(_request(headers={"x-api-key": "secret"}))).status_code == 200


@pytest.mark.asyncio
async def test_gateway_exempt_paths_skip_auth(monkeypatch):
    _configure(monkeypatch, auth_mode="api_key", api_key=None)

    assert (await _call(_request(path="/health"))).status_code == 200


def test_compose_edge_binds_loopback_and_dev_auth_is_explicit():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load((root / "docker-compose.yml").read_text())
    ports = compose["services"]["caddy"]["ports"]
    assert ports == [
        "${EDGE_BIND_ADDRESS:-127.0.0.1}:80:80",
        "${EDGE_BIND_ADDRESS:-127.0.0.1}:443:443",
    ]
    env_example = (root / ".env.example").read_text()
    assert "\nSECURITY_AUTH_MODE=disabled\n" in env_example
