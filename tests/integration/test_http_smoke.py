import os

import httpx
import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get("INTEGRATION_BASE_URL"),
    reason="requires INTEGRATION_BASE_URL and a running integration stack",
)


_RANKING_DOWN = os.environ.get("INTEGRATION_EXPECT_RANKING_DOWN") == "1"


def _base_url() -> str:
    return os.environ.get("INTEGRATION_BASE_URL", "http://localhost:8000").rstrip("/")


def _edge_url() -> str | None:
    value = os.environ.get("EDGE_BASE_URL")
    return value.rstrip("/") if value else None


def _headers() -> dict:
    headers = {}
    api_key = os.environ.get("API_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def test_gateway_livez():
    response = httpx.get(f"{_base_url()}/livez", timeout=10.0, trust_env=False)

    assert response.status_code == 200
    assert response.json()["service"] == "gateway-api"


def test_gateway_readyz():
    response = httpx.get(f"{_base_url()}/readyz", timeout=10.0, trust_env=False)

    # Readiness covers only the gateway's own state, so it stays 200 while a
    # shared downstream is degraded or down.
    assert response.status_code == 200
    assert response.json()["status"] in {"ready", "degraded"}


def test_gateway_full_readiness_reports_downstreams():
    response = httpx.get(
        f"{_base_url()}/readyz/full", headers=_headers(), timeout=20.0, trust_env=False
    )

    checks = response.json()["checks"]
    assert {"recommendation_service", "ranking_service", "kafka"} <= set(checks)
    if _RANKING_DOWN:
        assert checks["ranking_service"]["status"] != "healthy"


@pytest.mark.skipif(not os.environ.get("EDGE_BASE_URL"), reason="requires EDGE_BASE_URL")
def test_edge_serves_spa_and_its_own_health():
    health = httpx.get(f"{_edge_url()}/caddy-healthz", timeout=10.0, trust_env=False)
    index = httpx.get(f"{_edge_url()}/", timeout=10.0, trust_env=False)

    assert health.status_code == 200
    assert index.status_code == 200
    assert "<html" in index.text.lower()


@pytest.mark.skipif(_RANKING_DOWN, reason="ranking coordinator is stopped")
def test_recommendations_round_trip():
    response = httpx.post(
        f"{_base_url()}/api/recommendations",
        json={"user_id": "integration-user", "k": 3, "context": {"source": "integration-test"}},
        headers=_headers(),
        timeout=20.0,
        trust_env=False,
    )

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    payload = response.json()
    assert payload["user_id"] == "integration-user"
    assert len(payload["recommendations"]) == 3


def test_interactions_round_trip():
    response = httpx.post(
        f"{_base_url()}/api/interactions",
        json={
            "user_id": "integration-user",
            "product_id": "prod_1",
            "action": "click",
            "context": {"source": "integration-test"},
        },
        headers=_headers(),
        timeout=20.0,
        trust_env=False,
    )

    assert response.status_code == 202
    assert "X-Request-ID" in response.headers
