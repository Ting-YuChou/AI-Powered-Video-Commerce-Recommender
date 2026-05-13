import pytest
from types import SimpleNamespace

from cache_codec import json_loads
from ranking_coordinator_client import RankingCoordinatorTimeout
import ranking_proxy_asgi
from ranking_proxy_asgi import RankingProxyApp


class TimeoutCoordinatorClient:
    async def rank(self, body):
        raise RankingCoordinatorTimeout("ranking coordinator request timeout")


def test_proxy_body_with_deadline_preserves_earlier_deadline_and_fills_invalid(
    monkeypatch,
):
    app = RankingProxyApp()
    app.runtime.config = SimpleNamespace(
        service_topology_config=SimpleNamespace(
            ranking_coordinator_request_timeout_seconds=1.0
        )
    )
    monkeypatch.setattr(ranking_proxy_asgi.time, "time", lambda: 100.0)

    earlier = app._body_with_deadline(b'{"deadline_unix_seconds":100.2}')
    later = app._body_with_deadline(b'{"deadline_unix_seconds":200.0}')
    missing = app._body_with_deadline(b"{}")
    invalid = app._body_with_deadline(b'{"deadline_unix_seconds":"bad"}')

    assert json_loads(earlier)["deadline_unix_seconds"] == pytest.approx(100.2)
    assert json_loads(later)["deadline_unix_seconds"] == pytest.approx(100.9)
    assert json_loads(missing)["deadline_unix_seconds"] == pytest.approx(100.9)
    assert json_loads(invalid)["deadline_unix_seconds"] == pytest.approx(100.9)


@pytest.mark.asyncio
async def test_ranking_proxy_returns_clean_503_on_coordinator_timeout():
    app = RankingProxyApp()
    app.client = TimeoutCoordinatorClient()
    messages = []
    received = [{"type": "http.request", "body": b'{"request_id":"r1"}'}]

    async def receive():
        return received.pop(0)

    async def send(message):
        messages.append(message)

    await app._handle_http(
        {
            "type": "http",
            "method": "POST",
            "path": "/internal/rank",
            "headers": [],
        },
        receive,
        send,
    )

    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 503
    assert json_loads(messages[1]["body"]) == {"detail": "ranking_coordinator_timeout"}
    metrics = app.runtime.observability.prometheus_payload().decode("utf-8")
    assert (
        'video_commerce_ranking_coordinator_client_errors_total{reason="timeout"}'
        in metrics
    )
