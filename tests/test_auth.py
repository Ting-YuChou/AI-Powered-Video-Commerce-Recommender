import time

import pytest

jwt = pytest.importorskip("jwt")

from video_commerce.common.auth import AuthValidationError, BearerTokenValidator
from video_commerce.common.config import SecurityConfig


def test_bearer_token_validator_accepts_hs256_token():
    config = SecurityConfig(
        oidc_enabled=True,
        oidc_required=True,
        jwt_shared_secret="super-secret",
        jwt_algorithms=["HS256"],
        oidc_audience="video-commerce",
        oidc_issuer="https://issuer.example.com",
    )
    token = jwt.encode(
        {
            "sub": "user-123",
            "aud": "video-commerce",
            "iss": "https://issuer.example.com",
            "exp": int(time.time()) + 300,
        },
        "super-secret",
        algorithm="HS256",
    )

    claims = BearerTokenValidator(config).authenticate(f"Bearer {token}")

    assert claims["sub"] == "user-123"


def test_bearer_token_validator_requires_header_when_configured():
    config = SecurityConfig(
        oidc_enabled=True,
        oidc_required=True,
        jwt_shared_secret="super-secret",
        jwt_algorithms=["HS256"],
    )

    with pytest.raises(AuthValidationError):
        BearerTokenValidator(config).authenticate(None)


class _FakeJwkClient:
    def __init__(self):
        self.refreshes = 0

    def get_signing_keys(self, refresh=False):
        if refresh:
            self.refreshes += 1
        return []

    def match_kid(self, _keys, _kid):
        return None


def test_unknown_key_ids_force_at_most_one_jwks_refresh_per_interval():
    config = SecurityConfig(
        oidc_enabled=True,
        oidc_jwks_url="https://idp.example.com/jwks",
        jwks_min_refresh_interval_seconds=30,
    )
    validator = BearerTokenValidator(config)
    fake = _FakeJwkClient()
    validator._jwk_client = fake

    for kid in ("k1", "k2", "k3"):
        token = jwt.encode(
            {"sub": "u", "exp": int(time.time()) + 60},
            "x",
            algorithm="HS256",
            headers={"kid": kid},
        )
        with pytest.raises(AuthValidationError):
            validator.authenticate(f"Bearer {token}")

    assert fake.refreshes == 1


def test_bearer_tokens_without_exp_are_rejected():
    config = SecurityConfig(
        oidc_enabled=True, jwt_shared_secret="s", jwt_algorithms=["HS256"]
    )
    token = jwt.encode({"sub": "u"}, "s", algorithm="HS256")

    with pytest.raises(AuthValidationError):
        BearerTokenValidator(config).authenticate(f"Bearer {token}")


@pytest.mark.asyncio
async def test_jwks_validation_runs_off_the_event_loop(monkeypatch):
    import threading

    config = SecurityConfig(oidc_enabled=True, oidc_jwks_url="https://idp.example.com/jwks")
    validator = BearerTokenValidator(config)
    seen = {}

    def fake_authenticate(header):
        seen["thread"] = threading.current_thread()
        return {"sub": "u"}

    monkeypatch.setattr(validator, "authenticate", fake_authenticate)

    assert await validator.authenticate_async("Bearer t") == {"sub": "u"}
    assert seen["thread"] is not threading.main_thread()
