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
