"""
Authentication helpers for gateway bearer token validation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from video_commerce.common.config import SecurityConfig


class AuthValidationError(Exception):
    """Raised when client authentication fails."""


class BearerTokenValidator:
    """Validate bearer tokens using JWKS or a shared JWT secret."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.enabled = config.oidc_enabled
        self.required = config.oidc_required or config.auth_mode == "bearer"
        self._jwt = None
        self._jwk_client = None

        if not self.enabled:
            return

        import jwt

        self._jwt = jwt
        if config.oidc_jwks_url:
            self._jwk_client = jwt.PyJWKClient(config.oidc_jwks_url)
        elif not config.jwt_shared_secret:
            raise ValueError(
                "OIDC/JWT auth requires SECURITY_OIDC_JWKS_URL or SECURITY_JWT_SHARED_SECRET"
            )

    def authenticate(self, authorization_header: Optional[str]) -> Optional[Dict[str, Any]]:
        """Validate a bearer token when bearer auth is enabled."""
        if not self.enabled:
            return None

        if not authorization_header:
            if self.required:
                raise AuthValidationError("Missing bearer token")
            return None

        scheme, _, token = authorization_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise AuthValidationError("Invalid bearer token")

        return self._decode(token)

    def _decode(self, token: str) -> Dict[str, Any]:
        assert self._jwt is not None

        decode_kwargs: Dict[str, Any] = {
            "algorithms": self.config.jwt_algorithms,
        }
        options = {}
        if self.config.oidc_audience:
            decode_kwargs["audience"] = self.config.oidc_audience
        else:
            options["verify_aud"] = False
        if self.config.oidc_issuer:
            decode_kwargs["issuer"] = self.config.oidc_issuer
        else:
            options["verify_iss"] = False
        if options:
            decode_kwargs["options"] = options

        try:
            if self.config.jwt_shared_secret:
                signing_key = self.config.jwt_shared_secret
            else:
                signing_key = self._jwk_client.get_signing_key_from_jwt(token).key
            return self._jwt.decode(token, signing_key, **decode_kwargs)
        except self._jwt.PyJWTError as exc:
            raise AuthValidationError("Invalid bearer token") from exc
