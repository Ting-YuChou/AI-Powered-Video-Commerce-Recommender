import io
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from video_commerce.common.config import Config
from video_commerce.services.gateway import api as gateway_api
from video_commerce.services.gateway.api import (
    _proxy_json_request,
    stream_upload_to_temp_file,
    validate_upload_file,
)


def test_validate_upload_file_rejects_bad_extension():
    config = Config()
    file = UploadFile(
        file=io.BytesIO(b"abc"),
        filename="video.exe",
        headers=Headers({"content-type": "video/mp4"}),
    )

    with pytest.raises(HTTPException) as exc:
        validate_upload_file(file, config)

    assert exc.value.status_code == 400


def test_validate_upload_file_rejects_bad_mime():
    config = Config()
    file = UploadFile(
        file=io.BytesIO(b"abc"),
        filename="video.mp4",
        headers=Headers({"content-type": "application/octet-stream"}),
    )

    with pytest.raises(HTTPException) as exc:
        validate_upload_file(file, config)

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_stream_upload_to_temp_file_enforces_size_limit(tmp_path):
    file = UploadFile(
        file=io.BytesIO(b"a" * 20),
        filename="video.mp4",
        headers=Headers({"content-type": "video/mp4"}),
    )

    with pytest.raises(HTTPException) as exc:
        await stream_upload_to_temp_file(
            file=file,
            upload_dir=str(tmp_path),
            suffix=".mp4",
            max_size=10,
            chunk_size=4,
        )

    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_stream_upload_to_temp_file_streams_successfully(tmp_path):
    content = b"video-bytes"
    file = UploadFile(
        file=io.BytesIO(content),
        filename="video.mp4",
        headers=Headers({"content-type": "video/mp4"}),
    )

    path, size_bytes = await stream_upload_to_temp_file(
        file=file,
        upload_dir=str(tmp_path),
        suffix=".mp4",
        max_size=1024,
        chunk_size=4,
    )

    assert size_bytes == len(content)
    with open(path, "rb") as handle:
        assert handle.read() == content


@pytest.mark.asyncio
async def test_gateway_proxy_preserves_retry_after(monkeypatch):
    class FakePool:
        async def post(self, path, content, headers):
            return httpx.Response(
                429,
                content=b'{"detail":"ranking_overloaded","retry_after_seconds":1}',
                headers={"content-type": "application/json", "retry-after": "1"},
            )

    class FakeRequest:
        state = SimpleNamespace(request_id="req-1")

        async def body(self):
            return b"{}"

    monkeypatch.setattr(
        gateway_api.app.state.runtime,
        "config",
        SimpleNamespace(
            monitoring_config=SimpleNamespace(request_id_header="X-Request-ID"),
            security_config=SimpleNamespace(
                internal_service_header="X-Internal-Service-Key",
                internal_service_key="",
            ),
        ),
    )

    response = await _proxy_json_request(
        proxy_pool=FakePool(),
        path="/api/recommendations",
        target="recommendation-service",
        payload={},
        request=FakeRequest(),
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"
