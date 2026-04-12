import io

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from config import Config
from gateway_api import stream_upload_to_temp_file, validate_upload_file


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
