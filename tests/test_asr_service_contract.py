import io
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile

from asr_service import api


class FakeAsrModel:
    def __init__(self):
        self.paths = []

    def transcribe(self, audio, language=None):
        assert language is None
        assert Path(audio).exists()
        self.paths.append(Path(audio))
        return [SimpleNamespace(text="手機 headphones", language="Chinese")]


@pytest.mark.asyncio
async def test_multipart_transcription_contract_cleans_staged_audio():
    fake_model = FakeAsrModel()
    api.app.state.model = fake_model
    upload = UploadFile(file=io.BytesIO(b"wav-content"), filename="audio.wav")

    result = await api.transcribe(file=upload, model=api.ASR_MODEL)

    assert result == {
        "text": "手機 headphones",
        "language": "Chinese",
        "model": api.ASR_MODEL,
        "alignment_status": "degraded",
        "segments": [],
    }
    assert fake_model.paths
    assert not fake_model.paths[0].exists()


@pytest.mark.asyncio
async def test_transcription_contract_rejects_unserved_model():
    upload = UploadFile(file=io.BytesIO(b"wav-content"), filename="audio.wav")

    with pytest.raises(HTTPException) as exc_info:
        await api.transcribe(file=upload, model="different-model")

    assert exc_info.value.status_code == 400


class BlockingAsrModel:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()

    def transcribe(self, audio, language=None):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.entered.set()
        self.release.wait(timeout=1)
        with self.lock:
            self.active -= 1
        return [SimpleNamespace(text="phone", language="English")]


@pytest.mark.asyncio
async def test_transcription_serializes_model_inference():
    fake_model = BlockingAsrModel()
    api.app.state.model = fake_model
    api.app.state.inference_semaphore = asyncio.Semaphore(1)

    first = asyncio.create_task(
        api.transcribe(
            file=UploadFile(file=io.BytesIO(b"first"), filename="first.wav"),
            model=api.ASR_MODEL,
        )
    )
    await asyncio.to_thread(fake_model.entered.wait, 0.5)
    second = asyncio.create_task(
        api.transcribe(
            file=UploadFile(file=io.BytesIO(b"second"), filename="second.wav"),
            model=api.ASR_MODEL,
        )
    )
    await asyncio.sleep(0.05)
    fake_model.release.set()
    await asyncio.gather(first, second)

    assert fake_model.max_active == 1


@pytest.mark.asyncio
async def test_cancelled_transcription_retains_inference_slot_until_thread_finishes():
    fake_model = BlockingAsrModel()
    api.app.state.model = fake_model
    api.app.state.inference_semaphore = asyncio.Semaphore(1)

    first = asyncio.create_task(
        api.transcribe(
            file=UploadFile(file=io.BytesIO(b"first"), filename="first.wav"),
            model=api.ASR_MODEL,
        )
    )
    await asyncio.to_thread(fake_model.entered.wait, 0.5)
    first.cancel()
    second = asyncio.create_task(
        api.transcribe(
            file=UploadFile(file=io.BytesIO(b"second"), filename="second.wav"),
            model=api.ASR_MODEL,
        )
    )
    await asyncio.sleep(0.05)

    assert fake_model.max_active == 1

    fake_model.release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
