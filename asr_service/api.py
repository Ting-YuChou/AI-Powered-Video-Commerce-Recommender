"""Private Qwen3-ASR multipart transcription endpoint for content workers."""

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

ASR_MODEL = os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda:0")
ASR_MAX_AUDIO_BYTES = int(os.getenv("ASR_MAX_AUDIO_BYTES", str(16 * 1024 * 1024)))
ASR_MAX_INFERENCE_BATCH_SIZE = int(os.getenv("ASR_MAX_INFERENCE_BATCH_SIZE", "1"))
ASR_MAX_NEW_TOKENS = int(os.getenv("ASR_MAX_NEW_TOKENS", "2048"))
ASR_MAX_CONCURRENT_TRANSCRIPTIONS = max(
    1, int(os.getenv("ASR_MAX_CONCURRENT_TRANSCRIPTIONS", "1"))
)
ASR_QUEUE_WAIT_SECONDS = max(0.1, float(os.getenv("ASR_QUEUE_WAIT_SECONDS", "5")))


def _load_model() -> Any:
    import torch
    from qwen_asr import Qwen3ASRModel

    dtype = torch.bfloat16 if ASR_DEVICE.startswith("cuda") else torch.float32
    return Qwen3ASRModel.from_pretrained(
        ASR_MODEL,
        dtype=dtype,
        device_map=ASR_DEVICE,
        max_inference_batch_size=ASR_MAX_INFERENCE_BATCH_SIZE,
        max_new_tokens=ASR_MAX_NEW_TOKENS,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = await asyncio.to_thread(_load_model)
    app.state.inference_semaphore = asyncio.Semaphore(ASR_MAX_CONCURRENT_TRANSCRIPTIONS)
    yield


app = FastAPI(
    title="Video Commerce Internal ASR",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "model": ASR_MODEL}


async def _run_transcription(audio_path: Path):
    semaphore = getattr(app.state, "inference_semaphore", None)
    if semaphore is None:
        semaphore = asyncio.Semaphore(ASR_MAX_CONCURRENT_TRANSCRIPTIONS)
        app.state.inference_semaphore = semaphore
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=ASR_QUEUE_WAIT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=503, detail="ASR inference capacity exhausted") from exc
    try:
        inference_task = asyncio.create_task(
            asyncio.to_thread(
                app.state.model.transcribe,
                audio=str(audio_path),
                language=None,
            )
        )
        try:
            return await asyncio.shield(inference_task)
        except asyncio.CancelledError:
            # Keep the slot and staged audio until uncancellable thread work ends.
            try:
                await inference_task
            finally:
                raise
    finally:
        semaphore.release()


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=ASR_MODEL),
) -> dict[str, str]:
    if model != ASR_MODEL:
        raise HTTPException(status_code=400, detail="Requested ASR model is unavailable")

    temp_path: Path | None = None
    bytes_received = 0
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            while chunk := await file.read(1024 * 1024):
                bytes_received += len(chunk)
                if bytes_received > ASR_MAX_AUDIO_BYTES:
                    raise HTTPException(status_code=413, detail="Audio upload exceeds ASR limit")
                temp_file.write(chunk)

        results = await _run_transcription(temp_path)
        if not results:
            return {"text": "", "language": "", "model": ASR_MODEL}
        result = results[0]
        return {
            "text": str(getattr(result, "text", "") or ""),
            "language": str(getattr(result, "language", "") or ""),
            "model": ASR_MODEL,
        }
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
