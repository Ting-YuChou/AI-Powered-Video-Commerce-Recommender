"""Private Qwen3-ASR multipart transcription endpoint for content workers."""

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

ASR_MODEL = os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
ASR_ALIGNER_MODEL = os.getenv("ASR_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
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
        forced_aligner=ASR_ALIGNER_MODEL,
        forced_aligner_kwargs={"dtype": dtype, "device_map": ASR_DEVICE},
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
    return {
        "status": "healthy",
        "model": ASR_MODEL,
        "aligner_model": ASR_ALIGNER_MODEL,
    }


def _join_aligned_text(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    if left[-1].isascii() and right[0].isascii():
        if left[-1].isalpha() and right[0].isalpha():
            return f"{left} {right}"
    return left + right


def _merge_segment_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": _join_aligned_text(str(left["text"]), str(right["text"])),
        "start_seconds": float(left["start_seconds"]),
        "end_seconds": float(right["end_seconds"]),
    }


def group_alignment_items(
    items: Any,
    *,
    pause_seconds: float = 0.8,
    max_duration_seconds: float = 5.0,
    max_chars: int = 96,
    max_segments: int = 64,
) -> list[dict[str, Any]]:
    """Coalesce word/character alignment output into bounded temporal segments."""
    normalized: list[dict[str, Any]] = []
    for raw in items or []:
        text = str(getattr(raw, "text", "") or "").strip()
        start = max(0.0, float(getattr(raw, "start_time", 0.0) or 0.0))
        end = max(start, float(getattr(raw, "end_time", start) or start))
        if text:
            normalized.append(
                {"text": text, "start_seconds": start, "end_seconds": end}
            )
    normalized.sort(key=lambda item: (item["start_seconds"], item["end_seconds"]))

    segments: list[dict[str, Any]] = []
    for item in normalized:
        if not segments:
            segments.append(item)
            continue
        current = segments[-1]
        merged_text = _join_aligned_text(current["text"], item["text"])
        should_split = (
            item["start_seconds"] - current["end_seconds"] >= pause_seconds
            or item["end_seconds"] - current["start_seconds"] > max_duration_seconds
            or len(merged_text) > max_chars
        )
        if should_split:
            segments.append(item)
        else:
            segments[-1] = _merge_segment_pair(current, item)

    max_segments = max(1, int(max_segments))
    while len(segments) > max_segments:
        merge_index = min(
            range(len(segments) - 1),
            key=lambda index: (
                segments[index + 1]["end_seconds"] - segments[index]["start_seconds"]
            ),
        )
        segments[merge_index : merge_index + 2] = [
            _merge_segment_pair(segments[merge_index], segments[merge_index + 1])
        ]
    return segments


def build_transcription_payload(result: Any, *, model: str) -> dict[str, Any]:
    text = str(getattr(result, "text", "") or "")
    language = str(getattr(result, "language", "") or "")
    alignment = getattr(result, "time_stamps", None)
    segments = group_alignment_items(getattr(alignment, "items", alignment))
    if not text:
        alignment_status = "no_speech"
    elif segments:
        alignment_status = "completed"
    else:
        alignment_status = "degraded"
    return {
        "text": text,
        "language": language,
        "model": model,
        "alignment_status": alignment_status,
        "segments": segments,
    }


async def _run_transcription(audio_path: Path):
    semaphore = getattr(app.state, "inference_semaphore", None)
    if semaphore is None:
        semaphore = asyncio.Semaphore(ASR_MAX_CONCURRENT_TRANSCRIPTIONS)
        app.state.inference_semaphore = semaphore
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=ASR_QUEUE_WAIT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=503, detail="ASR inference capacity exhausted"
        ) from exc
    try:

        def transcribe_with_timestamps():
            try:
                return app.state.model.transcribe(
                    audio=str(audio_path),
                    language=None,
                    return_time_stamps=True,
                )
            except TypeError as exc:
                if "return_time_stamps" not in str(exc):
                    raise
                # Preserve compatibility with older internal/fake ASR clients
                # while production Qwen ASR receives timestamp alignment.
                return app.state.model.transcribe(
                    audio=str(audio_path),
                    language=None,
                )

        inference_task = asyncio.create_task(
            asyncio.to_thread(transcribe_with_timestamps)
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
) -> dict[str, Any]:
    if model != ASR_MODEL:
        raise HTTPException(
            status_code=400, detail="Requested ASR model is unavailable"
        )

    temp_path: Path | None = None
    bytes_received = 0
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            while chunk := await file.read(1024 * 1024):
                bytes_received += len(chunk)
                if bytes_received > ASR_MAX_AUDIO_BYTES:
                    raise HTTPException(
                        status_code=413, detail="Audio upload exceeds ASR limit"
                    )
                temp_file.write(chunk)

        results = await _run_transcription(temp_path)
        if not results:
            return build_transcription_payload(
                SimpleNamespace(text="", language="", time_stamps=None),
                model=ASR_MODEL,
            )
        return build_transcription_payload(results[0], model=ASR_MODEL)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
