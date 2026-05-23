import importlib.util
import importlib.machinery
import json
import sys
import types

import pytest
import numpy as np

if importlib.util.find_spec("pytesseract") is None:
    pytesseract_stub = types.ModuleType("pytesseract")
    pytesseract_stub.__spec__ = importlib.machinery.ModuleSpec("pytesseract", None)
    pytesseract_stub.Output = types.SimpleNamespace(DICT="dict")
    pytesseract_stub.image_to_data = lambda *args, **kwargs: {
        "conf": [],
        "text": [],
    }
    sys.modules["pytesseract"] = pytesseract_stub

from config import ModelConfig
from content_processor import ContentProcessor


def _processor(**overrides):
    config = ModelConfig(
        device="cpu",
        enable_gpu=False,
        num_keyframes=3,
        max_video_length=10,
        ffmpeg_timeout_seconds=1,
        ffmpeg_target_width=4,
        **overrides,
    )
    return ContentProcessor(config)


def _ppm_frame(width, height, values):
    return (
        f"P6\n{width} {height}\n255\n".encode("ascii")
        + np.array(values, dtype=np.uint8).reshape(height, width, 3).tobytes()
    )


def _ffprobe_payload(*, duration="12.5", fps="30000/1001", frames="300", audio=True):
    streams = [
        {
            "codec_type": "video",
            "duration": duration,
            "avg_frame_rate": fps,
            "nb_frames": frames,
            "width": 1920,
            "height": 1080,
        }
    ]
    if audio:
        streams.append({"codec_type": "audio"})
    return json.dumps(
        {
            "streams": streams,
            "format": {"duration": duration},
        }
    ).encode("utf-8")


@pytest.mark.asyncio
async def test_ffprobe_metadata_parses_duration_fps_frame_count_and_audio(monkeypatch):
    processor = _processor()

    async def fake_run(command):
        assert command[0] == "ffprobe"
        return _ffprobe_payload()

    monkeypatch.setattr(processor, "_run_media_command", fake_run)

    metadata = await processor._probe_video_metadata("video.mp4")

    assert metadata["duration"] == 12.5
    assert metadata["fps"] == pytest.approx(29.97002997)
    assert metadata["frame_count"] == 300
    assert metadata["has_audio"] is True
    assert metadata["width"] == 1920
    assert metadata["height"] == 1080


@pytest.mark.asyncio
async def test_ffmpeg_extraction_parses_rgb_frames_and_respects_keyframe_count(monkeypatch):
    processor = _processor()
    ffmpeg_commands = []
    frames = [
        _ppm_frame(2, 1, [[[255, 0, 0], [0, 255, 0]]]),
        _ppm_frame(2, 1, [[[0, 0, 255], [255, 255, 0]]]),
        _ppm_frame(2, 1, [[[255, 0, 255], [0, 255, 255]]]),
    ]

    async def fake_run(command):
        if command[0] == "ffprobe":
            return _ffprobe_payload(duration="2.0", fps="5/1", frames="10", audio=False)
        ffmpeg_commands.append(command)
        return b"".join(frames)

    monkeypatch.setattr(processor, "_run_media_command", fake_run)

    keyframes, video_info = await processor._extract_keyframes_ffmpeg("video.mp4")

    assert video_info["duration"] == 2.0
    assert video_info["fps"] == 5.0
    assert video_info["frame_count"] == 10
    assert video_info["has_audio"] is False
    assert len(keyframes) == 3
    assert keyframes[0].shape == (1, 2, 3)
    assert keyframes[0].tolist() == [[[255, 0, 0], [0, 255, 0]]]
    assert len(ffmpeg_commands) == 1
    command = ffmpeg_commands[0]
    assert command[0] == "ffmpeg"
    assert command[command.index("-frames:v") + 1] == "3"
    assert any("select=eq(n\\,0)+eq(n\\,4)+eq(n\\,9)" in arg for arg in command)


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("ffmpeg"),
        TimeoutError("ffmpeg timed out"),
        RuntimeError("ffmpeg failed"),
    ],
)
@pytest.mark.asyncio
async def test_ffmpeg_failure_falls_back_to_opencv(monkeypatch, error):
    processor = _processor()
    fallback_frame = np.zeros((1, 1, 3), dtype=np.uint8)

    async def fake_ffmpeg(_video_path):
        raise error

    async def fake_opencv(video_path):
        assert video_path == "video.mp4"
        return [fallback_frame], {"duration": 1.0, "has_audio": True}

    monkeypatch.setattr(processor, "_extract_keyframes_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(processor, "_extract_keyframes_opencv", fake_opencv)

    keyframes, video_info = await processor._extract_keyframes("video.mp4")

    assert keyframes == [fallback_frame]
    assert video_info["duration"] == 1.0


@pytest.mark.asyncio
async def test_ffmpeg_disabled_uses_opencv_without_calling_ffmpeg(monkeypatch):
    processor = _processor(ffmpeg_frame_extraction_enabled=False)
    fallback_frame = np.ones((1, 1, 3), dtype=np.uint8)

    async def fail_ffmpeg(_video_path):
        raise AssertionError("ffmpeg should not be called")

    async def fake_opencv(video_path):
        assert video_path == "video.mp4"
        return [fallback_frame], {"duration": 2.0}

    monkeypatch.setattr(processor, "_extract_keyframes_ffmpeg", fail_ffmpeg)
    monkeypatch.setattr(processor, "_extract_keyframes_opencv", fake_opencv)

    keyframes, video_info = await processor._extract_keyframes("video.mp4")

    assert keyframes == [fallback_frame]
    assert video_info["duration"] == 2.0


@pytest.mark.asyncio
async def test_ffmpeg_empty_frames_falls_back_to_opencv(monkeypatch):
    processor = _processor()
    fallback_frame = np.full((1, 1, 3), 7, dtype=np.uint8)

    async def empty_ffmpeg(_video_path):
        return [], {"duration": 3.0}

    async def fake_opencv(video_path):
        assert video_path == "video.mp4"
        return [fallback_frame], {"duration": 3.0}

    monkeypatch.setattr(processor, "_extract_keyframes_ffmpeg", empty_ffmpeg)
    monkeypatch.setattr(processor, "_extract_keyframes_opencv", fake_opencv)

    keyframes, video_info = await processor._extract_keyframes("video.mp4")

    assert keyframes == [fallback_frame]
    assert video_info["duration"] == 3.0
