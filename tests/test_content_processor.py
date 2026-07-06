import importlib.util
import importlib.machinery
import json
import shutil
import subprocess
import sys
import types

import httpx
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

from video_commerce.common.config import ModelConfig
from video_commerce.ml.content_processor import ContentProcessor


def _processor(**overrides):
    defaults = {
        "device": "cpu",
        "enable_gpu": False,
        "num_keyframes": 3,
        "max_video_length": 10,
        "ffmpeg_timeout_seconds": 1,
        "ffmpeg_target_width": 4,
    }
    defaults.update(overrides)
    config = ModelConfig(**defaults)
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
    processor = _processor(keyframe_sampling_strategy="uniform")
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


@pytest.mark.asyncio
async def test_scene_adaptive_extraction_merges_scene_and_floor_candidates(monkeypatch):
    processor = _processor(
        num_keyframes=4,
        keyframe_floor_seconds=4.0,
        keyframe_min_scene_gap_seconds=2.0,
        keyframe_min_luma=0.0,
        keyframe_min_blur_laplacian_var=0.0,
        keyframe_dedupe_luma_diff_threshold=0.0,
    )
    ffmpeg_commands = []
    frames = [
        _ppm_frame(2, 1, [[[255, 0, 0], [0, 255, 0]]]),
        _ppm_frame(2, 1, [[[0, 0, 255], [255, 255, 0]]]),
        _ppm_frame(2, 1, [[[255, 0, 255], [0, 255, 255]]]),
        _ppm_frame(2, 1, [[[255, 255, 255], [128, 128, 128]]]),
    ]

    async def fake_scene_run(command):
        assert command[0] == "ffmpeg"
        assert any("select=gt(scene\\,0.350000),showinfo" in arg for arg in command)
        return (
            b"",
            b"n:0 pts_time:1.0\nn:1 pts_time:1.5\nn:2 pts_time:6.2\n",
        )

    async def fake_run(command):
        if command[0] == "ffprobe":
            return _ffprobe_payload(duration="10.0", fps="10/1", frames="100")
        ffmpeg_commands.append(command)
        return b"".join(frames)

    monkeypatch.setattr(processor, "_run_media_command_with_stderr", fake_scene_run)
    monkeypatch.setattr(processor, "_run_media_command", fake_run)

    keyframes, video_info = await processor._extract_keyframes_ffmpeg("video.mp4")

    assert video_info["duration"] == 10.0
    assert len(keyframes) == 4
    command = ffmpeg_commands[0]
    assert any(
        "select=eq(n\\,0)+eq(n\\,10)+eq(n\\,62)+eq(n\\,99)" in arg
        for arg in command
    )


@pytest.mark.asyncio
async def test_scene_adaptive_extraction_uses_floor_when_no_scene_changes(monkeypatch):
    processor = _processor(
        num_keyframes=3,
        max_video_length=20,
        keyframe_floor_seconds=8.0,
        keyframe_min_luma=0.0,
        keyframe_min_blur_laplacian_var=0.0,
        keyframe_dedupe_luma_diff_threshold=0.0,
    )
    ffmpeg_commands = []
    frames = [
        _ppm_frame(1, 1, [[[255, 0, 0]]]),
        _ppm_frame(1, 1, [[[0, 255, 0]]]),
        _ppm_frame(1, 1, [[[0, 0, 255]]]),
    ]

    async def fake_scene_run(command):
        assert command[0] == "ffmpeg"
        return b"", b""

    async def fake_run(command):
        if command[0] == "ffprobe":
            return _ffprobe_payload(duration="16.0", fps="1/1", frames="16")
        ffmpeg_commands.append(command)
        return b"".join(frames)

    monkeypatch.setattr(processor, "_run_media_command_with_stderr", fake_scene_run)
    monkeypatch.setattr(processor, "_run_media_command", fake_run)

    keyframes, _ = await processor._extract_keyframes_ffmpeg("video.mp4")

    assert len(keyframes) == 3
    command = ffmpeg_commands[0]
    assert any("select=eq(n\\,0)+eq(n\\,8)+eq(n\\,15)" in arg for arg in command)


def _rgb_from_luma(plane):
    return np.repeat(plane[:, :, None], 3, axis=2).astype(np.uint8)


def test_quality_filter_drops_dark_blurry_and_duplicate_frames():
    processor = _processor(
        keyframe_min_luma=8.0,
        keyframe_min_blur_laplacian_var=10.0,
        keyframe_dedupe_luma_diff_threshold=4.0,
    )
    checker = ((np.indices((16, 16)).sum(axis=0) % 2) * 255).astype(np.uint8)
    duplicate = np.clip(checker.astype(np.int16) + 2, 0, 255).astype(np.uint8)
    inverse = (255 - checker).astype(np.uint8)

    frames = [
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.full((16, 16, 3), 128, dtype=np.uint8),
        _rgb_from_luma(checker),
        _rgb_from_luma(duplicate),
        _rgb_from_luma(inverse),
    ]

    filtered = processor._filter_keyframes_for_quality(frames)

    assert len(filtered) == 2
    np.testing.assert_array_equal(filtered[0], _rgb_from_luma(checker))
    np.testing.assert_array_equal(filtered[1], _rgb_from_luma(inverse))


@pytest.mark.asyncio
async def test_scene_adaptive_zero_usable_frames_falls_back_to_uniform(monkeypatch):
    processor = _processor(
        num_keyframes=2,
        keyframe_min_luma=300.0,
        keyframe_min_blur_laplacian_var=0.0,
        keyframe_dedupe_luma_diff_threshold=0.0,
    )
    ffmpeg_commands = []
    dark_frames = [
        _ppm_frame(1, 1, [[[0, 0, 0]]]),
        _ppm_frame(1, 1, [[[0, 0, 0]]]),
    ]
    fallback_frames = [
        _ppm_frame(1, 1, [[[255, 0, 0]]]),
        _ppm_frame(1, 1, [[[0, 255, 0]]]),
    ]

    async def fake_scene_run(command):
        assert command[0] == "ffmpeg"
        return b"", b"n:0 pts_time:1.0\n"

    async def fake_run(command):
        if command[0] == "ffprobe":
            return _ffprobe_payload(duration="2.0", fps="1/1", frames="2")
        ffmpeg_commands.append(command)
        if len(ffmpeg_commands) == 1:
            return b"".join(dark_frames)
        return b"".join(fallback_frames)

    monkeypatch.setattr(processor, "_run_media_command_with_stderr", fake_scene_run)
    monkeypatch.setattr(processor, "_run_media_command", fake_run)

    keyframes, _ = await processor._extract_keyframes_ffmpeg("video.mp4")

    assert len(keyframes) == 2
    assert len(ffmpeg_commands) == 2
    assert keyframes[0].tolist() == [[[255, 0, 0]]]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
@pytest.mark.asyncio
async def test_scene_adaptive_real_ffmpeg_smoke(tmp_path):
    video_path = tmp_path / "scene-smoke.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x64:d=1:r=5",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=64x64:d=1:r=5",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:d=1:r=5",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
            "-map",
            "[v]",
            "-c:v",
            "mpeg4",
            str(video_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    processor = _processor(
        num_keyframes=4,
        max_video_length=4,
        ffmpeg_timeout_seconds=10,
        ffmpeg_target_width=16,
        keyframe_scene_threshold=0.1,
        keyframe_floor_seconds=1.0,
        keyframe_min_scene_gap_seconds=0.5,
        keyframe_min_luma=0.0,
        keyframe_min_blur_laplacian_var=0.0,
        keyframe_dedupe_luma_diff_threshold=1.0,
    )

    keyframes, _ = await processor._extract_keyframes_ffmpeg(str(video_path))

    assert 1 <= len(keyframes) <= 4
    mean_colors = {
        tuple(np.round(np.mean(frame, axis=(0, 1))).astype(int))
        for frame in keyframes
    }
    assert len(mean_colors) > 1


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


@pytest.mark.asyncio
async def test_speech_to_text_success_retains_transcript_and_model(monkeypatch, tmp_path):
    processor = _processor(
        speech_to_text_enabled=True,
        speech_to_text_model="Qwen/Qwen3-ASR-0.6B",
    )
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"wav")

    async def fake_extract_audio(_video_path, _video_info):
        return str(audio_file)

    async def fake_request(_audio_path):
        return "新品手機 headphones 特價"

    monkeypatch.setattr(processor, "_extract_audio_wav", fake_extract_audio)
    monkeypatch.setattr(processor, "_request_transcription", fake_request)

    features = await processor._extract_audio_features(
        "video.mp4", {"has_audio": True, "duration": 4.0}
    )

    assert features.transcription_status == "completed"
    assert features.audio_transcript == "新品手機 headphones 特價"
    assert features.speech_detected is True
    assert features.asr_model == "Qwen/Qwen3-ASR-0.6B"
    assert features.transcription_time_seconds is not None
    assert not audio_file.exists()


@pytest.mark.asyncio
async def test_speech_to_text_skips_video_without_audio(monkeypatch):
    processor = _processor(speech_to_text_enabled=True)

    async def fail_extract_audio(_video_path, _video_info):
        raise AssertionError("audio extraction should not run")

    monkeypatch.setattr(processor, "_extract_audio_wav", fail_extract_audio)

    features = await processor._extract_audio_features(
        "video.mp4", {"has_audio": False, "duration": 4.0}
    )

    assert features.transcription_status == "skipped_no_audio"
    assert features.audio_transcript is None


@pytest.mark.asyncio
async def test_speech_to_text_disabled_does_not_extract_audio(monkeypatch):
    processor = _processor(speech_to_text_enabled=False)

    async def fail_extract_audio(_video_path, _video_info):
        raise AssertionError("audio extraction should not run when STT is disabled")

    monkeypatch.setattr(processor, "_extract_audio_wav", fail_extract_audio)

    features = await processor._extract_audio_features(
        "video.mp4", {"has_audio": True, "duration": 4.0}
    )

    assert features.transcription_status == "disabled"
    assert features.audio_transcript is None


@pytest.mark.asyncio
async def test_speech_to_text_empty_result_records_no_speech(monkeypatch, tmp_path):
    processor = _processor(speech_to_text_enabled=True)
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"wav")

    async def fake_extract_audio(_video_path, _video_info):
        return str(audio_file)

    async def empty_request(_audio_path):
        return "  "

    monkeypatch.setattr(processor, "_extract_audio_wav", fake_extract_audio)
    monkeypatch.setattr(processor, "_request_transcription", empty_request)

    features = await processor._extract_audio_features(
        "video.mp4", {"has_audio": True, "duration": 4.0}
    )

    assert features.transcription_status == "no_speech"
    assert features.audio_transcript is None
    assert features.speech_detected is False
    assert not audio_file.exists()


@pytest.mark.parametrize(
    "asr_error",
    [TimeoutError("asr timeout"), RuntimeError("asr HTTP failure")],
)
@pytest.mark.asyncio
async def test_speech_to_text_failure_degrades_without_raising(
    monkeypatch, tmp_path, asr_error
):
    processor = _processor(speech_to_text_enabled=True)
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"wav")

    async def fake_extract_audio(_video_path, _video_info):
        return str(audio_file)

    async def fail_request(_audio_path):
        raise asr_error

    monkeypatch.setattr(processor, "_extract_audio_wav", fake_extract_audio)
    monkeypatch.setattr(processor, "_request_transcription", fail_request)

    features = await processor._extract_audio_features(
        "video.mp4", {"has_audio": True, "duration": 4.0}
    )

    assert features.transcription_status == "degraded"
    assert features.audio_transcript is None
    assert features.speech_detected is False
    assert not audio_file.exists()


@pytest.mark.asyncio
async def test_speech_to_text_read_timeout_is_not_retried(monkeypatch, tmp_path):
    processor = _processor(speech_to_text_enabled=True, speech_to_text_max_attempts=2)
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"wav")
    calls = 0

    class TimeoutClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

        async def post(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("inference is still running")

    monkeypatch.setattr(
        "video_commerce.ml.content_processor.httpx.AsyncClient",
        lambda **_kwargs: TimeoutClient(),
    )

    with pytest.raises(RuntimeError):
        await processor._request_transcription(str(audio_file))

    assert calls == 1


def test_bilingual_speech_and_ocr_map_to_canonical_categories():
    processor = _processor()

    commerce = processor._analyze_commerce_content(
        {"text_blocks": ["skincare"], "price_mentions": []},
        [],
        "這款手機搭配 headphones 現在有折扣",
    )

    assert commerce["speech_categories"] == ["electronics"]
    assert commerce["ocr_categories"] == ["beauty"]
    assert commerce["category_scores"]["electronics"] > commerce["category_scores"]["beauty"]
