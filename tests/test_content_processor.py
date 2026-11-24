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
import torch

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
from video_commerce.ml.visual_retrieval import VisualRetrievalOutput


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


def test_retrieval_pooling_upgrades_content_to_v3_without_reencoding_video():
    processor = _processor(
        retrieval_visual_model_version="retrieval-7",
        retrieval_visual_product_index_version="products-42",
    )

    class Pooler:
        def __call__(self, frames, starts, mask):
            assert frames.shape == (1, 2, 512)
            return VisualRetrievalOutput(
                embedding=torch.nn.functional.normalize(
                    torch.tensor([[0.0, 1.0] + [0.0] * 510]), dim=-1
                ),
                attention_weights=torch.tensor([[0.25, 0.75]]),
            )

    processor.retrieval_pooler = Pooler()
    fields = processor._retrieval_visual_fields(
        [[1.0] * 512, [0.5] * 512],
        [0.0, 3.0],
    )

    assert fields["multimodal_schema_version"] == "temporal_multimodal_v3"
    assert fields["retrieval_visual_embedding"][1] == pytest.approx(1.0)
    assert fields["retrieval_model_version"] == "retrieval-7"
    assert fields["retrieval_product_index_version"] == "products-42"


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


def _showinfo(*pts_times):
    return "".join(
        f"[Parsed_showinfo_1 @ 0x1] n:{n} pts:{n} pts_time:{t} duration:1\n"
        for n, t in enumerate(pts_times)
    ).encode()


def _scene_processor(**overrides):
    defaults = dict(
        keyframe_min_luma=0.0,
        keyframe_min_blur_laplacian_var=0.0,
        keyframe_dedupe_luma_diff_threshold=0.0,
    )
    defaults.update(overrides)
    return _processor(**defaults)


@pytest.mark.asyncio
async def test_scene_adaptive_extraction_is_one_ffmpeg_pass_with_pts_timestamps(
    monkeypatch,
):
    processor = _scene_processor(num_keyframes=4, ffmpeg_timeout_per_media_second=2.0)
    commands = []
    frames = [
        _ppm_frame(2, 1, [[[value, 0, 0], [0, value, 0]]])
        for value in (20, 60, 100, 140, 180, 220)
    ]

    async def fake_run(command, *, timeout=None):
        assert command[0] == "ffmpeg"
        commands.append((command, timeout))
        # A non-zero stream start and VFR spacing: timestamps must come from
        # the frames' own pts, relative to the first frame.
        return b"".join(frames), _showinfo(0.5, 1.5, 4.0, 6.7, 8.5, 9.9)

    async def fake_probe(command, **_kwargs):
        return _ffprobe_payload(duration="10.0", fps="10/1", frames="100")

    monkeypatch.setattr(processor, "_run_media_command_with_stderr", fake_run)
    monkeypatch.setattr(processor, "_run_media_command", fake_probe)

    keyframes, video_info = await processor._extract_keyframes_ffmpeg("video.mp4")

    assert len(commands) == 1
    command, timeout = commands[0]
    assert command[command.index("-t") + 1] == "10.000"
    # Input option: as an output option FFmpeg trims after showinfo.
    assert command.index("-t") < command.index("-i")
    graph = command[command.index("-vf") + 1]
    assert graph.startswith(
        "select=" + processor._scene_candidate_select_expression(10.0) + ",showinfo,"
    )
    assert command[command.index("-threads") + 1] == "1"
    # base 1s + 10s of video x 2s per media second
    assert timeout == pytest.approx(21.0)
    assert len(keyframes) == 4
    # Six candidates thinned evenly to four.
    assert video_info["frame_timestamps_seconds"] == pytest.approx([0.0, 1.0, 6.2, 9.4])


def test_scene_candidate_expression_bounds_candidates_for_long_videos():
    processor = _processor(
        num_keyframes=16,
        max_video_length=300,
        keyframe_floor_seconds=8.0,
        keyframe_min_scene_gap_seconds=1.0,
        keyframe_scene_threshold=0.35,
    )

    short = processor._scene_candidate_select_expression(15.0)
    long = processor._scene_candidate_select_expression(300.0)

    # Short videos keep the configured gap and floor; one frame is also taken
    # in the last second, as the old method always included the last frame.
    assert short == (
        "gt(isnan(prev_selected_t)+gte(t-prev_selected_t\\,8.000000)"
        "+gt(scene\\,0.350000)*gte(t-prev_selected_t\\,1.000000)"
        "+gte(t\\,14.000000)*gte(t-prev_selected_t\\,1.000000)\\,0)"
    )
    # 300 s / (3 x 16) candidates: every selection is at least 6.25 s after
    # the previous one, so at most ~50 frames are decoded into memory.
    assert "gt(scene\\,0.350000)*gte(t-prev_selected_t\\,6.250000)" in long
    assert "+gte(t-prev_selected_t\\,8.000000)" in long
    assert "gte(t\\,293.750000)*gte(t-prev_selected_t\\,6.250000)" in long


@pytest.mark.asyncio
async def test_scene_candidates_reject_frame_timestamp_mismatch(monkeypatch):
    processor = _scene_processor()

    async def fake_run(command, *, timeout=None):
        return _ppm_frame(1, 1, [[[9, 9, 9]]]) * 2, _showinfo(0.0)

    monkeypatch.setattr(processor, "_run_media_command_with_stderr", fake_run)

    with pytest.raises(ValueError, match="2 candidate frames but 1 timestamps"):
        await processor._extract_scene_candidates_ffmpeg("video.mp4", 5.0)


def test_showinfo_times_parse_only_showinfo_frame_lines():
    stderr = (
        b"  Metadata:\n    title           : pts_time:5 in a tag\n"
        b"[Parsed_showinfo_1 @ 0x5f] config in time_base: 1/30, frame_rate: 30/1\n"
        b"[Parsed_showinfo_1 @ 0x5f] n:   0 pts:     -2 pts_time:-0.0667 duration:1\n"
        b"[Parsed_showinfo_1 @ 0x5f] n:   1 pts:NOPTS pts_time:NOPTS\n"
        b"frame=    3 fps=0.0\r[Parsed_showinfo_1 @ 0x5f] n:   2 pts:    360 "
        b"pts_time:1.2e+01 duration:1\n"
        b"[Parsed_showinfo_1 @ 0x5f]   color_range:tv last_pts_time:3\n"
    )

    assert ContentProcessor._parse_showinfo_times(stderr) == [-0.0667, 12.0]


@pytest.mark.asyncio
async def test_scene_adaptive_extracts_extra_candidates_before_quality_filtering(
    monkeypatch,
):
    processor = _processor(
        num_keyframes=3,
        max_video_length=10,
        keyframe_floor_seconds=1.0,
        keyframe_min_luma=8.0,
        keyframe_min_blur_laplacian_var=0.0,
        keyframe_dedupe_luma_diff_threshold=4.0,
    )
    checker = ((np.indices((16, 16)).sum(axis=0) % 2) * 255).astype(np.uint8)
    duplicate = np.clip(checker.astype(np.int16) + 2, 0, 255).astype(np.uint8)
    inverse = (255 - checker).astype(np.uint8)

    async def fake_candidates(_video_path, _duration):
        frames = []
        for index in range(11):
            if index == 5:
                frames.append(_rgb_from_luma(duplicate))
            elif index == 10:
                frames.append(np.zeros((16, 16, 3), dtype=np.uint8))
            elif index % 2 == 0:
                frames.append(_rgb_from_luma(checker))
            else:
                frames.append(_rgb_from_luma(inverse))
        return frames, [float(index) for index in range(11)]

    monkeypatch.setattr(processor, "_extract_scene_candidates_ffmpeg", fake_candidates)

    (
        keyframes,
        timestamps,
    ) = await processor._extract_scene_adaptive_keyframes_with_timestamps_ffmpeg(
        "video.mp4", frame_count=11, fps=1.0, duration=10.0
    )

    assert len(keyframes) == processor.max_keyframes
    # The dark frame (10) and the near-duplicate (5) never survive.
    assert 10.0 not in timestamps and 5.0 not in timestamps


@pytest.mark.asyncio
async def test_uniform_extraction_seeks_each_sample_and_drops_missing_tail(
    monkeypatch,
):
    processor = _processor(keyframe_sampling_strategy="uniform")
    seeks = []

    async def fake_run(command, *, timeout=None):
        if command[0] == "ffprobe":
            return _ffprobe_payload(duration="2.0", fps="5/1", frames="10", audio=False)
        timestamp = float(command[command.index("-ss") + 1])
        assert command.index("-ss") < command.index("-i")
        assert command[command.index("-frames:v") + 1] == "1"
        seeks.append(timestamp)
        if timestamp >= 1.8:
            return b""  # past the last decodable frame
        return _ppm_frame(1, 1, [[[int(timestamp * 100), 0, 0]]])

    monkeypatch.setattr(processor, "_run_media_command", fake_run)

    keyframes, video_info = await processor._extract_keyframes_ffmpeg("video.mp4")

    assert seeks == pytest.approx([0.0, 0.8, 1.8])
    assert len(keyframes) == 2
    assert video_info["frame_timestamps_seconds"] == pytest.approx([0.0, 0.8])
    assert keyframes[1].tolist() == [[[80, 0, 0]]]


@pytest.mark.asyncio
async def test_uniform_extraction_raises_when_every_seek_fails(monkeypatch):
    processor = _processor()

    async def fake_run(command, *, timeout=None):
        raise RuntimeError("ffmpeg failed: corrupt")

    monkeypatch.setattr(processor, "_run_media_command", fake_run)

    with pytest.raises(RuntimeError, match="corrupt"):
        await processor._extract_ffmpeg_frames_at_times("video.mp4", [0.0, 1.0])


@pytest.mark.asyncio
async def test_scene_adaptive_zero_usable_frames_falls_back_to_uniform(monkeypatch):
    processor = _processor(
        num_keyframes=2,
        keyframe_min_luma=300.0,
        keyframe_min_blur_laplacian_var=0.0,
        keyframe_dedupe_luma_diff_threshold=0.0,
    )

    async def dark_candidates(_video_path, _duration):
        return [np.zeros((1, 1, 3), dtype=np.uint8)] * 2, [0.0, 1.0]

    async def fake_run(command, *, timeout=None):
        if command[0] == "ffprobe":
            return _ffprobe_payload(duration="2.0", fps="1/1", frames="2")
        return _ppm_frame(1, 1, [[[255, 0, 0]]])

    monkeypatch.setattr(processor, "_extract_scene_candidates_ffmpeg", dark_candidates)
    monkeypatch.setattr(processor, "_run_media_command", fake_run)

    keyframes, video_info = await processor._extract_keyframes_ffmpeg("video.mp4")

    assert len(keyframes) == 2
    assert keyframes[0].tolist() == [[[255, 0, 0]]]
    assert video_info["frame_timestamps_seconds"] == [0.0, 1.0]


@pytest.mark.asyncio
async def test_scene_pass_deadline_is_not_retried_as_uniform(monkeypatch):
    from video_commerce.ml.content_processor import KeyframeExtractionDeadlineExceeded

    processor = _processor()

    async def out_of_time(_video_path, _duration):
        raise KeyframeExtractionDeadlineExceeded("deadline")

    async def fail_uniform(*_args):
        raise AssertionError("no fallback once the deadline is gone")

    monkeypatch.setattr(processor, "_extract_scene_candidates_ffmpeg", out_of_time)
    monkeypatch.setattr(processor, "_extract_uniform_keyframes_ffmpeg", fail_uniform)

    with pytest.raises(KeyframeExtractionDeadlineExceeded):
        await processor._extract_keyframes_ffmpeg(
            "video.mp4", {"duration": 5.0, "fps": 10.0, "frame_count": 50}
        )


def _probe_ok(processor, monkeypatch, info=None):
    async def fake_probe(_video_path):
        return dict(info or {"duration": 1.0, "fps": 1.0, "frame_count": 1})

    monkeypatch.setattr(processor, "_probe_video_metadata", fake_probe)


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("ffprobe"),
        TimeoutError("ffprobe timed out"),
        RuntimeError("ffprobe failed: Invalid data"),
        ValueError("ffprobe did not report a video stream"),
    ],
)
@pytest.mark.asyncio
async def test_unreadable_probe_falls_back_to_opencv(monkeypatch, error):
    processor = _processor()
    fallback_frame = np.zeros((1, 1, 3), dtype=np.uint8)

    async def failing_probe(_video_path):
        raise error

    async def fail_ffmpeg(*_args):
        raise AssertionError("FFmpeg extraction needs probe metadata")

    async def fake_opencv(video_path):
        assert video_path == "video.mp4"
        return [fallback_frame], {"duration": 1.0, "has_audio": True}

    monkeypatch.setattr(processor, "_probe_video_metadata", failing_probe)
    monkeypatch.setattr(processor, "_extract_keyframes_ffmpeg", fail_ffmpeg)
    monkeypatch.setattr(processor, "_extract_keyframes_opencv", fake_opencv)

    keyframes, video_info = await processor._extract_keyframes("video.mp4")

    assert keyframes == [fallback_frame]
    assert video_info["duration"] == 1.0


@pytest.mark.parametrize("outcome", ["raises", "empty"])
@pytest.mark.asyncio
async def test_ffmpeg_without_frames_falls_back_to_opencv_keeping_probe_info(
    monkeypatch, outcome
):
    processor = _processor()
    _probe_ok(
        processor,
        monkeypatch,
        {"duration": 9.0, "fps": 3.0, "frame_count": 27, "has_audio": False},
    )
    frame = np.full((1, 1, 3), 7, dtype=np.uint8)

    async def ffmpeg(_video_path, video_info):
        if outcome == "raises":
            raise RuntimeError("ffmpeg failed: Invalid data found")
        return [], video_info

    async def opencv(video_path):
        assert video_path == "video.mp4"
        return [frame], {"has_audio": True, "frame_timestamps_seconds": [4.0]}

    monkeypatch.setattr(processor, "_extract_keyframes_ffmpeg", ffmpeg)
    monkeypatch.setattr(processor, "_extract_keyframes_opencv", opencv)

    keyframes, video_info = await processor._extract_keyframes("video.mp4")

    assert keyframes == [frame]
    # ffprobe knew there is no audio; OpenCV only supplies frames and times.
    assert video_info["has_audio"] is False
    assert video_info["duration"] == 9.0
    assert video_info["frame_timestamps_seconds"] == [4.0]


@pytest.mark.asyncio
async def test_opencv_fallback_reports_a_corrupt_file_as_invalid(monkeypatch):
    from video_commerce.ml.content_processor import InvalidVideoError

    processor = _processor()
    _probe_ok(processor, monkeypatch)

    async def ffmpeg(_video_path, _video_info):
        raise RuntimeError("ffmpeg failed: Invalid data found")

    async def opencv(_video_path):
        raise InvalidVideoError("OpenCV could not decode video")

    monkeypatch.setattr(processor, "_extract_keyframes_ffmpeg", ffmpeg)
    monkeypatch.setattr(processor, "_extract_keyframes_opencv", opencv)

    with pytest.raises(InvalidVideoError):
        await processor._extract_keyframes("video.mp4")


@pytest.mark.asyncio
async def test_ffmpeg_deadline_is_not_handed_to_opencv(monkeypatch):
    from video_commerce.ml.content_processor import KeyframeExtractionDeadlineExceeded

    processor = _processor()
    _probe_ok(processor, monkeypatch)

    async def ffmpeg(_video_path, _video_info):
        raise KeyframeExtractionDeadlineExceeded("deadline")

    async def fail_opencv(_video_path):
        raise AssertionError("no time is left for OpenCV")

    monkeypatch.setattr(processor, "_extract_keyframes_ffmpeg", ffmpeg)
    monkeypatch.setattr(processor, "_extract_keyframes_opencv", fail_opencv)

    with pytest.raises(KeyframeExtractionDeadlineExceeded):
        await processor._extract_keyframes("video.mp4")


@pytest.mark.asyncio
async def test_uniform_seeks_return_partial_frames_at_the_deadline(monkeypatch):
    from video_commerce.ml.content_processor import KeyframeExtractionDeadlineExceeded

    processor = _processor()
    calls = []

    async def fake_run(command, *, timeout=None):
        calls.append(command)
        if len(calls) > 2:
            raise KeyframeExtractionDeadlineExceeded("deadline")
        return _ppm_frame(1, 1, [[[len(calls), 0, 0]]])

    monkeypatch.setattr(processor, "_run_media_command", fake_run)

    sampled = await processor._extract_ffmpeg_frames_at_times(
        "video.mp4", [0.0, 1.0, 2.0, 3.0]
    )

    assert [timestamp for timestamp, _ in sampled] == [0.0, 1.0]
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_uniform_seeks_raise_the_deadline_when_nothing_decoded(monkeypatch):
    from video_commerce.ml.content_processor import KeyframeExtractionDeadlineExceeded

    processor = _processor()

    async def fake_run(command, *, timeout=None):
        raise KeyframeExtractionDeadlineExceeded("deadline")

    monkeypatch.setattr(processor, "_run_media_command", fake_run)

    with pytest.raises(KeyframeExtractionDeadlineExceeded):
        await processor._extract_ffmpeg_frames_at_times("video.mp4", [0.0, 1.0])


@pytest.mark.asyncio
async def test_unknown_duration_and_fps_still_run_the_scene_pass(monkeypatch):
    processor = _scene_processor(max_video_length=30)
    seen = {}

    async def fake_candidates(_video_path, duration):
        seen["duration"] = duration
        return [np.full((2, 2, 3), 99, dtype=np.uint8)], [0.0]

    monkeypatch.setattr(processor, "_extract_scene_candidates_ffmpeg", fake_candidates)

    keyframes, video_info = await processor._extract_keyframes_ffmpeg(
        "video.webm", {"duration": 0.0, "fps": 0.0, "frame_count": 0}
    )

    # A piped WebM reports no duration: decode up to the allowed length.
    assert seen["duration"] == 30.0
    assert len(keyframes) == 1
    assert video_info["frame_timestamps_seconds"] == [0.0]


@pytest.mark.asyncio
async def test_ffprobe_uses_r_frame_rate_when_avg_frame_rate_is_unknown(monkeypatch):
    processor = _processor()

    async def fake_run(command, **_kwargs):
        payload = json.loads(_ffprobe_payload(fps="0/0", frames="0"))
        payload["streams"][0]["r_frame_rate"] = "25/1"
        return json.dumps(payload).encode()

    monkeypatch.setattr(processor, "_run_media_command", fake_run)

    metadata = await processor._probe_video_metadata("video.mkv")

    assert metadata["fps"] == 25.0
    assert metadata["frame_count"] == int(12.5 * 25)


# --- time budget --------------------------------------------------------------


@pytest.mark.asyncio
async def test_extraction_deadline_caps_command_timeouts_and_is_reset(monkeypatch):
    from video_commerce.ml import content_processor as module

    processor = _processor(
        keyframe_extraction_deadline_seconds=40.0, ffmpeg_timeout_seconds=30.0
    )
    seen = {}

    async def fake_ffmpeg(_video_path, video_info):
        seen["command"] = processor._media_command_limit(None)[0]
        seen["long"] = processor._media_command_limit(500.0)[0]
        seen["scene"] = processor._scene_pass_timeout(300.0)
        return [np.zeros((1, 1, 3), dtype=np.uint8)], video_info

    _probe_ok(processor, monkeypatch)
    monkeypatch.setattr(processor, "_extract_keyframes_ffmpeg", fake_ffmpeg)

    await processor._extract_keyframes("video.mp4")

    assert seen["command"] == pytest.approx(30.0)
    assert 39.0 < seen["long"] <= 40.0
    # The scene pass leaves 20% of the deadline for the uniform fallback.
    assert 31.0 < seen["scene"] <= 32.0
    assert module._keyframe_extraction_deadline.get() is None
    assert processor._media_command_limit(500.0)[0] == 500.0


def test_expired_deadline_refuses_new_commands():
    from video_commerce.ml import content_processor as module
    from video_commerce.ml.content_processor import KeyframeExtractionDeadlineExceeded

    processor = _processor()
    token = module._keyframe_extraction_deadline.set(module.time.monotonic() - 1)
    try:
        with pytest.raises(KeyframeExtractionDeadlineExceeded):
            processor._media_command_limit(None)[0]
        with pytest.raises(TimeoutError):
            processor._scene_pass_timeout(10.0)
    finally:
        module._keyframe_extraction_deadline.reset(token)


def test_full_decode_timeout_scales_with_duration_and_threads_are_configurable():
    processor = _processor(ffmpeg_timeout_seconds=30, ffmpeg_timeout_per_media_second=0.5)

    assert processor._full_decode_timeout(300.0) == pytest.approx(180.0)
    assert processor._full_decode_timeout(0.0) == pytest.approx(30.0)
    assert processor._ffmpeg_input("v.mp4") == ["-threads", "1", "-i", "v.mp4"]
    assert _processor(ffmpeg_decode_threads=0)._ffmpeg_input("v.mp4") == ["-i", "v.mp4"]


@pytest.mark.skipif(shutil.which("sleep") is None, reason="sleep unavailable")
@pytest.mark.asyncio
async def test_media_command_is_killed_on_timeout_and_on_cancellation(monkeypatch):
    import asyncio

    from video_commerce.ml import content_processor as module

    processor = _processor()
    processes = []
    create = asyncio.create_subprocess_exec

    async def tracking_create(*args, **kwargs):
        process = await create(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", tracking_create)

    with pytest.raises(TimeoutError, match="timed out after 0.2s"):
        await processor._run_media_command(["sleep", "30"], timeout=0.2)
    assert processes[-1].returncode is not None

    task = asyncio.ensure_future(processor._run_media_command(["sleep", "30"], timeout=60))
    while not processes or processes[-1].returncode is not None:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert processes[-1].returncode is not None


@pytest.mark.skipif(shutil.which("sleep") is None, reason="sleep unavailable")
@pytest.mark.asyncio
async def test_command_running_into_the_deadline_raises_deadline_exceeded():
    from video_commerce.ml import content_processor as module
    from video_commerce.ml.content_processor import KeyframeExtractionDeadlineExceeded

    processor = _processor()
    token = module._keyframe_extraction_deadline.set(module.time.monotonic() + 0.2)
    try:
        with pytest.raises(KeyframeExtractionDeadlineExceeded):
            await processor._run_media_command(["sleep", "30"], timeout=60)
    finally:
        module._keyframe_extraction_deadline.reset(token)


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


def test_frame_timestamps_prefer_extraction_metadata_over_synthetic_spacing():
    frames = [np.zeros((1, 1, 3), dtype=np.uint8) for _ in range(3)]

    timestamps = ContentProcessor._frame_timestamps(
        frames,
        {
            "duration": 10.0,
            "frame_timestamps_seconds": [0.0, 1.25, 9.5],
        },
    )

    assert timestamps == [0.0, 1.25, 9.5]


def test_frame_timestamps_derive_from_real_frame_indices_when_available():
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]

    timestamps = ContentProcessor._frame_timestamps(
        frames,
        {
            "fps": 4.0,
            "sampled_frame_indices": [0, 5, 38],
            "duration": 10.0,
        },
    )

    assert timestamps == pytest.approx([0.0, 1.25, 9.5])


def test_frame_timestamps_do_not_synthesize_spacing_without_source_metadata():
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]

    timestamps = ContentProcessor._frame_timestamps(
        frames,
        {"duration": 10.0},
    )

    assert timestamps == [0.0, 0.0, 0.0]


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

    async def no_fallback(*_args):
        raise AssertionError("the scene pass should have produced frames")

    processor._extract_uniform_keyframes_ffmpeg = no_fallback

    keyframes, _ = await processor._extract_keyframes_ffmpeg(str(video_path))

    assert 1 <= len(keyframes) <= 4
    mean_colors = {
        tuple(np.round(np.mean(frame, axis=(0, 1))).astype(int)) for frame in keyframes
    }
    assert len(mean_colors) > 1


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
async def test_speech_to_text_success_retains_transcript_and_model(
    monkeypatch, tmp_path
):
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
async def test_speech_to_text_retains_aligned_segments_and_embeddings(
    monkeypatch, tmp_path
):
    processor = _processor(speech_to_text_enabled=True)
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"wav")

    async def fake_extract_audio(_video_path, _video_info):
        return str(audio_file)

    async def fake_request(_audio_path):
        return {
            "text": "新品手機",
            "alignment_status": "completed",
            "segments": [
                {"text": "新品", "start_seconds": 0.0, "end_seconds": 1.0},
                {"text": "手機", "start_seconds": 1.1, "end_seconds": 2.0},
            ],
        }

    monkeypatch.setattr(processor, "_extract_audio_wav", fake_extract_audio)
    monkeypatch.setattr(processor, "_request_transcription", fake_request)
    monkeypatch.setattr(
        processor,
        "_encode_multilingual_texts",
        lambda texts, prefix: [[float(index), 1.0] for index, _ in enumerate(texts)],
    )

    features = await processor._extract_audio_features(
        "video.mp4", {"has_audio": True, "duration": 4.0}
    )

    assert features.alignment_status == "completed"
    assert features.asr_segments == [
        {
            "text": "新品",
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "text_embedding": [0.0, 1.0],
        },
        {
            "text": "手機",
            "start_seconds": 1.1,
            "end_seconds": 2.0,
            "text_embedding": [1.0, 1.0],
        },
    ]


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
    assert (
        commerce["category_scores"]["electronics"]
        > commerce["category_scores"]["beauty"]
    )


def test_ocr_embeddings_use_multilingual_passage_encoder():
    processor = _processor()
    calls = []

    class FakeEmbedder:
        def encode(self, texts, *, prefix):
            calls.append((list(texts), prefix))
            return [[1.0, 0.0] for _ in texts]

    processor.text_embedder = FakeEmbedder()

    result = processor._encode_ocr_texts(["新品手機", "new phone"])

    assert calls == [(["新品手機", "new phone"], "passage")]
    assert result == [[1.0, 0.0], [1.0, 0.0]]


class _FakeClipInputs(dict):
    def to(self, _device):
        return self


class _FakeClipProcessor:
    def __call__(self, images=None, **_kwargs):
        return _FakeClipInputs(count=len(images or []))


class _FakeClipModel:
    def __init__(self, *, fail=False, delay_seconds=0.0, zero=False):
        self.fail = fail
        self.delay_seconds = delay_seconds
        self.zero = zero

    def get_image_features(self, count):
        import time as _time

        if self.delay_seconds:
            _time.sleep(self.delay_seconds)
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        if self.zero:
            # Opposite unit vectors average to an all-zero embedding.
            rows = [[1.0, 0.0], [-1.0, 0.0]][:count]
            return torch.tensor(rows * (count // 2) + rows[: count % 2])
        return torch.ones((count, 2))


def _frames(count=2):
    return [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(count)]


@pytest.mark.asyncio
async def test_clip_failure_raises_retryable_modality_error_instead_of_zeros():
    from video_commerce.ml.content_processor import ContentModalityError

    processor = _processor()
    processor.clip_processor = _FakeClipProcessor()
    processor.clip_model = _FakeClipModel(fail=True)

    with pytest.raises(ContentModalityError) as exc:
        await processor._extract_visual_features(_frames())

    assert exc.value.modality == "clip"
    assert not isinstance(exc.value, ValueError)


@pytest.mark.asyncio
async def test_clip_without_frames_raises_instead_of_zero_embedding():
    from video_commerce.ml.content_processor import ContentModalityError

    processor = _processor()
    processor.clip_processor = _FakeClipProcessor()
    processor.clip_model = _FakeClipModel()

    with pytest.raises(ContentModalityError):
        await processor._extract_visual_features([])


@pytest.mark.asyncio
async def test_ocr_failure_is_recorded_as_status_and_processing_continues(
    monkeypatch,
):
    from video_commerce.ml import content_processor as module

    processor = _processor(ocr_backend="tesseract")

    def failing_ocr(*_args, **_kwargs):
        raise RuntimeError("tesseract crashed")

    monkeypatch.setattr(module.pytesseract, "image_to_data", failing_ocr)

    result = await processor._extract_text_features(_frames(2), [0.0, 1.0])

    assert result["status"] == "failed"
    assert result["text_blocks"] == []


@pytest.mark.asyncio
async def test_ocr_partial_frame_failure_is_degraded(monkeypatch):
    from video_commerce.ml import content_processor as module

    processor = _processor(ocr_backend="tesseract")
    calls = {"count": 0}

    def flaky_ocr(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("tesseract crashed")
        return {"conf": [], "text": []}

    monkeypatch.setattr(module.pytesseract, "image_to_data", flaky_ocr)

    result = await processor._extract_text_features(_frames(2), [0.0, 1.0])

    assert result["status"] == "degraded"


@pytest.mark.asyncio
async def test_ocr_backend_error_is_failed_status_not_exception():
    processor = _processor(ocr_backend="paddle")
    processor.ocr_region_extractor = None

    result = await processor._extract_text_features(_frames(1), [0.0])

    assert result["status"] == "failed"


def _stub_pipeline(processor, monkeypatch, *, ocr_status="failed", detection=None):
    from video_commerce.common.models import AudioFeatures

    async def keyframes(_path):
        return _frames(2), {
            "duration": 2.0,
            "fps": 1.0,
            "frame_timestamps_seconds": [0.0, 1.0],
        }

    async def text(_frames, _timestamps):
        return {"text_blocks": [], "price_mentions": [], "status": ocr_status}

    async def audio(_path, _info):
        return AudioFeatures(transcription_status="disabled")

    async def detect(_frames):
        return detection or (["phone"], "completed")

    processor.is_initialized = True
    processor.clip_processor = _FakeClipProcessor()
    monkeypatch.setattr(processor, "_extract_keyframes", keyframes)
    monkeypatch.setattr(processor, "_extract_text_features", text)
    monkeypatch.setattr(processor, "_extract_audio_features", audio)
    monkeypatch.setattr(processor, "_detect_objects_with_status", detect)


@pytest.mark.asyncio
async def test_process_video_persists_modality_status(monkeypatch):
    processor = _processor()
    _stub_pipeline(processor, monkeypatch, detection=([], "failed"))
    processor.clip_model = _FakeClipModel()

    features = await processor.process_video("video.mp4", "content-1")

    assert features.modality_status == {
        "clip": "completed",
        "ocr": "failed",
        "object_detection": "failed",
    }
    assert features.dict()["modality_status"]["ocr"] == "failed"


@pytest.mark.asyncio
async def test_process_video_marks_zero_norm_clip_embedding(monkeypatch):
    processor = _processor()
    _stub_pipeline(processor, monkeypatch, ocr_status="completed")
    processor.clip_model = _FakeClipModel(zero=True)

    features = await processor.process_video("video.mp4", "content-1")

    assert features.modality_status["clip"] == "zero_norm"


@pytest.mark.asyncio
async def test_process_video_propagates_clip_failure(monkeypatch):
    from video_commerce.ml.content_processor import ContentModalityError

    processor = _processor()
    _stub_pipeline(processor, monkeypatch)
    processor.clip_model = _FakeClipModel(fail=True)

    with pytest.raises(ContentModalityError):
        await processor.process_video("video.mp4", "content-1")


@pytest.mark.asyncio
async def test_process_video_without_keyframes_is_invalid_video(monkeypatch):
    from video_commerce.ml.content_processor import InvalidVideoError

    processor = _processor()
    processor.is_initialized = True

    async def no_keyframes(_path):
        return [], {}

    monkeypatch.setattr(processor, "_extract_keyframes", no_keyframes)

    with pytest.raises(InvalidVideoError):
        await processor.process_video("video.mp4", "content-1")


@pytest.mark.asyncio
async def test_clip_inference_does_not_block_the_event_loop():
    import asyncio
    import time as _time

    processor = _processor()
    processor.clip_processor = _FakeClipProcessor()
    processor.clip_model = _FakeClipModel(delay_seconds=0.4)
    gaps = []
    stop = asyncio.Event()

    async def ticker():
        last = _time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = _time.perf_counter()
            gaps.append(now - last)
            last = now

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    started = _time.perf_counter()
    await processor._extract_visual_features(_frames(2))
    elapsed = _time.perf_counter() - started
    stop.set()
    await task

    assert elapsed >= 0.4
    # Lease renewal and heartbeats keep running while CLIP occupies a thread.
    assert len(gaps) >= 10
    assert max(gaps) < 0.2


@pytest.mark.asyncio
async def test_ocr_and_detection_do_not_block_the_event_loop(monkeypatch):
    import asyncio
    import time as _time

    from video_commerce.ml import content_processor as module

    processor = _processor(ocr_backend="tesseract")

    def slow_ocr(*_args, **_kwargs):
        _time.sleep(0.2)
        return {"conf": [], "text": []}

    def slow_detect(_frames):
        _time.sleep(0.2)
        return ["phone"]

    monkeypatch.setattr(module.pytesseract, "image_to_data", slow_ocr)
    monkeypatch.setattr(processor, "_detect_objects_sync", slow_detect)
    gaps = []
    stop = asyncio.Event()

    async def ticker():
        last = _time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = _time.perf_counter()
            gaps.append(now - last)
            last = now

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    text = await processor._extract_text_features(_frames(1), [0.0])
    objects, status = await processor._detect_objects_with_status(_frames(1))
    stop.set()
    await task

    assert text["status"] == "completed"
    assert (objects, status) == (["phone"], "completed")
    assert max(gaps) < 0.15


def test_embedding_has_signal_detects_zero_and_non_finite_vectors():
    from video_commerce.ml.content_processor import embedding_has_signal

    assert embedding_has_signal([0.6, 0.8])
    assert not embedding_has_signal([0.0, 0.0])
    assert not embedding_has_signal([])
    assert not embedding_has_signal([float("nan"), 1.0])


def _opencv_extract(path):
    from video_commerce.ml import opencv_keyframes

    return opencv_keyframes.extract_keyframes(
        str(path), max_keyframes=3, max_video_length=10
    )


def test_opencv_missing_file_is_not_classified_as_invalid_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        _opencv_extract(tmp_path / "missing.mp4")


def test_opencv_undecodable_file_is_invalid_video(tmp_path):
    from video_commerce.ml.content_processor import InvalidVideoError

    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"this is not a video container")

    with pytest.raises(InvalidVideoError):
        _opencv_extract(garbage)


def test_opencv_io_errors_stay_retryable(monkeypatch, tmp_path):
    from video_commerce.ml import content_processor as module
    from video_commerce.ml.content_processor import InvalidVideoError

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")

    class Capture:
        def isOpened(self):
            return True

        def get(self, _prop):
            raise MemoryError("cannot allocate frame buffer")

        def release(self):
            return None

    monkeypatch.setattr(module.cv2, "VideoCapture", lambda _path: Capture())

    with pytest.raises(MemoryError) as exc:
        _opencv_extract(source)

    assert not isinstance(exc.value, InvalidVideoError)


# --- OpenCV fallback in a child process ----------------------------------------


@pytest.mark.asyncio
async def test_opencv_child_reports_invalid_and_missing_videos(tmp_path):
    from video_commerce.ml.content_processor import InvalidVideoError

    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"this is not a video container")
    processor = _processor(ffmpeg_timeout_seconds=30)

    with pytest.raises(InvalidVideoError):
        await processor._extract_keyframes_opencv(str(garbage))
    with pytest.raises(FileNotFoundError):
        await processor._extract_keyframes_opencv(str(tmp_path / "missing.mp4"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
@pytest.mark.asyncio
async def test_opencv_child_returns_frames_from_a_real_video(tmp_path):
    video_path = tmp_path / "clip.avi"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=s=32x24:d=2:r=5",
            "-c:v", "mjpeg", str(video_path),
        ],
        check=True,
        capture_output=True,
    )
    processor = _processor(num_keyframes=3, ffmpeg_timeout_seconds=30)

    keyframes, video_info = await processor._extract_keyframes_opencv(str(video_path))

    assert len(keyframes) == 3
    assert keyframes[0].shape == (24, 32, 3)
    assert keyframes[0].dtype == np.uint8
    assert video_info["frame_timestamps_seconds"] == pytest.approx([0.0, 0.8, 1.8])


@pytest.mark.asyncio
async def test_opencv_child_unexpected_failure_stays_retryable(monkeypatch):
    from video_commerce.ml.content_processor import InvalidVideoError

    processor = _processor()

    async def fake_run(command, *, timeout=None):
        assert command[1] == "-I"
        assert command[2].endswith("opencv_keyframes.py")
        # Scaled like a full decode: base + max_video_length x per-second.
        assert timeout == pytest.approx(1 + 10 * 0.5)
        with open(command[5], "w") as handle:
            handle.write('{"error": "failed", "message": "MemoryError: frame buffer"}')
        return b"[ INFO:0@0.1] global logging noise on stdout\n"

    monkeypatch.setattr(processor, "_run_media_command", fake_run)

    with pytest.raises(RuntimeError, match="MemoryError") as exc:
        await processor._extract_keyframes_opencv("video.mp4")
    assert not isinstance(exc.value, InvalidVideoError)


# --- content handler time budget ---------------------------------------------


def _budget_config(**model):
    from video_commerce.common.config import Config

    config = Config()
    defaults = dict(
        keyframe_extraction_deadline_seconds=150.0,
        ffmpeg_timeout_seconds=30.0,
        speech_to_text_enabled=False,
        speech_to_text_timeout_seconds=120.0,
        speech_to_text_max_attempts=2,
    )
    defaults.update(model)
    for name, value in defaults.items():
        setattr(config.model_config, name, value)
    config.kafka_config.consumer_max_poll_interval_ms = 600000
    return config


def test_default_content_budget_fits_the_kafka_retry_budget():
    from video_commerce.common.config import content_handler_budget_errors

    # 150 extraction + 30 audio + 60 other = 240 <= 540
    assert content_handler_budget_errors(_budget_config()) == []
    # + 120 x 2 ASR = 480 <= 540
    assert (
        content_handler_budget_errors(_budget_config(speech_to_text_enabled=True))
        == []
    )


def test_content_budget_rejects_a_single_attempt_longer_than_the_poll_budget():
    from video_commerce.common.config import content_handler_budget_errors

    config = _budget_config(speech_to_text_enabled=True)
    config.kafka_config.consumer_max_poll_interval_ms = 300000

    errors = content_handler_budget_errors(config)

    assert len(errors) == 1
    assert "480s" in errors[0] and "(270s)" in errors[0]


def test_only_the_content_worker_refuses_an_oversized_budget(monkeypatch):
    from video_commerce.common.config import Config
    from video_commerce.services.content_worker.video_processor import (
        validate_content_worker_config,
    )

    monkeypatch.setenv("MODEL_KEYFRAME_EXTRACTION_DEADLINE_SECONDS", "600")

    config = Config()  # other services still start
    with pytest.raises(ValueError, match="Invalid content worker configuration"):
        validate_content_worker_config(config)


# --- real FFmpeg edge cases -----------------------------------------------------


def _encode(path, *args):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args, str(path)],
        check=True,
        capture_output=True,
    )


def _real_processor(**overrides):
    defaults = dict(
        num_keyframes=8,
        ffmpeg_timeout_seconds=30,
        ffmpeg_target_width=32,
        keyframe_floor_seconds=2.0,
        keyframe_min_luma=0.0,
        keyframe_min_blur_laplacian_var=0.0,
        keyframe_dedupe_luma_diff_threshold=0.0,
    )
    defaults.update(overrides)
    processor = _processor(**defaults)

    async def no_fallback(*_args):
        raise AssertionError("the scene pass should have produced frames")

    processor._extract_uniform_keyframes_ffmpeg = no_fallback
    return processor


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
@pytest.mark.asyncio
async def test_real_ffmpeg_video_longer_than_the_limit_uses_the_scene_pass(tmp_path):
    video = tmp_path / "long.mp4"
    _encode(
        video,
        "-f", "lavfi", "-i", "testsrc2=s=64x48:r=10:d=12",
        "-vf", "negate=enable='mod(floor(t/3),2)'",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
    )
    processor = _real_processor(max_video_length=5)

    keyframes, info = await processor._extract_keyframes_ffmpeg(str(video))

    assert keyframes
    timestamps = info["frame_timestamps_seconds"]
    assert timestamps[0] == 0.0 and max(timestamps) < 5.0
    assert 3.0 in timestamps  # the cut inside the processed span


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
@pytest.mark.asyncio
async def test_real_ffmpeg_video_starting_after_audio(tmp_path):
    video = tmp_path / "offset.mp4"
    _encode(
        video,
        "-itsoffset", "0.5", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=10:d=6",
        "-f", "lavfi", "-i", "sine=d=6.5",
        "-vf", "negate=enable='gte(t,5.7)'",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
    )
    processor = _real_processor(max_video_length=30)

    keyframes, info = await processor._extract_keyframes_ffmpeg(str(video))

    assert keyframes
    assert info["frame_timestamps_seconds"][0] == 0.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
@pytest.mark.asyncio
async def test_real_ffmpeg_piped_webm_without_duration(tmp_path):
    video = tmp_path / "recorded.webm"
    with open(video, "wb") as handle:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=s=64x48:r=10:d=4",
                "-c:v", "libvpx", "-deadline", "realtime", "-f", "webm", "pipe:1",
            ],
            stdout=handle,
            stderr=subprocess.PIPE,
        )
    if result.returncode != 0:
        pytest.skip("libvpx encoder unavailable")
    processor = _real_processor(max_video_length=30)

    keyframes, info = await processor._extract_keyframes(str(video))

    assert keyframes
    assert info["frame_timestamps_seconds"][0] == 0.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
@pytest.mark.asyncio
async def test_opencv_child_ignores_opencv_logging_on_stdout(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.avi"
    _encode(video_path, "-f", "lavfi", "-i", "testsrc=s=32x24:d=1:r=5", "-c:v", "mjpeg")
    monkeypatch.setenv("OPENCV_LOG_LEVEL", "DEBUG")
    processor = _processor(num_keyframes=2, ffmpeg_timeout_seconds=30)

    keyframes, _ = await processor._extract_keyframes_opencv(str(video_path))

    assert len(keyframes) == 2
