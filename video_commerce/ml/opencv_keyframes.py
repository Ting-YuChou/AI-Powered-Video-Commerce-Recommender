"""OpenCV keyframe extraction, run in a child process by the content processor.

OpenCV's decoder cannot be interrupted from Python. Run in a thread, it kept
going after ``asyncio.wait_for`` gave up and held one of the default executor
threads that CLIP and OCR inference also use. In a child process it can be
killed at the keyframe extraction deadline.

This module imports only the standard library, NumPy and OpenCV, so the child
starts quickly. The parent runs it by file path with ``python -I`` (isolated:
no PYTHONPATH, user site or script directory on ``sys.path``), so it needs no
``video_commerce`` import and sibling modules cannot shadow library imports.
"""

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class InvalidVideoError(ValueError):
    """The source media cannot be decoded into frames; retrying will not help."""


def uniform_frame_indices(frame_count: int, max_keyframes: int) -> List[int]:
    if frame_count <= 0 or max_keyframes <= 0:
        return []
    sample_count = min(max_keyframes, frame_count)
    return np.linspace(0, frame_count - 1, sample_count, dtype=int).tolist()


def extract_keyframes(
    video_path: str, *, max_keyframes: int, max_video_length: float
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Uniformly sample RGB keyframes with OpenCV."""
    if not os.path.exists(video_path):
        # Not a decode problem: let the caller retry or classify it.
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    keyframes: List[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise InvalidVideoError(f"Could not open video file: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        if duration > max_video_length:
            frame_count = int(max_video_length * fps)

        video_info: Dict[str, Any] = {
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "has_audio": True,  # OpenCV cannot tell; ASR reports no speech
            "frame_timestamps_seconds": [],
        }
        for frame_idx in uniform_frame_indices(frame_count, max_keyframes):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                keyframes.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                video_info["frame_timestamps_seconds"].append(
                    float(frame_idx) / fps if fps > 0 else 0.0
                )
        return keyframes, video_info
    except cv2.error as exc:
        # OpenCV decode/convert failures mean the media is unusable.
        raise InvalidVideoError(f"OpenCV could not decode video: {exc}") from exc
    finally:
        cap.release()


def main(argv: List[str]) -> int:
    """``<video> <frames.npz> <result.json> <max_keyframes> <max_video_length>``.

    Writes frames to the .npz as ``frame_0``..``frame_N`` and one JSON object to
    ``result.json``: ``{"info": ..., "frames": N}``, or ``{"error": kind,
    "message": ...}`` where kind is ``missing``, ``invalid_video`` or
    ``failed``. The result goes to a file, not stdout, because OpenCV can log to
    stdout (``OPENCV_LOG_LEVEL``).
    """
    video_path, frames_path, result_path, max_keyframes, max_video_length = argv
    try:
        frames, info = extract_keyframes(
            video_path,
            max_keyframes=int(max_keyframes),
            max_video_length=float(max_video_length),
        )
        np.savez(frames_path, **{f"frame_{i}": frame for i, frame in enumerate(frames)})
        result: Dict[str, Any] = {"info": info, "frames": len(frames)}
    except FileNotFoundError as exc:
        result = {"error": "missing", "message": str(exc)}
    except InvalidVideoError as exc:
        result = {"error": "invalid_video", "message": str(exc)}
    except Exception as exc:  # reported to the parent, which keeps it retryable
        result = {"error": "failed", "message": f"{type(exc).__name__}: {exc}"}
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
