#!/usr/bin/env python3
"""Compare keyframe extraction strategies on a set of sample videos.

Strategies:

- ``current``: scene detection (one full decode), then one FFmpeg run that
  decodes from the start up to the last candidate with ``select=eq(n,idx)``.
  Timestamps are reported as ``index / avg_fps``.
- ``seek``: the same scene detection, then one ``ffmpeg -ss <t>`` run per
  candidate timestamp (``-threads 1``, bounded concurrency). An alternative
  that was evaluated, not shipped.
- ``single_pass``: the production method. One FFmpeg run selects scene and
  floor frames with a ``select`` expression (minimum gap raised to
  ``duration / (3 * num_keyframes)`` to bound memory) and writes them as PPM,
  with ``showinfo`` timestamps.

``current`` is a frozen copy of the pre-change behaviour, kept as a baseline.

Timestamp error is measured against the file's own frame presentation times
(from ffprobe, relative to the first frame): the frame ``current`` decoded at
each index, the frame each ``seek`` actually returned (``showinfo`` with
``-copyts``), and for ``single_pass`` the nearest real frame to each reported
timestamp.

All three feed the same quality filter and thinning, so differences come from
which source frames are decoded and how long that takes.

Each (video, strategy) runs in its own subprocess so peak memory is per run.
Run it under the content worker's CPU limit to match production, e.g.::

    docker run --rm --cpus=1 -v "$PWD:/app" -w /app -e PYTHONPATH=/app \\
        <backend-tests-image> python scripts/benchmark_keyframe_extraction.py \\
        --generate --samples-dir /samples --output /samples/kf.json

Synthetic samples exercise decode cost, GOP length, VFR and audio handling.
Selection quality on real footage needs real uploads: pass them with
``--video``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STRATEGIES = ("current", "seek", "single_pass")

# name -> (seconds, size, fps, gop, audio, vfr, video bitrate)
# At 8 Mbps only ~100 s fits under the 100 MiB upload limit, so the longest
# upload-sized sample is 720p at 2.5 Mbps; the 300 s 1080p one is a stress case.
SAMPLES = {
    "short_15s_1080p": (15, "1920x1080", 30, 250, True, False, "8M"),
    "medium_60s_1080p": (60, "1920x1080", 30, 250, True, False, "8M"),
    "long_300s_720p": (300, "1280x720", 30, 250, True, False, "2500k"),
    "long_300s_1080p": (300, "1920x1080", 30, 250, True, False, "8M"),
    "long_gop_120s_1080p": (120, "1920x1080", 30, 3600, True, False, "8M"),
    "vfr_60s_1080p": (60, "1920x1080", 30, 250, True, True, "8M"),
    "no_audio_60s_720p": (60, "1280x720", 30, 250, False, False, "8M"),
}


def generate_samples(directory: Path, names: Sequence[str]) -> List[Path]:
    """Encode synthetic H.264 clips with hard cuts every 5 seconds."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        seconds, size, fps, gop, audio, vfr, bitrate = SAMPLES[name]
        path = directory / f"{name}.mp4"
        paths.append(path)
        if path.exists():
            continue
        # Moving test pattern, inverted and hue-shifted every 5 s (hard cuts),
        # with temporal noise so the bitrate resembles camera footage.
        video = (
            f"testsrc2=s={size}:r={fps}:d={seconds},"
            "negate=enable='mod(floor(t/5),2)',"
            "hue=H='floor(t/5)*1.3',"
            "noise=alls=12:allf=t"
        )
        if vfr:
            # 10 fps between 20 s and 40 s, 30 fps elsewhere.
            video += ",select='not(between(t,20,40))+not(mod(n,3))'"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", video,
        ]
        if audio:
            command += ["-f", "lavfi", "-i", f"sine=f=440:d={seconds}"]
        command += [
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", "2M",
            "-g", str(gop), "-keyint_min", str(min(gop, 25)),
        ]
        if gop >= 1000:
            # Otherwise x264 adds a keyframe at every cut and the GOP stays short.
            command += ["-sc_threshold", "0"]
        if vfr:
            command += ["-fps_mode", "vfr"]
        if audio:
            command += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
        command.append(str(path))
        started = time.monotonic()
        subprocess.run(command, check=True)
        print(
            f"generated {path.name} in {time.monotonic() - started:.1f}s",
            file=sys.stderr,
        )
    return paths


def _processor(args):
    """Only pure helpers (probe parsing, candidate selection, quality filter,
    PPM parsing) come from the production processor. Every FFmpeg command is
    built here, so results do not move when the production commands change."""
    from video_commerce.common.config import ModelConfig
    from video_commerce.ml.content_processor import ContentProcessor

    return ContentProcessor(
        ModelConfig(
            device="cpu", enable_gpu=False, ffmpeg_decode_threads=args.decode_threads
        )
    )


def _threads(args) -> List[str]:
    return ["-threads", str(args.decode_threads)] if args.decode_threads else []


def _pts_times(stderr: bytes) -> List[float]:
    text = stderr.decode("utf-8", errors="replace")
    return [float(m.group(1)) for m in re.finditer(r"pts_time:(-?[0-9.]+)", text)]


def _legacy_candidate_indices(
    config, *, frame_count: int, fps: float, duration: float, scene_times, limit: int
) -> List[int]:
    """Candidate frame indices as main selected them before the single-pass
    change: frame 0, a floor grid, the last frame, and scene changes at least
    ``keyframe_min_scene_gap_seconds`` apart, thinned evenly to ``limit``."""
    import numpy as np

    def index_at(timestamp: float) -> int:
        return max(0, min(int(round(max(0.0, timestamp) * fps)), frame_count - 1))

    candidates = {0}
    floor_time = 0.0
    while floor_time <= duration + 1e-6:
        candidates.add(index_at(floor_time))
        floor_time += float(config.keyframe_floor_seconds)
    candidates.add(index_at(duration))
    last_scene: Optional[float] = None
    for scene_time in sorted(scene_times):
        if scene_time < 0 or scene_time > duration:
            continue
        if last_scene is not None and scene_time - last_scene < float(
            config.keyframe_min_scene_gap_seconds
        ):
            continue
        candidates.add(index_at(scene_time))
        last_scene = scene_time
    ordered = sorted(candidates)
    if len(ordered) <= limit:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, limit, dtype=int)
    return [ordered[int(position)] for position in positions]


def _frame_pts(video_path: str, limit_seconds: float) -> List[float]:
    """Display-order presentation times of the video stream's frames, relative
    to the first frame (as production reports timestamps)."""
    output = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-read_intervals", f"%+{limit_seconds + 1:.1f}",
            "-show_entries", "packet=pts_time", "-of", "csv=p=0", video_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    times = sorted(float(line) for line in output.split() if line not in {"", "N/A"})
    return [value - times[0] for value in times] if times else []


# ru_maxrss of a child includes memory it shared with the parent before exec,
# and this process holds torch (~300 MB). A bare interpreter (-S, no site)
# launches FFmpeg instead and reports FFmpeg's own peak RSS on a marker line.
_RSS_WRAPPER = (
    "import resource,subprocess,sys;"
    "r=subprocess.run(sys.argv[1:]).returncode;"
    "sys.stderr.write('\\n__FFMPEG_MAXRSS_KB__ %d\\n'"
    "%resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss);"
    "sys.exit(r)"
)
# ru_maxrss units: KiB on Linux, bytes on macOS (scaled in run_single).
_FFMPEG_PEAK_RSS_KB: List[int] = []


def _first_source_pts(video_path: str) -> float:
    output = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-read_intervals", "%+5", "-show_entries", "packet=pts_time",
            "-of", "csv=p=0", video_path,
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    values = [float(v) for v in output.split() if v not in {"", "N/A"}]
    return min(values) if values else 0.0


async def _run(command: List[str], timeout: float) -> Tuple[bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-S", "-c", _RSS_WRAPPER, *command,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise TimeoutError(f"{command[0]} exceeded {timeout:.0f}s")
    marker = re.search(rb"__FFMPEG_MAXRSS_KB__ (\d+)", stderr)
    if marker:
        _FFMPEG_PEAK_RSS_KB.append(int(marker.group(1)))
    if process.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", "replace").strip()[-500:])
    return stdout, stderr


async def _candidates(processor, video_path: str, info: Dict[str, Any], args):
    fps = float(info["fps"])
    frame_count = int(info["frame_count"])
    duration = processor._effective_keyframe_duration(
        frame_count=frame_count, fps=fps, duration=float(info["duration"])
    )
    threshold = float(processor.config.keyframe_scene_threshold)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-nostdin",
        *_threads(args), "-i", video_path, "-t", f"{duration:.1f}",
        "-vf", f"select=gt(scene\\,{threshold:.6f}),showinfo",
        "-an", "-f", "null", "-",
    ]
    started = time.monotonic()
    _, stderr = await _run(command, args.command_timeout)
    scene_times = _pts_times(stderr)
    scene_seconds = time.monotonic() - started
    indices = _legacy_candidate_indices(
        processor.config,
        frame_count=frame_count,
        fps=fps,
        duration=duration,
        scene_times=scene_times,
        limit=max(processor.max_keyframes, processor.max_keyframes * 3),
    )
    return indices, scene_seconds, duration


def _finish(processor, frames, timestamps):
    positions = processor._quality_keyframe_positions(frames)
    sampled = [(frames[p], timestamps[p]) for p in positions]
    sampled = processor._thin_ordered_items(sampled, processor.max_keyframes)
    return [f for f, _ in sampled], [t for _, t in sampled]


async def strategy_current(processor, video_path, info, args):
    indices, scene_seconds, _ = await _candidates(processor, video_path, info, args)
    width = max(1, int(processor.config.ffmpeg_target_width))
    select = "+".join(f"eq(n\\,{index})" for index in indices)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        *_threads(args), "-i", video_path,
        "-vf", f"select={select},scale=min({width}\\,iw):-2,format=rgb24",
        "-vsync", "vfr", "-frames:v", str(len(indices)),
        "-f", "image2pipe", "-vcodec", "ppm", "pipe:1",
    ]
    started = time.monotonic()
    stdout, _ = await _run(command, args.command_timeout)
    frames = processor._parse_ppm_frames(stdout)
    extract_seconds = time.monotonic() - started
    if len(frames) != len(indices):
        raise RuntimeError(f"{len(frames)} of {len(indices)} frames")
    fps = float(info["fps"])
    frames, timestamps = _finish(processor, frames, [i / fps for i in indices])
    return frames, timestamps, {
        "scene_seconds": scene_seconds,
        "extract_seconds": extract_seconds,
        "candidates": len(indices),
        # Source frame index of each reported timestamp, to look up true PTS.
        "source_indices": [int(round(t * fps)) for t in timestamps],
    }


async def _seek_one(processor, video_path: str, timestamp: float, width: int, args):
    # -copyts keeps source timestamps so showinfo reports the returned frame's
    # own pts (only for measurement; production seeks the same way without it).
    command = [
        "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info", "-nostdin",
        "-threads", "1", "-ss", f"{timestamp:.3f}", "-copyts", "-i", video_path,
        "-frames:v", "1", "-an",
        "-vf", f"showinfo,scale=min({width}\\,iw):-2,format=rgb24",
        "-f", "image2pipe", "-vcodec", "ppm", "pipe:1",
    ]
    stdout, stderr = await _run(command, args.command_timeout)
    frames = processor._parse_ppm_frames(stdout)
    times = processor._parse_showinfo_times(stderr)
    if not frames:
        return None
    return frames[0], (times[0] if times else None)


async def strategy_seek(processor, video_path, info, args):
    indices, scene_seconds, _ = await _candidates(processor, video_path, info, args)
    fps = float(info["fps"])
    times = [i / fps for i in indices]
    width = max(1, int(processor.config.ffmpeg_target_width))
    semaphore = asyncio.Semaphore(max(1, args.seek_concurrency))

    async def one(timestamp):
        async with semaphore:
            return await _seek_one(processor, video_path, timestamp, width, args)

    started = time.monotonic()
    results = await asyncio.gather(*(one(t) for t in times))
    extract_seconds = time.monotonic() - started
    kept = [(r[0], t, r[1]) for r, t in zip(results, times) if r is not None]
    # Positions survive _finish via the timestamp -> returned-pts map.
    returned_pts = {t: pts for _, t, pts in kept}
    frames, timestamps = _finish(
        processor, [f for f, _, _ in kept], [t for _, t, _ in kept]
    )
    return frames, timestamps, {
        "scene_seconds": scene_seconds,
        "extract_seconds": extract_seconds,
        "candidates": len(indices),
        "missing_seeks": len(times) - len(kept),
        "returned_source_pts": [returned_pts[t] for t in timestamps],
    }


async def strategy_single_pass(processor, video_path, info, args):
    """The production command and timestamp normalization; candidates are
    bounded by the select expression, not thinned."""
    fps = float(info["fps"])
    duration = processor._effective_keyframe_duration(
        frame_count=int(info["frame_count"]), fps=fps, duration=float(info["duration"])
    )
    command = processor._scene_candidates_command(video_path, duration)
    started = time.monotonic()
    stdout, stderr = await _run(command, args.command_timeout)
    extract_seconds = time.monotonic() - started
    frames = processor._parse_ppm_frames(stdout)
    times = processor._parse_showinfo_times(stderr)
    if len(times) != len(frames):
        raise RuntimeError(f"{len(frames)} frames but {len(times)} timestamps")
    relative: List[float] = []
    for value in times:
        timestamp = max(0.0, value - times[0])
        relative.append(max(timestamp, relative[-1]) if relative else timestamp)
    candidates = len(frames)
    frames, timestamps = _finish(processor, frames, relative)
    return frames, timestamps, {
        "scene_seconds": 0.0,
        "extract_seconds": extract_seconds,
        "candidates": candidates,
        "pipe_bytes": len(stdout),
    }


def _true_pts(
    pts: List[float], strategy: str, timestamps, details, first_source_pts: float
) -> List[float]:
    """Presentation time (relative to the first frame) of each returned frame."""
    if not pts:
        return []
    if strategy == "current":
        return [pts[min(i, len(pts) - 1)] for i in details["source_indices"]]
    if strategy == "seek":
        return [
            (value - first_source_pts) if value is not None else float("nan")
            for value in details["returned_source_pts"]
        ]
    # single_pass: distance to the nearest real frame.
    return [min(pts, key=lambda p: abs(p - t)) for t in timestamps]


def run_single(video_path: str, strategy: str, args) -> Dict[str, Any]:
    processor = _processor(args)
    runner = {
        "current": strategy_current,
        "seek": strategy_seek,
        "single_pass": strategy_single_pass,
    }[strategy]

    async def main():
        started = time.monotonic()
        info = await processor._probe_video_metadata(video_path)
        if info["duration"] > processor.max_video_length and info["fps"] > 0:
            info["frame_count"] = min(
                int(info["frame_count"]),
                int(processor.max_video_length * info["fps"]),
            )
        frames, timestamps, details = await runner(processor, video_path, info, args)
        return info, frames, timestamps, details, time.monotonic() - started

    info, frames, timestamps, details, wall = asyncio.run(main())
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    scale = 1 if sys.platform == "darwin" else 1024
    pts = _frame_pts(video_path, float(processor.max_video_length))
    true_pts = _true_pts(
        pts, strategy, timestamps, details, _first_source_pts(video_path)
    )
    errors = [abs(a - b) for a, b in zip(timestamps, true_pts)]
    signatures = [
        _signature(frame).tolist() for frame in frames
    ]
    details.pop("source_indices", None)
    details.pop("returned_source_pts", None)
    return {
        "video": Path(video_path).name,
        "strategy": strategy,
        "wall_seconds": round(wall, 3),
        # FFmpeg plus the RSS launchers and ffprobe.
        "child_cpu_seconds": round(child_usage.ru_utime + child_usage.ru_stime, 3),
        "peak_rss_python_mb": round(self_usage.ru_maxrss * scale / 2**20, 1),
        # Largest single FFmpeg process (concurrent seeks add up).
        "peak_rss_ffmpeg_mb": round(
            max(_FFMPEG_PEAK_RSS_KB or [0]) * scale / 2**20, 1
        ),
        "duration": info["duration"],
        "fps": info["fps"],
        "frames": len(frames),
        "timestamps": [round(t, 3) for t in timestamps],
        "max_timestamp_error_seconds": round(max(errors), 3) if errors else None,
        "signatures": signatures,
        **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in details.items()},
    }


def _signature(frame):
    import cv2
    import numpy as np

    luma = frame[:, :, :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114])
    return cv2.resize(luma, (16, 16), interpolation=cv2.INTER_AREA).round(1)


def _similarity_to_current(results: List[Dict[str, Any]]) -> None:
    """Mean absolute 16x16 luma difference per slot against ``current``."""
    import numpy as np

    by_video: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for result in results:
        by_video.setdefault(result["video"], {})[result["strategy"]] = result
    for strategies in by_video.values():
        baseline = strategies.get("current")
        for result in strategies.values():
            if baseline is None or "error" in result or "error" in baseline:
                continue
            a, b = baseline["signatures"], result["signatures"]
            if a and len(a) == len(b):
                diffs = [
                    float(np.mean(np.abs(np.array(x) - np.array(y))))
                    for x, y in zip(a, b)
                ]
                result["mean_luma_diff_vs_current"] = round(float(np.mean(diffs)), 2)
            else:
                result["mean_luma_diff_vs_current"] = None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", action="append", default=[], help="Video file")
    parser.add_argument("--samples-dir", type=Path)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--sample", action="append", choices=sorted(SAMPLES))
    parser.add_argument("--strategy", action="append", choices=STRATEGIES)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seek-concurrency", type=int, default=1)
    parser.add_argument(
        "--decode-threads", type=int, default=0,
        help="-threads N on full-decode commands (0: FFmpeg default, one per core)",
    )
    parser.add_argument("--command-timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--single", nargs=2, metavar=("VIDEO", "STRATEGY"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.single:
        try:
            result = run_single(args.single[0], args.single[1], args)
        except Exception as exc:  # reported, not raised: other runs continue
            result = {
                "video": Path(args.single[0]).name,
                "strategy": args.single[1],
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(result))
        return 0

    videos = [Path(v) for v in args.video]
    if args.generate:
        if args.samples_dir is None:
            parser.error("--generate needs --samples-dir")
        videos += generate_samples(args.samples_dir, args.sample or list(SAMPLES))
    elif args.samples_dir is not None:
        videos += sorted(args.samples_dir.glob("*.mp4"))
    if not videos:
        parser.error("no videos: pass --video, --samples-dir or --generate")

    results = []
    for video in videos:
        for strategy in args.strategy or STRATEGIES:
            for repeat in range(args.repeat):
                command = [
                    sys.executable, os.path.abspath(__file__),
                    "--single", str(video), strategy,
                    "--seek-concurrency", str(args.seek_concurrency),
                    "--decode-threads", str(args.decode_threads),
                    "--command-timeout", str(args.command_timeout),
                ]
                completed = subprocess.run(command, capture_output=True, text=True)
                try:
                    result = json.loads(completed.stdout.strip().splitlines()[-1])
                except (IndexError, ValueError):
                    # The child crashed before reporting; keep the other runs.
                    result = {
                        "video": video.name,
                        "strategy": strategy,
                        "error": f"exit {completed.returncode}: "
                        + completed.stderr.strip()[-500:],
                    }
                result["repeat"] = repeat
                results.append(result)
                print(
                    f"{result['video']:<26} {strategy:<12} "
                    + (
                        result["error"]
                        if "error" in result
                        else f"wall={result['wall_seconds']:>7.2f}s "
                        f"cpu={result['child_cpu_seconds']:>7.2f}s "
                        f"ffmpeg_rss={result['peak_rss_ffmpeg_mb']:>6.1f}MB "
                        f"frames={result['frames']:>2} "
                        f"ts_err={result['max_timestamp_error_seconds']}"
                    ),
                    file=sys.stderr,
                )
    _similarity_to_current(results)
    payload = {
        "cpu_count": os.cpu_count(),
        "seek_concurrency": args.seek_concurrency,
        "decode_threads": args.decode_threads,
        "ffmpeg": subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True
        ).stdout.splitlines()[0],
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
