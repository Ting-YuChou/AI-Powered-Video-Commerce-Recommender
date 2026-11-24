# Keyframe Extraction

The content worker turns each uploaded video into at most `MODEL_NUM_KEYFRAMES`
(16) RGB frames for CLIP, OCR and object detection
(`video_commerce/ml/content_processor.py`, `_extract_keyframes`).

## Pipeline

1. **ffprobe** reads duration, frame rate (`avg_frame_rate`, else
   `r_frame_rate`), frame count and whether there is audio. Videos longer than
   `MODEL_MAX_VIDEO_LENGTH` (300 s) are processed up to that length. A file
   with no duration (for example a WebM written by a browser recorder) is
   decoded up to that length too.
2. **Scene pass: one FFmpeg decode.** A `select` expression keeps:
   - the first frame;
   - any frame `MODEL_KEYFRAME_FLOOR_SECONDS` (8 s) after the previous pick;
   - scene changes (`MODEL_KEYFRAME_SCENE_THRESHOLD`) at least a minimum gap
     after the previous pick;
   - one frame in the last minimum gap (at least 1 s), so the end of the video
     is covered.

   The minimum gap is `MODEL_KEYFRAME_MIN_SCENE_GAP_SECONDS` (1 s), raised to
   `duration / (3 × num_keyframes)` for long videos, so a 300 s video yields at
   most about 50 candidates however often its scene changes. The candidates are
   scaled to `MODEL_FFMPEG_TARGET_WIDTH` and piped out as PPM. `showinfo`
   supplies their presentation times, measured from the first frame, so
   variable-frame-rate video gets correct timestamps.
3. **Quality filter** drops dark (`MODEL_KEYFRAME_MIN_LUMA`), blurry
   (`MODEL_KEYFRAME_MIN_BLUR_LAPLACIAN_VAR`) and near-duplicate
   (`MODEL_KEYFRAME_DEDUPE_LUMA_DIFF_THRESHOLD`) candidates. The survivors are
   then thinned evenly to `num_keyframes`.
4. **Uniform fallback.** If the scene pass fails, or no candidate survives the
   filter, the worker seeks to `num_keyframes` evenly spaced timestamps (one
   `ffmpeg -ss` per sample). A seek only decodes from the nearest keyframe, so
   this still works when a full decode is too slow.
   - A sample that cannot be decoded is dropped.
   - At the deadline, the samples decoded so far are kept.
   - `MODEL_KEYFRAME_SAMPLING_STRATEGY=uniform` uses this path directly.
5. **OpenCV** runs when any of these holds:
   - `MODEL_FFMPEG_FRAME_EXTRACTION_ENABLED=false`;
   - ffprobe cannot read the file;
   - FFmpeg produced no frames.

   The opencv-python wheel bundles its own FFmpeg build, so it sometimes
   decodes what the system FFmpeg cannot. When it cannot decode the file
   either, it reports `InvalidVideoError`. That error is not retryable, so a
   corrupt upload is dead-lettered at once.

   When it runs after ffprobe, the ffprobe metadata (duration, audio) is kept.
   OpenCV runs in a child process (`video_commerce/ml/opencv_keyframes.py`,
   started with `python -I`, result written to a file), so it can be killed at
   the deadline and does not occupy the thread pool CLIP and OCR use.

Earlier versions ran a separate scene-detection decode and then decoded again
up to the last candidate (`select=eq(n,idx)`). They fell back to a third full
decode and then to OpenCV in a thread. Timestamps were `index / avg_fps`, which
is wrong for variable-frame-rate video.

## Time Budget

| Setting | Default | Applies to |
|---|---|---|
| `MODEL_KEYFRAME_EXTRACTION_DEADLINE_SECONDS` | 150 | All of extraction, including fallbacks. Each command's timeout is capped by what remains. |
| `MODEL_FFMPEG_TIMEOUT_SECONDS` | 30 | ffprobe, each seek, and audio extraction (300 s of audio takes about 1 s) |
| `MODEL_FFMPEG_TIMEOUT_PER_MEDIA_SECOND` | 0.5 | Added per second of video for commands that decode the whole video (the scene pass and OpenCV) |
| `MODEL_FFMPEG_DECODE_THREADS` | 1 | Decoder threads per command. 0 means one per visible core, which ignores container CPU quotas. |

- **Fallback reserve:** the scene pass stops in time to leave 20% of the
  deadline for the uniform fallback.
- **Process cleanup:** a command that times out, or whose task is cancelled, is
  killed, so abandoned FFmpeg processes never keep running.
- **Startup check:** the content worker refuses to start if one worst-case
  processing attempt cannot fit in 90% of `KAFKA_CONSUMER_MAX_POLL_INTERVAL_MS`.
  - **Why 90%:** that is the consumer's retry budget.
  - **The worst case adds up:**
    1. the extraction deadline;
    2. `MODEL_FFMPEG_TIMEOUT_SECONDS` for audio;
    3. the ASR timeout times its attempts (when ASR is enabled);
    4. 60 s for download, inference and persistence.
  - **Defaults:** 240 s with ASR off and 480 s with ASR on, against a 540 s
    budget.
  - **Limits:** the 60 s is an allowance, not an enforced timeout (OCR has
    none). Other services skip the check, so a content-only setting cannot stop
    them.
- **Where to set these:**
  - Compose passes all four settings (and their defaults) to the
    `content-worker` service.
  - In Helm, set them under `backend.workloads.contentWorker.env`.

When the deadline is exceeded, extraction raises
`KeyframeExtractionDeadlineExceeded`, a `TimeoutError`. It is retryable: the
consumer retries within its budget and then dead-letters the message.

## Benchmark

`scripts/benchmark_keyframe_extraction.py` compares three strategies on a set of
videos (each run in its own process, child failures recorded rather than
aborting the run):
- `current`: the pre-change two-decode method, kept as a frozen baseline.
- `seek`: scene detection, then one seek per candidate.
- `single_pass`: the production method.

The script records:
- wall time and child-process CPU time (FFmpeg plus the launchers and ffprobe);
- FFmpeg's own peak RSS;
- how far each reported timestamp is from the frame's true presentation time;
- a per-slot luma difference against `current`.

Run it under the worker's CPU limit:

```bash
docker run --rm --cpus=1 -v "$PWD:/app" -v /tmp/kf-samples:/samples -w /app \
  -e PYTHONPATH=/app vc-backend-tests python scripts/benchmark_keyframe_extraction.py \
  --generate --samples-dir /samples --decode-threads 1 --output /samples/kf.json
```

`--generate` encodes synthetic H.264 samples: a moving test pattern with a hard
cut every 5 s, temporal noise, and 720p or 1080p at 30 fps. The set covers
long GOPs, variable frame rate, and videos with and without audio. Pass real
uploads with `--video` to check frame selection on real footage.

### Results (2026-11-24)

**Setup:**
- Docker with `--cpus=1` (the content worker's Compose limit).
- FFmpeg 7.1.5 with `-threads 1`.
- One run per cell; wall time in seconds.
- The `single_pass` column is the final production command.

| Sample | `current` | `seek` | `single_pass` |
|---|---|---|---|
| 15 s, 1080p, 8 Mbps | 5.0 | 6.2 | 2.6 |
| 60 s, 1080p, 8 Mbps | 20.0 | 22.4 | 10.2 |
| 60 s, 720p, 8 Mbps, no audio | 17.7 | 13.9 | 8.2 |
| 60 s, 1080p, VFR (10 fps for 20 s) | 18.2 | 26.2 | 9.4 |
| 300 s, 720p, 2.5 Mbps (fits the 100 MiB limit) | 37.2 | 30.5 | 17.3 |
| 300 s, 1080p, 8 Mbps (305 MB, stress case) | 96.1 | 81.8 | 50.2 |
| 120 s, 1080p, a single keyframe | 39.1 | 297.0 | 18.1 |

- **Speed:** `single_pass` was 1.9–2.2× faster than `current` on every sample.
- **Seek scales with GOP length:** `seek` was faster than `current` only on
  some samples. Each seek decodes from the previous keyframe, so on the
  single-keyframe file it took 297 s.
- **Timestamps on VFR** (error against the file's own frame times, measured as
  described above):
  - `current` was up to 5.55 s off on the VFR sample, and only 9 of 16 frames
    survived deduplication, because `index / avg_fps` pointed at the wrong
    frames.
  - `seek` was within 0.09 s (the first frame at or after the seek point).
  - `single_pass` had zero error on every sample: each reported timestamp is a
    real frame time.
- **FFmpeg peak RSS:**
  - With `-threads 1`: 76–95 MB per process.
  - With FFmpeg's default thread count (11 visible cores in the container),
    300 s 1080p was slower and needed twice the memory: `current` took 119 s
    at 189 MB (versus 96 s at 88 MB), and `single_pass` took 72 s at 190 MB.
- **Candidate memory:** the bounded gap keeps a 300 s video to 41 candidates,
  27 MB of 640×360 PPM. Portrait video at the same width is about 3× that per
  frame. The old method held up to 48 candidates of the same size.
- **Main's behaviour at the old 30 s per-command timeout:** main ran FFmpeg with
  default threads. Its scene-detection pass alone took 65 s on 300 s of 1080p
  (about 0.22 s per second of video). At 1 CPU, 1080p uploads longer than about
  two minutes therefore timed out:
  1. first the scene pass;
  2. then the uniform full decode;
  3. and finally fell back to OpenCV in a thread.

**Not measured:**
- **Selection quality on real footage.** Frame selection differs from `current`
  on long videos, because a bounded minimum gap replaces even thinning of up to
  48 candidates. On the 300 s samples, the per-slot luma difference is large
  because the synthetic content inverts at every cut. That is not a quality
  signal. Run the script with `--video` on real uploads (and compare CLIP
  embeddings) before tuning the defaults.
- **The uniform fallback on long-GOP files.** It needs one seek per sample. On a
  file with one keyframe per minute or more, 16 seeks do not fit in the 30 s
  reserve. The frames decoded by the deadline are kept, so such a file ends up
  with fewer frames; it fails only if no seek finished. The scene pass itself
  handles these files (18 s above), so this matters only when the scene pass
  fails. `MODEL_KEYFRAME_SAMPLING_STRATEGY=uniform` always takes this path.
- **GPUs and CPU limits above 1 core.**

### Pipeline Version

`MODEL_CONTENT_PIPELINE_VERSION` is unchanged. The feature schema and models are
the same. Only which frames are sampled changes, and timestamps on VFR video
become correct. Existing content features stay valid. Bumping the version would
require deploying the gateway and the workers together for no contract change.
