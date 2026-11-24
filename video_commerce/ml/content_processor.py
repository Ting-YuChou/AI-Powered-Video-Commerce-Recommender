"""
AI-Powered Video Commerce Recommender - Content Processor
=========================================================

This module handles multi-modal video content processing using CLIP for visual understanding,
OCR for text extraction, and basic audio processing. It extracts rich features from video
content that can be used for product recommendations.
"""

import cv2
import numpy as np
import torch
import asyncio
import contextvars
import json
import logging
import re
import sys
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image, ImageEnhance
import io
import base64
import time
from pathlib import Path
import tempfile
import os
import httpx

# ML/AI imports
from transformers import CLIPModel, CLIPProcessor
import pytesseract
from pytesseract import Output

# Local imports
from video_commerce.common.models import AudioFeatures, ContentFeatures
from video_commerce.common.config import ModelConfig
from video_commerce.ml import opencv_keyframes
from video_commerce.ml.opencv_keyframes import InvalidVideoError  # noqa: F401 (re-export)
from video_commerce.ml.text_embeddings import MultilingualTextEmbedder
from video_commerce.ml.visual_retrieval import load_visual_retrieval_checkpoint

logger = logging.getLogger(__name__)


ZERO_NORM_EPSILON = 1e-6

# Monotonic deadline for the keyframe extraction in progress (None outside it).
# Every media command's timeout is capped by what remains, so scene detection,
# frame extraction and fallbacks share one budget.
_keyframe_extraction_deadline: contextvars.ContextVar[Optional[float]] = (
    contextvars.ContextVar("keyframe_extraction_deadline", default=None)
)


class KeyframeExtractionDeadlineExceeded(TimeoutError):
    """Keyframe extraction used up MODEL_KEYFRAME_EXTRACTION_DEADLINE_SECONDS."""


_SHOWINFO_FRAME_LINE = re.compile(
    r"(?:^|\r)\[Parsed_showinfo_\d+ @ [^\]]*\]\s+n:\s*\d+\s+pts:\s*-?\d+\s+"
    r"pts_time:(-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)",
    re.MULTILINE,
)

# Share of the extraction deadline the scene pass leaves for the uniform
# fallback, so a scene pass that runs out of time still leaves room for seeks.
UNIFORM_FALLBACK_DEADLINE_SHARE = 0.2


class ContentModalityError(RuntimeError):
    """A required modality failed; the task is retryable (never silently zeroed)."""

    def __init__(self, modality: str, message: str):
        super().__init__(f"{modality} extraction failed: {message}")
        self.modality = modality


def embedding_has_signal(embedding: Any) -> bool:
    """True when an embedding has a non-trivial L2 norm."""
    try:
        values = np.asarray(embedding, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    if values.size == 0 or not np.all(np.isfinite(values)):
        return False
    return float(np.linalg.norm(values)) > ZERO_NORM_EPSILON


class ContentProcessor:
    """
    Multi-modal content processor for video commerce recommendations.

    Processes video content to extract:
    - Visual features using CLIP
    - Text features using OCR
    - Basic audio features
    - Object detection and scene analysis
    """

    def __init__(self, config: ModelConfig):
        """Initialize the content processor with configuration."""
        self.config = config
        self.clip_model = None
        self.clip_processor = None
        self.ocr_region_extractor = None
        self.text_embedder = None
        self.retrieval_pooler = None
        self.device = torch.device(self._get_device())
        self.is_initialized = False
        self.lazy_load = os.getenv("MODEL_LAZY_LOAD", "false").lower() == "true"

        # Processing parameters
        self.max_keyframes = config.num_keyframes
        self.max_video_length = config.max_video_length
        self.batch_size = config.batch_size

        # OCR configuration
        self.ocr_config = r"--oem 3 --psm 6"

        # Commerce-related keywords for filtering
        self.commerce_keywords = {
            "price_indicators": [
                "$",
                "€",
                "£",
                "¥",
                "price",
                "cost",
                "buy",
                "sale",
                "discount",
                "%",
                "off",
                "價格",
                "特價",
                "折扣",
            ],
            "product_keywords": [
                "product",
                "item",
                "model",
                "brand",
                "new",
                "latest",
                "review",
                "產品",
                "商品",
                "品牌",
                "新品",
                "開箱",
            ],
            "action_words": [
                "buy",
                "purchase",
                "order",
                "shop",
                "get",
                "available",
                "stock",
                "購買",
                "下單",
                "現貨",
            ],
        }
        self.category_keywords = {
            "electronics": [
                "electronics",
                "phone",
                "smartphone",
                "headphone",
                "headphones",
                "earbud",
                "earbuds",
                "laptop",
                "camera",
                "tablet",
                "speaker",
                "手機",
                "耳機",
                "筆電",
                "電腦",
                "电脑",
                "相機",
                "相机",
                "平板",
                "音響",
            ],
            "fashion": [
                "clothing",
                "shirt",
                "dress",
                "shoes",
                "sneakers",
                "jacket",
                "bag",
                "watch",
                "jewelry",
                "衣服",
                "上衣",
                "洋裝",
                "连衣裙",
                "鞋",
                "球鞋",
                "外套",
                "包包",
                "手錶",
                "手表",
                "珠寶",
                "珠宝",
            ],
            "home": [
                "furniture",
                "kitchen",
                "home decor",
                "sofa",
                "chair",
                "家具",
                "廚房",
                "厨房",
                "家飾",
                "家饰",
                "沙發",
                "沙发",
                "椅子",
            ],
            "beauty": [
                "cosmetics",
                "skincare",
                "perfume",
                "makeup",
                "化妝品",
                "化妆品",
                "保養",
                "保养",
                "香水",
                "彩妝",
                "彩妆",
            ],
            "books": ["book", "books", "novel", "書", "书", "小說", "小说"],
            "toys": ["toy", "gaming", "game", "玩具", "遊戲", "游戏"],
        }

        logger.info(f"ContentProcessor initialized with device: {self.device}")

    def _get_device(self) -> str:
        """Determine the best device for processing."""
        if self.config.device == "auto":
            if torch.cuda.is_available() and self.config.enable_gpu:
                return "cuda"
            else:
                return "cpu"
        return self.config.device

    async def load_models(self):
        """Load all required ML models."""
        try:
            logger.info(f"Loading CLIP model: {self.config.clip_model}")

            # Load CLIP model and processor
            self.clip_model = CLIPModel.from_pretrained(
                self.config.clip_model,
                revision=self.config.clip_revision,
                cache_dir=self.config.cache_dir,
            ).to(self.device)

            self.clip_processor = CLIPProcessor.from_pretrained(
                self.config.clip_model,
                revision=self.config.clip_revision,
                cache_dir=self.config.cache_dir,
            )

            self.text_embedder = MultilingualTextEmbedder(
                model_id=self.config.multilingual_text_model,
                revision=self.config.multilingual_text_revision,
                device=self.device,
                cache_dir=self.config.cache_dir,
            )
            self.text_embedder.load()

            checkpoint_path = Path(self.config.retrieval_visual_checkpoint_path)
            if (
                (
                    self.config.retrieval_visual_attention_shadow
                    or self.config.retrieval_visual_attention_enabled
                )
                and checkpoint_path.exists()
            ):
                try:
                    expected_sha256 = (
                        self.config.retrieval_visual_checkpoint_sha256 or None
                    )
                    pooler, metadata = load_visual_retrieval_checkpoint(
                        checkpoint_path,
                        expected_sha256=expected_sha256,
                        map_location=self.device,
                    )
                    lineage = metadata["lineage"]
                    if (
                        lineage["clip_model_id"] != self.config.clip_model
                        or lineage["clip_revision"] != self.config.clip_revision
                        or metadata["model_version"]
                        != self.config.retrieval_visual_model_version
                        or lineage["product_index_version"]
                        != self.config.retrieval_visual_product_index_version
                    ):
                        raise ValueError(
                            "visual retrieval checkpoint serving lineage mismatch"
                        )
                    self.retrieval_pooler = pooler.to(self.device).eval()
                except (OSError, ValueError):
                    if self.config.retrieval_visual_attention_enabled:
                        raise
                    logger.warning(
                        "Visual retrieval shadow checkpoint is incompatible; "
                        "continuing with v2 mean-pooled features",
                        exc_info=True,
                    )

            # Set to evaluation mode
            self.clip_model.eval()

            if self.config.ocr_backend == "paddle":
                from video_commerce.ml.video_ocr import PaddleOCRRegionExtractor

                self.ocr_region_extractor = PaddleOCRRegionExtractor(
                    language=self.config.ocr_language,
                    detection_threshold=self.config.ocr_detection_threshold,
                    recognition_threshold=self.config.ocr_recognition_threshold,
                )

            # Enable quantization if configured
            if self.config.enable_quantization and self.device == "cpu":
                self.clip_model = torch.quantization.quantize_dynamic(
                    self.clip_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Model quantization enabled")

            self.is_initialized = True
            logger.info("CLIP model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    async def process_video(
        self, video_path: str, content_id: Optional[str] = None
    ) -> ContentFeatures:
        """
        Process a video file and extract multi-modal features.

        Args:
            video_path: Path to the video file
            content_id: Optional content identifier

        Returns:
            ContentFeatures object with extracted features
        """
        if not self.is_initialized:
            await self.load_models()

        start_time = time.time()

        try:
            logger.info(f"Processing video: {video_path}")

            # Generate content ID if not provided
            if content_id is None:
                content_id = f"content_{int(time.time())}"

            # Extract keyframes from video
            keyframes, video_info = await self._extract_keyframes(video_path)

            if not keyframes:
                raise InvalidVideoError("No keyframes could be extracted from video")

            # Model inference runs in worker threads, one stage at a time, so the
            # event loop keeps renewing the processing lease and heartbeats.
            # CLIP is required: a failure raises ContentModalityError (retryable).
            visual_features = await self._extract_visual_features(keyframes)
            modality_status = {
                "clip": (
                    "completed"
                    if embedding_has_signal(visual_features["embedding"])
                    else "zero_norm"
                )
            }

            # OCR is optional: failures are recorded and processing continues.
            frame_timestamps = self._frame_timestamps(keyframes, video_info)
            text_features = await self._extract_text_features(
                keyframes, frame_timestamps
            )
            modality_status["ocr"] = str(text_features.get("status") or "completed")

            # Extract audio speech without making it a hard dependency for the job.
            audio_features = await self._extract_audio_features(video_path, video_info)

            # Basic object detection using CLIP
            detected_objects, detection_status = await self._detect_objects_with_status(
                keyframes
            )
            modality_status["object_detection"] = detection_status

            # Analyze commerce-related content
            commerce_features = self._analyze_commerce_content(
                text_features,
                detected_objects,
                audio_features.audio_transcript or "",
            )
            audio_features.speech_categories = commerce_features.get(
                "speech_categories", []
            )

            processing_time = time.time() - start_time

            # Create ContentFeatures object
            retrieval_fields = await asyncio.to_thread(
                self._retrieval_visual_fields,
                visual_features.get("individual_embeddings", []),
                frame_timestamps,
            )
            content_features = ContentFeatures(
                content_id=content_id,
                visual_embedding=visual_features["embedding"].tolist(),
                frame_embeddings=visual_features.get("individual_embeddings", []),
                frame_timestamps_seconds=frame_timestamps,
                ocr_tracks=text_features.get("ocr_tracks", []),
                modality_status=modality_status,
                text_embedding_model=self.config.multilingual_text_model,
                text_embedding_revision=self.config.multilingual_text_revision,
                duration_seconds=video_info.get("duration", 0),
                detected_objects=detected_objects,
                extracted_text=text_features.get("text_blocks", []),
                product_mentions=commerce_features.get("product_mentions", []),
                category_scores=commerce_features.get("category_scores", {}),
                processing_time=processing_time,
                audio_features=audio_features,
                text_features={
                    "total_text_regions": len(text_features.get("text_blocks", [])),
                    "commerce_score": commerce_features.get("commerce_score", 0.0),
                    "price_mentions": text_features.get("price_mentions", []),
                    "ocr_categories": commerce_features.get("ocr_categories", []),
                },
                **retrieval_fields,
            )

            logger.info(
                f"Video processing completed in {processing_time:.2f}s for {content_id}"
            )
            return content_features

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            raise

    def _retrieval_visual_fields(
        self,
        frame_embeddings: List[List[float]],
        frame_timestamps_seconds: List[float],
    ) -> Dict[str, Any]:
        """Pool existing frozen CLIP frames; never re-decode video for v3."""
        if self.retrieval_pooler is None or not frame_embeddings:
            return {}
        if len(frame_embeddings) != len(frame_timestamps_seconds):
            raise ValueError("retrieval frame embeddings and timestamps are misaligned")
        frames = torch.as_tensor(
            frame_embeddings[:16], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        starts = torch.as_tensor(
            frame_timestamps_seconds[:16],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask = torch.ones(starts.shape, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            output = self.retrieval_pooler(frames, starts, mask)
        return {
            "multimodal_schema_version": "temporal_multimodal_v3",
            "retrieval_visual_embedding": output.embedding[0].cpu().tolist(),
            "retrieval_model_version": self.config.retrieval_visual_model_version,
            "retrieval_product_index_version": (
                self.config.retrieval_visual_product_index_version
            ),
            "retrieval_clip_model": self.config.clip_model,
            "retrieval_clip_revision": self.config.clip_revision,
        }

    async def _extract_keyframes(
        self, video_path: str
    ) -> Tuple[List[np.ndarray], Dict]:
        """Extract keyframes within MODEL_KEYFRAME_EXTRACTION_DEADLINE_SECONDS.

        FFmpeg (scene pass, then uniform seeks) runs first. OpenCV, in a child
        process bounded by the same deadline, runs when FFmpeg extraction is
        disabled, ffprobe cannot read the file, or FFmpeg produced no frames.
        The opencv-python wheel bundles its own FFmpeg build, so it sometimes
        decodes what the system FFmpeg cannot. When it also fails it reports
        ``InvalidVideoError`` (not retryable), so a corrupt upload is
        dead-lettered at once instead of using the whole retry budget.
        Running out of time raises ``KeyframeExtractionDeadlineExceeded``.
        """
        deadline = time.monotonic() + float(
            self.config.keyframe_extraction_deadline_seconds
        )
        token = _keyframe_extraction_deadline.set(deadline)
        try:
            if not self.config.ffmpeg_frame_extraction_enabled:
                return await self._extract_keyframes_opencv(video_path)
            try:
                video_info = await self._probe_video_metadata(video_path)
            except KeyframeExtractionDeadlineExceeded:
                raise
            except Exception as exc:
                logger.warning(
                    "ffprobe could not read the video; falling back to OpenCV: %s",
                    exc,
                )
                return await self._extract_keyframes_opencv(video_path)
            try:
                keyframes, video_info = await self._extract_keyframes_ffmpeg(
                    video_path, video_info
                )
            except KeyframeExtractionDeadlineExceeded:
                raise
            except Exception as exc:
                logger.warning(
                    "FFmpeg keyframe extraction failed; trying OpenCV: %s", exc
                )
                keyframes = []
            if keyframes:
                logger.info(
                    "Extracted %s keyframes from video using FFmpeg", len(keyframes)
                )
                return keyframes, video_info
            logger.warning("FFmpeg extracted no keyframes; trying OpenCV")
            keyframes, opencv_info = await self._extract_keyframes_opencv(video_path)
            # Keep ffprobe's metadata (duration, audio); OpenCV supplies frames.
            video_info.pop("sampled_frame_indices", None)
            video_info["frame_timestamps_seconds"] = opencv_info.get(
                "frame_timestamps_seconds", []
            )
            return keyframes, video_info
        finally:
            _keyframe_extraction_deadline.reset(token)

    async def _extract_keyframes_ffmpeg(
        self, video_path: str, video_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[np.ndarray], Dict]:
        """Extract RGB keyframes using ffprobe and FFmpeg."""
        if video_info is None:
            video_info = await self._probe_video_metadata(video_path)
        frame_count = int(video_info.get("frame_count") or 0)
        fps = float(video_info.get("fps") or 0)
        duration = float(video_info.get("duration") or 0)

        if frame_count <= 0 and duration > 0 and fps > 0:
            frame_count = int(duration * fps)

        if duration > self.max_video_length and fps > 0:
            logger.warning(
                "Video duration %ss exceeds limit %ss",
                duration,
                self.max_video_length,
            )
            frame_count = min(frame_count, int(self.max_video_length * fps))
            video_info["frame_count"] = frame_count

        if self.config.keyframe_sampling_strategy == "uniform":
            return await self._extract_uniform_keyframes_ffmpeg(
                video_path,
                video_info,
                frame_count,
            )

        try:
            (
                keyframes,
                frame_timestamps,
            ) = await self._extract_scene_adaptive_keyframes_with_timestamps_ffmpeg(
                video_path,
                frame_count,
                fps,
                duration,
            )
            if keyframes:
                video_info["frame_timestamps_seconds"] = frame_timestamps
                return keyframes, video_info
            logger.warning(
                "Scene-adaptive FFmpeg extraction returned no usable frames; "
                "falling back to uniform FFmpeg extraction"
            )
        except KeyframeExtractionDeadlineExceeded:
            raise
        except Exception as exc:
            logger.warning(
                "Scene-adaptive FFmpeg extraction failed; "
                "falling back to uniform FFmpeg extraction: %s",
                exc,
            )

        return await self._extract_uniform_keyframes_ffmpeg(
            video_path,
            video_info,
            frame_count,
        )

    async def _extract_uniform_keyframes_ffmpeg(
        self,
        video_path: str,
        video_info: Dict[str, Any],
        frame_count: int,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Uniformly sampled RGB keyframes, one input seek per sample.

        Seeking decodes only from the nearest keyframe, so this stays cheap
        when the scene pass failed because a full decode was too slow.
        """
        frame_indices = self._keyframe_indices(frame_count)
        fps = float(video_info.get("fps") or 0.0)
        duration = float(video_info.get("duration") or 0.0)
        if fps > 0:
            timestamps = [float(index) / fps for index in frame_indices]
        elif duration > 0 and frame_count > 0:
            timestamps = [float(index) * duration / frame_count for index in frame_indices]
        else:
            timestamps = []
        if not timestamps:
            return [], video_info
        sampled = await self._extract_ffmpeg_frames_at_times(video_path, timestamps)
        video_info["frame_timestamps_seconds"] = [ts for ts, _ in sampled]
        video_info.pop("sampled_frame_indices", None)
        return [frame for _, frame in sampled], video_info

    async def _extract_ffmpeg_frames_at_times(
        self,
        video_path: str,
        timestamps: List[float],
    ) -> List[Tuple[float, np.ndarray]]:
        """Decode one frame at (or just after) each timestamp.

        Returns ``(timestamp, frame)`` for the samples that decoded. A seek at
        the very end of a stream can legitimately return no frame, and one bad
        seek should not discard the rest, so failures drop that sample. If
        every seek fails, the last error is raised. At the extraction deadline
        the samples decoded so far are returned (raising only if there are
        none), so a slow file degrades to fewer frames instead of failing the
        same way on every retry.
        """
        target_width = max(1, int(self.config.ffmpeg_target_width))
        sampled: List[Tuple[float, np.ndarray]] = []
        last_error: Optional[Exception] = None
        for timestamp in timestamps:
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-ss",
                f"{max(0.0, timestamp):.3f}",
                *self._ffmpeg_input(video_path),
                "-frames:v",
                "1",
                "-an",
                "-sn",
                "-dn",
                "-vf",
                f"scale=min({target_width}\\,iw):-2,format=rgb24",
                "-f",
                "image2pipe",
                "-vcodec",
                "ppm",
                "pipe:1",
            ]
            try:
                frames = self._parse_ppm_frames(await self._run_media_command(command))
            except KeyframeExtractionDeadlineExceeded:
                if not sampled:
                    raise
                logger.warning(
                    "Keyframe extraction deadline reached after %s of %s seeks",
                    len(sampled),
                    len(timestamps),
                )
                return sampled
            except Exception as exc:
                last_error = exc
                logger.warning("FFmpeg seek to %.3fs failed: %s", timestamp, exc)
                continue
            if frames:
                sampled.append((float(timestamp), frames[0]))
        if not sampled and last_error is not None:
            raise last_error
        return sampled

    async def _extract_scene_adaptive_keyframes_with_timestamps_ffmpeg(
        self,
        video_path: str,
        frame_count: int,
        fps: float,
        duration: float,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Scene-aware keyframes from one decode, with source timestamps.

        One FFmpeg pass selects scene-change and floor candidates, then the
        quality filter drops dark, blurry and duplicate frames and the rest are
        thinned to ``num_keyframes``. (This replaced a scene-detection pass
        followed by a second decode up to the last candidate.)
        """
        # Only the duration matters here; fps and frame_count may be unknown.
        if self.max_keyframes <= 0:
            return [], []

        effective_duration = self._effective_keyframe_duration(
            frame_count=frame_count,
            fps=fps,
            duration=duration,
        )
        candidates, timestamps = await self._extract_scene_candidates_ffmpeg(
            video_path, effective_duration
        )
        retained_positions = self._quality_keyframe_positions(candidates)
        sampled = [
            (candidates[position], timestamps[position])
            for position in retained_positions
        ]
        sampled = self._thin_ordered_items(sampled, self.max_keyframes)
        return [frame for frame, _ in sampled], [ts for _, ts in sampled]

    def _scene_candidate_select_expression(self, effective_duration: float) -> str:
        """FFmpeg ``select`` expression for scene-change and floor candidates.

        Selects the first frame; any frame ``floor`` seconds after the previous
        selection; scene changes at least ``min_gap`` after it; and one frame in
        the last ``min_gap`` (at least 1 s) of the processed span, so the end of
        the video is covered as the first frame is. Raising ``min_gap`` to
        ``duration / candidate_limit`` bounds the output to about
        ``candidate_limit`` frames however often the scene changes, which bounds
        the memory the decoded candidates take.
        """
        candidate_limit = max(1, self.max_keyframes * 3)
        min_gap = max(
            float(self.config.keyframe_min_scene_gap_seconds),
            float(effective_duration) / candidate_limit,
        )
        floor = max(float(self.config.keyframe_floor_seconds), min_gap)
        threshold = float(self.config.keyframe_scene_threshold)
        tail_start = max(0.0, float(effective_duration) - max(min_gap, 1.0))
        since = "t-prev_selected_t"
        spaced = f"gte({since}\\,{min_gap:.6f})"
        return (
            "gt(isnan(prev_selected_t)"
            f"+gte({since}\\,{floor:.6f})"
            f"+gt(scene\\,{threshold:.6f})*{spaced}"
            f"+gte(t\\,{tail_start:.6f})*{spaced}"
            "\\,0)"
        )

    def _scene_candidates_command(
        self, video_path: str, effective_duration: float
    ) -> List[str]:
        """The scene-pass FFmpeg command (also run by the benchmark script).

        ``-t`` is an input option: as an output option FFmpeg trims after
        ``select,showinfo``, so ``showinfo`` would log a frame the trim drops.
        """
        target_width = max(1, int(self.config.ffmpeg_target_width))
        filter_graph = (
            f"select={self._scene_candidate_select_expression(effective_duration)},"
            "showinfo,"
            f"scale=min({target_width}\\,iw):-2,"
            "format=rgb24"
        )
        return [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "info",
            "-nostdin",
            *self._ffmpeg_input(video_path, duration=effective_duration),
            "-an",
            "-sn",
            "-dn",
            "-vf",
            filter_graph,
            "-vsync",
            "vfr",
            "-f",
            "image2pipe",
            "-vcodec",
            "ppm",
            "pipe:1",
        ]

    async def _extract_scene_candidates_ffmpeg(
        self,
        video_path: str,
        effective_duration: float,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Decode once and return candidate frames with their timestamps.

        Timestamps come from ``showinfo`` (the frames' own presentation times,
        so variable-frame-rate video is placed correctly) relative to the first
        frame, which is always selected.
        """
        command = self._scene_candidates_command(video_path, effective_duration)
        stdout, stderr = await self._run_media_command_with_stderr(
            command, timeout=self._scene_pass_timeout(effective_duration)
        )
        frames = self._parse_ppm_frames(stdout)
        presentation_times = self._parse_showinfo_times(stderr)
        if len(presentation_times) != len(frames):
            raise ValueError(
                f"FFmpeg returned {len(frames)} candidate frames but "
                f"{len(presentation_times)} timestamps"
            )
        if not frames:
            return [], []
        origin = presentation_times[0]
        timestamps: List[float] = []
        for value in presentation_times:
            # Monotonic and non-negative, as the OCR tracker and pooler expect.
            timestamp = max(0.0, value - origin)
            if timestamps and timestamp < timestamps[-1]:
                timestamp = timestamps[-1]
            timestamps.append(timestamp)
        return frames, timestamps

    @staticmethod
    def _parse_showinfo_times(payload: bytes) -> List[float]:
        """``pts_time`` of each frame ``showinfo`` logged to stderr.

        Anchored to showinfo's per-frame line (``[Parsed_showinfo_N @ 0x..] n:
        pts: pts_time:``) so container metadata that happens to contain
        ``pts_time:`` is not counted. ``NOPTS`` frames are not matched.
        """
        text = payload.decode("utf-8", errors="replace")
        return [
            float(match.group(1))
            for match in _SHOWINFO_FRAME_LINE.finditer(text)
        ]

    def _effective_keyframe_duration(
        self,
        *,
        frame_count: int,
        fps: float,
        duration: float,
    ) -> float:
        """Seconds of video to sample, capped at ``max_video_length``.

        Uses the probed duration, else ``frame_count / fps``. When neither is
        known (a WebM/MKV written to a pipe reports no duration), the whole
        allowed length is decoded; FFmpeg stops at the end of the file anyway.
        """
        limit = float(self.max_video_length)
        if duration > 0:
            return min(float(duration), limit)
        if fps > 0 and frame_count > 0:
            return min(float(frame_count) / float(fps), limit)
        return limit

    def _thin_ordered_items(self, items: List[Any], limit: int) -> List[Any]:
        if limit <= 0:
            return []
        if len(items) <= limit:
            return items

        positions = np.linspace(
            0,
            len(items) - 1,
            limit,
            dtype=int,
        )
        return [items[int(position)] for position in positions]

    def _filter_keyframes_for_quality(
        self,
        keyframes: List[np.ndarray],
    ) -> List[np.ndarray]:
        return [
            keyframes[position]
            for position in self._quality_keyframe_positions(keyframes)
        ]

    def _quality_keyframe_positions(
        self,
        keyframes: List[np.ndarray],
    ) -> List[int]:
        retained: List[int] = []
        previous_luma: Optional[np.ndarray] = None

        for position, frame in enumerate(keyframes):
            luma = self._frame_luma(frame)
            if luma.size == 0:
                continue
            if float(np.mean(luma)) < float(self.config.keyframe_min_luma):
                continue
            if (
                self.config.keyframe_min_blur_laplacian_var > 0
                and min(luma.shape[:2]) >= 3
                and cv2.Laplacian(luma.astype(np.uint8), cv2.CV_64F).var()
                < float(self.config.keyframe_min_blur_laplacian_var)
            ):
                continue

            luma_signature = cv2.resize(
                luma,
                (32, 32),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            if previous_luma is not None:
                diff = float(np.mean(np.abs(luma_signature - previous_luma)))
                if diff < float(self.config.keyframe_dedupe_luma_diff_threshold):
                    continue

            retained.append(position)
            previous_luma = luma_signature

        return retained

    def _frame_luma(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] < 3:
            return np.array([], dtype=np.float32)
        rgb = frame[:, :, :3].astype(np.float32)
        return rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114

    async def _probe_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Read video metadata using ffprobe."""
        command = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            video_path,
        ]
        output = await self._run_media_command(command)
        payload = json.loads(output.decode("utf-8"))
        streams = payload.get("streams") or []
        video_stream = next(
            (stream for stream in streams if stream.get("codec_type") == "video"),
            None,
        )
        if not video_stream:
            raise ValueError("ffprobe did not report a video stream")

        format_info = payload.get("format") or {}
        # avg_frame_rate is "0/0" (truthy, parses to 0) for some streams.
        fps = self._parse_frame_rate(video_stream.get("avg_frame_rate"))
        if fps <= 0:
            fps = self._parse_frame_rate(video_stream.get("r_frame_rate"))
        duration = self._parse_float(
            video_stream.get("duration"),
            default=self._parse_float(format_info.get("duration"), default=0.0),
        )
        frame_count = self._parse_int(video_stream.get("nb_frames"), default=0)
        if frame_count <= 0 and duration > 0 and fps > 0:
            frame_count = int(duration * fps)

        return {
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
            "width": self._parse_int(video_stream.get("width"), default=0),
            "height": self._parse_int(video_stream.get("height"), default=0),
        }

    def _ffmpeg_input(
        self, video_path: str, *, duration: Optional[float] = None
    ) -> List[str]:
        """Input options (decoder threads, read duration) and the input."""
        threads = int(self.config.ffmpeg_decode_threads)
        args = ["-threads", str(threads)] if threads > 0 else []
        if duration is not None:
            args += ["-t", f"{float(duration):.3f}"]
        return [*args, "-i", video_path]

    def _full_decode_timeout(self, effective_duration: float) -> float:
        """Timeout for a command that decodes the whole (length-capped) video."""
        return float(self.config.ffmpeg_timeout_seconds) + max(
            0.0, float(effective_duration)
        ) * float(self.config.ffmpeg_timeout_per_media_second)

    def _scene_pass_timeout(self, effective_duration: float) -> float:
        """Full-decode timeout, leaving the uniform fallback its share."""
        timeout = self._full_decode_timeout(effective_duration)
        deadline = _keyframe_extraction_deadline.get()
        if deadline is None:
            return timeout
        reserve = UNIFORM_FALLBACK_DEADLINE_SHARE * float(
            self.config.keyframe_extraction_deadline_seconds
        )
        available = deadline - time.monotonic() - reserve
        if available <= 0:
            raise TimeoutError("no time left for the scene pass before the fallback")
        return min(timeout, available)

    def _media_command_limit(self, timeout: Optional[float]) -> Tuple[float, bool]:
        """``(limit, deadline_bound)``: the effective timeout, and whether the
        extraction deadline (rather than the command's own timeout) set it."""
        requested = float(
            self.config.ffmpeg_timeout_seconds if timeout is None else timeout
        )
        deadline = _keyframe_extraction_deadline.get()
        if deadline is None:
            return requested, False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise KeyframeExtractionDeadlineExceeded(
                "keyframe extraction deadline of "
                f"{self.config.keyframe_extraction_deadline_seconds:g}s exceeded"
            )
        return min(requested, remaining), remaining < requested

    async def _run_media_command(
        self, command: List[str], *, timeout: Optional[float] = None
    ) -> bytes:
        """Run an FFmpeg/ffprobe command and return stdout."""
        stdout, _ = await self._run_media_command_with_stderr(command, timeout=timeout)
        return stdout

    async def _run_media_command_with_stderr(
        self,
        command: List[str],
        *,
        timeout: Optional[float] = None,
    ) -> Tuple[bytes, bytes]:
        """Run an FFmpeg/ffprobe command and return stdout and stderr.

        ``timeout`` defaults to ``ffmpeg_timeout_seconds`` and is capped by the
        keyframe extraction deadline. The process is killed on timeout and on
        cancellation, so an abandoned command never keeps decoding.
        """
        limit, deadline_bound = self._media_command_limit(timeout)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=limit)
        except asyncio.TimeoutError as exc:
            await self._kill_media_process(process)
            if deadline_bound:
                raise KeyframeExtractionDeadlineExceeded(
                    f"{command[0]} stopped at the keyframe extraction deadline"
                ) from exc
            raise TimeoutError(f"{command[0]} timed out after {limit:.1f}s") from exc
        except BaseException:
            await self._kill_media_process(process)
            raise

        if process.returncode != 0:
            # The tail holds the error; info-level logs can be many KB.
            message = stderr.decode("utf-8", errors="replace").strip()[-2000:]
            raise RuntimeError(f"{command[0]} failed: {message}")
        return stdout, stderr

    @staticmethod
    async def _kill_media_process(process) -> None:
        if process.returncode is not None:
            return
        try:
            process.kill()
        except ProcessLookupError:
            return
        # Drain and close the pipes while reaping. If this is itself
        # cancelled, the kill was already sent.
        await process.communicate()

    async def _extract_audio_features(
        self, video_path: str, video_info: Dict[str, Any]
    ) -> AudioFeatures:
        """Transcribe audio through the internal ASR service when configured."""
        has_audio = bool(video_info.get("has_audio", False))
        audio_features = AudioFeatures(
            has_audio=has_audio,
            audio_length=float(video_info.get("duration") or 0),
            transcription_status="not_attempted",
            alignment_status="not_attempted",
        )
        if not has_audio:
            audio_features.transcription_status = "skipped_no_audio"
            audio_features.alignment_status = "skipped_no_audio"
            return audio_features
        if not self.config.speech_to_text_enabled:
            audio_features.transcription_status = "disabled"
            audio_features.alignment_status = "disabled"
            return audio_features

        audio_features.asr_model = self.config.speech_to_text_model
        started_at = time.perf_counter()
        temp_audio_path: Optional[str] = None
        try:
            temp_audio_path = await self._extract_audio_wav(video_path, video_info)
            response = await self._request_transcription(temp_audio_path)
            if isinstance(response, str):
                response = {
                    "text": response,
                    "alignment_status": "degraded" if response.strip() else "no_speech",
                    "segments": [],
                }
            transcript = str(response.get("text") or "")
            transcript = transcript.strip()[
                : self.config.speech_to_text_max_transcript_chars
            ]
            audio_features.audio_transcript = transcript or None
            audio_features.speech_detected = bool(transcript)
            audio_features.transcription_status = (
                "completed" if transcript else "no_speech"
            )
            audio_features.alignment_status = str(
                response.get("alignment_status")
                or ("degraded" if transcript else "no_speech")
            )
            try:
                raw_segments = list(response.get("segments") or [])[
                    : int(self.config.max_asr_temporal_tokens)
                ]
                normalized_segments = []
                previous_start = 0.0
                duration = max(0.0, float(video_info.get("duration") or 0.0))
                for segment in raw_segments:
                    start = float(segment.get("start_seconds") or 0.0)
                    end = float(segment.get("end_seconds") or start)
                    if not np.isfinite(start) or not np.isfinite(end):
                        raise ValueError("ASR alignment timestamp is not finite")
                    if start < previous_start or end < start:
                        raise ValueError("ASR alignment timestamps are not ordered")
                    if duration > 0:
                        start = min(max(0.0, start), duration)
                        end = min(max(start, end), duration)
                    normalized_segments.append(
                        {
                            "text": str(segment.get("text") or ""),
                            "start_seconds": start,
                            "end_seconds": end,
                        }
                    )
                    previous_start = start
                segment_embeddings = await asyncio.to_thread(
                    self._encode_multilingual_texts,
                    [segment["text"] for segment in normalized_segments],
                    prefix="passage",
                )
                audio_features.asr_segments = [
                    {
                        **segment,
                        "text_embedding": segment_embeddings[index],
                    }
                    for index, segment in enumerate(normalized_segments)
                ]
            except Exception as alignment_exc:
                audio_features.alignment_status = "degraded"
                audio_features.asr_segments = []
                logger.warning(
                    "ASR alignment unavailable; retaining transcript: %s",
                    type(alignment_exc).__name__,
                )
        except Exception as exc:
            audio_features.speech_detected = False
            audio_features.transcription_status = "degraded"
            audio_features.alignment_status = "degraded"
            logger.warning(
                "Speech transcription unavailable; preserving other features: %s",
                type(exc).__name__,
            )
        finally:
            audio_features.transcription_time_seconds = time.perf_counter() - started_at
            if temp_audio_path:
                try:
                    os.remove(temp_audio_path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning(
                        "Failed to clean temporary audio extraction: %s", exc
                    )
        return audio_features

    async def _extract_audio_wav(
        self, video_path: str, video_info: Dict[str, Any]
    ) -> str:
        """Extract bounded mono PCM audio suitable for speech recognition."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output:
            audio_path = output.name
        duration = float(video_info.get("duration") or self.max_video_length)
        duration = max(0.1, min(duration, float(self.max_video_length)))
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            video_path,
            "-t",
            str(duration),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            audio_path,
        ]
        try:
            await self._run_media_command(command)
        except Exception:
            try:
                os.remove(audio_path)
            except OSError:
                pass
            raise
        return audio_path

    async def _request_transcription(self, audio_path: str) -> Dict[str, Any]:
        """Call the internal OpenAI-compatible ASR endpoint with finite retries."""
        endpoint = (
            self.config.speech_to_text_base_url.rstrip("/") + "/v1/audio/transcriptions"
        )
        max_attempts = max(1, int(self.config.speech_to_text_max_attempts))
        timeout = httpx.Timeout(float(self.config.speech_to_text_timeout_seconds))
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(
                    timeout=timeout, trust_env=False
                ) as client:
                    with open(audio_path, "rb") as audio_file:
                        response = await client.post(
                            endpoint,
                            data={"model": self.config.speech_to_text_model},
                            files={"file": ("audio.wav", audio_file, "audio/wav")},
                        )
                response.raise_for_status()
                payload = response.json()
                return {
                    "text": str(payload.get("text") or ""),
                    "alignment_status": str(
                        payload.get("alignment_status") or "degraded"
                    ),
                    "segments": list(payload.get("segments") or []),
                }
            except httpx.TimeoutException as exc:
                last_error = exc
                # A timed-out request may still be running inference on the GPU.
                break
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code not in {429, 503}:
                    break
            except httpx.ConnectError as exc:
                last_error = exc
            except (httpx.HTTPError, ValueError, OSError) as exc:
                last_error = exc
                break
            if attempt + 1 < max_attempts:
                await asyncio.sleep(0.1 * (attempt + 1))
        raise RuntimeError("Internal speech-to-text request failed") from last_error

    def _parse_ppm_frames(self, payload: bytes) -> List[np.ndarray]:
        """Parse concatenated binary PPM frames from FFmpeg image2pipe output."""
        frames: List[np.ndarray] = []
        index = 0
        length = len(payload)

        while index < length:
            index = self._skip_ppm_whitespace_and_comments(payload, index)
            if index >= length:
                break
            if payload[index : index + 2] != b"P6":
                raise ValueError("Unexpected FFmpeg frame format")
            index += 2

            width_token, index = self._read_ppm_token(payload, index)
            height_token, index = self._read_ppm_token(payload, index)
            max_value_token, index = self._read_ppm_token(payload, index)
            width = int(width_token)
            height = int(height_token)
            max_value = int(max_value_token)
            if width <= 0 or height <= 0 or max_value != 255:
                raise ValueError("Unsupported FFmpeg PPM frame")
            if index >= length or payload[index] not in b" \t\r\n":
                raise ValueError("Invalid FFmpeg PPM frame payload")
            index += 1

            frame_size = width * height * 3
            if index + frame_size > length:
                raise ValueError("Truncated FFmpeg PPM frame")
            frame = np.frombuffer(
                payload[index : index + frame_size],
                dtype=np.uint8,
            ).copy()
            frames.append(frame.reshape((height, width, 3)))
            index += frame_size

        return frames

    def _skip_ppm_whitespace_and_comments(self, payload: bytes, index: int) -> int:
        while index < len(payload):
            if payload[index] in b" \t\r\n":
                index += 1
                continue
            if payload[index] == ord("#"):
                while index < len(payload) and payload[index] not in b"\r\n":
                    index += 1
                continue
            break
        return index

    def _read_ppm_token(self, payload: bytes, index: int) -> Tuple[str, int]:
        index = self._skip_ppm_whitespace_and_comments(payload, index)
        start = index
        while index < len(payload) and payload[index] not in b" \t\r\n":
            index += 1
        if start == index:
            raise ValueError("Invalid FFmpeg PPM header")
        return payload[start:index].decode("ascii"), index

    def _keyframe_indices(self, frame_count: int) -> List[int]:
        if frame_count <= 0 or self.max_keyframes <= 0:
            return []
        sample_count = min(self.max_keyframes, frame_count)
        return np.linspace(0, frame_count - 1, sample_count, dtype=int).tolist()

    def _parse_frame_rate(self, value: Any) -> float:
        if not value:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_value = float(denominator)
            if denominator_value == 0:
                return 0.0
            return float(numerator) / denominator_value
        return self._parse_float(value, default=0.0)

    def _parse_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _parse_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    async def _extract_keyframes_opencv(
        self, video_path: str
    ) -> Tuple[List[np.ndarray], Dict]:
        """Extract uniform keyframes with OpenCV in a child process.

        A child process (unlike a thread) can be killed at the extraction
        deadline and does not hold an executor thread CLIP/OCR need.
        """
        with tempfile.TemporaryDirectory(prefix="opencv-keyframes-") as directory:
            frames_path = os.path.join(directory, "frames.npz")
            result_path = os.path.join(directory, "result.json")
            command = [
                sys.executable,
                "-I",
                opencv_keyframes.__file__,
                video_path,
                frames_path,
                result_path,
                str(self.max_keyframes),
                str(self.max_video_length),
            ]
            await self._run_media_command(
                command,
                timeout=self._full_decode_timeout(float(self.max_video_length)),
            )
            try:
                with open(result_path, encoding="utf-8") as handle:
                    result = json.load(handle)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"OpenCV keyframe extraction wrote no result: {exc}"
                ) from exc
            error = result.get("error")
            if error == "missing":
                raise FileNotFoundError(result.get("message"))
            if error == "invalid_video":
                raise InvalidVideoError(result.get("message"))
            if error:
                raise RuntimeError(
                    f"OpenCV keyframe extraction failed: {result.get('message')}"
                )
            with np.load(frames_path, allow_pickle=False) as archive:
                keyframes = [
                    archive[f"frame_{index}"] for index in range(int(result["frames"]))
                ]
        logger.info("Extracted %s keyframes from video using OpenCV", len(keyframes))
        return keyframes, result["info"]

    async def _extract_visual_features(
        self, keyframes: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Extract visual features using CLIP; failures raise, never zero-fill."""
        try:
            features = await asyncio.to_thread(self._clip_visual_features, keyframes)
        except Exception as exc:
            logger.error(f"Error extracting visual features: {exc}")
            raise ContentModalityError("clip", type(exc).__name__) from exc
        if features["frame_count"] == 0:
            raise ContentModalityError("clip", "no frame embeddings were produced")
        return features

    def _clip_visual_features(self, keyframes: List[np.ndarray]) -> Dict[str, Any]:
        # Convert numpy arrays to PIL Images
        pil_images = []
        for frame in keyframes:
            # Enhance image quality
            img = Image.fromarray(frame)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)  # Slight sharpness increase
            pil_images.append(img)

        # Process images in batches
        all_embeddings = []

        for i in range(0, len(pil_images), self.batch_size):
            batch_images = pil_images[i : i + self.batch_size]

            # Preprocess images
            inputs = self.clip_processor(
                images=batch_images, return_tensors="pt", padding=True
            ).to(self.device)

            # Extract features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(
                    dim=1, keepdim=True
                )
                all_embeddings.append(image_features.cpu())

        if not all_embeddings:
            return {"embedding": None, "frame_count": 0, "individual_embeddings": []}
        # Combine all embeddings and average pool across frames
        stacked = torch.cat(all_embeddings, dim=0)
        return {
            "embedding": stacked.mean(dim=0),
            "frame_count": len(keyframes),
            "individual_embeddings": stacked.numpy().tolist(),
        }

    @staticmethod
    def _frame_timestamps(keyframes: List[np.ndarray], video_info: Dict) -> List[float]:
        """Return deterministic ordered timestamps when extraction metadata is unavailable."""
        count = len(keyframes)
        if count == 0:
            return []
        extracted = video_info.get("frame_timestamps_seconds") or []
        if len(extracted) == count:
            return [float(value) for value in extracted]
        frame_indices = video_info.get("sampled_frame_indices") or []
        fps = float(video_info.get("fps", 0.0) or 0.0)
        if len(frame_indices) == count and fps > 0:
            return [float(index) / fps for index in frame_indices]
        logger.warning(
            "Frame timestamp metadata unavailable; marking sampled timestamps unknown"
        )
        return [0.0] * count

    async def _extract_text_features(
        self,
        keyframes: List[np.ndarray],
        frame_timestamps: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Extract text using OCR from keyframes.

        OCR is optional for the job: an OCR failure is reported through
        ``status`` ("failed", or "degraded" when only some frames failed) and the
        remaining modalities still complete.
        """
        from video_commerce.ml.video_ocr import TemporalOCRTracker

        frame_timestamps = frame_timestamps or [float(i) for i in range(len(keyframes))]

        try:
            (
                regions,
                price_mentions,
                failed_frames,
            ) = await asyncio.to_thread(
                self._ocr_regions_sync, keyframes, frame_timestamps
            )

            tracker = TemporalOCRTracker(
                text_similarity_threshold=float(
                    self.config.ocr_temporal_text_similarity
                ),
                polygon_iou_threshold=float(self.config.ocr_temporal_iou_threshold),
                max_missed_frames=int(self.config.ocr_temporal_max_missed_frames),
                max_gap_seconds=float(self.config.ocr_temporal_max_gap_seconds),
            )
            tracks = tracker.track(regions)
            unique_text_blocks = [track.text for track in tracks if len(track.text) > 2]
            text_embeddings = await asyncio.to_thread(
                self._encode_ocr_texts, [track.text for track in tracks]
            )
            track_payloads = [
                {
                    "text": track.text,
                    "polygon": track.polygon,
                    "first_seen_seconds": track.first_seen_seconds,
                    "last_seen_seconds": track.last_seen_seconds,
                    "occurrence_count": track.occurrence_count,
                    "confidence": track.max_confidence,
                    "text_embedding": text_embeddings[index],
                }
                for index, track in enumerate(tracks)
            ]
            if keyframes and failed_frames >= len(keyframes):
                status = "failed"
            elif failed_frames:
                status = "degraded"
            else:
                status = "completed"

            return {
                "text_blocks": unique_text_blocks,
                "price_mentions": price_mentions,
                "total_text_regions": len(regions),
                "ocr_tracks": track_payloads,
                "status": status,
            }

        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return {
                "text_blocks": [],
                "price_mentions": [],
                "total_text_regions": 0,
                "status": "failed",
            }

    def _ocr_regions_sync(
        self, keyframes: List[np.ndarray], frame_timestamps: List[float]
    ) -> Tuple[List[Any], List[Dict[str, Any]], int]:
        """Run PaddleOCR or pytesseract over every keyframe (blocking)."""
        from video_commerce.ml.video_ocr import OCRRegion

        regions = []
        price_mentions = []
        failed_frames = 0
        for i, frame in enumerate(keyframes):
            if self.config.ocr_backend == "paddle":
                if self.ocr_region_extractor is None:
                    raise RuntimeError("PaddleOCR region extractor is not initialized")
                frame_regions = self.ocr_region_extractor.extract(
                    frame,
                    frame_index=i,
                    timestamp_seconds=float(frame_timestamps[i]),
                )
                regions.extend(frame_regions)
                for region in frame_regions:
                    if self._is_price_mention(region.text):
                        price_mentions.append(
                            {
                                "text": region.text,
                                "frame": i,
                                "confidence": region.recognition_confidence,
                            }
                        )
                continue
            # Convert to PIL Image for OCR
            pil_image = Image.fromarray(frame)

            # Enhance image for better OCR
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced_image = enhancer.enhance(1.5)

            # Perform OCR
            try:
                ocr_data = pytesseract.image_to_data(
                    enhanced_image, config=self.ocr_config, output_type=Output.DICT
                )

                # Extract text blocks with confidence > 30
                for j, confidence in enumerate(ocr_data["conf"]):
                    if int(confidence) > 30:
                        text = ocr_data["text"][j].strip()
                        if text and len(text) > 1:
                            width = max(1, int(enhanced_image.width))
                            height = max(1, int(enhanced_image.height))
                            left = int(ocr_data.get("left", [0])[j]) / width
                            top = int(ocr_data.get("top", [0])[j]) / height
                            right = left + int(ocr_data.get("width", [0])[j]) / width
                            bottom = top + int(ocr_data.get("height", [0])[j]) / height
                            score = max(0.0, min(1.0, float(confidence) / 100.0))
                            regions.append(
                                OCRRegion(
                                    text=text,
                                    polygon=[
                                        (left, top),
                                        (right, top),
                                        (right, bottom),
                                        (left, bottom),
                                    ],
                                    detection_confidence=score,
                                    recognition_confidence=score,
                                    frame_index=i,
                                    timestamp_seconds=float(frame_timestamps[i]),
                                )
                            )

                            # Check for price patterns
                            if self._is_price_mention(text):
                                price_mentions.append(
                                    {
                                        "text": text,
                                        "frame": i,
                                        "confidence": confidence,
                                    }
                                )

            except Exception as ocr_error:
                failed_frames += 1
                logger.warning(f"OCR failed for frame {i}: {ocr_error}")
        return regions, price_mentions, failed_frames

    def _encode_ocr_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed OCR strings in the frozen multilingual passage space."""
        return self._encode_multilingual_texts(texts, prefix="passage")

    def _encode_multilingual_texts(
        self, texts: List[str], *, prefix: str
    ) -> List[List[float]]:
        if not texts:
            return []
        if self.text_embedder is None:
            return [[] for _ in texts]
        return self.text_embedder.encode(texts, prefix=prefix)

    def _is_price_mention(self, text: str) -> bool:
        """Check if text contains price information."""
        text_lower = text.lower()

        # Check for currency symbols and price patterns
        price_patterns = ["$", "€", "£", "¥", "usd", "eur", "gbp"]
        for pattern in price_patterns:
            if pattern in text_lower:
                return True

        # Check for price-like numbers
        import re

        if re.search(r"\d+\.?\d*\s*(dollars?|euros?|pounds?)", text_lower):
            return True

        return False

    async def _detect_objects_with_status(
        self, keyframes: List[np.ndarray]
    ) -> Tuple[List[str], str]:
        """CLIP zero-shot object detection off the event loop.

        Detection is optional: a failure returns no objects with status
        "failed" instead of aborting the job.
        """
        try:
            return await asyncio.to_thread(self._detect_objects_sync, keyframes), (
                "completed"
            )
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return [], "failed"

    def _detect_objects_sync(self, keyframes: List[np.ndarray]) -> List[str]:
        # Common product categories for e-commerce
        candidate_objects = [
            "clothing",
            "shoes",
            "electronics",
            "phone",
            "laptop",
            "headphones",
            "watch",
            "jewelry",
            "bag",
            "furniture",
            "book",
            "toy",
            "food",
            "cosmetics",
            "skincare",
            "perfume",
            "glasses",
            "hat",
            "shirt",
            "dress",
            "jacket",
            "pants",
            "sneakers",
            "boots",
            "camera",
            "tablet",
            "speaker",
            "gaming",
            "kitchen",
            "home decor",
        ]

        detected_objects = set()

        # Sample a few representative frames
        sample_frames = keyframes[:: max(1, len(keyframes) // 3)][:3]

        for frame in sample_frames:
            pil_image = Image.fromarray(frame)

            # Prepare text queries for zero-shot classification
            text_queries = [f"a photo of {obj}" for obj in candidate_objects]

            # Process with CLIP
            inputs = self.clip_processor(
                images=[pil_image],
                text=text_queries,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # Get top predictions above threshold
            top_probs, top_indices = torch.topk(probs[0], k=5)

            for prob, idx in zip(top_probs, top_indices):
                if prob.item() > 0.1:  # Threshold for object detection
                    detected_objects.add(candidate_objects[idx.item()])

        return list(detected_objects)

    def _categories_from_text(self, text_values: List[str]) -> Dict[str, float]:
        """Map bilingual commerce words to canonical catalog categories."""
        matches: Dict[str, float] = {}
        for category, keywords in self.category_keywords.items():
            hit_count = 0
            for text in text_values:
                text_lower = str(text).lower()
                if any(keyword in text_lower for keyword in keywords):
                    hit_count += 1
            if hit_count:
                matches[category] = min(1.0, 0.5 + (0.1 * hit_count))
        return matches

    def _analyze_commerce_content(
        self,
        text_features: Dict,
        detected_objects: List[str],
        audio_transcript: str = "",
    ) -> Dict[str, Any]:
        """Analyze commerce-related aspects of the content."""
        try:
            text_blocks = text_features.get("text_blocks", [])
            price_mentions = text_features.get("price_mentions", [])
            semantic_text_blocks = list(text_blocks)
            if audio_transcript:
                semantic_text_blocks.append(audio_transcript)

            # Calculate commerce score based on various factors
            commerce_score = 0.0

            # Text-based commerce indicators
            commerce_text_count = 0
            for text in semantic_text_blocks:
                text_lower = text.lower()
                for keyword_category in self.commerce_keywords.values():
                    if any(keyword in text_lower for keyword in keyword_category):
                        commerce_text_count += 1
                        break

            if semantic_text_blocks:
                commerce_score += (
                    commerce_text_count / len(semantic_text_blocks)
                ) * 0.4

            # Price mentions boost
            if price_mentions:
                commerce_score += min(len(price_mentions) * 0.1, 0.3)

            # Product objects boost
            product_objects = [
                obj
                for obj in detected_objects
                if obj
                in [
                    "clothing",
                    "shoes",
                    "electronics",
                    "phone",
                    "laptop",
                    "watch",
                    "jewelry",
                    "bag",
                ]
            ]
            if product_objects:
                commerce_score += min(len(product_objects) * 0.05, 0.3)

            # Normalize score
            commerce_score = min(commerce_score, 1.0)

            # Categorize content
            category_scores = {}
            category_mapping = {
                "electronics": [
                    "phone",
                    "laptop",
                    "headphones",
                    "camera",
                    "tablet",
                    "speaker",
                ],
                "fashion": ["clothing", "shoes", "jewelry", "watch", "bag", "hat"],
                "home": ["furniture", "kitchen", "home decor"],
                "beauty": ["cosmetics", "skincare", "perfume"],
                "books": ["book"],
                "toys": ["toy", "gaming"],
            }

            for category, keywords in category_mapping.items():
                score = len([obj for obj in detected_objects if obj in keywords]) / max(
                    len(detected_objects), 1
                )
                if score > 0:
                    category_scores[category] = score

            ocr_category_scores = self._categories_from_text(text_blocks)
            speech_category_scores = self._categories_from_text(
                [audio_transcript] if audio_transcript else []
            )
            for category, score in ocr_category_scores.items():
                category_scores[category] = max(
                    category_scores.get(category, 0.0), score
                )
            for category, score in speech_category_scores.items():
                category_scores[category] = max(
                    category_scores.get(category, 0.0),
                    min(1.0, score + 0.2),
                )

            # Extract potential product mentions from text
            product_mentions = []
            for text in text_blocks:
                if any(
                    keyword in text.lower()
                    for keyword in self.commerce_keywords["product_keywords"]
                ):
                    if len(text.split()) <= 5:  # Likely product names are short
                        product_mentions.append(text)

            return {
                "commerce_score": commerce_score,
                "category_scores": category_scores,
                "ocr_categories": list(ocr_category_scores.keys()),
                "speech_categories": list(speech_category_scores.keys()),
                "product_mentions": product_mentions[:10],  # Limit to top 10
                "has_pricing_info": len(price_mentions) > 0,
                "commerce_text_ratio": commerce_text_count
                / max(len(semantic_text_blocks), 1),
            }

        except Exception as e:
            logger.error(f"Error analyzing commerce content: {e}")
            return {
                "commerce_score": 0.0,
                "category_scores": {},
                "ocr_categories": [],
                "speech_categories": [],
                "product_mentions": [],
                "has_pricing_info": False,
                "commerce_text_ratio": 0.0,
            }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the content processor."""
        try:
            status = {
                "initialized": self.is_initialized,
                "lazy_load_enabled": self.lazy_load,
                "device": str(self.device),
                "clip_model_loaded": self.clip_model is not None,
                "clip_processor_loaded": self.clip_processor is not None,
            }

            # Test basic functionality if initialized
            if self.is_initialized:
                try:
                    # Create dummy image and test processing
                    dummy_image = Image.new("RGB", (224, 224), color="white")
                    inputs = self.clip_processor(
                        images=[dummy_image], return_tensors="pt"
                    ).to(self.device)

                    with torch.no_grad():
                        outputs = self.clip_model.get_image_features(**inputs)

                    status["clip_test_passed"] = True
                    status["output_shape"] = list(outputs.shape)

                except Exception as test_error:
                    status["clip_test_passed"] = False
                    status["test_error"] = str(test_error)

            return status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"error": str(e), "initialized": False}

    async def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """Process a single image (useful for testing or single-frame processing)."""
        if not self.is_initialized:
            await self.load_models()

        try:
            # Process single image
            inputs = self.clip_processor(images=[image], return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                normalized_features = image_features / image_features.norm(
                    dim=1, keepdim=True
                )

            return {
                "embedding": normalized_features[0].cpu().numpy(),
                "shape": list(normalized_features.shape),
            }

        except Exception as e:
            logger.error(f"Error processing single image: {e}")
            return {"embedding": np.zeros(self.config.embedding_dim), "error": str(e)}
