"""Structured OCR regions and temporal deduplication for sampled video frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import base64
import json
import os
import subprocess
import threading
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


def normalize_ocr_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(normalized.casefold().split())


def normalized_edit_similarity(left: str, right: str) -> float:
    left_value = normalize_ocr_text(left)
    right_value = normalize_ocr_text(right)
    if not left_value and not right_value:
        return 1.0
    if not left_value or not right_value:
        return 0.0
    return float(SequenceMatcher(None, left_value, right_value).ratio())


def polygon_iou(left: Sequence[Point], right: Sequence[Point]) -> float:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if left_array.shape[0] < 3 or right_array.shape[0] < 3:
        return 0.0
    left_hull = cv2.convexHull(left_array)
    right_hull = cv2.convexHull(right_array)
    left_area = float(cv2.contourArea(left_hull))
    right_area = float(cv2.contourArea(right_hull))
    if left_area <= 0.0 or right_area <= 0.0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(left_hull, right_hull)
    union = left_area + right_area - float(intersection)
    return float(intersection / union) if union > 0.0 else 0.0


@dataclass(frozen=True)
class OCRRegion:
    text: str
    polygon: List[Point]
    detection_confidence: float
    recognition_confidence: float
    frame_index: int
    timestamp_seconds: float
    text_embedding: Optional[List[float]] = None


class PaddleOCRRegionExtractor:
    """Lazy PaddleOCR 3.x adapter returning normalized, trackable text regions.

    Paddle's OCR pipeline performs text detection first and sends only detected
    crops to recognition. Keeping the dependency lazy lets ranking-only images
    avoid loading Paddle and makes the content worker fail with a clear message.
    """

    def __init__(
        self,
        *,
        language: str = "en",
        detection_model: str = "PP-OCRv5_mobile_det",
        recognition_model: str = "PP-OCRv5_mobile_rec",
        detection_threshold: float = 0.3,
        recognition_threshold: float = 0.5,
        pipeline=None,
    ) -> None:
        self.detection_threshold = float(detection_threshold)
        self.recognition_threshold = float(recognition_threshold)
        if pipeline is None:
            pipeline = _PaddleOCRSubprocessPipeline(
                language=language,
                detection_model=detection_model,
                recognition_model=recognition_model,
            )
        self.pipeline = pipeline

    def extract(
        self, frame: np.ndarray, *, frame_index: int, timestamp_seconds: float
    ) -> List[OCRRegion]:
        regions: List[OCRRegion] = []
        for result in self.pipeline.predict(frame):
            payload = getattr(result, "json", result)
            if callable(payload):
                payload = payload()
            if isinstance(payload, dict) and "res" in payload:
                payload = payload["res"]
            polygons = payload.get("dt_polys", [])
            texts = payload.get("rec_texts", [])
            scores = payload.get("rec_scores", [])
            detection_scores = payload.get("dt_scores", [1.0] * len(texts))
            height, width = frame.shape[:2]
            for polygon, text, rec_score, det_score in zip(
                polygons, texts, scores, detection_scores
            ):
                if (
                    float(det_score) < self.detection_threshold
                    or float(rec_score) < self.recognition_threshold
                    or not normalize_ocr_text(text)
                ):
                    continue
                normalized_polygon = [
                    (float(point[0]) / max(1, width), float(point[1]) / max(1, height))
                    for point in polygon
                ]
                regions.append(
                    OCRRegion(
                        text=str(text),
                        polygon=normalized_polygon,
                        detection_confidence=float(det_score),
                        recognition_confidence=float(rec_score),
                        frame_index=int(frame_index),
                        timestamp_seconds=float(timestamp_seconds),
                    )
                )
        return regions


class _PaddleOCRSubprocessPipeline:
    """Keep Paddle's conflicting dependencies in a persistent isolated venv."""

    def __init__(self, *, language: str, detection_model: str, recognition_model: str):
        python = os.getenv("PADDLE_OCR_PYTHON", "/opt/paddleocr/bin/python")
        env = dict(os.environ)
        env["PADDLE_OCR_LANGUAGE"] = language
        env["PADDLE_OCR_DETECTION_MODEL"] = detection_model
        env["PADDLE_OCR_RECOGNITION_MODEL"] = recognition_model
        self._process = subprocess.Popen(
            [python, "-m", "video_commerce.ml.paddle_ocr_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )
        self._lock = threading.Lock()

    def predict(self, frame: np.ndarray):
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Failed to encode OCR frame")
        request = json.dumps({"image": base64.b64encode(encoded).decode("ascii")})
        with self._lock:
            if self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("PaddleOCR subprocess pipes are unavailable")
            self._process.stdin.write(request + "\n")
            self._process.stdin.flush()
            response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError("PaddleOCR subprocess exited unexpectedly")
        response = json.loads(response_line)
        if response.get("error"):
            raise RuntimeError(f"PaddleOCR failed: {response['error']}")
        return response.get("results", [])

    def close(self) -> None:
        if self._process.poll() is None:
            if self._process.stdin is not None:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()

    def __del__(self):  # pragma: no cover - interpreter shutdown safety
        try:
            self.close()
        except Exception:
            pass


@dataclass
class OCRTrack:
    text: str
    normalized_text: str
    polygon: List[Point]
    first_seen_seconds: float
    last_seen_seconds: float
    first_frame_index: int
    last_frame_index: int
    occurrence_count: int
    max_confidence: float
    text_embedding: Optional[List[float]] = None
    _regions: List[OCRRegion] = field(default_factory=list, repr=False)


class TemporalOCRTracker:
    def __init__(
        self,
        *,
        text_similarity_threshold: float = 0.8,
        polygon_iou_threshold: float = 0.3,
        max_missed_frames: int = 1,
        max_gap_seconds: float = 5.0,
    ) -> None:
        self.text_similarity_threshold = float(text_similarity_threshold)
        self.polygon_iou_threshold = float(polygon_iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.max_gap_seconds = float(max_gap_seconds)

    def track(self, regions: Iterable[OCRRegion]) -> List[OCRTrack]:
        ordered = sorted(
            regions,
            key=lambda region: (
                int(region.frame_index),
                float(region.timestamp_seconds),
                normalize_ocr_text(region.text),
            ),
        )
        by_frame: Dict[int, List[OCRRegion]] = {}
        for region in ordered:
            if normalize_ocr_text(region.text):
                by_frame.setdefault(int(region.frame_index), []).append(region)

        tracks: List[OCRTrack] = []
        for frame_index in sorted(by_frame):
            frame_regions = by_frame[frame_index]
            candidates = []
            for track_index, track in enumerate(tracks):
                frame_gap = frame_index - track.last_frame_index
                time_gap = min(
                    region.timestamp_seconds - track.last_seen_seconds
                    for region in frame_regions
                )
                if (
                    frame_gap > self.max_missed_frames + 1
                    or time_gap > self.max_gap_seconds
                ):
                    continue
                previous = track._regions[-1]
                for region_index, region in enumerate(frame_regions):
                    similarity = normalized_edit_similarity(previous.text, region.text)
                    overlap = polygon_iou(previous.polygon, region.polygon)
                    if (
                        similarity >= self.text_similarity_threshold
                        and overlap >= self.polygon_iou_threshold
                    ):
                        candidates.append(
                            (similarity + overlap, track_index, region_index)
                        )

            matched_tracks = set()
            matched_regions = set()
            for _, track_index, region_index in sorted(candidates, reverse=True):
                if track_index in matched_tracks or region_index in matched_regions:
                    continue
                self._append(tracks[track_index], frame_regions[region_index])
                matched_tracks.add(track_index)
                matched_regions.add(region_index)

            for region_index, region in enumerate(frame_regions):
                if region_index not in matched_regions:
                    tracks.append(self._new_track(region))

        for track in tracks:
            self._finalize_consensus(track)
        return sorted(
            tracks,
            key=lambda track: (track.first_seen_seconds, track.first_frame_index),
        )

    @staticmethod
    def _new_track(region: OCRRegion) -> OCRTrack:
        confidence = max(
            float(region.detection_confidence), float(region.recognition_confidence)
        )
        return OCRTrack(
            text=region.text,
            normalized_text=normalize_ocr_text(region.text),
            polygon=list(region.polygon),
            first_seen_seconds=float(region.timestamp_seconds),
            last_seen_seconds=float(region.timestamp_seconds),
            first_frame_index=int(region.frame_index),
            last_frame_index=int(region.frame_index),
            occurrence_count=1,
            max_confidence=confidence,
            text_embedding=region.text_embedding,
            _regions=[region],
        )

    @staticmethod
    def _append(track: OCRTrack, region: OCRRegion) -> None:
        track._regions.append(region)
        track.last_seen_seconds = float(region.timestamp_seconds)
        track.last_frame_index = int(region.frame_index)
        track.occurrence_count += 1
        confidence = max(
            float(region.detection_confidence), float(region.recognition_confidence)
        )
        track.max_confidence = max(track.max_confidence, confidence)

    @staticmethod
    def _finalize_consensus(track: OCRTrack) -> None:
        votes: Dict[str, float] = {}
        best_region: Dict[str, OCRRegion] = {}
        for region in track._regions:
            normalized = normalize_ocr_text(region.text)
            confidence = max(0.0, float(region.recognition_confidence))
            votes[normalized] = votes.get(normalized, 0.0) + confidence
            existing = best_region.get(normalized)
            if (
                existing is None
                or region.recognition_confidence > existing.recognition_confidence
            ):
                best_region[normalized] = region
        winner = max(votes, key=lambda value: (votes[value], value))
        representative = best_region[winner]
        track.text = representative.text
        track.normalized_text = winner
        track.text_embedding = representative.text_embedding
