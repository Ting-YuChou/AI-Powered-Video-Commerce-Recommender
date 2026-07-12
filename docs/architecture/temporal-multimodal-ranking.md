# Temporal multimodal video ranking

## Data flow

1. The content worker samples up to 16 scene-aware frames and preserves their
   order, timestamps, and normalized 512-dimensional CLIP ViT-B/16 embeddings.
2. PaddleOCR runs in an isolated persistent process. Its detector finds text
   polygons and recognition operates on those regions. The content worker then
   joins adjacent observations only when normalized edit similarity and polygon
   IoU both pass their configured thresholds.
3. The compatibility `visual_embedding` remains available for recall and old
   checkpoints. Ordered frames and OCR tracks are stored in Redis for serving
   and in Postgres `content_feature_artifacts` for durable offline training.
4. A trainable temporal Transformer contextualizes frame tokens. At ranking
   time the candidate image, candidate text, and two-tower embedding form the
   query for visual and OCR cross-attention. The fused representation is joined
   with existing ranking features before the multi-objective heads.

## Version and rollout contract

- Content schema: `temporal_multimodal_v1`.
- Ranking feature schema: `ranking_v3_00_temporal_multimodal`.
- Ranking runtime supports payload versions 1, 2, and 3. New multimodal inputs
  travel in `multimodal_context`; v1/v2 callers retain the existing path.
- The v3 model changes checkpoint structure and must be retrained. Do not load
  a v2 checkpoint into `TemporalMultimodalRankingModel`.
- Apply `migrations/postgres/007_temporal_multimodal_features.sql` before
  enabling content backfill. Preview with
  `python scripts/backfill_temporal_multimodal.py --limit 100`; add `--enqueue`
  to re-enqueue existing durable `content_jobs` from their original
  `storage_path`. Activation should wait until coverage and offline ranking
  metrics meet the rollout gate.

## OCR deployment

Build the dedicated target with `docker compose build content-worker`. Paddle
is installed under `/opt/paddleocr`; the application environment remains on
Pydantic 1 and Transformers 4.35. The long-lived JSON-lines worker prevents a
model reload per frame and isolates PaddleOCR 3's conflicting dependencies.

Use `MODEL_OCR_BACKEND=tesseract` only as an explicit fallback. Paddle model
downloads happen during first initialization unless they are pre-cached in the
image or model volume.
