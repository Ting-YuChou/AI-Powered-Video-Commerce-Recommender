# Temporal multimodal video ranking

## Data flow

1. The content worker samples up to 16 scene-aware frames and preserves their
   real frame indices, timestamps, and normalized 512-dimensional CLIP
   embeddings. The compatibility mean embedding remains available to recall.
2. PaddleOCR runs in an isolated persistent process. Its detector finds text
   polygons and recognition operates on those regions. The content worker then
   joins adjacent observations only when normalized edit similarity and polygon
   IoU both pass their configured thresholds.
3. Qwen3-ASR uses the pinned forced aligner to return word/character timestamps.
   The worker groups alignment items by pause, duration, and text length, then
   embeds at most 64 segments. Alignment failure retains the whole transcript
   for compatibility but marks the ASR temporal branch absent.
4. OCR, ASR, and catalog text use frozen
   `intfloat/multilingual-e5-small` revision
   `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Content uses the `passage:`
   prefix; candidate text uses `query:`.
5. Every completed extraction is written as a canonical, checksummed immutable
   JSON object before Postgres and Redis current pointers advance. PIT examples
   store only the artifact URI, SHA-256, schema, and creation time.
6. Three trainable temporal Transformers contextualize visual, OCR, and ASR
   sequences using normalized time, delta, and span duration. Candidate image,
   text, and two-tower sidecar embeddings form the cross-attention query.
   Candidate-attended and global pooled representations are combined by a
   presence-aware modality gate and injected as a near-zero residual into the
   existing multi-objective ranker.

## Version and rollout contract

- Content schema: `temporal_multimodal_v2`; immutable envelope schema:
  `content_feature_artifact_v2`.
- Ranking feature schema: `ranking_v4_00_temporal_trimodal`.
- Ranking runtime supports payload versions 1 through 4. V4 carries bounded
  fp16 tensors plus timestamps and masks in `multimodal_context`; it never
  sends transcript text. V1-v3 remain parseable and retain base ranking.
- Candidate image/text/two-tower embeddings are stored in a checksummed NPZ
  sidecar. Its checksum and model version are locked inside the ranking
  checkpoint. Missing modalities use presence masks; missing products never
  receive random fallback embeddings.
- `RANKING_TRIMODAL_SHADOW=true` and `RANKING_TRIMODAL_ENABLED=false` are the
  initial rollout defaults. Shadow training warm-starts the base ranker, freezes
  it for epoch one, then fine-tunes it at 0.1 times the new-layer learning rate.
  Serving only activates V4 when checkpoint, sidecar, and feature schemas match.
- Apply `migrations/postgres/007_temporal_multimodal_features.sql` before
  enabling content backfill. Preview with
  `python scripts/backfill_temporal_multimodal.py --limit 100`; add `--enqueue`
  to re-enqueue missing or pre-v2 durable `content_jobs` from their original
  `storage_path`. Activation waits for the documented offline quality and
  latency promotion gates. The V4 artifact preserves the prior V3 checkpoint
  reference for a fast rollback.

## OCR deployment

Build the dedicated target with `docker compose build content-worker`. Paddle
is installed under `/opt/paddleocr`; the application environment remains on
Pydantic 1 and Transformers 4.35. The long-lived JSON-lines worker prevents a
model reload per frame and isolates PaddleOCR 3's conflicting dependencies.

Use `MODEL_OCR_BACKEND=tesseract` only as an explicit fallback. Paddle model
downloads happen during first initialization unless they are pre-cached in the
image or model volume.
