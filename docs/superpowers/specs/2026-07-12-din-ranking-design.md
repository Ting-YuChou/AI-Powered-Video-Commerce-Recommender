# PIT-Based DIN Ranking Design

## Goal

Add a candidate-aware Deep Interest Network (DIN) to ranking. Ranking training
uses only the point-in-time (PIT) offline dataset. Retrieval models and their
training remain out of scope.

## Decisions

- Keep separate `click`, `add_to_cart`, and `purchase` histories.
- Use the latest 60 events per action from the preceding 30 days.
- Require `event_time < as_of_ts` and `available_at <= as_of_ts`.
- Preserve repeated interactions and left-pad sequences to a fixed length.
- Carry product IDs and timestamps through PIT and serving contracts; do not
  carry embedding vectors in request payloads or Parquet rows.
- Freeze the exact 128-dimensional trained two-tower embedding sidecar pinned
  by each ranking training run. Synthetic and unknown embeddings are excluded
  in v1.
- Share one DIN local-activation network across the three action sequences.
  Use a trainable eight-dimensional action embedding and log-recency.
- Replace the previous averaged history vectors and candidate dot-products,
  while retaining count, coverage, recency, and has-signal summaries.
- Train attention, projection, and ranking parameters using the existing CTR,
  CVR, CTCVR, GMV, pairwise, and listwise losses. Add no DIN auxiliary loss.
- Activate the first valid DIN artifact directly after runtime, data-contract,
  and coverage gates pass; do not add a shadow-quality gate.

## Contracts

Introduce `din_sequence_v1`, `ranking_ltr_v2_din`,
`ranking_feature_assembler_v2_din`, `ranking_v3_din`, and internal ranking
payload version 3.

Each action sequence contains fixed-length `product_ids[60]`,
`event_times[60]`, and `mask[60]`. PIT materialization and online Redis reads
must produce identical canonical sequences. Flink maintains namespace-aware
action ZSETs and a DIN freshness token used by recommendation-cache keys.

The model consumes structured inputs: dense ranking features, request-level
history indices/recencies/masks, candidate indices, and a candidate-to-request
mapping. Histories are not duplicated for every candidate in a microbatch.

## DIN Model

For each valid history item, the shared local-activation unit receives the
candidate embedding, history embedding, their difference and element-wise
product, the action embedding, and normalized log-recency. The MLP is
`256 -> 64 -> 1` with Dice activation and masked softmax within each action.

The three 128-dimensional weighted history vectors are concatenated and
projected through `Linear(384, 128)`, LayerNorm, and ReLU. The resulting
candidate-aware `interest_vector` and twelve retained history summaries are
appended to the existing dense ranking representation.

Missing history embeddings are masked. If the candidate embedding is missing,
the DIN branch produces zeros and base ranking features remain usable.

## Artifacts And Activation

The trainer pins one PIT manifest and one exact two-tower artifact. The ranking
artifact atomically contains the ranking checkpoint, product-index mapping,
and checksummed frozen embedding sidecar. Activation metadata records dataset,
feature, sequence, label, assembler, two-tower, and checksum lineage.

Artifact validation failures retain the previous model. With no valid trained
model, serving retains the existing `combined_score` fallback. An activatable
DIN artifact requires the configured minimum training samples and at least
30 percent of rows to contain a non-empty DIN history.

## Verification

Use TDD for sequence contracts, PIT leakage, offline-online parity, attention
masking and gradients, structured batching, payload v3, artifact atomicity,
and fallback behavior. Compare warm p95 latency at 100 and 250 candidates; DIN
may add at most 10 ms under identical batch settings.

After primary verification, a dedicated read-only subagent must review the
exact implementation diff against this design. The main agent does not perform
code review. All Critical and Important findings require fixes, re-verification,
and reviewer re-review before handoff.
