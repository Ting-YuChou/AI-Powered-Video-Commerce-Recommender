import numpy as np
import base64
import pytest
import torch
from fastapi import HTTPException

from video_commerce.common.config import RankingConfig
from video_commerce.ml.ranking import (
    TemporalMultimodalRankingModel,
    TemporalTrimodalRankingModel,
)
from video_commerce.ml.temporal_multimodal import (
    CandidateMultimodalAttention,
    TemporalSequenceEncoder,
    TrimodalCandidateAttention,
    TemporalContentEncoder,
    decode_float16_matrix,
    encode_float16_matrix,
    pack_content_features,
    unpack_temporal_multimodal_context,
)
from video_commerce.ml.video_ocr import (
    OCRRegion,
    PaddleOCRRegionExtractor,
    TemporalOCRTracker,
)
from video_commerce.ranking_runtime.ranking_payloads import coerce_rank_payload


def test_float16_matrix_codec_validates_shape_and_round_trips():
    values = np.arange(12, dtype=np.float32).reshape(3, 4) / 7.0

    payload = encode_float16_matrix(values)
    decoded = decode_float16_matrix(payload, rows=3, columns=4)

    assert decoded.shape == (3, 4)
    np.testing.assert_allclose(decoded, values, atol=1e-3)
    with pytest.raises(ValueError, match="byte length"):
        decode_float16_matrix(payload[:-1], rows=3, columns=4)


def test_temporal_encoder_masks_padding_and_returns_normalized_pooling():
    torch.manual_seed(7)
    encoder = TemporalContentEncoder(
        input_dim=4,
        model_dim=8,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
    ).eval()
    frames = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [9.0, 9.0, 9.0, 9.0]]]
    )
    timestamps = torch.tensor([[0.0, 2.0, 99.0]])
    mask = torch.tensor([[True, True, False]])

    output = encoder(frames, timestamps, mask)

    assert output.tokens.shape == (1, 3, 8)
    assert output.pooled_embedding.shape == (1, 4)
    assert output.attention_weights.shape == (1, 3)
    assert output.attention_weights[0, 2].item() == 0.0
    assert output.attention_weights[0, :2].sum().item() == pytest.approx(1.0)
    assert torch.linalg.vector_norm(
        output.pooled_embedding, dim=-1
    ).item() == pytest.approx(1.0)


def test_temporal_sequence_encoder_supports_spans_and_fully_missing_rows():
    torch.manual_seed(17)
    encoder = TemporalSequenceEncoder(
        input_dim=4, model_dim=8, num_layers=1, num_heads=2, dropout=0.0
    ).eval()
    output = encoder(
        torch.randn(2, 3, 4),
        torch.tensor([[0.0, 2.0, 4.0], [0.0, 0.0, 0.0]]),
        torch.tensor([[True, True, False], [False, False, False]]),
        end_timestamps_seconds=torch.tensor([[1.0, 3.5, 4.0], [0.0, 0.0, 0.0]]),
    )

    assert output.tokens.shape == (2, 3, 8)
    assert output.pooled_embedding.shape == (2, 8)
    assert torch.count_nonzero(output.tokens[1]) == 0
    assert torch.count_nonzero(output.pooled_embedding[1]) == 0
    assert torch.isfinite(output.tokens).all()


def test_trimodal_attention_masks_absent_modalities_and_backpropagates():
    torch.manual_seed(19)
    module = TrimodalCandidateAttention(
        model_dim=8,
        candidate_image_dim=4,
        candidate_text_dim=6,
        candidate_two_tower_dim=3,
        num_heads=2,
        dropout=0.0,
    )
    visual = torch.randn(2, 2, 8, requires_grad=True)
    ocr = torch.randn(2, 2, 8, requires_grad=True)
    asr = torch.randn(2, 2, 8, requires_grad=True)
    output = module(
        visual_tokens=visual,
        visual_pooled=visual.mean(1),
        visual_mask=torch.tensor([[True, True], [False, False]]),
        ocr_tokens=ocr,
        ocr_pooled=ocr.mean(1),
        ocr_mask=torch.tensor([[True, False], [False, False]]),
        asr_tokens=asr,
        asr_pooled=asr.mean(1),
        asr_mask=torch.tensor([[True, True], [False, False]]),
        candidate_image=torch.randn(2, 4),
        candidate_text=torch.randn(2, 6),
        candidate_two_tower=torch.randn(2, 3),
        candidate_presence=torch.ones(2, 3, dtype=torch.bool),
    )

    assert output.fused.shape == (2, 8)
    assert output.modality_gate.shape == (2, 3)
    assert output.modality_gate[0].sum().item() == pytest.approx(1.0)
    assert torch.count_nonzero(output.modality_gate[1]) == 0
    assert torch.count_nonzero(output.fused[1]) == 0
    output.fused[0].sum().backward()
    assert visual.grad is not None and torch.count_nonzero(visual.grad)
    assert ocr.grad is not None and torch.count_nonzero(ocr.grad)
    assert asr.grad is not None and torch.count_nonzero(asr.grad)


def test_candidate_attention_is_candidate_specific_and_masks_missing_text():
    torch.manual_seed(11)
    module = CandidateMultimodalAttention(
        visual_input_dim=4,
        text_input_dim=4,
        two_tower_dim=2,
        model_dim=8,
        num_heads=2,
        dropout=0.0,
    ).eval()
    visual_tokens = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]).repeat(
        2, 1, 1
    )
    visual_mask = torch.ones(2, 2, dtype=torch.bool)
    candidate_image = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    candidate_text = torch.zeros(2, 4)
    candidate_two_tower = torch.zeros(2, 2)
    ocr_tokens = torch.zeros(2, 1, 4)
    ocr_mask = torch.zeros(2, 1, dtype=torch.bool)

    output = module(
        visual_tokens=visual_tokens,
        visual_mask=visual_mask,
        candidate_image=candidate_image,
        candidate_text=candidate_text,
        candidate_two_tower=candidate_two_tower,
        ocr_tokens=ocr_tokens,
        ocr_mask=ocr_mask,
    )

    assert output.fused.shape == (2, 8)
    assert output.visual_attention.shape == (2, 2)
    assert not torch.allclose(output.visual_attention[0], output.visual_attention[1])
    assert torch.count_nonzero(output.text_attention) == 0


def _region(text, *, frame, timestamp, x=0.1, confidence=0.9):
    return OCRRegion(
        text=text,
        polygon=[(x, 0.1), (x + 0.4, 0.1), (x + 0.4, 0.2), (x, 0.2)],
        detection_confidence=confidence,
        recognition_confidence=confidence,
        frame_index=frame,
        timestamp_seconds=timestamp,
    )


def test_temporal_ocr_tracker_deduplicates_short_tracks_and_keeps_first_anchor():
    tracker = TemporalOCRTracker(
        text_similarity_threshold=0.8,
        polygon_iou_threshold=0.3,
        max_missed_frames=1,
        max_gap_seconds=5.0,
    )

    tracks = tracker.track(
        [
            _region("SALE 50%", frame=0, timestamp=0.0, confidence=0.7),
            _region("SALE 50%", frame=1, timestamp=1.0, confidence=0.95),
            _region("SALE 50%", frame=4, timestamp=10.0, confidence=0.9),
        ]
    )

    assert len(tracks) == 2
    assert tracks[0].text == "SALE 50%"
    assert tracks[0].first_seen_seconds == 0.0
    assert tracks[0].last_seen_seconds == 1.0
    assert tracks[0].occurrence_count == 2
    assert tracks[0].polygon[0] == (0.1, 0.1)


def test_temporal_ocr_tracker_requires_text_and_spatial_match():
    tracker = TemporalOCRTracker()

    tracks = tracker.track(
        [
            _region("BUY NOW", frame=0, timestamp=0.0, x=0.1),
            _region("BUY NOW", frame=1, timestamp=1.0, x=0.7),
            _region("SOLD OUT", frame=1, timestamp=1.0, x=0.1),
        ]
    )

    assert len(tracks) == 3


def test_paddle_region_extractor_filters_scores_and_normalizes_polygons():
    class FakePipeline:
        def predict(self, frame):
            return [
                {
                    "res": {
                        "dt_polys": [
                            [[10, 10], [50, 10], [50, 20], [10, 20]],
                            [[0, 0], [5, 0], [5, 5], [0, 5]],
                        ],
                        "dt_scores": [0.9, 0.2],
                        "rec_texts": ["SALE", "noise"],
                        "rec_scores": [0.8, 0.9],
                    }
                }
            ]

    extractor = PaddleOCRRegionExtractor(pipeline=FakePipeline())
    regions = extractor.extract(
        np.zeros((100, 200, 3), dtype=np.uint8),
        frame_index=3,
        timestamp_seconds=1.5,
    )

    assert len(regions) == 1
    assert regions[0].text == "SALE"
    assert regions[0].polygon[0] == (0.05, 0.1)
    assert regions[0].frame_index == 3


def test_temporal_multimodal_ranker_returns_candidate_attention_and_scores():
    model = TemporalMultimodalRankingModel(
        6,
        RankingConfig(hidden_dims=[16], architecture="mlp", dropout_rate=0.0),
        visual_token_dim=4,
        clip_dim=4,
        two_tower_dim=2,
        multimodal_dim=8,
    ).eval()
    output = model(
        torch.zeros(2, 6),
        visual_tokens=torch.randn(2, 3, 4),
        visual_mask=torch.tensor([[True, True, False], [True, True, True]]),
        candidate_image=torch.randn(2, 4),
        candidate_text=torch.randn(2, 4),
        candidate_two_tower=torch.randn(2, 2),
        ocr_tokens=torch.randn(2, 2, 4),
        ocr_mask=torch.tensor([[True, False], [True, True]]),
    )

    assert output["ranking_score"].shape == (2, 1)
    assert output["visual_attention"].shape == (2, 3)
    assert output["text_attention"].shape == (2, 2)


def test_trimodal_ranker_zero_residual_matches_base_and_all_branches_get_gradients():
    torch.manual_seed(23)
    model = TemporalTrimodalRankingModel(
        6,
        RankingConfig(hidden_dims=[16], architecture="mlp", dropout_rate=0.0),
        model_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    base = torch.randn(2, 6)
    multimodal = {
        "visual_embeddings": torch.randn(2, 2, 512),
        "visual_starts": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        "visual_ends": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        "visual_mask": torch.ones(2, 2, dtype=torch.bool),
        "ocr_embeddings": torch.randn(2, 2, 384),
        "ocr_starts": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        "ocr_ends": torch.tensor([[0.5, 1.5], [0.5, 1.5]]),
        "ocr_mask": torch.ones(2, 2, dtype=torch.bool),
        "asr_embeddings": torch.randn(2, 2, 384),
        "asr_starts": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        "asr_ends": torch.tensor([[0.5, 1.5], [0.5, 1.5]]),
        "asr_mask": torch.ones(2, 2, dtype=torch.bool),
        "candidate_image": torch.randn(2, 512),
        "candidate_text": torch.randn(2, 384),
        "candidate_two_tower": torch.randn(2, 128),
        "candidate_presence": torch.ones(2, 3, dtype=torch.bool),
    }
    with torch.no_grad():
        model.residual_gate_logit.fill_(-100.0)
        expected = model.ranker(base)["ranking_score"]
        actual = model(base, **multimodal)["ranking_score"]
    torch.testing.assert_close(actual, expected)

    model.residual_gate_logit.data.fill_(-2.0)
    model(base, **multimodal)["ranking_score"].sum().backward()
    for encoder in (model.visual_encoder, model.ocr_encoder, model.asr_encoder):
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for parameter in encoder.parameters()
        )


def test_rank_payload_accepts_bounded_v4_trimodal_context():
    packed = base64.b64encode(encode_float16_matrix(np.ones((2, 4)))).decode()
    payload = coerce_rank_payload(
        {
            "payload_version": 4,
            "k": 1,
            "candidates": [],
            "user_features": {},
            "multimodal_context": {
                "schema_version": "temporal_multimodal_v2",
                "visual": {
                    "data": packed,
                    "rows": 2,
                    "columns": 4,
                    "starts": [0.0, 1.0],
                    "ends": [0.0, 1.0],
                    "mask": [True, True],
                },
            },
        }
    )
    assert payload.payload_version == 4


def test_rank_payload_rejects_unknown_multimodal_protocol_version():
    with pytest.raises(HTTPException) as exc_info:
        coerce_rank_payload({"payload_version": 5, "k": 1, "candidates": []})
    assert exc_info.value.detail == "Invalid payload_version"


def test_rank_payload_v4_rejects_sequence_over_cap():
    with pytest.raises(HTTPException, match="visual rows exceed 16"):
        coerce_rank_payload(
            {
                "payload_version": 4,
                "k": 1,
                "candidates": [],
                "user_features": {},
                "multimodal_context": {
                    "schema_version": "temporal_multimodal_v2",
                    "visual": {
                        "data": "",
                        "rows": 17,
                        "columns": 512,
                        "starts": [0.0] * 17,
                        "ends": [0.0] * 17,
                        "mask": [True] * 17,
                    },
                },
            }
        )


def test_content_features_pack_to_bounded_v4_tensors():
    from video_commerce.common.models import AudioFeatures, ContentFeatures

    content = ContentFeatures(
        content_id="v1",
        visual_embedding=[0.0] * 512,
        frame_embeddings=[[1.0] * 512] * 20,
        frame_timestamps_seconds=list(range(20)),
        ocr_tracks=[
            {
                "first_seen_seconds": 1.0,
                "last_seen_seconds": 2.0,
                "text_embedding": [1.0] * 384,
            }
        ],
        audio_features=AudioFeatures(
            asr_segments=[
                {
                    "start_seconds": 2.0,
                    "end_seconds": 3.0,
                    "text_embedding": [1.0] * 384,
                }
            ]
        ),
    )
    packed = pack_content_features(content)
    assert packed["visual"]["rows"] == 16
    decoded = unpack_temporal_multimodal_context(packed)
    assert decoded["visual_embeddings"].shape == (16, 512)
    assert decoded["ocr_embeddings"].shape == (1, 384)
    assert decoded["asr_embeddings"].shape == (1, 384)
