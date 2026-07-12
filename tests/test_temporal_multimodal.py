import numpy as np
import pytest
import torch
from fastapi import HTTPException

from video_commerce.common.config import RankingConfig
from video_commerce.ml.ranking import TemporalMultimodalRankingModel
from video_commerce.ml.temporal_multimodal import (
    CandidateMultimodalAttention,
    TemporalContentEncoder,
    decode_float16_matrix,
    encode_float16_matrix,
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


def test_rank_payload_rejects_unknown_multimodal_protocol_version():
    with pytest.raises(HTTPException) as exc_info:
        coerce_rank_payload({"payload_version": 4, "k": 1, "candidates": []})
    assert exc_info.value.detail == "Invalid payload_version"
