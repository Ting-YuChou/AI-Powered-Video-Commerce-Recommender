from types import SimpleNamespace

import numpy as np
import pytest
import torch

from video_commerce.common.config import RankingConfig
from video_commerce.ml.din import DeepInterestNetwork
from video_commerce.ml.ranking import MultiObjectiveRankingModel
from video_commerce.ml.ranking_onnx import export_ranking_onnx


def _config(**overrides):
    values = {
        "architecture": "dcn",
        "hidden_dims": [32, 16],
        "dropout_rate": 0.0,
        "cross_layers": 2,
        "low_rank_dim": 4,
    }
    values.update(overrides)
    return RankingConfig(**values)


def test_din_export_path_matches_eager_for_empty_sparse_and_full_histories():
    torch.manual_seed(7)
    embeddings = torch.randn(17, 128)
    embeddings[0].zero_()
    din = DeepInterestNetwork(embeddings).eval()
    candidates = torch.tensor([1, 2], dtype=torch.long)
    recency = torch.rand(2, 3, 60)

    for mask in (
        torch.zeros(2, 3, 60, dtype=torch.bool),
        torch.cat(
            [
                torch.zeros(2, 3, 50, dtype=torch.bool),
                torch.ones(2, 3, 10, dtype=torch.bool),
            ],
            dim=-1,
        ),
        torch.ones(2, 3, 60, dtype=torch.bool),
    ):
        history = torch.randint(1, 17, (2, 3, 60), dtype=torch.long)
        history = torch.where(mask, history, torch.zeros_like(history))

        eager = din(candidates, history, recency, mask)
        exported = din.forward_onnx(candidates, history, recency, mask)

        torch.testing.assert_close(exported, eager, atol=1e-5, rtol=1e-4)


def test_exported_base_onnx_matches_pytorch_for_required_batch_sizes(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(11)
    model = MultiObjectiveRankingModel(28, _config()).eval()
    ranking = SimpleNamespace(
        model=model,
        is_trained=True,
        artifact_verified=True,
        feature_schema_version="ranking_v3_00_temporal_multimodal",
        model_version="ranking-export-test",
        config=_config(din_enabled=False, trimodal_enabled=False),
        feature_extractor=SimpleNamespace(total_feature_dim=28),
        din_sidecar_metadata={},
    )
    target = tmp_path / "ranking.onnx"

    metadata = export_ranking_onnx(ranking, target)

    assert target.is_file()
    assert metadata["opset"] == 17
    assert metadata["parity_verified"] is True
    assert metadata["validated_batch_sizes"] == [1, 20, 64]
    assert metadata["model_architecture"] == "dcn"
    assert metadata["target_triton_version"] == "26.07"
    assert metadata["target_onnxruntime_version"] == "1.27.0"
    assert metadata["output_contract"] == {
        "ctr": [None, 1],
        "cvr": [None, 1],
        "ctcvr": [None, 1],
        "gmv": [None, 1],
        "ranking_score": [None, 1],
    }
    assert metadata["output_names"] == [
        "ctr",
        "cvr",
        "ctcvr",
        "gmv",
        "ranking_score",
    ]


def test_exported_din_onnx_matches_empty_sparse_and_full_history(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(23)
    config = _config(din_enabled=True, trimodal_enabled=False)
    embeddings = torch.randn(17, 128)
    embeddings[0].zero_()
    model = MultiObjectiveRankingModel(
        28,
        config,
        din_item_embeddings=embeddings,
    ).eval()
    ranking = SimpleNamespace(
        model=model,
        is_trained=True,
        artifact_verified=True,
        feature_schema_version="ranking_v3_din",
        model_version="ranking-din-export-test",
        config=config,
        feature_extractor=SimpleNamespace(total_feature_dim=28),
        din_sidecar_metadata={"sha256": "a" * 64},
    )

    metadata = export_ranking_onnx(ranking, tmp_path / "ranking-din.onnx")

    assert metadata["din_enabled"] is True
    assert metadata["din_sidecar_sha256"] == "a" * 64
    assert metadata["input_contract"]["history_indices"] == [None, 3, 60]


def test_export_refuses_untrained_or_trimodal_models(tmp_path):
    model = MultiObjectiveRankingModel(28, _config()).eval()
    ranking = SimpleNamespace(
        model=model,
        is_trained=False,
        artifact_verified=True,
        feature_schema_version="ranking_v3_00_temporal_multimodal",
        model_version="ranking-export-test",
        config=_config(trimodal_enabled=False),
        feature_extractor=SimpleNamespace(total_feature_dim=28),
        din_sidecar_metadata={},
    )
    with pytest.raises(ValueError, match="trained"):
        export_ranking_onnx(ranking, tmp_path / "untrained.onnx")

    ranking.is_trained = True
    ranking.config.trimodal_enabled = True
    with pytest.raises(ValueError, match="trimodal"):
        export_ranking_onnx(ranking, tmp_path / "trimodal.onnx")


def test_export_refuses_unsupported_model_architecture(tmp_path):
    config = _config(architecture="mlp")
    model = MultiObjectiveRankingModel(28, config).eval()
    ranking = SimpleNamespace(
        model=model,
        is_trained=True,
        artifact_verified=True,
        feature_schema_version="ranking_v3_00_temporal_multimodal",
        model_version="ranking-export-test",
        config=config,
        feature_extractor=SimpleNamespace(total_feature_dim=28),
        din_sidecar_metadata={},
    )

    with pytest.raises(ValueError, match="DCN"):
        export_ranking_onnx(ranking, tmp_path / "unsupported.onnx")
