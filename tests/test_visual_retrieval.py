import torch
import torch.nn.functional as F
import pytest
from types import SimpleNamespace
import numpy as np
import faiss

from video_commerce.common.models import ContentFeatures
from video_commerce.ml.ranking_training import (
    AttributionFacts,
    RankingTrainingExample,
)
from video_commerce.ml.visual_retrieval import (
    VISUAL_RETRIEVAL_SCHEMA_VERSION,
    VisualRetrievalPooler,
    VisualRetrievalTrainingExample,
    add_product_index_hard_negatives,
    build_visual_retrieval_training_examples,
    evaluate_visual_retrieval,
    load_visual_retrieval_checkpoint,
    save_visual_retrieval_checkpoint,
    select_content_retrieval_embedding,
    weighted_multi_positive_info_nce,
)


def test_visual_retrieval_pooler_preserves_mean_clip_space_at_zero_gate():
    model = VisualRetrievalPooler(
        input_dim=4,
        model_dim=4,
        num_layers=1,
        num_heads=1,
        dropout=0.0,
    )
    model.residual_gate_logit.data.fill_(-30.0)
    frames = F.normalize(
        torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ]
        ),
        dim=-1,
    )
    starts = torch.tensor([[0.0, 2.0, 0.0]])
    mask = torch.tensor([[True, True, False]])

    output = model(frames, starts, mask)

    expected = F.normalize(frames[:, :2].mean(dim=1), dim=-1)
    torch.testing.assert_close(output.embedding, expected, atol=1e-6, rtol=1e-6)
    assert output.embedding.shape == (1, 4)
    torch.testing.assert_close(
        output.attention_weights.sum(dim=1),
        torch.ones(1),
    )
    assert output.attention_weights[0, 2].item() == 0.0


def test_visual_retrieval_pooler_handles_missing_rows_without_nan():
    model = VisualRetrievalPooler(
        input_dim=4,
        model_dim=4,
        num_layers=1,
        num_heads=1,
        dropout=0.0,
    )

    output = model(
        torch.zeros(2, 1, 4),
        torch.zeros(2, 1),
        torch.tensor([[False], [True]]),
    )

    assert torch.isfinite(output.embedding).all()
    torch.testing.assert_close(output.embedding[0], torch.zeros(4))
    assert output.attention_weights[0, 0].item() == 0.0


def test_visual_retrieval_pooler_trains_temporal_scores_and_residual_gate():
    torch.manual_seed(7)
    model = VisualRetrievalPooler(
        input_dim=4,
        model_dim=4,
        num_layers=1,
        num_heads=1,
        dropout=0.0,
    )
    frames = F.normalize(torch.randn(2, 3, 4), dim=-1)
    starts = torch.tensor([[0.0, 1.0, 3.0], [0.0, 2.0, 4.0]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    positives = F.normalize(torch.randn(2, 4), dim=-1)

    output = model(frames, starts, mask)
    loss = -(output.embedding * positives).sum(dim=1).mean()
    loss.backward()

    assert model.residual_gate_logit.grad is not None
    assert model.residual_gate_logit.grad.abs().item() > 0.0
    score_gradients = [
        parameter.grad
        for parameter in model.temporal_encoder.pool_score.parameters()
        if parameter.grad is not None
    ]
    assert score_gradients
    assert sum(gradient.abs().sum().item() for gradient in score_gradients) > 0.0


def test_weighted_multi_positive_info_nce_masks_same_content_false_negatives():
    queries = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            requires_grad=True,
        ),
        dim=-1,
    )
    products = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
            ]
        ),
        dim=-1,
    )

    result = weighted_multi_positive_info_nce(
        queries,
        products,
        content_ids=["video-1", "video-1", "video-2"],
        positive_weights=torch.tensor([1.0, 3.0, 1.0]),
        temperature=0.1,
    )

    assert torch.isfinite(result.loss)
    assert result.valid_negative_mask[0, 1].item() is False
    assert result.valid_negative_mask[1, 0].item() is False
    assert result.valid_negative_mask[0, 2].item() is True
    result.loss.backward()


def test_info_nce_masks_same_product_across_different_content():
    result = weighted_multi_positive_info_nce(
        F.normalize(torch.eye(2), dim=-1),
        F.normalize(torch.eye(2), dim=-1),
        content_ids=["video-1", "video-2"],
        positive_product_ids=["popular", "popular"],
        positive_weights=torch.ones(2),
        temperature=0.1,
    )

    assert result.valid_negative_mask[0, 1].item() is False
    assert result.valid_negative_mask[1, 0].item() is False


def test_retrieval_embedding_selection_is_lineage_checked_and_canary_stable():
    features = ContentFeatures(
        content_id="video-1",
        visual_embedding=[1.0, 0.0],
        retrieval_visual_embedding=[0.0, 1.0],
        retrieval_model_version="retrieval-7",
        retrieval_product_index_version="products-42",
        multimodal_schema_version="temporal_multimodal_v3",
    )

    first, first_source = select_content_retrieval_embedding(
        features,
        enabled=True,
        canary_percent=100.0,
        expected_model_version="retrieval-7",
        expected_product_index_version="products-42",
    )
    second, second_source = select_content_retrieval_embedding(
        features,
        enabled=True,
        canary_percent=100.0,
        expected_model_version="retrieval-7",
        expected_product_index_version="products-42",
    )

    assert first == second == [0.0, 1.0]
    assert first_source == second_source == "attention"

    fallback, fallback_source = select_content_retrieval_embedding(
        features,
        enabled=True,
        canary_percent=100.0,
        expected_model_version="retrieval-8",
        expected_product_index_version="products-42",
    )
    assert fallback == [1.0, 0.0]
    assert fallback_source == "mean"


def test_retrieval_embedding_selection_defaults_to_mean_when_disabled():
    features = ContentFeatures(
        content_id="video-1",
        visual_embedding=[1.0, 0.0],
        retrieval_visual_embedding=[0.0, 1.0],
        retrieval_model_version="retrieval-7",
        retrieval_product_index_version="products-42",
        multimodal_schema_version="temporal_multimodal_v3",
    )

    embedding, source = select_content_retrieval_embedding(
        features,
        enabled=False,
        canary_percent=100.0,
        expected_model_version="retrieval-7",
        expected_product_index_version="products-42",
    )

    assert embedding == [1.0, 0.0]
    assert source == "mean"

    zero_canary, zero_source = select_content_retrieval_embedding(
        features,
        enabled=True,
        canary_percent=0.0,
        expected_model_version="retrieval-7",
        expected_product_index_version="products-42",
    )
    assert zero_canary == [1.0, 0.0]
    assert zero_source == "mean"


def _ranking_example(bundle, *, action, content, image, available_at):
    clicked = action in {"click", "add_to_cart", "purchase"}
    return RankingTrainingExample(
        observation_id=f"obs-{action}",
        impression_id="imp-1",
        bundle=bundle,
        attribution=AttributionFacts(
            attributed_action=action,
            attributed_click=clicked,
            attributed_purchase=action == "purchase",
        ),
        multimodal_content=content,
        candidate_embeddings={
            "image": image,
            "available_at": available_at,
        },
    )


def test_visual_retrieval_training_examples_are_pit_checked_and_weighted():
    ranking_feature_bundle = SimpleNamespace(
        as_of_ts=100.0,
        candidate=SimpleNamespace(product_id="product-1"),
    )
    content = ContentFeatures(
        content_id="video-1",
        visual_embedding=[1.0] * 512,
        frame_embeddings=[[1.0] * 512, [0.5] * 512],
        frame_timestamps_seconds=[1.0, 4.0],
    )
    examples = [
        _ranking_example(
            ranking_feature_bundle,
            action="click",
            content=content,
            image=[1.0] * 512,
            available_at=ranking_feature_bundle.as_of_ts,
        ),
        _ranking_example(
            ranking_feature_bundle,
            action="add_to_cart",
            content=content,
            image=[1.0] * 512,
            available_at=ranking_feature_bundle.as_of_ts - 1,
        ),
        _ranking_example(
            ranking_feature_bundle,
            action="purchase",
            content=content,
            image=[1.0] * 512,
            available_at=ranking_feature_bundle.as_of_ts - 2,
        ),
        _ranking_example(
            ranking_feature_bundle,
            action="view",
            content=content,
            image=[1.0] * 512,
            available_at=ranking_feature_bundle.as_of_ts,
        ),
    ]

    built = build_visual_retrieval_training_examples(examples)

    assert [example.positive_weight for example in built] == [1.0, 2.0, 3.0]
    assert all(example.content_id == "video-1" for example in built)
    assert all(len(example.frame_embeddings) == 2 for example in built)


def test_visual_retrieval_training_rejects_future_product_embedding():
    ranking_feature_bundle = SimpleNamespace(
        as_of_ts=100.0,
        candidate=SimpleNamespace(product_id="product-1"),
    )
    content = ContentFeatures(
        content_id="video-1",
        visual_embedding=[1.0] * 512,
        frame_embeddings=[[1.0] * 512],
        frame_timestamps_seconds=[1.0],
    )
    leaking = _ranking_example(
        ranking_feature_bundle,
        action="click",
        content=content,
        image=[1.0] * 512,
        available_at=ranking_feature_bundle.as_of_ts + 1,
    )

    with pytest.raises(ValueError, match="future"):
        build_visual_retrieval_training_examples([leaking])


def test_visual_retrieval_checkpoint_is_strict_and_lineage_locked(tmp_path):
    model = VisualRetrievalPooler(
        input_dim=4,
        model_dim=4,
        num_layers=1,
        num_heads=1,
        dropout=0.0,
    )
    lineage = {
        "parent_ranker_checkpoint": "ranker-7",
        "clip_model_id": "clip-test",
        "clip_revision": "clip-revision",
        "product_index_version": "products-42",
        "product_index_sha256": "a" * 64,
        "catalog_activation_id": "catalog-42",
        "pit_manifest_uri": "s3://bucket/pit/manifest.json",
        "pit_sidecar_sha256": "b" * 64,
        "training_cutoff": 1234.0,
    }
    checkpoint_path = tmp_path / "retrieval.pt"

    digest = save_visual_retrieval_checkpoint(
        checkpoint_path,
        model=model,
        model_version="retrieval-7",
        lineage=lineage,
    )

    assert len(digest) == 64
    loaded, metadata = load_visual_retrieval_checkpoint(
        checkpoint_path,
        expected_sha256=digest,
        expected_lineage=lineage,
    )
    assert metadata["schema_version"] == VISUAL_RETRIEVAL_SCHEMA_VERSION
    assert metadata["model_version"] == "retrieval-7"
    assert loaded.input_dim == 4

    with pytest.raises(ValueError, match="lineage"):
        load_visual_retrieval_checkpoint(
            checkpoint_path,
            expected_sha256=digest,
            expected_lineage={**lineage, "clip_revision": "wrong"},
        )


def test_visual_retrieval_evaluation_reports_recall_mrr_and_coverage():
    product_ids = ["p1", "p2", "p3"]
    products = np.eye(3, dtype=np.float32)
    queries = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    metrics = evaluate_visual_retrieval(
        queries,
        relevant_product_ids=[{"p1"}, {"p3"}],
        product_embeddings=products,
        product_ids=product_ids,
        cutoffs=(1, 2),
        expected_catalog_size=4,
    )

    assert metrics.recall_at == {1: 0.5, 2: 0.5}
    assert metrics.mrr == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    assert metrics.catalog_coverage == pytest.approx(0.75)


def test_product_index_hard_negatives_exclude_all_same_content_positives():
    matrix = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(2)
    index.add(matrix)
    bundle = SimpleNamespace(
        index=index,
        product_index_map={0: "p1", 1: "p2", 2: "p3"},
        product_embeddings={
            product_id: matrix[row] for row, product_id in enumerate(("p1", "p2", "p3"))
        },
        product_metadata={
            "p1": {"category": "shoes", "available_at": 0.0},
            "p2": {"category": "shoes", "available_at": 0.0},
            "p3": {"category": "bags", "available_at": 0.0},
        },
        manifest={"catalog_available_at": 0.0},
    )
    base = dict(
        observation_id="obs",
        impression_id="imp",
        content_id="video",
        as_of_ts=10.0,
        product_available_at=5.0,
        frame_embeddings=((1.0, 0.0),),
        frame_timestamps_seconds=(0.0,),
        positive_embedding=(1.0, 0.0),
        positive_weight=1.0,
    )
    positives = [
        VisualRetrievalTrainingExample(product_id="p1", **base),
        VisualRetrievalTrainingExample(product_id="p2", **base),
    ]

    negatives = add_product_index_hard_negatives(
        {},
        positives,
        product_index_bundle=bundle,
        neighbors_per_content=2,
    )

    assert [product_id for product_id, _ in negatives[("video", "imp")]] == ["p3"]

    bundle.product_metadata["p3"]["available_at"] = 20.0
    future_filtered = add_product_index_hard_negatives(
        {},
        positives,
        product_index_bundle=bundle,
        neighbors_per_content=2,
    )
    assert ("video", "imp") not in future_filtered
