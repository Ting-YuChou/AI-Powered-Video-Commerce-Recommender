import numpy as np
import pytest
import torch

from video_commerce.common.config import RankingConfig
from video_commerce.ml.ranking import RankingModel


def test_training_relevance_label_uses_action_strength_and_purchase_gmv():
    ranking = RankingModel(RankingConfig())

    purchase_relevance = ranking._training_relevance_label(
        {"action": "purchase", "value": 99.0},
        {"price": 10.0},
        "purchase",
    )

    assert purchase_relevance == pytest.approx(4.0 + np.log1p(99.0))
    assert ranking._training_relevance_label({}, {}, "add_to_cart") == 3.0
    assert ranking._training_relevance_label({}, {}, "click") == 2.0
    assert ranking._training_relevance_label({}, {}, "view") == 1.0
    assert ranking._training_relevance_label({}, {}, "share") == 0.0


def test_training_pairwise_group_key_precedence_and_time_bucket_fallback():
    ranking = RankingModel(RankingConfig())

    assert (
        ranking._training_pairwise_group_key(
            {
                "request_id": "req-1",
                "user_id": "user-1",
                "timestamp": 3600.0,
                "context": {"session_id": "session-1"},
            }
        )
        == "request:req-1"
    )
    assert (
        ranking._training_pairwise_group_key(
            {
                "user_id": "user-1",
                "timestamp": 3600.0,
                "context": {"session_id": "session-1"},
            }
        )
        == "session:session-1"
    )
    assert (
        ranking._training_pairwise_group_key(
            {"user_id": "user-1", "timestamp": 3700.0, "context": {}}
        )
        == "user_time:user-1:2"
    )


def test_pairwise_pair_builder_respects_cap_and_relevance_gap():
    ranking = RankingModel(
        RankingConfig(
            ltr_max_pairs_per_group=2,
            ltr_min_relevance_gap=0.5,
        )
    )
    relevance = torch.tensor([4.0, 3.0, 2.0, 1.0])
    group_ids = torch.tensor([0, 0, 0, 0])

    positive_indices, negative_indices = ranking._build_pairwise_ltr_pairs(
        relevance,
        group_ids,
    )

    assert positive_indices.numel() == 2
    assert negative_indices.numel() == 2
    assert torch.all(relevance[positive_indices] > relevance[negative_indices])


def test_pairwise_ltr_loss_prefers_higher_scores_for_higher_relevance():
    ranking = RankingModel(RankingConfig(ltr_min_relevance_gap=0.5))
    relevance = torch.tensor([[3.0], [1.0]])
    group_ids = torch.tensor([0, 0])

    good_scores = torch.tensor([[2.0], [0.0]], requires_grad=True)
    bad_scores = torch.tensor([[0.0], [2.0]], requires_grad=True)

    good_loss = ranking._compute_pairwise_ltr_loss(
        good_scores,
        relevance,
        group_ids,
    )
    bad_loss = ranking._compute_pairwise_ltr_loss(
        bad_scores,
        relevance,
        group_ids,
    )

    assert good_loss < bad_loss


def test_pairwise_ltr_loss_is_zero_when_no_valid_pairs():
    ranking = RankingModel(RankingConfig(ltr_min_relevance_gap=0.5))
    scores = torch.tensor([[0.5], [0.2]], requires_grad=True)
    relevance = torch.tensor([[1.0], [1.0]])
    group_ids = torch.tensor([0, 0])

    loss = ranking._compute_pairwise_ltr_loss(scores, relevance, group_ids)
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert scores.grad is not None
    assert torch.all(scores.grad == 0)


@pytest.mark.parametrize("pairwise_enabled", [False, True])
@pytest.mark.asyncio
async def test_ranking_training_with_and_without_pairwise_ltr_infers(pairwise_enabled):
    ranking = RankingModel(
        RankingConfig(
            architecture="dcn",
            hidden_dims=[16, 8],
            training_min_samples=1,
            epochs=1,
            batch_size=2,
            ltr_pairwise_enabled=pairwise_enabled,
            ltr_pairwise_weight=0.25,
            ltr_max_pairs_per_group=4,
        )
    )
    await ranking.load_model()

    samples = [
        {
            "request_id": "req-1",
            "user_id": "u1",
            "product_id": "p1",
            "action": "purchase",
            "value": 40.0,
            "timestamp": 1_700_000_000.0,
            "context": {"device": "mobile", "session_position": 1},
            "combined_score": 0.7,
        },
        {
            "request_id": "req-1",
            "user_id": "u1",
            "product_id": "p2",
            "action": "view",
            "timestamp": 1_700_000_001.0,
            "context": {"device": "mobile", "session_position": 2},
            "combined_score": 0.2,
        },
    ]
    product_metadata_map = {
        "p1": {
            "price": 40.0,
            "rating": 4.5,
            "num_reviews": 12,
            "in_stock": True,
            "created_at": 1_700_000_000.0,
            "tags": ["featured"],
            "brand": "brand",
            "category": "cat",
        },
        "p2": {
            "price": 20.0,
            "rating": 3.5,
            "num_reviews": 4,
            "in_stock": True,
            "created_at": 1_700_000_000.0,
            "tags": [],
            "brand": "brand",
            "category": "cat",
        },
    }

    ranking._train_model_sync(
        samples,
        user_features_map={"u1": {"user_id": "u1", "total_interactions": 10}},
        product_metadata_map=product_metadata_map,
    )
    predictions, _ = ranking.run_inference_batch(
        np.zeros((2, ranking.feature_extractor.total_feature_dim), dtype=np.float32)
    )

    assert ranking.is_trained
    assert set(predictions) == {"ctr", "cvr", "gmv", "ranking_score"}
    assert all(values.shape == (2,) for values in predictions.values())
