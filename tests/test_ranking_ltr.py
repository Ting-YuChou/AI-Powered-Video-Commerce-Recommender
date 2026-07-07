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


@pytest.mark.asyncio
async def test_ranking_loads_checkpoint_without_business_objective_metadata(tmp_path):
    source = RankingModel(RankingConfig(hidden_dims=[8]))
    await source.load_model()
    checkpoint_path = tmp_path / "legacy-ranking.pt"
    torch.save(
        {
            "model_state_dict": source.model.state_dict(),
            "config": {"architecture": "dcn", "hidden_dims": [8]},
        },
        checkpoint_path,
    )

    loaded = RankingModel(RankingConfig(hidden_dims=[8]))
    await loaded.load_model(str(checkpoint_path))

    assert loaded.is_trained
    assert loaded.ranking_objective_version == "legacy_multi_objective"
    assert loaded.value_transform_stats["global"]["count"] == 0


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


def test_forward_returns_derived_ctcvr_probability():
    ranking = RankingModel(RankingConfig(hidden_dims=[8]))
    ranking._initialize_model()
    features = torch.zeros((3, ranking.feature_extractor.total_feature_dim))

    predictions = ranking.model(features)

    assert "ctcvr" in predictions
    torch.testing.assert_close(predictions["ctcvr"], predictions["ctr"] * predictions["cvr"])


@pytest.mark.asyncio
async def test_inference_business_score_uses_ctcvr_times_predicted_value():
    ranking = RankingModel(RankingConfig(hidden_dims=[8]))
    await ranking.load_model()
    predictions, _ = ranking.run_inference_batch(
        np.zeros((3, ranking.feature_extractor.total_feature_dim), dtype=np.float32)
    )

    np.testing.assert_allclose(predictions["ctcvr"], predictions["ctr"] * predictions["cvr"])
    np.testing.assert_allclose(
        predictions["business_score"],
        predictions["ctcvr"] * predictions["predicted_value"],
    )


def test_business_predictions_inverse_transform_each_value_bucket():
    ranking = RankingModel(
        RankingConfig(value_min_bucket_purchases=1, value_clip_quantile=1.0)
    )
    ranking._fit_value_transform(
        [
            {"business_value": 10.0, "value_bucket": 0},
            {"business_value": 100.0, "value_bucket": 1},
        ]
    )
    predictions = {
        "ctr": np.array([1.0, 1.0], dtype=np.float32),
        "cvr": np.array([1.0, 1.0], dtype=np.float32),
        "ctcvr": np.array([1.0, 1.0], dtype=np.float32),
        "gmv": np.array(
            [
                ranking._transform_business_value(10.0, 0),
                ranking._transform_business_value(100.0, 1),
            ],
            dtype=np.float32,
        ),
        "ranking_score": np.array([0.0, 0.0], dtype=np.float32),
    }

    ranking._add_business_predictions(predictions, value_bucket_ids=[0, 1])

    np.testing.assert_allclose(predictions["predicted_value"], [10.0, 100.0], rtol=1e-5)
    np.testing.assert_allclose(predictions["business_score"], [10.0, 100.0], rtol=1e-5)


def test_prepare_training_data_labels_ctcvr_for_all_impression_rows():
    ranking = RankingModel(
        RankingConfig(value_min_bucket_purchases=1, value_clip_quantile=1.0)
    )
    samples = [
        {"user_id": "u1", "product_id": "p-view", "action": "view", "context": {}},
        {"user_id": "u1", "product_id": "p-click", "action": "click", "context": {}},
        {
            "user_id": "u1",
            "product_id": "p-purchase",
            "action": "purchase",
            "context": {"profit": 25.0},
        },
    ]

    _, labels = ranking._prepare_training_data(
        samples,
        product_metadata_map={
            "p-view": {"price": 20.0, "category": "cat"},
            "p-click": {"price": 30.0, "category": "cat"},
            "p-purchase": {"price": 40.0, "category": "cat"},
        },
    )

    assert labels["ctr"].squeeze(1).tolist() == [0.0, 1.0, 1.0]
    assert labels["cvr"].squeeze(1).tolist() == [0.0, 0.0, 1.0]
    assert labels["cvr_mask"].squeeze(1).tolist() == [0.0, 1.0, 1.0]
    assert labels["ctcvr"].squeeze(1).tolist() == [0.0, 0.0, 1.0]
    assert labels["value_mask"].squeeze(1).tolist() == [0.0, 0.0, 1.0]


def test_prepare_training_data_treats_attributed_purchase_as_implicit_click():
    ranking = RankingModel(
        RankingConfig(value_min_bucket_purchases=1, value_clip_quantile=1.0)
    )

    _, labels = ranking._prepare_training_data(
        [
            {
                "user_id": "u1",
                "product_id": "p1",
                "action": "view",
                "context": {"attributed_purchase": True, "purchase_value": 75.0},
            }
        ],
        product_metadata_map={"p1": {"price": 30.0, "category": "cat"}},
    )

    assert labels["ctr"].item() == 1.0
    assert labels["cvr"].item() == 1.0
    assert labels["cvr_mask"].item() == 1.0
    assert labels["ctcvr"].item() == 1.0
    assert labels["value_mask"].item() == 1.0


def test_ctcvr_pos_weight_must_be_positive_when_configured():
    assert RankingConfig(ctcvr_pos_weight=None).ctcvr_pos_weight is None
    with pytest.raises(ValueError):
        RankingConfig(ctcvr_pos_weight=0)
    with pytest.raises(ValueError):
        RankingConfig(ctcvr_pos_weight=-1)


def test_ctcvr_loss_trains_unclicked_rows_through_derived_product():
    ranking = RankingModel(
        RankingConfig(
            ctcvr_weight=1.0,
            direct_cvr_weight=1.0,
            gmv_weight=0.0,
            ltr_pairwise_enabled=False,
        )
    )
    predictions = {
        "ctr": torch.tensor([[0.9], [0.9]], dtype=torch.float32),
        "cvr": torch.tensor([[0.9], [0.1]], dtype=torch.float32),
        "ctcvr": torch.tensor([[0.81], [0.09]], dtype=torch.float32),
        "gmv": torch.zeros((2, 1), dtype=torch.float32),
        "ranking_score": torch.zeros((2, 1), dtype=torch.float32),
    }
    labels = {
        "ctr": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "cvr": torch.tensor([[0.0], [0.0]], dtype=torch.float32),
        "cvr_mask": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "ctcvr": torch.tensor([[0.0], [0.0]], dtype=torch.float32),
        "value": torch.zeros((2, 1), dtype=torch.float32),
        "value_mask": torch.zeros((2, 1), dtype=torch.float32),
    }

    high_unclicked_cvr_loss = ranking._compute_loss(predictions, labels)

    predictions["cvr"][0, 0] = 0.1
    predictions["ctcvr"] = predictions["ctr"] * predictions["cvr"]
    low_unclicked_cvr_loss = ranking._compute_loss(predictions, labels)

    assert low_unclicked_cvr_loss < high_unclicked_cvr_loss


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


def test_listwise_ltr_loss_prefers_higher_scores_for_higher_relevance():
    ranking = RankingModel(
        RankingConfig(
            ltr_min_relevance_gap=0.5,
            ltr_listwise_min_group_size=2,
        )
    )
    relevance = torch.tensor([[3.0], [1.0]])
    group_ids = torch.tensor([0, 0])
    slate_mask = torch.tensor([True, True])

    good_scores = torch.tensor([[2.0], [0.0]], requires_grad=True)
    bad_scores = torch.tensor([[0.0], [2.0]], requires_grad=True)

    good_loss = ranking._compute_listwise_ltr_loss(
        good_scores,
        relevance,
        group_ids,
        slate_mask,
    )
    bad_loss = ranking._compute_listwise_ltr_loss(
        bad_scores,
        relevance,
        group_ids,
        slate_mask,
    )

    assert good_loss < bad_loss


def test_listwise_ltr_loss_is_zero_when_no_valid_slate_group():
    ranking = RankingModel(
        RankingConfig(
            ltr_min_relevance_gap=0.5,
            ltr_listwise_min_group_size=2,
        )
    )
    scores = torch.tensor([[0.5], [0.2]], requires_grad=True)
    relevance = torch.tensor([[3.0], [1.0]])
    group_ids = torch.tensor([0, 0])
    slate_mask = torch.tensor([False, False])

    loss = ranking._compute_listwise_ltr_loss(
        scores,
        relevance,
        group_ids,
        slate_mask,
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert scores.grad is not None
    assert torch.all(scores.grad == 0)


def test_ltr_training_batches_keep_impression_group_together():
    ranking = RankingModel(
        RankingConfig(
            batch_size=2,
            ltr_listwise_enabled=True,
            ltr_listwise_weight=0.25,
        )
    )
    features = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    labels = {
        "pairwise_group": torch.tensor([0, 0, 0, 1]),
        "ltr_group": torch.tensor([0, 0, 0, 1]),
        "ltr_is_slate_sample": torch.tensor([True, True, True, True]),
    }

    batches = list(ranking._iter_training_batches(features, labels))

    assert [batch_features.size(0) for batch_features, _ in batches] == [3, 1]
    assert torch.equal(batches[0][1]["ltr_group"], torch.tensor([0, 0, 0]))


def test_event_level_impression_context_is_not_marked_as_listwise_slate():
    ranking = RankingModel(RankingConfig())
    samples = [
        {
            "user_id": "u1",
            "product_id": "p1",
            "action": "click",
            "context": {"impression_id": "imp-1"},
        },
        {
            "user_id": "u1",
            "product_id": "p2",
            "action": "view",
            "context": {"impression_id": "imp-1"},
        },
    ]

    _, labels = ranking._prepare_training_data(
        samples,
        training_sample_source="interaction_events",
    )

    assert labels["ltr_is_slate_sample"].tolist() == [False, False]
    assert labels["ltr_group"].tolist() == [0, 1]


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
    assert set(predictions) == {
        "ctr",
        "cvr",
        "ctcvr",
        "gmv",
        "predicted_value",
        "business_score",
        "ranking_score",
    }
    assert all(values.shape == (2,) for values in predictions.values())


@pytest.mark.asyncio
async def test_ranking_training_with_listwise_ltr_infers():
    ranking = RankingModel(
        RankingConfig(
            architecture="dcn",
            hidden_dims=[16, 8],
            training_min_samples=1,
            epochs=1,
            batch_size=1,
            ltr_pairwise_enabled=False,
            ltr_listwise_enabled=True,
            ltr_listwise_weight=0.25,
            ltr_listwise_min_group_size=2,
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
            "context": {"impression_id": "imp-1", "session_position": 1},
            "combined_score": 0.7,
        },
        {
            "request_id": "req-1",
            "user_id": "u1",
            "product_id": "p2",
            "action": "view",
            "timestamp": 1_700_000_001.0,
            "context": {"impression_id": "imp-1", "session_position": 2},
            "combined_score": 0.2,
        },
    ]
    product_metadata_map = {
        "p1": {"price": 40.0, "in_stock": True},
        "p2": {"price": 20.0, "in_stock": True},
    }

    ranking._train_model_sync(
        samples,
        user_features_map={"u1": {"user_id": "u1", "total_interactions": 10}},
        product_metadata_map=product_metadata_map,
        training_sample_source="recommendation_impressions",
    )
    predictions, _ = ranking.run_inference_batch(
        np.zeros((2, ranking.feature_extractor.total_feature_dim), dtype=np.float32)
    )

    assert ranking.is_trained
    assert set(predictions) == {"ctr", "cvr", "gmv", "ranking_score"}
    assert all(values.shape == (2,) for values in predictions.values())
