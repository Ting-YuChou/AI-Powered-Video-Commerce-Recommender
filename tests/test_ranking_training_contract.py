import math

import pytest

from video_commerce.ml.ranking_training import (
    RANKING_LABEL_DEFINITION_VERSION,
    AttributionFacts,
    RankingLabelBuilder,
    RankingTrainingExample,
    TrainingTensorBuilder,
)
from video_commerce.common.models import CandidateProduct, UserFeatures
from video_commerce.ml.ranking_features import FeatureBundle, RankingFeatureAssembler
from video_commerce.ml.ranking import FeatureExtractor


@pytest.mark.parametrize(
    "action,click,purchase,ctr,cvr,cvr_mask,relevance",
    [
        ("view", False, False, 0.0, 0.0, 0.0, 1.0),
        ("click", True, False, 1.0, 0.0, 1.0, 2.0),
        ("add_to_cart", True, False, 1.0, 0.0, 1.0, 3.0),
        ("purchase", True, True, 1.0, 1.0, 1.0, 4.0),
    ],
)
def test_label_builder_uses_only_finalized_attribution_facts(
    action, click, purchase, ctr, cvr, cvr_mask, relevance
):
    labels = RankingLabelBuilder().build(
        AttributionFacts(
            attributed_action=action,
            attributed_click=click,
            attributed_purchase=purchase,
        )
    )

    assert labels.label_definition_version == RANKING_LABEL_DEFINITION_VERSION
    assert labels.ctr == ctr
    assert labels.cvr == cvr
    assert labels.ctcvr == cvr
    assert labels.cvr_mask == cvr_mask
    assert labels.value_mask == 0.0
    assert labels.relevance == relevance


def test_label_builder_masks_missing_actual_purchase_value():
    labels = RankingLabelBuilder().build(
        AttributionFacts(
            attributed_action="purchase",
            attributed_click=True,
            attributed_purchase=True,
            attributed_value=None,
            attributed_value_source=None,
        )
    )

    assert labels.business_value == 0.0
    assert labels.value_mask == 0.0
    assert labels.relevance == 4.0


def test_label_builder_uses_actual_purchase_value_without_catalog_fallback():
    labels = RankingLabelBuilder().build(
        AttributionFacts(
            attributed_action="purchase",
            attributed_click=True,
            attributed_purchase=True,
            attributed_value=35.0,
            attributed_value_source="purchase_value",
        )
    )

    assert labels.business_value == 35.0
    assert labels.value_mask == 1.0
    assert labels.relevance == pytest.approx(4.0 + math.log1p(35.0))


def test_attribution_facts_reject_inconsistent_purchase_state():
    with pytest.raises(ValueError, match="purchase attribution"):
        AttributionFacts(
            attributed_action="purchase",
            attributed_click=False,
            attributed_purchase=True,
        )


def test_training_tensor_builder_uses_shared_assembler_and_impression_groups():
    assembler = RankingFeatureAssembler(FeatureExtractor())
    bundle = FeatureBundle(
        as_of_ts=100.0,
        feature_definition_version="ranking_ltr_v1",
        user_features=UserFeatures(user_id="u1", last_active=90.0),
        product_metadata={"price": 9.0, "created_at": 50.0},
        context={},
        candidate=CandidateProduct(product_id="p1", combined_score=0.5, source="pit"),
    )
    examples = [
        RankingTrainingExample(
            observation_id="imp-1:p1",
            impression_id="imp-1",
            bundle=bundle,
            attribution=AttributionFacts("click", True, False),
        ),
        RankingTrainingExample(
            observation_id="imp-1:p2",
            impression_id="imp-1",
            bundle=bundle,
            attribution=AttributionFacts("view", False, False),
        ),
    ]

    features, labels = TrainingTensorBuilder(assembler).build(examples)

    assert features.shape == (2, assembler.extractor.total_feature_dim)
    assert labels["ctr"].squeeze(1).tolist() == [1.0, 0.0]
    assert labels["pairwise_group"].tolist() == [0, 0]
    assert labels["ltr_group"].tolist() == [0, 0]
    assert labels["ltr_is_slate_sample"].tolist() == [True, True]


@pytest.mark.asyncio
async def test_public_ranking_train_model_rejects_untyped_rows():
    from video_commerce.common.config import RankingConfig
    from video_commerce.ml.ranking import RankingModel

    ranking = RankingModel(RankingConfig(training_min_samples=1))

    with pytest.raises(TypeError, match="typed training examples"):
        await ranking.train_model(
            [{"user_id": "u1", "product_id": "p1", "action": "click"}]
        )
