import math

import pytest
import torch

from video_commerce.common.feature_history_contracts import (
    RANKING_LTR_DIN_FEATURE_DEFINITION_VERSION,
)
from video_commerce.ml.ranking_training import (
    RANKING_LABEL_DEFINITION_VERSION,
    AttributionFacts,
    RankingLabelBuilder,
    RankingTrainingExample,
    TrainingTensorBuilder,
)
from video_commerce.common.models import (
    AudioFeatures,
    CandidateProduct,
    ContentFeatures,
    UserFeatures,
)
from video_commerce.ml.ranking_features import FeatureBundle, RankingFeatureAssembler
from video_commerce.ml.ranking import FeatureExtractor
from video_commerce.ml.din import (
    build_din_behavior_sequences,
    save_din_embedding_sidecar,
)


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


def test_training_tensor_builder_pads_three_modalities_and_presence_masks():
    assembler = RankingFeatureAssembler(FeatureExtractor())
    bundle = FeatureBundle(
        as_of_ts=100.0,
        feature_definition_version="ranking_ltr_v1",
        user_features=UserFeatures(user_id="u1", last_active=90.0),
        product_metadata={"price": 9.0, "created_at": 50.0},
        context={},
        candidate=CandidateProduct(product_id="p1", combined_score=0.5, source="pit"),
    )
    content = ContentFeatures(
        content_id="v1",
        visual_embedding=[0.0] * 512,
        frame_embeddings=[[1.0] * 512, [2.0] * 512],
        frame_timestamps_seconds=[0.0, 2.0],
        ocr_tracks=[
            {
                "first_seen_seconds": 1.0,
                "last_seen_seconds": 3.0,
                "text_embedding": [1.0] * 384,
            }
        ],
        audio_features=AudioFeatures(
            asr_segments=[
                {
                    "start_seconds": 0.5,
                    "end_seconds": 1.5,
                    "text_embedding": [2.0] * 384,
                }
            ]
        ),
    )
    example = RankingTrainingExample(
        observation_id="imp-1:p1",
        impression_id="imp-1",
        bundle=bundle,
        attribution=AttributionFacts("click", True, False),
        multimodal_content=content,
        candidate_embeddings={
            "image": [1.0] * 512,
            "text": [1.0] * 384,
            "two_tower": [1.0] * 128,
        },
    )

    _, _, multimodal = TrainingTensorBuilder(assembler).build_trimodal([example])
    assert multimodal["visual_embeddings"].shape == (1, 16, 512)
    assert multimodal["ocr_embeddings"].shape == (1, 32, 384)
    assert multimodal["asr_embeddings"].shape == (1, 64, 384)
    assert multimodal["visual_mask"].sum().item() == 2
    assert multimodal["ocr_mask"].sum().item() == 1
    assert multimodal["asr_mask"].sum().item() == 1
    assert torch.equal(
        multimodal["candidate_presence"], torch.ones(1, 3, dtype=torch.bool)
    )


@pytest.mark.asyncio
async def test_public_ranking_train_model_rejects_untyped_rows():
    from video_commerce.common.config import RankingConfig
    from video_commerce.ml.ranking import RankingModel

    ranking = RankingModel(RankingConfig(training_min_samples=1))

    with pytest.raises(TypeError, match="typed training examples"):
        await ranking.train_model(
            [{"user_id": "u1", "product_id": "p1", "action": "click"}]
        )


def test_ranking_loss_contract_errors_fail_fast():
    from video_commerce.common.config import RankingConfig
    from video_commerce.ml.ranking import RankingModel

    ranking = RankingModel(RankingConfig())

    with pytest.raises(KeyError):
        ranking._compute_loss({}, {})


def test_trimodal_training_config_requires_joint_finetune_epoch():
    from video_commerce.common.config import RankingConfig

    with pytest.raises(ValueError, match="joint fine-tuning"):
        RankingConfig(
            trimodal_enabled=True,
            epochs=1,
            trimodal_warmup_epochs=1,
        )


def test_trimodal_modality_dropout_is_batch_local_and_preserves_visual_inputs():
    from video_commerce.common.config import RankingConfig
    from video_commerce.ml.ranking import RankingModel

    ranking = RankingModel(RankingConfig(trimodal_enabled=True))
    inputs = {
        "visual_embeddings": torch.ones(2, 2, 512),
        "visual_starts": torch.ones(2, 2),
        "visual_ends": torch.ones(2, 2),
        "visual_mask": torch.ones(2, 2, dtype=torch.bool),
        "ocr_embeddings": torch.ones(2, 2, 384),
        "ocr_starts": torch.ones(2, 2),
        "ocr_ends": torch.ones(2, 2),
        "ocr_mask": torch.ones(2, 2, dtype=torch.bool),
        "asr_embeddings": torch.ones(2, 2, 384),
        "asr_starts": torch.ones(2, 2),
        "asr_ends": torch.ones(2, 2),
        "asr_mask": torch.ones(2, 2, dtype=torch.bool),
    }

    dropped = ranking._apply_trimodal_modality_dropout(
        inputs,
        probability=1.0,
        generator=torch.Generator().manual_seed(7),
    )

    assert torch.equal(dropped["visual_embeddings"], inputs["visual_embeddings"])
    assert torch.equal(dropped["visual_mask"], inputs["visual_mask"])
    assert torch.count_nonzero(dropped["ocr_embeddings"]) == 0
    assert torch.count_nonzero(dropped["ocr_mask"]) == 0
    assert torch.count_nonzero(dropped["asr_embeddings"]) == 0
    assert torch.count_nonzero(dropped["asr_mask"]) == 0
    assert torch.count_nonzero(inputs["ocr_embeddings"]) > 0
    assert torch.count_nonzero(inputs["asr_embeddings"]) > 0


@pytest.mark.asyncio
async def test_trimodal_training_writes_v4_checkpoint_locked_to_sidecar(tmp_path):
    from video_commerce.common.config import RankingConfig
    from video_commerce.ml.ranking import RankingModel, TemporalTrimodalRankingModel

    config = RankingConfig(
        trimodal_enabled=True,
        training_min_samples=2,
        epochs=1,
        trimodal_warmup_epochs=0,
        training_min_epochs=1,
        batch_size=2,
        hidden_dims=[8],
        architecture="mlp",
        dropout_rate=0.0,
        learning_rate=0.001,
    )
    ranking = RankingModel(config)
    checkpoint = tmp_path / "ranking.pt"
    sidecar = tmp_path / "ranking.candidates.npz"
    ranking.loaded_model_path = str(checkpoint)
    ranking.configure_candidate_sidecar_for_training(
        {"p1": {"text": [1.0] * 384}}, path=str(sidecar)
    )
    bundle = FeatureBundle(
        as_of_ts=100.0,
        feature_definition_version="ranking_ltr_v1",
        user_features=UserFeatures(user_id="u1", last_active=90.0),
        product_metadata={"price": 9.0, "created_at": 50.0},
        context={},
        candidate=CandidateProduct(product_id="p1", combined_score=0.5, source="pit"),
    )
    content = ContentFeatures(
        content_id="v1",
        visual_embedding=[0.0] * 512,
        frame_embeddings=[[1.0] * 512],
        frame_timestamps_seconds=[0.0],
    )
    example = RankingTrainingExample(
        observation_id="imp-1:p1",
        impression_id="imp-1",
        bundle=bundle,
        attribution=AttributionFacts("click", True, False),
        multimodal_content=content,
        candidate_embeddings={"text": [1.0] * 384},
    )

    second = RankingTrainingExample(
        observation_id="imp-1:p1-second",
        impression_id="imp-1",
        bundle=bundle,
        attribution=AttributionFacts("view", False, False),
        multimodal_content=content,
        candidate_embeddings={"text": [1.0] * 384},
    )
    await ranking.train_model([example, second])

    assert isinstance(ranking.model, TemporalTrimodalRankingModel)
    assert checkpoint.exists() and sidecar.exists()
    saved = torch.load(checkpoint, map_location="cpu")
    assert (
        saved["config"]["feature_schema_version"] == "ranking_v4_01_temporal_trimodal"
    )
    assert len(saved["config"]["candidate_sidecar_sha256"]) == 64
    assert sorted(group["lr"] for group in ranking.optimizer.param_groups) == [
        0.0001,
        0.001,
    ]


@pytest.mark.asyncio
async def test_trimodal_and_din_training_share_one_batch_contract(tmp_path):
    from video_commerce.common.config import RankingConfig
    from video_commerce.ml.ranking import RankingModel, TemporalTrimodalRankingModel

    din_sidecar = tmp_path / "ranking-din.npz"
    save_din_embedding_sidecar(
        str(din_sidecar),
        {
            "history-product": torch.ones(128).numpy(),
            "p1": torch.full((128,), 0.5).numpy(),
        },
        two_tower_model_version="two-tower-1",
    )
    config = RankingConfig(
        trimodal_enabled=True,
        din_enabled=True,
        din_embedding_sidecar_path=str(din_sidecar),
        din_sequence_last_n=60,
        training_min_samples=2,
        epochs=2,
        batch_size=2,
        hidden_dims=[8],
        architecture="mlp",
        dropout_rate=0.0,
        learning_rate=0.001,
    )
    ranking = RankingModel(config)
    ranking.loaded_model_path = str(tmp_path / "ranking.pt")
    ranking.configure_candidate_sidecar_for_training(
        {"p1": {"text": [1.0] * 384}},
        path=str(tmp_path / "ranking.candidates.npz"),
    )
    as_of_ts = 1000.0
    behavior_sequences = build_din_behavior_sequences(
        [
            {
                "product_id": "history-product",
                "action": "click",
                "occurred_at": as_of_ts - 10.0,
                "available_at": as_of_ts - 9.0,
                "event_id": "event-1",
            }
        ],
        as_of_ts=as_of_ts,
        last_n=60,
    )
    bundle = FeatureBundle(
        as_of_ts=as_of_ts,
        feature_definition_version=RANKING_LTR_DIN_FEATURE_DEFINITION_VERSION,
        user_features=UserFeatures(user_id="u1", last_active=900.0),
        product_metadata={"price": 9.0, "created_at": 500.0},
        context={},
        candidate=CandidateProduct(product_id="p1", combined_score=0.5, source="pit"),
        behavior_sequences=behavior_sequences,
    )
    content = ContentFeatures(
        content_id="v1",
        visual_embedding=[0.0] * 512,
        frame_embeddings=[[1.0] * 512],
        frame_timestamps_seconds=[0.0],
    )
    examples = [
        RankingTrainingExample(
            observation_id=f"imp-1:p1:{index}",
            impression_id="imp-1",
            bundle=bundle,
            attribution=AttributionFacts(
                "click" if index == 0 else "view",
                index == 0,
                False,
            ),
            multimodal_content=content,
            candidate_embeddings={"text": [1.0] * 384},
        )
        for index in range(2)
    ]

    await ranking.train_model(examples)

    assert isinstance(ranking.model, TemporalTrimodalRankingModel)
    assert ranking.model.ranker.din is not None
    assert ranking.is_trained is True


@pytest.mark.asyncio
async def test_trimodal_training_cannot_stop_before_joint_finetuning(tmp_path):
    from video_commerce.common.config import RankingConfig
    from video_commerce.ml.ranking import RankingModel

    config = RankingConfig(
        trimodal_enabled=True,
        training_min_samples=2,
        epochs=3,
        batch_size=2,
        training_min_epochs=2,
        trimodal_warmup_epochs=1,
        early_stopping_patience=1,
        validation_fraction=0.5,
        hidden_dims=[8],
        architecture="mlp",
        dropout_rate=0.0,
    )
    ranking = RankingModel(config)
    ranking.loaded_model_path = str(tmp_path / "ranking.pt")
    ranking.configure_candidate_sidecar_for_training(
        {"p1": {"text": [1.0] * 384}},
        path=str(tmp_path / "ranking.candidates.npz"),
    )
    content = ContentFeatures(
        content_id="v1",
        visual_embedding=[0.0] * 512,
        frame_embeddings=[[1.0] * 512],
        frame_timestamps_seconds=[0.0],
    )
    examples = []
    for impression_index, as_of_ts in enumerate((100.0, 200.0)):
        bundle = FeatureBundle(
            as_of_ts=as_of_ts,
            feature_definition_version="ranking_ltr_v1",
            user_features=UserFeatures(user_id="u1", last_active=90.0),
            product_metadata={"price": 9.0, "created_at": 50.0},
            context={},
            candidate=CandidateProduct(
                product_id="p1",
                combined_score=0.5,
                source="pit",
            ),
        )
        for candidate_index in range(2):
            clicked = candidate_index == 0
            examples.append(
                RankingTrainingExample(
                    observation_id=(f"imp-{impression_index}:p1:{candidate_index}"),
                    impression_id=f"imp-{impression_index}",
                    bundle=bundle,
                    attribution=AttributionFacts(
                        "click" if clicked else "view",
                        clicked,
                        False,
                    ),
                    multimodal_content=content,
                    candidate_embeddings={"text": [1.0] * 384},
                )
            )

    def zero_loss(predictions, _labels):
        return predictions["ranking_score"].sum() * 0.0

    ranking._compute_loss = zero_loss
    await ranking.train_model(examples)

    metrics = ranking.training_history[-1]
    assert metrics["epochs_completed"] >= 2
    assert metrics["validation_loss"] == pytest.approx(0.0)
    assert metrics["modality_coverage"] == {
        "visual": pytest.approx(1.0),
        "ocr": pytest.approx(0.0),
        "asr": pytest.approx(0.0),
    }
    assert metrics["candidate_modality_coverage"] == {
        "image": pytest.approx(0.0),
        "text": pytest.approx(1.0),
        "two_tower": pytest.approx(0.0),
    }
    assert metrics["modality_gate_mean"] == {
        "visual": pytest.approx(1.0),
        "ocr": pytest.approx(0.0),
        "asr": pytest.approx(0.0),
    }


@pytest.mark.asyncio
async def test_training_rechecks_minimum_samples_after_time_holdout(tmp_path):
    from video_commerce.common.config import RankingConfig
    from video_commerce.ml.ranking import RankingModel

    ranking = RankingModel(
        RankingConfig(
            training_min_samples=4,
            epochs=1,
            training_min_epochs=1,
            validation_fraction=0.5,
            batch_size=2,
            hidden_dims=[8],
            architecture="mlp",
        )
    )
    ranking.loaded_model_path = str(tmp_path / "ranking.pt")
    examples = []
    for impression_id, as_of_ts, row_count in (
        ("old", 100.0, 1),
        ("new", 200.0, 3),
    ):
        bundle = FeatureBundle(
            as_of_ts=as_of_ts,
            feature_definition_version="ranking_ltr_v1",
            user_features=UserFeatures(user_id="u1"),
            product_metadata={"price": 9.0},
            context={},
            candidate=CandidateProduct(
                product_id="p1",
                combined_score=0.5,
                source="pit",
            ),
        )
        for row in range(row_count):
            examples.append(
                RankingTrainingExample(
                    observation_id=f"{impression_id}:{row}",
                    impression_id=impression_id,
                    bundle=bundle,
                    attribution=AttributionFacts("view", False, False),
                )
            )

    await ranking.train_model(examples)

    assert ranking.is_trained is False
    assert ranking.training_history == []
    assert not (tmp_path / "ranking.pt").exists()


@pytest.mark.asyncio
async def test_trimodal_activation_rejects_incomplete_checkpoint_state(tmp_path):
    from video_commerce.common.config import RankingConfig
    from video_commerce.ml.candidate_embedding_sidecar import (
        write_candidate_embedding_sidecar,
    )
    from video_commerce.ml.ranking import RankingModel

    config = RankingConfig(
        trimodal_enabled=True,
        epochs=2,
        trimodal_warmup_epochs=1,
        training_min_epochs=2,
        hidden_dims=[8],
        architecture="mlp",
    )
    checkpoint_path = tmp_path / "ranking.pt"
    sidecar_path = checkpoint_path.with_suffix(".candidates.npz")
    sidecar_sha256 = write_candidate_embedding_sidecar(
        sidecar_path,
        {"p1": {"text": [1.0] * 384}},
        model_version="ranking-v1",
    )
    ranking = RankingModel(config)
    ranking._initialize_model(architecture="mlp")
    ranking.is_trained = True
    ranking.candidate_sidecar_sha256 = sidecar_sha256
    ranking.candidate_sidecar_model_version = "ranking-v1"
    await ranking.save_model(str(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing_key = next(
        key
        for key in checkpoint["model_state_dict"]
        if key.startswith("visual_encoder.")
    )
    checkpoint["model_state_dict"].pop(missing_key)
    torch.save(checkpoint, checkpoint_path)

    loader = RankingModel(config.copy(deep=True))
    with pytest.raises(RuntimeError, match="incomplete"):
        await loader.load_model(str(checkpoint_path))
