from types import SimpleNamespace

import pytest

from video_commerce.common.models import CandidateProduct, UserFeatures
from video_commerce.ml.ranking_features import FeatureBundle
from video_commerce.ml.ranking_training import (
    AttributionFacts,
    RankingTrainingExample,
)
from video_commerce.services.model_trainer.main import ModelTrainerService


def test_model_trainer_initializes_runtime_state():
    service = ModelTrainerService(SimpleNamespace())

    assert service.recommendation_engine is None
    assert service.ranking_model is None
    assert service.system_store is None
    assert service.object_storage is None
    assert service.artifact_manager is None
    assert service.pit_dataset_reader is None
    assert service.observability is not None
    assert service.running is False
    assert service.instance_id.startswith("model-trainer-")


@pytest.mark.asyncio
async def test_model_trainer_reuses_pit_pinned_candidate_embeddings(tmp_path):
    service = ModelTrainerService(
        SimpleNamespace(
            model_config=SimpleNamespace(
                ranking_model_path=str(tmp_path / "ranking.pt"),
            )
        )
    )

    class RankingModel:
        config = SimpleNamespace(trimodal_enabled=True)
        loaded_model_path = str(tmp_path / "ranking.pt")

        def configure_candidate_sidecar_for_training(self, records, *, path):
            self.records = records
            self.path = path

    ranking_model = RankingModel()
    bundle = FeatureBundle(
        as_of_ts=100.0,
        feature_definition_version="ranking_ltr_v1",
        user_features=UserFeatures(user_id="u1"),
        product_metadata={"title": "product"},
        context={},
        candidate=CandidateProduct(
            product_id="p1",
            combined_score=0.5,
            source="pit",
        ),
    )
    examples = [
        RankingTrainingExample(
            observation_id="imp-1:p1",
            impression_id="imp-1",
            bundle=bundle,
            attribution=AttributionFacts("click", True, False),
            candidate_embeddings={"text": [1.0] * 384},
        )
    ]

    attached = await service._attach_trimodal_candidate_embeddings(
        ranking_model,
        examples,
    )

    assert attached == examples
    assert ranking_model.records == {"p1": {"text": [1.0] * 384}}
    assert ranking_model.path.endswith(".candidates.npz")
