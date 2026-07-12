from types import SimpleNamespace

import pytest

from video_commerce.common.config import RankingConfig
from video_commerce.ml.legacy_training_adapter import LegacyTrainingDatasetAdapter
from video_commerce.ml.ranking import RankingModel
from video_commerce.ml.ranking_training import RankingTrainingExample


@pytest.mark.asyncio
async def test_legacy_adapter_owns_current_state_lookups_and_returns_typed_examples():
    class Store:
        calls = 0

        async def get_all_user_features_map(self):
            self.calls += 1
            return {"u1": {"total_interactions": 7}}

    store = Store()
    ranking = RankingModel(RankingConfig())
    adapter = LegacyTrainingDatasetAdapter(
        feature_store=store,
        vector_search=SimpleNamespace(product_metadata={"p1": {"price": 9.0}}),
        ranking_model=ranking,
        recommendation_engine=None,
    )

    examples = await adapter.build(
        [
            {
                "event_id": "e1",
                "user_id": "u1",
                "product_id": "p1",
                "action": "click",
                "as_of_ts": 100.0,
                "context": {},
            }
        ],
        training_sample_source="interaction_events",
    )

    assert store.calls == 1
    assert len(examples) == 1
    assert isinstance(examples[0], RankingTrainingExample)
    assert examples[0].bundle.user_features.total_interactions == 7
    assert examples[0].bundle.product_metadata["price"] == 9.0
    assert examples[0].attribution.attributed_action == "click"
    assert examples[0].is_slate_sample is False
