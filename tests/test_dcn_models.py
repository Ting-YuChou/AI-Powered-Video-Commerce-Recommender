import asyncio
import hashlib

import faiss
import numpy as np
import pytest
import torch

from video_commerce.common.config import RankingConfig
from video_commerce.ml.dcn import DeepAndCrossNetwork, LowRankCrossLayer
from video_commerce.common.models import CandidateProduct, UserFeatures
from video_commerce.ml.ranking import MultiObjectiveRankingModel, RankingModel
from video_commerce.ranking_runtime.ranking_batcher import RankingBatcher
from video_commerce.ml.two_tower import ItemFeatureEncoder, TwoTowerModel, TwoTowerTrainer


def test_deep_and_cross_network_outputs_finite_expected_shape():
    model = DeepAndCrossNetwork(
        input_dim=6,
        deep_hidden_dims=[8, 4],
        output_dim=5,
        cross_layers=2,
        dropout=0.0,
    )
    model.eval()

    output = model(torch.randn(3, 6))

    assert output.shape == (3, 5)
    assert torch.isfinite(output).all()


def test_low_rank_cross_layer_outputs_finite_shape_and_expected_parameter_count():
    layer = LowRankCrossLayer(input_dim=28, low_rank_dim=8)
    x0 = torch.randn(4, 28)
    xl = torch.randn(4, 28)

    output = layer(x0, xl)
    parameter_count = sum(parameter.numel() for parameter in layer.parameters())

    assert output.shape == (4, 28)
    assert torch.isfinite(output).all()
    assert parameter_count == 28 * 8 + 8 * 28 + 28


def test_two_tower_dcn_outputs_normalized_embeddings():
    model = TwoTowerModel(
        num_users=3,
        num_items=4,
        clip_dim=5,
        output_dim=7,
        user_hidden_dims=[9, 8],
        item_hidden_dims=[10, 8],
        architecture="dcn",
        cross_layers=2,
    )
    model.eval()

    user_embeddings = model.encode_users(
        torch.tensor([1, 2], dtype=torch.long),
        torch.zeros(2, 10),
    )
    item_embeddings = model.encode_items(
        torch.tensor([1, 2], dtype=torch.long),
        torch.zeros(2, 5),
        torch.zeros(2, 8),
    )

    assert user_embeddings.shape == (2, 7)
    assert item_embeddings.shape == (2, 7)
    np.testing.assert_allclose(
        torch.linalg.norm(user_embeddings, dim=1).detach().numpy(),
        np.ones(2),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        torch.linalg.norm(item_embeddings, dim=1).detach().numpy(),
        np.ones(2),
        rtol=1e-5,
    )


def test_two_tower_legacy_checkpoint_without_architecture_loads_as_mlp(tmp_path):
    path = tmp_path / "legacy_two_tower.pt"
    legacy_model = TwoTowerModel(
        num_users=2,
        num_items=3,
        clip_dim=5,
        output_dim=7,
        user_hidden_dims=[8],
        item_hidden_dims=[9],
        architecture="mlp",
    )
    torch.save(
        {
            "model_state_dict": legacy_model.state_dict(),
            "user_mapping": {"u1": 1, "u2": 2},
            "item_mapping": {"p1": 1, "p2": 2, "p3": 3},
            "reverse_item_mapping": {1: "p1", 2: "p2", 3: "p3"},
            "config": {
                "clip_dim": 5,
                "output_dim": 7,
                "temperature": 0.07,
                "num_users": 2,
                "num_items": 3,
            },
        },
        path,
    )

    trainer = TwoTowerTrainer(
        clip_dim=5,
        output_dim=7,
        user_hidden_dims=[8],
        item_hidden_dims=[9],
        architecture="dcn",
    )

    assert trainer.load_checkpoint(str(path)) is True
    assert trainer.model is not None
    assert trainer.model.architecture == "mlp"


def test_two_tower_dcn_checkpoint_round_trips_architecture(tmp_path):
    path = tmp_path / "two_tower_dcn.pt"
    trainer = TwoTowerTrainer(
        clip_dim=5,
        output_dim=7,
        user_hidden_dims=[8],
        item_hidden_dims=[9],
        architecture="dcn",
        cross_layers=2,
    )
    trainer.user_mapping = {"u1": 1}
    trainer.item_mapping = {"p1": 1}
    trainer.reverse_item_mapping = {1: "p1"}
    trainer.model = TwoTowerModel(
        num_users=1,
        num_items=1,
        clip_dim=5,
        output_dim=7,
        user_hidden_dims=[8],
        item_hidden_dims=[9],
        architecture="dcn",
        cross_layers=2,
    )

    trainer.save_checkpoint(str(path))
    loaded = TwoTowerTrainer(
        clip_dim=5,
        output_dim=7,
        user_hidden_dims=[8],
        item_hidden_dims=[9],
        architecture="mlp",
    )

    assert loaded.load_checkpoint(str(path)) is True
    assert loaded.model is not None
    assert loaded.model.architecture == "dcn"
    assert loaded.model.cross_layers == 2


def test_two_tower_item_side_features_use_stable_hash_buckets():
    features = ItemFeatureEncoder.encode(
        {
            "price": 10.0,
            "rating": 4.0,
            "num_reviews": 2,
            "in_stock": True,
            "created_at": 1_700_000_000.0,
            "tags": [],
            "category": "shoes",
            "brand": "Acme",
        }
    )
    expected_category = (
        int.from_bytes(
            hashlib.sha256(b"shoes").digest()[:8], byteorder="big", signed=False
        )
        % 64
    ) / 64
    expected_brand = (
        int.from_bytes(
            hashlib.sha256(b"Acme").digest()[:8], byteorder="big", signed=False
        )
        % 128
    ) / 128

    np.testing.assert_allclose(features[6], expected_category, rtol=1e-6)
    np.testing.assert_allclose(features[7], expected_brand, rtol=1e-6)


def test_two_tower_training_passes_user_embedding_to_hard_negative_sampler():
    trainer = TwoTowerTrainer(
        clip_dim=2,
        output_dim=2,
        batch_size=2,
        epochs=1,
        num_hard_negatives=1,
        num_random_negatives=0,
        hard_ratio_start=1.0,
        hard_ratio_end=1.0,
        user_hidden_dims=[8],
        item_hidden_dims=[8],
        architecture="dcn",
    )
    trainer.prepare(
        interactions=[
            {"user_id": "u1", "product_id": "p1", "action": "view"},
            {"user_id": "u2", "product_id": "p2", "action": "view"},
        ],
        product_metadata={
            "p1": {"brand": "a", "category": "cat"},
            "p2": {"brand": "b", "category": "cat"},
            "p3": {"brand": "c", "category": "cat"},
        },
        product_clip_embeddings={
            "p1": np.array([1.0, 0.0], dtype=np.float32),
            "p2": np.array([0.0, 1.0], dtype=np.float32),
            "p3": np.array([1.0, 1.0], dtype=np.float32),
        },
        user_features_map={"u1": {}, "u2": {}},
    )
    index = faiss.IndexFlatIP(2)
    index.add(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.707, 0.707],
            ],
            dtype=np.float32,
        )
    )
    saw_user_embedding = []

    def sample_with_probe(user_embedding, positive_items, epoch=0, total_epochs=1):
        saw_user_embedding.append(user_embedding is not None)
        for item_idx in range(1, len(trainer.item_mapping) + 1):
            if item_idx not in positive_items:
                return [item_idx]
        return [0]

    trainer.negative_sampler.sample = sample_with_probe

    trainer.train(existing_cf_index=index)

    assert any(saw_user_embedding)


def test_two_tower_warm_start_keeps_configured_dcn_architecture(tmp_path):
    path = tmp_path / "legacy_two_tower.pt"
    legacy_model = TwoTowerModel(
        num_users=1,
        num_items=1,
        clip_dim=5,
        output_dim=7,
        user_hidden_dims=[8],
        item_hidden_dims=[9],
        architecture="mlp",
    )
    torch.save(
        {
            "model_state_dict": legacy_model.state_dict(),
            "user_mapping": {"u1": 1},
            "item_mapping": {"p1": 1},
            "reverse_item_mapping": {1: "p1"},
            "config": {
                "clip_dim": 5,
                "output_dim": 7,
                "temperature": 0.07,
                "num_users": 1,
                "num_items": 1,
            },
        },
        path,
    )
    trainer = TwoTowerTrainer(
        clip_dim=5,
        output_dim=7,
        user_hidden_dims=[8],
        item_hidden_dims=[9],
        architecture="dcn",
        cross_layers=2,
    )
    trainer.user_mapping = {"u1": 1}
    trainer.item_mapping = {"p1": 1}
    trainer.reverse_item_mapping = {1: "p1"}
    trainer.model = TwoTowerModel(
        num_users=1,
        num_items=1,
        clip_dim=5,
        output_dim=7,
        user_hidden_dims=[8],
        item_hidden_dims=[9],
        architecture="dcn",
        cross_layers=2,
    )

    assert trainer.warm_start_from_checkpoint(str(path)) is True
    assert trainer.model.architecture == "dcn"


@pytest.mark.asyncio
async def test_ranking_legacy_raw_state_dict_loads_as_mlp(tmp_path):
    path = tmp_path / "legacy_ranking.pt"
    legacy_config = RankingConfig(architecture="mlp", hidden_dims=[16, 8])
    legacy_model = MultiObjectiveRankingModel(28, legacy_config)
    torch.save(legacy_model.state_dict(), path)

    ranking = RankingModel(
        RankingConfig(architecture="dcn_v2_low_rank", hidden_dims=[16, 8])
    )
    await ranking.load_model(str(path))

    assert ranking.model is not None
    assert ranking.model.architecture == "mlp"


@pytest.mark.asyncio
async def test_ranking_dcn_checkpoint_round_trips_architecture(tmp_path):
    path = tmp_path / "ranking_dcn.pt"
    config = RankingConfig(architecture="dcn", hidden_dims=[16, 8], cross_layers=2)
    ranking = RankingModel(config)
    await ranking.load_model()
    ranking.is_trained = True

    await ranking.save_model(str(path))
    loaded = RankingModel(RankingConfig(architecture="mlp", hidden_dims=[16, 8]))
    await loaded.load_model(str(path))

    assert loaded.model is not None
    assert loaded.model.architecture == "dcn"
    assert loaded.model.cross_layers == 2


@pytest.mark.asyncio
async def test_ranking_dcn_v2_low_rank_checkpoint_round_trips_architecture(tmp_path):
    path = tmp_path / "ranking_dcn_v2.pt"
    config = RankingConfig(
        architecture="dcn_v2_low_rank",
        hidden_dims=[16, 8],
        cross_layers=2,
        low_rank_dim=4,
    )
    ranking = RankingModel(config)
    await ranking.load_model()
    ranking.is_trained = True

    await ranking.save_model(str(path))
    loaded = RankingModel(RankingConfig(architecture="mlp", hidden_dims=[16, 8]))
    await loaded.load_model(str(path))

    assert loaded.model is not None
    assert loaded.model.architecture == "dcn_v2_low_rank"
    assert loaded.model.cross_layers == 2
    assert loaded.model.low_rank_dim == 4


@pytest.mark.asyncio
async def test_ranking_dcn_direct_and_microbatch_inference_outputs_shapes():
    config = RankingConfig(
        architecture="dcn",
        hidden_dims=[16, 8],
        cross_layers=2,
        enable_async_batching=True,
        batch_max_requests=2,
        batch_wait_ms=25.0,
        batch_runner_count=1,
        offload_inference_to_thread=False,
    )
    ranking = RankingModel(config)
    await ranking.load_model()
    feature_matrix = np.zeros(
        (4, ranking.feature_extractor.total_feature_dim), dtype=np.float32
    )

    predictions, _ = ranking.run_inference_batch(feature_matrix)

    assert set(predictions) == {"ctr", "cvr", "gmv", "ranking_score"}
    assert all(values.shape == (4,) for values in predictions.values())

    user_features = UserFeatures(user_id="u1")
    product_metadata = {
        "p1": {
            "title": "Product 1",
            "price": 10.0,
            "category": "cat",
            "brand": "brand",
        },
        "p2": {
            "title": "Product 2",
            "price": 12.0,
            "category": "cat",
            "brand": "brand",
        },
    }
    first_candidates = [
        CandidateProduct(product_id="p1", combined_score=0.4, source="test")
    ]
    second_candidates = [
        CandidateProduct(product_id="p2", combined_score=0.5, source="test")
    ]
    batcher = RankingBatcher(ranking, config)

    first_task = asyncio.create_task(
        batcher.rank_candidates(
            first_candidates,
            user_features,
            {},
            k=1,
            product_metadata_map=product_metadata,
            include_profile=True,
        )
    )
    second_task = asyncio.create_task(
        batcher.rank_candidates(
            second_candidates,
            user_features,
            {},
            k=1,
            product_metadata_map=product_metadata,
            include_profile=True,
        )
    )
    await asyncio.sleep(0)
    await batcher.start()
    try:
        first_result, second_result = await asyncio.gather(first_task, second_task)
    finally:
        await batcher.close()

    assert first_result[1]["path"] == "torch_microbatch"
    assert second_result[1]["path"] == "torch_microbatch"
    assert len(first_result[0]) == 1
    assert len(second_result[0]) == 1


@pytest.mark.asyncio
async def test_ranking_torch_compile_disabled_uses_eager_inference(monkeypatch):
    def fail_compile(*args, **kwargs):
        raise AssertionError("torch.compile should not be called")

    monkeypatch.setattr(torch, "compile", fail_compile)
    ranking = RankingModel(
        RankingConfig(
            architecture="dcn",
            hidden_dims=[16, 8],
            torch_compile_enabled=False,
        )
    )

    await ranking.load_model()
    predictions, profile = ranking.run_inference_batch(
        np.zeros((4, ranking.feature_extractor.total_feature_dim), dtype=np.float32)
    )
    status = ranking.health_check()

    assert ranking._compiled_model is None
    assert status["torch_compile_enabled"] is False
    assert status["torch_compile_active"] is False
    assert status["torch_compile_warmup_ms"] is None
    assert status["torch_compile_fallback_count"] == 0
    assert status["torch_compile_last_fallback_error"] is None
    assert status["torch_compile_last_inference_path"] == "eager"
    assert profile["inference_path"] == "eager"
    assert set(predictions) == {"ctr", "cvr", "gmv", "ranking_score"}


@pytest.mark.asyncio
async def test_ranking_torch_compile_enabled_uses_compiled_wrapper(monkeypatch):
    wrappers = []
    compile_calls = []

    class FakeCompiledModel:
        def __init__(self, model):
            self.model = model
            self.calls = []

        def __call__(self, features):
            self.calls.append(tuple(features.shape))
            return self.model(features)

    def fake_compile(model, **kwargs):
        compile_calls.append((model, kwargs))
        wrapper = FakeCompiledModel(model)
        wrappers.append(wrapper)
        return wrapper

    monkeypatch.setattr(torch, "compile", fake_compile)
    ranking = RankingModel(
        RankingConfig(
            architecture="dcn",
            hidden_dims=[16, 8],
            batch_target_requests=3,
            torch_compile_enabled=True,
            torch_compile_backend="inductor",
            torch_compile_mode="default",
            torch_compile_dynamic=True,
        )
    )

    await ranking.load_model()
    predictions, profile = ranking.run_inference_batch(
        np.zeros((4, ranking.feature_extractor.total_feature_dim), dtype=np.float32)
    )
    stats = ranking.get_stats()
    health = ranking.health_check()

    assert len(compile_calls) == 1
    assert compile_calls[0][0] is ranking.model
    assert compile_calls[0][1] == {
        "backend": "inductor",
        "mode": "default",
        "dynamic": True,
        "fullgraph": True,
    }
    assert ranking._compiled_model is wrappers[0]
    assert wrappers[0].calls[0] == (3, ranking.feature_extractor.total_feature_dim)
    assert wrappers[0].calls[-1] == (4, ranking.feature_extractor.total_feature_dim)
    assert stats["torch_compile_enabled"] is True
    assert stats["torch_compile_active"] is True
    assert stats["torch_compile_error"] is None
    assert stats["torch_compile_warmup_ms"] is not None
    assert stats["torch_compile_warmup_ms"] >= 0.0
    assert stats["torch_compile_fallback_count"] == 0
    assert stats["torch_compile_last_fallback_error"] is None
    assert stats["torch_compile_last_inference_path"] == "compiled"
    assert health["torch_compile_warmup_ms"] == stats["torch_compile_warmup_ms"]
    assert health["torch_compile_fallback_count"] == 0
    assert profile["inference_path"] == "compiled"
    assert set(predictions) == {"ctr", "cvr", "gmv", "ranking_score"}


@pytest.mark.asyncio
async def test_ranking_torch_compile_failure_falls_back_to_eager(monkeypatch):
    def fail_compile(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(torch, "compile", fail_compile)
    ranking = RankingModel(
        RankingConfig(
            architecture="dcn",
            hidden_dims=[16, 8],
            torch_compile_enabled=True,
        )
    )

    await ranking.load_model()
    predictions, profile = ranking.run_inference_batch(
        np.zeros((2, ranking.feature_extractor.total_feature_dim), dtype=np.float32)
    )
    status = ranking.health_check()

    assert ranking._compiled_model is None
    assert status["torch_compile_enabled"] is True
    assert status["torch_compile_active"] is False
    assert "compile failed" in status["torch_compile_error"]
    assert status["torch_compile_warmup_ms"] is None
    assert status["torch_compile_fallback_count"] == 0
    assert status["torch_compile_last_fallback_error"] is None
    assert status["torch_compile_last_inference_path"] == "eager"
    assert profile["inference_path"] == "eager"
    assert set(predictions) == {"ctr", "cvr", "gmv", "ranking_score"}


@pytest.mark.asyncio
async def test_ranking_torch_compile_inference_failure_records_fallback(monkeypatch):
    wrappers = []

    class FailingAfterWarmupCompiledModel:
        def __init__(self, model):
            self.model = model
            self.calls = 0

        def __call__(self, features):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("compiled inference failed")
            return self.model(features)

    def fake_compile(model, **kwargs):
        wrapper = FailingAfterWarmupCompiledModel(model)
        wrappers.append(wrapper)
        return wrapper

    monkeypatch.setattr(torch, "compile", fake_compile)
    ranking = RankingModel(
        RankingConfig(
            architecture="dcn",
            hidden_dims=[16, 8],
            torch_compile_enabled=True,
        )
    )

    await ranking.load_model()
    predictions, profile = ranking.run_inference_batch(
        np.zeros((2, ranking.feature_extractor.total_feature_dim), dtype=np.float32)
    )
    status = ranking.health_check()

    assert wrappers[0].calls == 2
    assert ranking._compiled_model is None
    assert profile["inference_path"] == "eager"
    assert status["torch_compile_active"] is False
    assert status["torch_compile_fallback_count"] == 1
    assert "compiled inference failed" in status["torch_compile_error"]
    assert "compiled inference failed" in status["torch_compile_last_fallback_error"]
    assert status["torch_compile_last_inference_path"] == "eager"
    assert set(predictions) == {"ctr", "cvr", "gmv", "ranking_score"}


@pytest.mark.asyncio
async def test_ranking_model_rebuild_invalidates_compiled_wrapper(monkeypatch):
    class FakeCompiledModel:
        def __init__(self, model):
            self.model = model

        def __call__(self, features):
            return self.model(features)

    monkeypatch.setattr(
        torch,
        "compile",
        lambda model, **kwargs: FakeCompiledModel(model),
    )
    ranking = RankingModel(
        RankingConfig(
            architecture="dcn",
            hidden_dims=[16, 8],
            torch_compile_enabled=True,
        )
    )
    await ranking.load_model()

    assert ranking._compiled_model is not None
    ranking._initialize_model(architecture="mlp", hidden_dims=[16, 8])

    assert ranking._compiled_model is None
    assert ranking.get_stats()["torch_compile_active"] is False


@pytest.mark.asyncio
async def test_ranking_dcn_v2_low_rank_trains_and_infers_with_batch_size_one():
    config = RankingConfig(
        architecture="dcn_v2_low_rank",
        hidden_dims=[16, 8],
        cross_layers=2,
        low_rank_dim=4,
        training_min_samples=1,
        epochs=1,
        batch_size=1,
        enable_async_batching=True,
        batch_max_requests=2,
        batch_wait_ms=25.0,
        batch_runner_count=1,
        offload_inference_to_thread=False,
    )
    ranking = RankingModel(config)
    await ranking.load_model()
    samples = [
        {
            "user_id": "u1",
            "product_id": "p1",
            "action": "click",
            "context": {"device": "mobile", "session_position": 1, "time_on_page": 10},
            "combined_score": 0.7,
        },
        {
            "user_id": "u1",
            "product_id": "p2",
            "action": "view",
            "context": {"device": "desktop", "session_position": 2, "time_on_page": 3},
            "combined_score": 0.2,
        },
    ]
    user_features_map = {
        "u1": {
            "user_id": "u1",
            "total_interactions": 5,
            "avg_session_length": 120.0,
            "last_active": 1_700_000_000.0,
        }
    }
    product_metadata_map = {
        "p1": {
            "price": 10.0,
            "rating": 4.5,
            "num_reviews": 12,
            "in_stock": True,
            "created_at": 1_700_000_000.0,
            "tags": ["a"],
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
        user_features_map=user_features_map,
        product_metadata_map=product_metadata_map,
    )
    feature_matrix = np.zeros(
        (4, ranking.feature_extractor.total_feature_dim), dtype=np.float32
    )
    predictions, _ = ranking.run_inference_batch(feature_matrix)

    assert ranking.is_trained
    assert ranking.model is not None
    assert ranking.model.architecture == "dcn_v2_low_rank"
    assert set(predictions) == {"ctr", "cvr", "gmv", "ranking_score"}
    assert all(values.shape == (4,) for values in predictions.values())

    user_features = UserFeatures(
        user_id="u1",
        total_interactions=5,
        avg_session_length=120.0,
        last_active=1_700_000_000.0,
    )
    first_candidates = [
        CandidateProduct(product_id="p1", combined_score=0.7, source="test")
    ]
    second_candidates = [
        CandidateProduct(product_id="p2", combined_score=0.2, source="test")
    ]
    batcher = RankingBatcher(ranking, config)

    first_task = asyncio.create_task(
        batcher.rank_candidates(
            first_candidates,
            user_features,
            {"device": "mobile", "session_position": 1, "time_on_page": 10},
            k=1,
            product_metadata_map=product_metadata_map,
            include_profile=True,
        )
    )
    second_task = asyncio.create_task(
        batcher.rank_candidates(
            second_candidates,
            user_features,
            {"device": "mobile", "session_position": 1, "time_on_page": 10},
            k=1,
            product_metadata_map=product_metadata_map,
            include_profile=True,
        )
    )
    await asyncio.sleep(0)
    await batcher.start()
    try:
        first_result, second_result = await asyncio.gather(first_task, second_task)
    finally:
        await batcher.close()

    assert first_result[1]["path"] == "torch_microbatch"
    assert second_result[1]["path"] == "torch_microbatch"
    assert len(first_result[0]) == 1
    assert len(second_result[0]) == 1
