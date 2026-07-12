import pytest
import torch

from video_commerce.common.config import RankingConfig

from video_commerce.ml.din import (
    DIN_ACTIONS,
    DIN_SEQUENCE_CONTRACT_VERSION,
    build_din_behavior_sequences,
    build_din_freshness_token,
    DeepInterestNetwork,
    build_din_batch_inputs,
    load_din_embedding_sidecar,
    save_din_embedding_sidecar,
    parse_din_behavior_sequences,
)
from video_commerce.common.feature_history_contracts import (
    RANKING_LTR_FEATURE_DEFINITION_VERSION,
)
from video_commerce.common.models import CandidateProduct, UserFeatures
from video_commerce.ml.ranking_features import FeatureBundle
from video_commerce.ml.ranking import MultiObjectiveRankingModel, RankingModel


def _event(product_id, action, event_time, *, available_at=None, event_id=None):
    return {
        "product_id": product_id,
        "action": action,
        "occurred_at": event_time,
        "available_at": event_time if available_at is None else available_at,
        "event_id": event_id or f"{action}-{product_id}-{event_time}",
    }


def test_build_din_sequences_filters_pit_leakage_and_left_pads_per_action():
    as_of = 4_000_000.0
    events = [
        _event("old", "click", as_of - 31 * 86400),
        _event("c1", "click", as_of - 30),
        _event("c1", "click", as_of - 20, event_id="repeat"),
        _event("cart", "add_to_cart", as_of - 10),
        _event("late", "purchase", as_of - 5, available_at=as_of + 1),
        _event("future", "click", as_of),
        _event("view", "view", as_of - 1),
    ]

    built = build_din_behavior_sequences(events, as_of_ts=as_of, last_n=3)

    assert built.contract_version == DIN_SEQUENCE_CONTRACT_VERSION
    assert tuple(built.actions) == DIN_ACTIONS
    assert built.actions["click"].product_ids == ("", "c1", "c1")
    assert built.actions["click"].mask == (False, True, True)
    assert built.actions["cart"].product_ids == ("", "", "cart")
    assert built.actions["purchase"].mask == (False, False, False)
    assert built.actions["click"].event_times[-2:] == (
        pytest.approx(as_of - 30),
        pytest.approx(as_of - 20),
    )


def test_din_freshness_token_is_stable_and_action_sensitive():
    as_of = 2_000.0
    base = build_din_behavior_sequences(
        [_event("p1", "purchase", as_of - 5, event_id="e1")],
        as_of_ts=as_of,
        last_n=2,
    )
    changed = build_din_behavior_sequences(
        [
            _event("p1", "purchase", as_of - 5, event_id="e1"),
            _event("c1", "click", as_of - 4, event_id="e2"),
        ],
        as_of_ts=as_of,
        last_n=2,
    )

    first = build_din_freshness_token(base)
    second = build_din_freshness_token(base)
    third = build_din_freshness_token(changed)

    assert first == second
    assert first["contract_version"] == DIN_SEQUENCE_CONTRACT_VERSION
    assert first["actions"]["purchase"]["length"] == 1
    assert first["sequence_hash"] != third["sequence_hash"]


def test_din_attention_masks_padding_and_backpropagates_without_embedding_gradients():
    torch.manual_seed(7)
    embeddings = torch.randn(6, 128)
    embeddings[0].zero_()
    model = DeepInterestNetwork(embeddings)
    candidate_indices = torch.tensor([1, 0], dtype=torch.long)
    history_indices = torch.tensor(
        [
            [[0, 2, 3], [0, 0, 4], [0, 0, 5]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.long,
    )
    history_mask = history_indices.ne(0)
    recency = torch.tensor(
        [
            [[0.0, 0.2, 0.1], [0.0, 0.0, 0.3], [0.0, 0.0, 0.4]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )

    interest, attention = model(
        candidate_indices,
        history_indices,
        recency,
        history_mask,
        return_attention=True,
    )

    assert interest.shape == (2, 128)
    assert attention.shape == (2, 3, 3)
    assert torch.allclose(attention[0].sum(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.count_nonzero(attention[0][~history_mask[0]]) == 0
    assert torch.count_nonzero(interest[1]) == 0

    interest[0].sum().backward()
    assert model.item_embedding.weight.requires_grad is False
    assert model.item_embedding.weight.grad is None
    assert model.action_embedding.weight.grad is not None
    assert any(
        parameter.grad is not None for parameter in model.attention_mlp.parameters()
    )


def test_din_ranking_config_uses_approved_contract_defaults():
    config = RankingConfig()

    assert config.din_enabled is False
    assert config.din_sequence_last_n == 60
    assert config.din_sequence_lookback_days == 30
    assert config.din_min_nonempty_ratio == pytest.approx(0.30)

    with pytest.raises(ValueError, match="replaces averaged"):
        RankingConfig(din_enabled=True, history_embeddings_enabled=True)


def test_din_batch_reuses_request_history_and_emits_candidate_mapping():
    as_of = 4_000_000.0
    sequences = build_din_behavior_sequences(
        [_event("h1", "click", as_of - 60)], as_of_ts=as_of, last_n=3
    )
    user = UserFeatures(user_id="u1")
    bundles = [
        FeatureBundle(
            as_of_ts=as_of,
            feature_definition_version=RANKING_LTR_FEATURE_DEFINITION_VERSION,
            user_features=user,
            product_metadata={"price": 1.0},
            context={},
            candidate=CandidateProduct(product_id=product_id, source="test"),
            behavior_sequences=(
                sequences
                if product_id == "c1"
                else parse_din_behavior_sequences(
                    sequences.to_dict(), expected_as_of_ts=as_of, last_n=3
                )
            ),
        )
        for product_id in ("c1", "unknown")
    ]

    batch = build_din_batch_inputs(
        bundles, {"h1": 1, "c1": 2, "c2": 3}, sequence_length=3
    )

    assert batch.request_history_indices.shape == (1, 3, 3)
    assert batch.candidate_indices.tolist() == [2, 0]
    assert batch.candidate_to_request.tolist() == [0, 0]
    assert batch.summary_features.shape == (2, 12)
    assert batch.summary_features[0, 0].item() == pytest.approx(1.0)
    assert torch.count_nonzero(batch.summary_features[1]) == 0


def test_ranking_model_backpropagates_existing_loss_into_din_attention():
    torch.manual_seed(9)
    config = RankingConfig(
        din_enabled=True,
        architecture="mlp",
        hidden_dims=[16],
        dropout_rate=0.0,
    )
    embeddings = torch.randn(5, 128)
    model = MultiObjectiveRankingModel(7, config, din_item_embeddings=embeddings)
    dense = torch.randn(2, 7)
    candidate_indices = torch.tensor([1, 2])
    history_indices = torch.tensor([[[0, 3], [0, 0], [0, 4]], [[0, 3], [0, 0], [0, 4]]])
    mask = history_indices.ne(0)

    predictions = model(
        dense,
        candidate_indices=candidate_indices,
        history_indices=history_indices,
        history_recency=torch.zeros_like(history_indices, dtype=torch.float32),
        history_mask=mask,
        summary_features=torch.zeros(2, 12),
    )
    predictions["ranking_score"].sum().backward()

    assert predictions["ranking_score"].shape == (2, 1)
    assert model.din.item_embedding.weight.grad is None
    assert any(
        parameter.grad is not None for parameter in model.din.attention_mlp.parameters()
    )


def test_din_embedding_sidecar_has_padding_mapping_and_checksum(tmp_path):
    path = tmp_path / "ranking-din.npz"
    lineage = save_din_embedding_sidecar(
        str(path),
        {"p2": torch.ones(128).numpy(), "p1": torch.zeros(128).numpy()},
        two_tower_model_version="two-tower-7",
    )

    embeddings, mapping, metadata = load_din_embedding_sidecar(str(path))

    assert embeddings.shape == (3, 128)
    assert torch.count_nonzero(embeddings[0]) == 0
    assert mapping == {"p1": 1, "p2": 2}
    assert metadata["two_tower_model_version"] == "two-tower-7"
    assert metadata["sha256"] == lineage["sha256"]


@pytest.mark.asyncio
async def test_din_checkpoint_rejects_missing_trainable_tensor(tmp_path):
    sidecar = tmp_path / "din.npz"
    checkpoint = tmp_path / "ranking.pt"
    save_din_embedding_sidecar(
        str(sidecar), {"p1": torch.ones(128).numpy()}, two_tower_model_version="tt-1"
    )
    config = RankingConfig(
        din_enabled=True,
        din_embedding_sidecar_path=str(sidecar),
        architecture="mlp",
        hidden_dims=[8],
    )
    source = RankingModel(config)
    await source.load_model(str(checkpoint))
    source.is_trained = True
    await source.save_model(str(checkpoint))
    payload = torch.load(checkpoint)
    payload["model_state_dict"].pop("din.attention_mlp.0.weight")
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="trainable state is incomplete"):
        await RankingModel(config).load_model(str(checkpoint))


@pytest.mark.asyncio
async def test_din_checkpoint_rejects_mismatched_sidecar_lineage(tmp_path):
    sidecar = tmp_path / "din.npz"
    checkpoint = tmp_path / "ranking.pt"
    save_din_embedding_sidecar(
        str(sidecar), {"p1": torch.ones(128).numpy()}, two_tower_model_version="tt-1"
    )
    config = RankingConfig(
        din_enabled=True,
        din_embedding_sidecar_path=str(sidecar),
        architecture="mlp",
        hidden_dims=[8],
    )
    source = RankingModel(config)
    await source.load_model(str(checkpoint))
    source.is_trained = True
    await source.save_model(str(checkpoint))
    save_din_embedding_sidecar(
        str(sidecar), {"p2": torch.ones(128).numpy()}, two_tower_model_version="tt-2"
    )

    with pytest.raises(RuntimeError, match="lineage mismatch"):
        await RankingModel(config).load_model(str(checkpoint))


@pytest.mark.asyncio
async def test_ranking_checkpoint_save_propagates_failure_without_replacing_target(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "ranking.pt"
    checkpoint.write_bytes(b"previous")
    model = RankingModel(RankingConfig(architecture="mlp", hidden_dims=[8]))
    await model.load_model(str(tmp_path / "missing.pt"))
    model.is_trained = True

    def fail_save(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        await model.save_model(str(checkpoint))

    assert checkpoint.read_bytes() == b"previous"
