import sys
import types

import pytest
import torch
import numpy as np

try:
    import faiss  # noqa: F401
except ModuleNotFoundError:
    fake_faiss = types.ModuleType("faiss")
    fake_faiss.Index = object
    sys.modules["faiss"] = fake_faiss

from video_commerce.ml.two_tower import TwoTowerTrainer, logq_correct_logits


def test_logq_correction_uses_only_known_source_probabilities():
    logits = torch.tensor([[1.0, 1.0, 1.0]])
    probabilities = torch.tensor([[0.5, 0.25, 0.0]])
    known_probability = torch.tensor([[True, True, False]])

    corrected = logq_correct_logits(logits, probabilities, known_probability)

    assert corrected[0, 0].item() == pytest.approx(
        1.0 - torch.log(torch.tensor(0.5)).item()
    )
    assert corrected[0, 1].item() == pytest.approx(
        1.0 - torch.log(torch.tensor(0.25)).item()
    )
    assert corrected[0, 2].item() == 1.0


def test_logq_correction_rejects_non_positive_known_probability():
    with pytest.raises(ValueError, match="strictly positive"):
        logq_correct_logits(
            torch.ones((1, 1)),
            torch.zeros((1, 1)),
            torch.ones((1, 1), dtype=torch.bool),
        )


def test_false_negative_mask_uses_only_positives_known_at_query_cutoff():
    trainer = TwoTowerTrainer(clip_dim=2, output_dim=2, epochs=1)
    trainer.prepare(
        [
            {
                "event_id": "e1",
                "user_id": "u1",
                "product_id": "p1",
                "action": "click",
                "as_of_ts": 10.0,
            },
            {
                "event_id": "e2",
                "user_id": "u1",
                "product_id": "p2",
                "action": "click",
                "as_of_ts": 20.0,
            },
        ],
        {"p1": {}, "p2": {}},
        {"p1": np.asarray([1.0, 0.0]), "p2": np.asarray([0.0, 1.0])},
        {"u1": {}},
        as_of_ts=20.0,
    )

    first_known = trainer._train_samples[0][5]
    second_known = trainer._train_samples[1][5]
    p1 = trainer.item_mapping["p1"]
    p2 = trainer.item_mapping["p2"]
    assert first_known == frozenset({p1})
    assert second_known == frozenset({p1, p2})


def test_teacher_soft_negative_is_preserved_for_auxiliary_loss():
    trainer = TwoTowerTrainer(clip_dim=2, output_dim=2, epochs=1)
    trainer.prepare(
        [
            {
                "event_id": "e1",
                "user_id": "u1",
                "product_id": "p1",
                "action": "click",
                "as_of_ts": 10.0,
            }
        ],
        {"p1": {}, "p2": {}},
        {"p1": np.asarray([1.0, 0.0]), "p2": np.asarray([0.0, 1.0])},
        {"u1": {}},
        external_negatives=[
            {
                "user_id": "u1",
                "product_id": "p2",
                "source": "ranker_rejected",
                "as_of_ts": 10.0,
                "teacher_target": 0.25,
            }
        ],
        as_of_ts=10.0,
    )

    candidate = trainer._external_negatives_by_user[trainer.user_mapping["u1"]][0]
    assert candidate.teacher_target == 0.25
