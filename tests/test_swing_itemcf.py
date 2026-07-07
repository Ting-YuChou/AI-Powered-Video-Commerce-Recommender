from pathlib import Path

from video_commerce.common.models import CandidateProduct
from video_commerce.ml.swing_itemcf import (
    SwingItemCFCandidateEngine,
    SwingItemCFIndex,
    SwingItemCFTrainer,
)


def _event(user_id, product_id, action="click", occurred_at=1_700_000_000.0):
    return {
        "user_id": user_id,
        "product_id": product_id,
        "action": action,
        "occurred_at": occurred_at,
        "timestamp": occurred_at,
    }


def test_swing_builds_symmetric_neighbors_for_shared_item_pairs():
    trainer = SwingItemCFTrainer(alpha=5.0, max_neighbors_per_item=10)
    index = trainer.fit(
        {
            "u1": [_event("u1", "a"), _event("u1", "b")],
            "u2": [_event("u2", "a"), _event("u2", "b")],
        },
        model_version="swing-test",
    )

    assert index.model_version == "swing-test"
    assert index.get_neighbors("a")[0].product_id == "b"
    assert index.get_neighbors("b")[0].product_id == "a"
    assert index.get_neighbors("a")[0].normalized_score == 1.0


def test_swing_downweights_broad_user_pair_overlap():
    trainer = SwingItemCFTrainer(alpha=5.0, max_neighbors_per_item=10)
    index = trainer.fit(
        {
            "narrow-1": [_event("narrow-1", "a"), _event("narrow-1", "b")],
            "narrow-2": [_event("narrow-2", "a"), _event("narrow-2", "b")],
            "broad-1": [
                _event("broad-1", "x"),
                _event("broad-1", "y"),
                _event("broad-1", "z"),
                _event("broad-1", "w"),
            ],
            "broad-2": [
                _event("broad-2", "x"),
                _event("broad-2", "y"),
                _event("broad-2", "z"),
                _event("broad-2", "w"),
            ],
        },
        model_version="swing-test",
    )

    narrow_score = index.get_neighbors("a")[0].score
    broad_score = index.get_neighbors("x")[0].score

    assert narrow_score > broad_score


def test_candidate_engine_applies_action_and_recency_weights():
    index = SwingItemCFIndex(
        model_version="swing-test",
        neighbors={
            "seed-click": [
                {"product_id": "candidate", "score": 1.0, "normalized_score": 1.0}
            ],
            "seed-view": [
                {"product_id": "candidate", "score": 1.0, "normalized_score": 1.0}
            ],
            "seed-old-purchase": [
                {"product_id": "old-candidate", "score": 1.0, "normalized_score": 1.0}
            ],
            "seed-dup": [
                {"product_id": "dup-candidate", "score": 1.0, "normalized_score": 1.0}
            ],
        },
        metadata={"built_at": 1.0},
    )
    engine = SwingItemCFCandidateEngine(
        index,
        max_seed_items=20,
        score_weight=1.0,
    )

    candidates = engine.get_candidates(
        [
            _event("u1", "seed-view", "view", 1_700_000_000.0),
            _event("u1", "seed-click", "click", 1_700_000_000.0),
            _event("u1", "seed-old-purchase", "purchase", 1_700_000_000.0 - 60 * 86400),
            _event("u1", "seed-dup", "purchase", 1_700_000_000.0 - 60 * 86400),
            _event("u1", "seed-dup", "view", 1_700_000_000.0),
        ],
        k=5,
        exclude_items=set(),
        current_time=1_700_000_000.0,
    )

    by_id = {candidate.product_id: candidate for candidate in candidates}
    assert by_id["candidate"].source == "swing_itemcf"
    assert by_id["candidate"].collaborative_score == 3.0
    assert by_id["dup-candidate"].collaborative_score == 5.0
    assert by_id["old-candidate"].collaborative_score < by_id["candidate"].collaborative_score


def test_candidate_engine_excludes_already_interacted_seed_items():
    index = SwingItemCFIndex(
        model_version="swing-test",
        neighbors={
            "seed": [
                {"product_id": "seed", "score": 1.0, "normalized_score": 1.0},
                {"product_id": "fresh", "score": 0.5, "normalized_score": 0.5},
            ],
        },
        metadata={"built_at": 1.0},
    )
    engine = SwingItemCFCandidateEngine(index, max_seed_items=20)

    candidates = engine.get_candidates(
        [_event("u1", "seed", "purchase")],
        k=10,
        exclude_items={"seed"},
        current_time=1_700_000_000.0,
    )

    assert [candidate.product_id for candidate in candidates] == ["fresh"]
    assert isinstance(candidates[0], CandidateProduct)


def test_swing_index_round_trips_compressed_json_artifact(tmp_path):
    index = SwingItemCFIndex(
        model_version="swing-test",
        neighbors={
            "a": [{"product_id": "b", "score": 0.25, "normalized_score": 1.0}],
        },
        metadata={"sample_count": 2},
    )
    path = Path(tmp_path / "swing_itemcf.json.gz")

    index.save(str(path))
    restored = SwingItemCFIndex.load(str(path))

    assert restored.model_version == "swing-test"
    assert restored.metadata["sample_count"] == 2
    assert restored.get_neighbors("a")[0].product_id == "b"
    assert restored.get_neighbors("a")[0].score == 0.25
