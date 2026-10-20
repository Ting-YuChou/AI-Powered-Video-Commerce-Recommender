from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import numpy as np

from video_commerce.ml.model_release_quality import (
    GateDecision,
    QUALITY_GATE_POLICY_VERSION,
    RankingEvaluationRow,
    RankingEvaluationSpool,
    RankingPrediction,
    RankingQualityGateConfig,
    _cluster_bootstrap_ndcg,
    _informative_impression_count,
    _mean_ndcg,
    evaluate_quality_gate,
    split_fixed_holdout,
    split_training_examples_fixed_holdout,
)


def test_ranking_evaluation_spool_round_trips_compact_records(tmp_path):
    row = _row(
        impression="i1",
        user="u1",
        product="p1",
        timestamp=100.0,
        relevance=2.0,
        clicked=True,
    )
    prediction = RankingPrediction(0.8, 0.7, 0.4, 12.0)
    spool = RankingEvaluationSpool(tmp_path / "evaluation.sqlite3")
    spool.add(row, prediction, prediction)

    rows, champion, challenger = spool.load()
    spool.close()

    assert rows == [row]
    assert champion[row.observation_id] == prediction
    assert challenger[row.observation_id] == prediction


def _row(
    *,
    impression: str,
    user: str,
    product: str,
    timestamp: float,
    relevance: float,
    clicked: bool = False,
    purchased: bool = False,
    value: float = 0.0,
) -> RankingEvaluationRow:
    return RankingEvaluationRow(
        observation_id=f"{impression}:{product}",
        impression_id=impression,
        user_id=user,
        product_id=product,
        as_of_ts=timestamp,
        relevance=relevance,
        clicked=clicked,
        purchased=purchased,
        business_value=value,
        total_interactions=10,
        price=100.0,
        product_created_at=timestamp - 60 * 86400,
        pre_holdout_exposures=100,
        modality_presence={
            "visual": True,
            "ocr": True,
            "asr": True,
            "candidate_image": True,
            "candidate_text": True,
            "candidate_two_tower": True,
        },
    )


def _fixture_rows() -> list[RankingEvaluationRow]:
    rows: list[RankingEvaluationRow] = []
    for index in range(8):
        timestamp = 1_700_000_000.0 + index * 86400
        rows.extend(
            [
                _row(
                    impression=f"i{index}",
                    user=f"u{index % 4}",
                    product=f"p{index}:positive",
                    timestamp=timestamp,
                    relevance=4.0,
                    clicked=True,
                    purchased=True,
                    value=20.0,
                ),
                _row(
                    impression=f"i{index}",
                    user=f"u{index % 4}",
                    product=f"p{index}:negative",
                    timestamp=timestamp,
                    relevance=1.0,
                ),
            ]
        )
    return rows


def _prediction(row: RankingEvaluationRow, *, correct: bool) -> RankingPrediction:
    positive = row.relevance > 1.0
    score = 0.9 if positive == correct else 0.1
    return RankingPrediction(
        serving_score=score,
        ctr=0.9 if positive == correct else 0.1,
        ctcvr=0.8 if positive == correct else 0.05,
        predicted_value=20.0 if positive else 1.0,
    )


def _small_config() -> RankingQualityGateConfig:
    return RankingQualityGateConfig(
        holdout_days=2,
        bootstrap_samples=100,
        random_seed=42,
        min_impressions=2,
        min_users=2,
        min_clicks=2,
        min_purchases=2,
        min_slice_impressions=1,
        min_slice_users=1,
        min_slice_positives=1,
        require_candidate_recall=False,
    )


def test_fixed_holdout_reserves_latest_complete_days():
    rows = _fixture_rows()

    training, holdout, window = split_fixed_holdout(
        rows,
        holdout_days=2,
        attribution_cutoff=max(row.as_of_ts for row in rows),
    )

    assert window.end_ts == 1_700_604_800.0
    assert window.start_ts == 1_700_432_000.0
    assert {row.impression_id for row in holdout} == {"i5", "i6", "i7"}
    assert max(row.as_of_ts for row in training) < window.start_ts


def test_training_examples_use_same_fixed_holdout_boundary():
    examples = [
        type(
            "Example", (), {"bundle": type("Bundle", (), {"as_of_ts": row.as_of_ts})()}
        )()
        for row in _fixture_rows()
    ]

    training, holdout, window = split_training_examples_fixed_holdout(
        examples,
        holdout_days=2,
        attribution_cutoff=max(example.bundle.as_of_ts for example in examples),
    )

    assert max(example.bundle.as_of_ts for example in training) < window.start_ts
    assert min(example.bundle.as_of_ts for example in holdout) >= window.start_ts
    assert len(training) + len(holdout) == len(examples)


def test_quality_gate_passes_non_regressing_challenger_deterministically():
    rows = _fixture_rows()[-6:]
    champion = {row.observation_id: _prediction(row, correct=True) for row in rows}
    challenger = dict(champion)

    first = evaluate_quality_gate(rows, champion, challenger, _small_config())
    second = evaluate_quality_gate(rows, champion, challenger, _small_config())

    assert first.decision is GateDecision.PASSED
    assert first.overall["ndcg_at_10_delta"] == pytest.approx(0.0)
    assert first.bootstrap == second.bootstrap
    assert first.policy_version == "ranking_quality_gate_v4"
    assert first.policy_version == QUALITY_GATE_POLICY_VERSION
    assert first.overall["ndcg_informative_impressions"] == 3


def test_quality_gate_rejects_serving_order_regression():
    rows = _fixture_rows()[-6:]
    champion = {row.observation_id: _prediction(row, correct=True) for row in rows}
    challenger = {row.observation_id: _prediction(row, correct=False) for row in rows}

    result = evaluate_quality_gate(rows, champion, challenger, _small_config())

    assert result.decision is GateDecision.FAILED
    assert "overall_ndcg_point_delta" in result.reasons
    assert result.overall["ndcg_at_10_delta"] < 0


def test_quality_gate_blocks_when_required_evidence_is_missing():
    rows = _fixture_rows()[-2:]
    predictions = {row.observation_id: _prediction(row, correct=True) for row in rows}
    config = replace(_small_config(), min_users=10)

    result = evaluate_quality_gate(rows, predictions, predictions, config)

    assert result.decision is GateDecision.INSUFFICIENT_EVIDENCE
    assert "overall_users" in result.reasons


def test_quality_gate_rejects_missing_or_non_finite_predictions():
    rows = _fixture_rows()[-6:]
    champion = {row.observation_id: _prediction(row, correct=True) for row in rows}
    challenger = dict(champion)
    challenger[rows[0].observation_id] = RankingPrediction(
        serving_score=float("nan"),
        ctr=0.5,
        ctcvr=0.2,
        predicted_value=10.0,
    )

    result = evaluate_quality_gate(rows, champion, challenger, _small_config())

    assert result.decision is GateDecision.FAILED
    assert result.reasons == ("non_finite_prediction",)


def test_quality_gate_v2_rejects_ece_regression():
    rows = _fixture_rows()[-6:]
    champion = {row.observation_id: _prediction(row, correct=True) for row in rows}
    challenger = {
        row.observation_id: replace(
            champion[row.observation_id],
            ctr=0.6 if row.clicked else 0.4,
            ctcvr=0.6 if row.purchased else 0.4,
        )
        for row in rows
    }

    result = evaluate_quality_gate(rows, champion, challenger, _small_config())

    assert result.decision is GateDecision.FAILED
    assert "overall_ctr_ece_absolute_regression" in result.reasons
    assert result.overall["ctr_ece_absolute_regression"] > 0.005


def test_quality_gate_v2_requires_candidate_recall_evidence_when_enabled():
    rows = _fixture_rows()[-6:]
    predictions = {row.observation_id: _prediction(row, correct=True) for row in rows}
    config = replace(_small_config(), require_candidate_recall=True)

    result = evaluate_quality_gate(rows, predictions, predictions, config)

    assert result.decision is GateDecision.INSUFFICIENT_EVIDENCE
    assert "candidate_recall_at_500" in result.reasons


def test_quality_gate_ties_use_product_id_independently_of_row_order():
    from video_commerce.ml.model_release_quality import _mean_ndcg
    rows = [_row(impression="i1", user="u1", product="a", timestamp=1.0, relevance=4.0),
            _row(impression="i1", user="u1", product="b", timestamp=1.0, relevance=1.0)]
    predictions = {row.observation_id: RankingPrediction(0.5, 0.5, 0.5, 1.0) for row in rows}
    assert _mean_ndcg(rows, predictions) == pytest.approx(1.0)
    assert _mean_ndcg(list(reversed(rows)), predictions) == pytest.approx(1.0)


def _score(serving_score: float) -> RankingPrediction:
    return RankingPrediction(
        serving_score=serving_score, ctr=0.5, ctcvr=0.1, predicted_value=1.0
    )


def test_ndcg_excludes_all_view_slates():
    rows = [
        _row(impression="engaged", user="u1", product="a", timestamp=1.0, relevance=1.0),
        _row(
            impression="engaged",
            user="u1",
            product="b",
            timestamp=1.0,
            relevance=2.0,
            clicked=True,
        ),
    ]
    for index in range(9):
        rows.extend(
            _row(
                impression=f"views{index}",
                user="u2",
                product=product,
                timestamp=1.0,
                relevance=1.0,
            )
            for product in ("a", "b", "c")
        )
    # The view-only row "a" outranks the clicked row "b" in the engaged slate.
    misranked = {
        row.observation_id: _score(1.0 if row.product_id == "a" else 0.0)
        for row in rows
    }

    expected = 1.0 / np.log2(3.0)
    assert _informative_impression_count(rows) == 1
    # All-view slates would otherwise score a perfect 1.0 and dilute the miss.
    assert _mean_ndcg(rows, misranked) == pytest.approx(expected)


def test_ndcg_delta_is_clearly_negative_for_random_order_challenger():
    rng = np.random.default_rng(7)
    rows: list[RankingEvaluationRow] = []
    for index in range(200):
        clicked_position = int(rng.integers(5)) if index % 5 == 0 else None
        for position in range(5):
            clicked = position == clicked_position
            rows.append(
                _row(
                    impression=f"i{index}",
                    user=f"u{index % 40}",
                    product=f"p{position}",
                    timestamp=1.0,
                    relevance=2.0 if clicked else 1.0,
                    clicked=clicked,
                )
            )
    champion = {
        row.observation_id: _score(1.0 if row.clicked else 0.0) for row in rows
    }
    challenger = {row.observation_id: _score(float(rng.random())) for row in rows}

    delta = _mean_ndcg(rows, challenger) - _mean_ndcg(rows, champion)
    bootstrap = _cluster_bootstrap_ndcg(rows, champion, challenger, _small_config())

    assert _informative_impression_count(rows) == 40
    assert _mean_ndcg(rows, champion) == pytest.approx(1.0)
    assert delta < -0.25
    assert bootstrap["ndcg_at_10_delta_ci_upper"] < -0.15


# --- ranking_quality_gate_v4 slice policy -------------------------------------------


def _categorised(rows, category):
    return [replace(row, category=category) for row in rows]


def _tiny_category_rows():
    rows = _categorised(_fixture_rows(), "main")
    rows.extend(
        _categorised(
            [
                _row(
                    impression="tiny-impression",
                    user="u-tiny",
                    product="tiny:positive",
                    timestamp=1_700_000_000.0 + 3 * 86400,
                    relevance=2.0,
                    clicked=True,
                ),
                _row(
                    impression="tiny-impression",
                    user="u-tiny",
                    product="tiny:negative",
                    timestamp=1_700_000_000.0 + 3 * 86400,
                    relevance=1.0,
                ),
            ],
            "tiny",
        )
    )
    return rows


def test_long_tail_category_is_reported_without_blocking_the_gate():
    rows = _tiny_category_rows()
    predictions = {row.observation_id: _prediction(row, correct=True) for row in rows}
    config = replace(_small_config(), min_slice_impressions=4)

    result = evaluate_quality_gate(rows, predictions, predictions, config)

    assert result.decision is GateDecision.PASSED
    assert result.policy_version == "ranking_quality_gate_v4"
    tiny = result.slices["category_tiny"]
    assert tiny["status"] == GateDecision.INSUFFICIENT_EVIDENCE.value
    assert tiny["blocking"] is False
    assert result.slices["price_mid"]["blocking"] is True


def test_pre_registered_slice_without_evidence_still_blocks():
    rows = _tiny_category_rows()
    predictions = {row.observation_id: _prediction(row, correct=True) for row in rows}
    config = replace(_small_config(), min_slice_impressions=100)

    result = evaluate_quality_gate(rows, predictions, predictions, config)

    assert result.decision is GateDecision.INSUFFICIENT_EVIDENCE
    assert "slice_price_mid" in result.reasons
    assert "slice_category_tiny" not in result.reasons


def test_category_slices_are_capped_to_top_n_plus_other():
    rows = []
    for index, row in enumerate(_fixture_rows()):
        rows.append(replace(row, category=f"c{(index // 2) % 8}"))
    predictions = {row.observation_id: _prediction(row, correct=True) for row in rows}
    config = replace(_small_config(), slice_top_n_categories=3)

    result = evaluate_quality_gate(rows, predictions, predictions, config)

    category_slices = sorted(name for name in result.slices if name.startswith("category_"))
    assert len(category_slices) == 4
    assert "category_other" in category_slices


def test_retrain_kind_does_not_require_candidate_recall_evidence():
    from video_commerce.ml.model_release_quality import default_gate_config

    rows = _fixture_rows()
    predictions = {row.observation_id: _prediction(row, correct=True) for row in rows}
    small = _small_config()
    retrain = replace(
        default_gate_config("retrain"),
        **{
            name: getattr(small, name)
            for name in (
                "holdout_days",
                "bootstrap_samples",
                "min_impressions",
                "min_users",
                "min_clicks",
                "min_purchases",
                "min_slice_impressions",
                "min_slice_users",
                "min_slice_positives",
            )
        },
    )
    combined = replace(retrain, experiment_kind="combined", require_candidate_recall=True)

    assert evaluate_quality_gate(rows, predictions, predictions, retrain).decision is (
        GateDecision.PASSED
    )
    assert "candidate_recall_at_500" in evaluate_quality_gate(
        rows, predictions, predictions, combined
    ).reasons


def test_retrain_snapshot_validates_against_retrain_defaults_only():
    from dataclasses import asdict

    from video_commerce.ml.model_release_quality import default_gate_config
    from video_commerce.ml.phase_c_evidence import quality_gate_snapshot_valid

    retrain = asdict(default_gate_config("retrain"))

    assert quality_gate_snapshot_valid(retrain, experiment_kind="retrain")
    assert not quality_gate_snapshot_valid(retrain, experiment_kind="combined")
    # v2/v3 snapshots predate the top-N fields and remain valid.
    legacy = asdict(default_gate_config("combined"))
    legacy.pop("slice_top_n_categories")
    legacy.pop("slice_top_n_retrieval_sources")
    assert quality_gate_snapshot_valid(legacy)


def test_equivalent_challenger_rarely_fails_across_many_category_slices():
    """A/A-style check: equal-quality models with noisy scores, 30 categories.

    Only slice-driven failures are counted: the overall NDCG non-inferiority CI
    is a power question that depends on holdout size, not on slice policy.
    """
    slice_failures = 0
    category_failures = 0
    trials = 20
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        rows = []
        for impression in range(240):
            user = f"u{impression % 120}"
            category = f"c{impression % 30}"
            for slot in range(4):
                clicked = bool(rng.random() < 0.3)
                rows.append(
                    replace(
                        _row(
                            impression=f"i{impression}",
                            user=user,
                            product=f"i{impression}:p{slot}",
                            timestamp=1_700_000_000.0 + impression * 600,
                            relevance=2.0 if clicked else 1.0,
                            clicked=clicked,
                            purchased=clicked and rng.random() < 0.3,
                            value=10.0,
                        ),
                        category=category,
                    )
                )
        base = {
            row.observation_id: float(rng.random()) + (0.3 if row.clicked else 0.0)
            for row in rows
        }

        def predictions(noise_seed):
            noise = np.random.default_rng(noise_seed)
            out = {}
            for row in rows:
                score = base[row.observation_id] + float(noise.normal(0.0, 0.01))
                probability = min(0.99, max(0.01, score / 1.5))
                out[row.observation_id] = RankingPrediction(
                    serving_score=score,
                    ctr=probability,
                    ctcvr=probability * 0.3,
                    predicted_value=10.0,
                )
            return out

        config = RankingQualityGateConfig(
            bootstrap_samples=100,
            min_impressions=50,
            min_users=20,
            min_clicks=10,
            min_purchases=5,
            min_slice_impressions=5,
            min_slice_users=3,
            min_slice_positives=1,
            require_candidate_recall=False,
            experiment_kind="retrain",
        )
        result = evaluate_quality_gate(
            rows, predictions(1000 + seed), predictions(2000 + seed), config
        )
        slice_reasons = [reason for reason in result.reasons if reason.startswith("slice_")]
        slice_failures += bool(slice_reasons)
        category_failures += any("category_" in reason for reason in result.reasons)

    # 30 per-category point thresholds no longer multiply false failures.
    assert category_failures == 0
    assert slice_failures / trials <= 0.05


def test_fixed_holdout_purges_rows_within_label_reach_of_the_holdout():
    rows = _fixture_rows()

    training, holdout, window = split_fixed_holdout(
        rows,
        holdout_days=2,
        attribution_cutoff=rows[-1].as_of_ts + 86400,
        purge_seconds=2 * 86400,
    )
    examples = [SimpleNamespace(bundle=SimpleNamespace(as_of_ts=row.as_of_ts)) for row in rows]
    example_training, _, _ = split_training_examples_fixed_holdout(
        examples,
        holdout_days=2,
        attribution_cutoff=rows[-1].as_of_ts + 86400,
        purge_seconds=2 * 86400,
    )

    assert holdout
    assert all(row.as_of_ts < window.start_ts - 2 * 86400 for row in training)
    assert len(example_training) == len(training)
    unpurged, _, _ = split_fixed_holdout(
        rows, holdout_days=2, attribution_cutoff=rows[-1].as_of_ts + 86400
    )
    assert len(unpurged) > len(training)
