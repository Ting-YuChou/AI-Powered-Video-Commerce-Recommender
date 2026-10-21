from __future__ import annotations

from dataclasses import replace
import json
import os
import resource
import sys
import time
import tracemalloc
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
    _GATED_METRICS,
    _GateColumns,
    _cluster_bootstrap,
    _cluster_bootstrap_rows,
    _comparison_metrics,
    _gate_columns,
    _gate_statistics,
    _informative_impression_count,
    _mean_ndcg,
    _metric_failures,
    _ndcg_underpowered,
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
    assert first.policy_version == "ranking_quality_gate_v6"
    assert first.policy_version == QUALITY_GATE_POLICY_VERSION
    assert first.overall["ndcg_informative_impressions"] == 3


def test_quality_gate_rejects_serving_order_regression():
    rows = _fixture_rows()[-6:]
    champion = {row.observation_id: _prediction(row, correct=True) for row in rows}
    challenger = {row.observation_id: _prediction(row, correct=False) for row in rows}

    result = evaluate_quality_gate(rows, champion, challenger, _small_config())

    assert result.decision is GateDecision.FAILED
    assert "overall_ndcg_regression" in result.reasons
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
    bootstrap = _cluster_bootstrap_rows(rows, champion, challenger, _small_config())

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
    assert result.policy_version == "ranking_quality_gate_v6"
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


def _equal_quality_fixture(
    seed: int, *, impressions: int = 240, categories: int = 30, noise: float = 0.01
):
    """Rows plus a factory for equal-quality predictions with independent noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for impression in range(impressions):
        user = f"u{impression % (impressions // 2)}"
        category = f"c{impression % categories}"
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
        generator = np.random.default_rng(noise_seed)
        out = {}
        for row in rows:
            score = base[row.observation_id] + float(generator.normal(0.0, noise))
            probability = min(0.99, max(0.01, score / 1.5))
            out[row.observation_id] = RankingPrediction(
                serving_score=score,
                ctr=probability,
                ctcvr=probability * 0.3,
                predicted_value=10.0,
            )
        return out

    return rows, predictions


def _aa_config() -> RankingQualityGateConfig:
    return RankingQualityGateConfig(
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


def test_equivalent_challenger_rarely_fails_across_many_category_slices():
    """A/A-style check: equal-quality models with noisy scores, 30 categories."""
    slice_failures = 0
    category_failures = 0
    ndcg_failures = 0
    underpowered = 0
    trials = 20
    for seed in range(trials):
        rows, predictions = _equal_quality_fixture(seed)
        result = evaluate_quality_gate(
            rows, predictions(1000 + seed), predictions(2000 + seed), _aa_config()
        )
        slice_reasons = [
            reason
            for reason in result.reasons
            if reason.startswith("slice_") and not reason.endswith("_underpowered")
        ]
        slice_failures += bool(slice_reasons)
        category_failures += any("category_" in reason for reason in result.reasons)
        ndcg_failures += any(
            reason.endswith(("_ndcg_regression", "_ndcg_minimum_improvement"))
            for reason in result.reasons
        )
        underpowered += "overall_ndcg_underpowered" in result.reasons

    # 30 per-category point thresholds no longer multiply false failures.
    assert category_failures == 0
    assert slice_failures / trials <= 0.05
    # v5: an equal-quality challenger never fails on NDCG; a holdout this small
    # often cannot resolve the -0.005 margin and is reported as underpowered.
    assert ndcg_failures == 0
    assert underpowered > 0


def _ndcg_metrics(delta: float, lower: float, upper: float) -> dict:
    metrics = {
        "ndcg_informative_impressions": 400,
        "users": 400,
        "ctr_positives": 400,
        "ctr_negatives": 400,
        "ctcvr_positives": 400,
        "ctcvr_negatives": 400,
        "value_rows": 400,
    }
    for metric, _, _ in _GATED_METRICS:
        metrics.update(
            {
                metric: 0.0,
                f"{metric}_ci_lower": 0.0,
                f"{metric}_ci_upper": 0.0,
                f"{metric}_ci_half_width": 0.0,
            }
        )
    metrics.update(
        {
            "ndcg_at_10_delta": delta,
            "ndcg_at_10_delta_ci_lower": lower,
            "ndcg_at_10_delta_ci_upper": upper,
            "ndcg_at_10_delta_ci_half_width": (upper - lower) / 2.0,
        }
    )
    return metrics


@pytest.mark.parametrize(
    ("delta", "lower", "upper", "failed", "underpowered"),
    [
        (0.001, -0.002, 0.004, False, False),  # CI clears the margin
        (-0.001, -0.009, 0.007, False, True),  # straddles the margin
        (-0.02, -0.03, -0.01, True, False),  # wholly below the margin
        (-0.003, -0.004, -0.002, False, False),  # worse, but within the margin
    ],
)
def test_ndcg_is_a_three_way_non_inferiority_test(
    delta, lower, upper, failed, underpowered
):
    config = RankingQualityGateConfig(experiment_kind="retrain")
    metrics = _ndcg_metrics(delta, lower, upper)

    failures = _metric_failures(metrics, config, slice_name=None)

    assert ("overall_ndcg_regression" in failures) is failed
    assert "overall_ndcg_minimum_improvement" not in failures
    assert _ndcg_underpowered(metrics, config, slice_name=None) is underpowered


def test_positive_minimum_delta_is_still_a_superiority_requirement():
    config = RankingQualityGateConfig(overall_ndcg_min_delta=0.005)

    failures = _metric_failures(
        _ndcg_metrics(0.001, -0.002, 0.004), config, slice_name=None
    )

    assert "overall_ndcg_minimum_improvement" in failures


def test_underpowered_holdout_is_insufficient_with_a_sample_size_estimate():
    for seed in range(20):
        rows, predictions = _equal_quality_fixture(seed)
        result = evaluate_quality_gate(
            rows, predictions(1000 + seed), predictions(2000 + seed), _aa_config()
        )
        if (
            result.decision is GateDecision.INSUFFICIENT_EVIDENCE
            and "overall_ndcg_underpowered" in result.reasons
        ):
            break
    else:
        pytest.fail("no seed produced an NDCG underpowered result")

    assert result.decision is GateDecision.INSUFFICIENT_EVIDENCE
    assert result.overall["ndcg_at_10_delta_ci_half_width"] > 0.0
    estimate = result.overall["ndcg_informative_impressions_required_estimate"]
    if result.overall["ndcg_at_10_delta"] > -0.005:
        assert estimate > result.overall["ndcg_informative_impressions"]
    else:
        assert estimate is None


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


# --- ranking_quality_gate_v6: CI-based metrics and vectorized bootstrap -------------


def _slow_mean_ndcg(rows, predictions) -> float:
    """The v5 per-row implementation, kept as the equivalence reference."""
    import math

    grouped: dict[str, list[RankingEvaluationRow]] = {}
    for row in rows:
        grouped.setdefault(row.impression_id, []).append(row)

    def gain(row):
        return max(float(row.relevance) - 1.0, 0.0)

    values = []
    for impression_rows in grouped.values():
        if len(impression_rows) < 2 or not any(gain(row) > 0.0 for row in impression_rows):
            continue
        ranked = sorted(
            impression_rows,
            key=lambda row: (-predictions[row.observation_id].serving_score, row.product_id),
        )[:10]
        ideal = sorted(impression_rows, key=lambda row: (-row.relevance, row.product_id))[:10]

        def dcg(selected):
            return sum(
                gain(row) / math.log2(position + 2) for position, row in enumerate(selected)
            )

        values.append(dcg(ranked) / dcg(ideal))
    return float(np.mean(values)) if values else 0.0


def _slow_resamples(rows, config):
    """v5 resampling: copies of each sampled user's rows with fresh impression ids."""
    rows_by_user: dict[str, list[RankingEvaluationRow]] = {}
    for row in rows:
        rows_by_user.setdefault(row.user_id, []).append(row)
    users = sorted(rows_by_user)
    generator = np.random.default_rng(config.random_seed)
    for _ in range(config.bootstrap_samples):
        sampled = []
        for index, user in enumerate(generator.choice(users, size=len(users), replace=True)):
            sampled.extend(
                replace(row, impression_id=f"{index}:{row.impression_id}")
                for row in rows_by_user[str(user)]
            )
        yield sampled


def _slow_metric_values(rows, predictions, bin_of) -> dict:
    """sklearn metrics on materialized rows; ECE uses the evaluated rows' bins."""
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

    labels = {
        "ctr": np.array([float(row.clicked) for row in rows]),
        "ctcvr": np.array([float(row.purchased) for row in rows]),
    }
    values = {"ndcg_at_10": _slow_mean_ndcg(rows, predictions)}
    for head, y in labels.items():
        p = np.clip([getattr(predictions[row.observation_id], head) for row in rows], 1e-7, 1 - 1e-7)
        values[f"{head}_pr_auc"] = float(average_precision_score(y, p))
        values[f"{head}_log_loss"] = float(log_loss(y, p, labels=[0.0, 1.0]))
        values[f"{head}_brier"] = float(brier_score_loss(y, p))
        gaps: dict[int, float] = {}
        for row, probability, label in zip(rows, p, y):
            key = bin_of[head][row.observation_id]
            gaps[key] = gaps.get(key, 0.0) + probability - label
        values[f"{head}_ece"] = sum(abs(gap) for gap in gaps.values()) / len(rows)
    value_rows = [row for row in rows if row.purchased and row.value_mask]
    errors = sum(
        abs(row.business_value - predictions[row.observation_id].predicted_value)
        for row in value_rows
    )
    values["value_wape"] = errors / max(sum(row.business_value for row in value_rows), 1e-12)
    return values


def _slow_bins(rows, predictions, bins: int) -> dict:
    out = {}
    for head in ("ctr", "ctcvr"):
        p = np.clip([getattr(predictions[row.observation_id], head) for row in rows], 1e-7, 1 - 1e-7)
        groups = np.array_split(np.argsort(p, kind="mergesort"), min(bins, len(rows)))
        out[head] = {
            rows[index].observation_id: number
            for number, group in enumerate(groups)
            for index in group
        }
    return out


def _slow_gated(champion_values, challenger_values) -> dict:
    def relative(name):
        old = champion_values[name]
        return (challenger_values[name] - old) / max(abs(old), 1e-12)

    gated = {"ndcg_at_10_delta": challenger_values["ndcg_at_10"] - champion_values["ndcg_at_10"]}
    for head in ("ctr", "ctcvr"):
        gated[f"{head}_pr_auc_delta"] = (
            challenger_values[f"{head}_pr_auc"] - champion_values[f"{head}_pr_auc"]
        )
        gated[f"{head}_log_loss_relative_regression"] = relative(f"{head}_log_loss")
        gated[f"{head}_brier_relative_regression"] = relative(f"{head}_brier")
        gated[f"{head}_ece_absolute_regression"] = (
            challenger_values[f"{head}_ece"] - champion_values[f"{head}_ece"]
        )
    gated["value_wape_relative_regression"] = relative("value_wape")
    return gated


def _slow_bootstrap(rows, champion, challenger, config, *, alpha=0.05) -> dict:
    champion_bins = _slow_bins(rows, champion, config.ece_bins)
    challenger_bins = _slow_bins(rows, challenger, config.ece_bins)
    draws: dict[str, list[float]] = {}
    for sampled in _slow_resamples(rows, config):
        gated = _slow_gated(
            _slow_metric_values(sampled, champion, champion_bins),
            _slow_metric_values(sampled, challenger, challenger_bins),
        )
        for name, value in gated.items():
            draws.setdefault(name, []).append(value)
    result = {}
    for name, values in draws.items():
        result[f"{name}_ci_lower"] = float(np.quantile(values, alpha / 2.0))
        result[f"{name}_ci_upper"] = float(np.quantile(values, 1.0 - alpha / 2.0))
    return result


def _random_order_rows():
    rng = np.random.default_rng(7)
    rows = []
    for index in range(60):
        clicked_position = int(rng.integers(5)) if index % 3 == 0 else None
        for position in range(5):
            clicked = position == clicked_position
            rows.append(
                _row(
                    impression=f"i{index}",
                    user=f"u{index % 17}",
                    product=f"p{position}",
                    timestamp=1.0,
                    relevance=(3.0 if position % 2 else 2.0) if clicked else 1.0,
                    clicked=clicked,
                    purchased=clicked and position % 2 == 1,
                    value=5.0 + position,
                )
            )
    champion = {row.observation_id: _score(1.0 if row.clicked else 0.0) for row in rows}
    challenger = {
        row.observation_id: RankingPrediction(
            serving_score=float(rng.random()),
            ctr=float(rng.random()),
            ctcvr=float(rng.random()) * 0.3,
            predicted_value=float(rng.random()) * 10.0,
        )
        for row in rows
    }
    return rows, champion, challenger


def _equivalence_cases():
    for seed in (0, 1):
        rows, predictions = _equal_quality_fixture(seed, impressions=80, categories=5)
        yield rows, predictions(1000 + seed), predictions(2000 + seed)
    yield _random_order_rows()


@pytest.mark.parametrize("case", range(3))
def test_vectorized_ndcg_bootstrap_matches_the_v5_slow_implementation(case):
    rows, champion, challenger = list(_equivalence_cases())[case]
    config = replace(_small_config(), bootstrap_samples=60)

    slow = [
        _slow_mean_ndcg(sampled, challenger) - _slow_mean_ndcg(sampled, champion)
        for sampled in _slow_resamples(rows, config)
    ]
    fast = _cluster_bootstrap_rows(rows, champion, challenger, config)

    # The same seed draws the same resamples as v5's generator.choice().
    assert fast["ndcg_at_10_delta_ci_lower"] == pytest.approx(
        float(np.quantile(slow, 0.025)), abs=1e-12
    )
    assert fast["ndcg_at_10_delta_ci_upper"] == pytest.approx(
        float(np.quantile(slow, 0.975)), abs=1e-12
    )
    assert _mean_ndcg(rows, champion) == pytest.approx(_slow_mean_ndcg(rows, champion), abs=1e-12)


@pytest.mark.parametrize("case", range(3))
def test_vectorized_point_metrics_match_the_sklearn_definitions(case):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

    from video_commerce.ml.probability_calibration import expected_calibration_error

    rows, champion, challenger = list(_equivalence_cases())[case]
    columns = _gate_columns(rows, champion, challenger)
    metrics = _comparison_metrics(columns, _gate_statistics(columns, ece_bins=15))

    def reference(predictions):
        values = {"ndcg_at_10": _slow_mean_ndcg(rows, predictions)}
        for head, attribute in (("ctr", "clicked"), ("ctcvr", "purchased")):
            y = np.array([float(getattr(row, attribute)) for row in rows])
            p = np.clip([getattr(predictions[row.observation_id], head) for row in rows], 1e-7, 1 - 1e-7)
            values[f"{head}_pr_auc"] = average_precision_score(y, p)
            values[f"{head}_log_loss"] = log_loss(y, p, labels=[0.0, 1.0])
            values[f"{head}_brier"] = brier_score_loss(y, p)
            values[f"{head}_ece"] = expected_calibration_error(p, y, bins=15)
        value_rows = [row for row in rows if row.purchased and row.value_mask]
        errors = [
            abs(row.business_value - predictions[row.observation_id].predicted_value)
            for row in value_rows
        ]
        values["value_wape"] = sum(errors) / max(sum(row.business_value for row in value_rows), 1e-12)
        values["value_mae"] = float(np.mean(errors)) if errors else 0.0
        return values

    old, new = reference(champion), reference(challenger)
    for name, value in _slow_gated(old, new).items():
        assert metrics[name] == pytest.approx(value, abs=1e-12), name
    for head in ("ctr", "ctcvr"):
        assert metrics[f"{head}_ece_champion"] == pytest.approx(old[f"{head}_ece"], abs=1e-12)
    assert metrics["value_mae_challenger"] == pytest.approx(new["value_mae"], abs=1e-12)
    assert metrics["ndcg_at_10_champion"] == pytest.approx(old["ndcg_at_10"], abs=1e-12)


@pytest.mark.parametrize("case", range(3))
def test_vectorized_bootstrap_matches_row_resampling_for_every_metric(case):
    rows, champion, challenger = list(_equivalence_cases())[case]
    config = replace(_small_config(), bootstrap_samples=40)

    slow = _slow_bootstrap(rows, champion, challenger, config)
    fast = _cluster_bootstrap_rows(rows, champion, challenger, config)

    assert set(slow) <= set(fast)
    for name, value in slow.items():
        assert fast[name] == pytest.approx(value, abs=1e-9), name


def test_bootstrap_is_deterministic_and_independent_of_chunking(monkeypatch):
    import video_commerce.ml.model_release_quality as quality

    rows, champion, challenger = _random_order_rows()
    config = replace(_small_config(), bootstrap_samples=50)
    statistics = _gate_statistics(_gate_columns(rows, champion, challenger), ece_bins=15)

    first = _cluster_bootstrap(statistics, config)
    monkeypatch.setattr(quality, "_BOOTSTRAP_CHUNK_ELEMENTS", 7)
    chunked = _cluster_bootstrap(statistics, config)
    reseeded = _cluster_bootstrap(statistics, replace(config, random_seed=43))

    assert chunked == pytest.approx(first, abs=1e-12)
    assert _cluster_bootstrap(statistics, config) == pytest.approx(first, abs=1e-12)
    assert reseeded != first


def test_gate_reports_ci_bounds_for_every_gated_metric():
    rows, predictions = _equal_quality_fixture(0)

    result = evaluate_quality_gate(rows, predictions(1000), predictions(2000), _aa_config())

    for metric, _, _ in _GATED_METRICS:
        for bound in ("lower", "upper", "half_width"):
            key = f"{metric}_ci_{bound}"
            assert key in result.bootstrap
            assert key in result.overall
            assert key in result.slices["price_mid"]["metrics"]
        assert result.overall[f"{metric}_ci_lower"] <= result.overall[f"{metric}_ci_upper"]


def _ci_metrics(metric: str, lower: float, upper: float) -> dict:
    metrics = _ndcg_metrics(0.0, 0.0, 0.0)
    metrics.update(
        {
            metric: (lower + upper) / 2.0,
            f"{metric}_ci_lower": lower,
            f"{metric}_ci_upper": upper,
            f"{metric}_ci_half_width": (upper - lower) / 2.0,
        }
    )
    return metrics


@pytest.mark.parametrize(("metric", "stem", "higher_is_better"), _GATED_METRICS)
@pytest.mark.parametrize("slice_name", [None, "price_mid"])
def test_every_gated_metric_is_a_three_way_ci_test(
    metric, stem, higher_is_better, slice_name
):
    from video_commerce.ml.model_release_quality import (
        _metric_threshold,
        _underpowered_reasons,
    )

    config = RankingQualityGateConfig(experiment_kind="retrain")
    prefix = f"slice_{slice_name}" if slice_name else "overall"
    threshold = _metric_threshold(metric, config, slice_name=slice_name)
    worse = -1.0 if higher_is_better else 1.0
    failure = f"{prefix}_ndcg_regression" if stem == "ndcg" else f"{prefix}_{metric}"
    underpowered = f"{prefix}_{stem}_underpowered"
    beyond = sorted((threshold + worse * 0.01, threshold + worse * 0.03))
    straddling = sorted((threshold + worse * 0.01, threshold - worse * 0.01))
    within = sorted((threshold - worse * 0.001, threshold - worse * 0.01))

    for (lower, upper), failed, insufficient in (
        (beyond, True, False),
        (straddling, False, True),
        (within, False, False),
    ):
        metrics = _ci_metrics(metric, lower, upper)
        failures = _metric_failures(metrics, config, slice_name=slice_name)
        reasons = _underpowered_reasons(metrics, config, slice_name=slice_name)
        assert (failure in failures) is failed
        assert reasons == ([underpowered] if insufficient else [])


@pytest.mark.parametrize("noise", [0.01, 0.05])
def test_equal_quality_challenger_false_fail_rate_is_under_five_percent(noise):
    """A/A: equal-quality models with independent score noise, 240 impressions.

    Under v5 the point thresholds failed 2/20 runs at noise 0.01 and most runs
    at 0.05. Under v6 no gated metric fails unless its whole CI is beyond the
    threshold, so small holdouts are insufficient evidence instead.
    """
    trials = 40
    failed = 0
    insufficient = 0
    config = replace(_aa_config(), bootstrap_samples=200)
    for seed in range(trials):
        rows, predictions = _equal_quality_fixture(seed, noise=noise)
        result = evaluate_quality_gate(
            rows, predictions(1000 + seed), predictions(2000 + seed), config
        )
        failed += result.decision is GateDecision.FAILED
        insufficient += result.decision is GateDecision.INSUFFICIENT_EVIDENCE
        assert all(reason.endswith("_underpowered") for reason in result.reasons)

    assert failed / trials < 0.05
    assert insufficient > 0


def test_clear_regression_still_fails_on_ci_based_metrics():
    rows, predictions = _equal_quality_fixture(0)
    champion = predictions(1000)
    rng = np.random.default_rng(5)
    challenger = {
        key: RankingPrediction(
            serving_score=float(rng.random()),
            ctr=min(0.99, value.ctr * 2.5),
            ctcvr=min(0.99, value.ctcvr * 2.5),
            predicted_value=20.0,
        )
        for key, value in champion.items()
    }

    result = evaluate_quality_gate(
        rows, champion, challenger, replace(_aa_config(), bootstrap_samples=500)
    )

    assert result.decision is GateDecision.FAILED
    for reason in (
        "overall_ndcg_regression",
        "overall_ctr_pr_auc_delta",
        "overall_ctr_log_loss_relative_regression",
        "overall_ctr_brier_relative_regression",
        "overall_ctr_ece_absolute_regression",
        "overall_value_wape_relative_regression",
        "slice_price_mid",
    ):
        assert reason in result.reasons



@pytest.mark.parametrize("policy", ["ranking_quality_gate_v5", "ranking_quality_gate_v6"])
def test_v5_and_v6_evidence_both_remain_valid_for_phase_c(policy):
    from dataclasses import asdict

    from video_commerce.ml.model_release_quality import (
        PHASE_C_QUALITY_GATE_POLICY_VERSIONS,
        SUPPORTED_QUALITY_GATE_POLICY_VERSIONS,
        default_gate_config,
    )
    from video_commerce.ml.phase_c_evidence import (
        active_rrf_release_valid,
        quality_gate_snapshot_valid,
    )

    assert policy in PHASE_C_QUALITY_GATE_POLICY_VERSIONS
    assert policy in SUPPORTED_QUALITY_GATE_POLICY_VERSIONS
    assert active_rrf_release_valid(
        {
            "payload": {
                "model_release_id": "release-1",
                "active_generation": 1,
                "quality_gate_policy_version": policy,
                "candidate_fusion_policy_version": "candidate_rrf_v1",
                "candidate_rrf_k": 60,
                "candidate_source_registry_version": "retrieval_sources_v1",
            }
        }
    )
    # v6 changes decision rules only, not the config, so a v5 snapshot of the
    # frozen defaults still validates.
    assert quality_gate_snapshot_valid(asdict(default_gate_config("combined")))



def _calibrated_fixture(seed: int, *, impressions: int, buy_rate: float = 0.3):
    """Rows drawn from a known per-row CTR; models predict it with logit noise.

    Lower noise is strictly better on ranking, calibration and value error, so
    a less noisy challenger is a genuine non-regressing improvement.
    """
    rng = np.random.default_rng(seed)
    rows, truth = [], {}
    for impression in range(impressions):
        for slot in range(4):
            rate = float(rng.beta(1.5, 4.0))
            clicked = bool(rng.random() < rate)
            purchased = clicked and bool(rng.random() < buy_rate)
            value = float(rng.gamma(4.0, 5.0)) if purchased else 0.0
            row = _row(
                impression=f"i{impression}",
                user=f"u{impression // 2}",
                product=f"i{impression}:p{slot}",
                timestamp=1_700_000_000.0 + impression * 60,
                relevance=3.0 if purchased else 2.0 if clicked else 1.0,
                clicked=clicked,
                purchased=purchased,
                value=value,
            )
            rows.append(row)
            truth[row.observation_id] = (rate, value)

    def predictions(noise_seed: int, noise: float):
        generator = np.random.default_rng(noise_seed)
        out = {}
        for row in rows:
            rate, value = truth[row.observation_id]
            logit = np.log(rate / (1.0 - rate)) + generator.normal(0.0, noise)
            probability = float(1.0 / (1.0 + np.exp(-logit)))
            predicted_value = (
                value * float(np.exp(generator.normal(0.0, noise))) if value else 20.0
            )
            out[row.observation_id] = RankingPrediction(
                serving_score=probability,
                ctr=probability,
                ctcvr=probability * buy_rate,
                predicted_value=predicted_value,
            )
        return out

    return rows, predictions


def test_non_regressing_better_challenger_passes_on_a_large_holdout():
    """16k rows / 2k users: a genuinely better challenger clears every CI."""
    config = replace(_aa_config(), bootstrap_samples=500)
    decisions = []
    for seed in range(6):
        rows, predictions = _calibrated_fixture(seed, impressions=4000)
        result = evaluate_quality_gate(
            rows, predictions(1000 + seed, 0.5), predictions(2000 + seed, 0.2), config
        )
        decisions.append(result.decision)
        assert result.decision is not GateDecision.FAILED
        assert result.overall["ndcg_at_10_delta_ci_lower"] > 0.0

    assert decisions.count(GateDecision.PASSED) >= 4


def test_equal_quality_challenger_can_pass_at_a_realistic_holdout():
    """A/A at 16k rows / 2k users: v6 is passable, not only underpowered."""
    config = replace(_aa_config(), bootstrap_samples=500)
    decisions = []
    for seed in range(4):
        rows, predictions = _equal_quality_fixture(seed, impressions=4000, categories=5)
        result = evaluate_quality_gate(
            rows, predictions(1000 + seed), predictions(2000 + seed), config
        )
        decisions.append(result.decision)

    assert GateDecision.FAILED not in decisions
    assert decisions.count(GateDecision.PASSED) > 0


def test_underpowered_metrics_report_a_required_user_estimate():
    rows, predictions = _equal_quality_fixture(0)

    result = evaluate_quality_gate(
        rows, predictions(1000), predictions(2000), _aa_config()
    )

    underpowered = [
        reason for reason in result.reasons if reason.startswith("overall_")
    ]
    assert underpowered
    for metric, stem, _ in _GATED_METRICS:
        key = f"{metric}_required_users_estimate"
        if f"overall_{stem}_underpowered" in result.reasons:
            assert key in result.overall
            estimate = result.overall[key]
            assert estimate is None or estimate > result.overall["users"]
        else:
            assert key not in result.overall


def test_head_without_positives_is_no_evidence_not_a_pass():
    """A blocking slice with clicks but no purchases cannot vouch for ctcvr."""
    rows, predictions = _equal_quality_fixture(0)
    rows = [
        replace(row, price=300.0, purchased=False)
        if int(row.impression_id[1:]) % 4 == 0
        else row
        for row in rows
    ]
    champion = predictions(1000)

    result = evaluate_quality_gate(rows, champion, dict(champion), _aa_config())

    high = result.slices["price_high"]
    assert high["blocking"] is True
    assert high["status"] == GateDecision.INSUFFICIENT_EVIDENCE.value
    for stem in ("ctcvr_pr_auc", "ctcvr_log_loss", "ctcvr_brier", "ctcvr_ece", "value_wape"):
        assert f"slice_price_high_{stem}_no_evidence" in high["reasons"]
        assert f"slice_price_high_{stem}_no_evidence" in result.reasons
    assert "slice_price_high_ctr_pr_auc_no_evidence" not in high["reasons"]
    assert result.decision is GateDecision.INSUFFICIENT_EVIDENCE


def test_definite_failure_takes_precedence_over_a_slice_without_evidence():
    rows, predictions = _equal_quality_fixture(0)
    # One brand-new item makes a cold_item slice far below its evidence floor.
    rows = [
        replace(row, product_created_at=row.as_of_ts, pre_holdout_exposures=0)
        if row.impression_id == "i0"
        else row
        for row in rows
    ]
    champion = predictions(1000)
    rng = np.random.default_rng(5)
    challenger = {
        key: replace(value, serving_score=float(rng.random()))
        for key, value in champion.items()
    }

    result = evaluate_quality_gate(
        rows, champion, challenger, replace(_aa_config(), bootstrap_samples=500)
    )

    assert result.slices["cold_item"]["status"] == GateDecision.INSUFFICIENT_EVIDENCE.value
    assert result.decision is GateDecision.FAILED
    assert "overall_ndcg_regression" in result.reasons
    assert "slice_cold_item" not in result.reasons


def _requirement_metrics(**cis) -> dict:
    metrics = _ndcg_metrics(0.0, 0.0, 0.0)
    for head in ("ctr", "ctcvr"):
        metrics[f"{head}_ece_champion"] = 0.05
        metrics[f"{head}_ece_relative_regression"] = 0.0
        metrics[f"{head}_ece_relative_regression_ci_lower"] = 0.0
        metrics[f"{head}_ece_relative_regression_ci_upper"] = 0.0
    for metric, (lower, upper) in cis.items():
        metrics.update(
            {
                metric: (lower + upper) / 2.0,
                f"{metric}_ci_lower": lower,
                f"{metric}_ci_upper": upper,
                f"{metric}_ci_half_width": (upper - lower) / 2.0,
            }
        )
    return metrics


@pytest.mark.parametrize(
    ("kind", "overall_cis", "cold_cis", "failed", "underpowered"),
    [
        # Features: overall NDCG >= 0.005 or cold-user NDCG >= 0.01.
        ("features", {"ndcg_at_10_delta": (0.006, 0.01)}, None, False, False),
        ("features", {"ndcg_at_10_delta": (-0.002, 0.01)}, None, False, True),
        ("features", {"ndcg_at_10_delta": (-0.002, 0.004)}, (-0.01, 0.005), True, False),
        ("features", {"ndcg_at_10_delta": (-0.002, 0.004)}, (0.011, 0.03), False, False),
        # Calibration: noise around zero Brier change is underpowered, not a failure.
        (
            "calibration",
            {
                "ctr_ece_relative_regression": (-0.4, -0.2),
                "ctcvr_ece_relative_regression": (-0.4, -0.2),
                "ctr_brier_relative_regression": (-0.01, 0.01),
                "ctcvr_brier_relative_regression": (-0.02, -0.01),
            },
            None,
            False,
            True,
        ),
        (
            "calibration",
            {
                "ctr_ece_relative_regression": (-0.05, 0.05),
                "ctcvr_ece_relative_regression": (-0.4, -0.2),
                "ctr_brier_relative_regression": (-0.02, -0.01),
                "ctcvr_brier_relative_regression": (-0.02, -0.01),
            },
            None,
            True,
            False,
        ),
    ],
)
def test_experiment_requirements_are_three_way_ci_tests(
    kind, overall_cis, cold_cis, failed, underpowered
):
    from video_commerce.ml.model_release_quality import _experiment_requirements

    config = RankingQualityGateConfig(experiment_kind=kind)
    slices = {}
    if cold_cis is not None:
        slices["cold_user"] = {
            "metrics": _requirement_metrics(ndcg_at_10_delta=cold_cis)
        }

    failures, insufficient = _experiment_requirements(
        _requirement_metrics(**overall_cis), slices, config
    )

    assert bool(failures) is failed
    assert bool(insufficient) is underpowered
    assert all(reason.endswith("_underpowered") for reason in insufficient)


def _priced_rows():
    rows, predictions = _equal_quality_fixture(3, impressions=120, categories=3)
    prices = (20.0, 100.0, 300.0)
    rows = [
        replace(row, price=prices[int(row.impression_id[1:]) % 3]) for row in rows
    ]
    return rows, predictions(1003), predictions(2003)


def test_blocking_slices_share_one_bonferroni_alpha_and_match_the_row_path():
    rows, champion, challenger = _priced_rows()
    config = replace(_aa_config(), bootstrap_samples=2000)

    result = evaluate_quality_gate(rows, champion, challenger, config)

    assert result.overall["ci_alpha"] == pytest.approx(0.05)
    for name in ("price_low", "price_mid", "price_high"):
        metrics = result.slices[name]["metrics"]
        assert result.slices[name]["blocking"] is True
        assert metrics["ci_alpha"] == pytest.approx(0.05 / 3)
        selected = [row for row in rows if _price_slice(row) == name]
        expected = _cluster_bootstrap_rows(
            selected, champion, challenger, config, alpha=0.05 / 3
        )
        for key, value in expected.items():
            assert metrics[key] == pytest.approx(value, abs=1e-12), (name, key)


def _price_slice(row) -> str:
    return "price_low" if row.price < 50 else "price_mid" if row.price < 200 else "price_high"


def test_slice_path_matches_row_level_resampling():
    rows, champion, challenger = _priced_rows()
    config = replace(_small_config(), bootstrap_samples=40)

    result = evaluate_quality_gate(rows, champion, challenger, config)

    metrics = result.slices["price_high"]["metrics"]
    selected = [row for row in rows if _price_slice(row) == "price_high"]
    slow = _slow_bootstrap(selected, champion, challenger, config, alpha=0.05 / 3)
    for key, value in slow.items():
        assert metrics[key] == pytest.approx(value, abs=1e-9), key


# --- opt-in scale benchmark ----------------------------------------------------


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _synthetic_columns(rows: int, users: int, slate_size: int = 10) -> _GateColumns:
    """Slates of `slate_size` rows; each impression belongs to one user."""
    rng = np.random.default_rng(2026)
    impressions = rows // slate_size
    impression = np.repeat(np.arange(impressions), slate_size)[:rows]
    user = np.sort(rng.integers(0, users, size=impressions))[impression]
    user = np.unique(user, return_inverse=True)[1].reshape(-1).astype(np.int64)
    clicked = (rng.random(rows) < 0.08).astype(np.float64)
    purchased = clicked * (rng.random(rows) < 0.2)
    relevance = 1.0 + clicked + 2.0 * purchased
    base = rng.random(rows) + 0.4 * clicked + 0.4 * purchased
    noise = rng.normal(0.0, 0.05, size=(2, rows))
    score = base + noise
    ctr = np.clip(score / 2.0, 1e-7, 1 - 1e-7)
    value = np.where(purchased > 0, rng.gamma(2.0, 20.0, size=rows), 0.0)
    return _GateColumns(
        user=user,
        impression=impression.astype(np.int64),
        product_rank=rng.permutation(rows).astype(np.int64),
        relevance=relevance,
        clicked=clicked,
        purchased=purchased,
        value_row=purchased > 0,
        business_value=value,
        serving_score=score,
        ctr=ctr,
        ctcvr=np.clip(ctr * 0.2, 1e-7, 1 - 1e-7),
        predicted_value=value + rng.normal(0.0, 5.0, size=(2, rows)),
    )


@pytest.mark.skipif(
    os.getenv("RANKING_GATE_BENCHMARK") != "1",
    reason="set RANKING_GATE_BENCHMARK=1 to run the ranking gate benchmark",
)
def test_cluster_bootstrap_scales_to_a_million_rows():
    """Opt-in: RANKING_GATE_BENCHMARK=1 python -m pytest -s -k scales_to_a_million.

    Sizes and bounds: RANKING_GATE_BENCHMARK_{ROWS,USERS,SAMPLES,MAX_SECONDS,
    MAX_TRACED_MB}. Traced (numpy) peak memory does not grow with SAMPLES.
    """
    rows = _env_int("RANKING_GATE_BENCHMARK_ROWS", 1_000_000)
    users = _env_int("RANKING_GATE_BENCHMARK_USERS", 100_000)
    samples = _env_int("RANKING_GATE_BENCHMARK_SAMPLES", 2000)
    max_seconds = float(os.getenv("RANKING_GATE_BENCHMARK_MAX_SECONDS", "300"))
    max_traced_mb = float(os.getenv("RANKING_GATE_BENCHMARK_MAX_TRACED_MB", "1536"))
    config = RankingQualityGateConfig(bootstrap_samples=samples)
    columns = _synthetic_columns(rows, users)
    timings: dict[str, float] = {}

    tracemalloc.start()
    started = time.perf_counter()
    statistics = _gate_statistics(columns, ece_bins=config.ece_bins)
    timings["statistics_seconds"] = time.perf_counter() - started
    point_started = time.perf_counter()
    point = _comparison_metrics(columns, statistics, ece_bins=config.ece_bins)
    timings["point_seconds"] = time.perf_counter() - point_started
    bootstrap_started = time.perf_counter()
    bootstrap = _cluster_bootstrap(statistics, config)
    timings["bootstrap_seconds"] = time.perf_counter() - bootstrap_started
    timings["total_seconds"] = time.perf_counter() - started
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    report = {
        "rows": rows,
        "users": statistics.users,
        "informative_slates": point["ndcg_informative_impressions"],
        "samples": samples,
        **{name: round(value, 2) for name, value in timings.items()},
        "traced_peak_mb": round(traced_peak / (1024 * 1024), 1),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "ndcg_ci": [
            bootstrap["ndcg_at_10_delta_ci_lower"],
            bootstrap["ndcg_at_10_delta_ci_upper"],
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    assert all(np.isfinite(value) for value in bootstrap.values())
    assert timings["total_seconds"] < max_seconds
    assert traced_peak / (1024 * 1024) < max_traced_mb
