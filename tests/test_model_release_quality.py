from __future__ import annotations

from dataclasses import replace

import pytest

from video_commerce.ml.model_release_quality import (
    GateDecision,
    RankingEvaluationRow,
    RankingPrediction,
    RankingQualityGateConfig,
    evaluate_quality_gate,
    split_fixed_holdout,
    split_training_examples_fixed_holdout,
)


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
    assert first.policy_version == "ranking_quality_gate_v1"


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
