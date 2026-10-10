"""Deterministic offline quality gates for immutable ranking releases."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    ndcg_score,
)


QUALITY_GATE_POLICY_VERSION = "ranking_quality_gate_v1"


class GateDecision(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class HoldoutWindow:
    start_ts: float
    end_ts: float


@dataclass(frozen=True)
class RankingEvaluationRow:
    observation_id: str
    impression_id: str
    user_id: str
    product_id: str
    as_of_ts: float
    relevance: float
    clicked: bool
    purchased: bool
    business_value: float
    total_interactions: int
    price: float
    product_created_at: float
    pre_holdout_exposures: int
    modality_presence: Mapping[str, bool] = field(default_factory=dict)
    is_long_tail: bool = False
    is_high_price: bool = False


@dataclass(frozen=True)
class RankingPrediction:
    serving_score: float
    ctr: float
    ctcvr: float
    predicted_value: float


@dataclass(frozen=True)
class RankingQualityGateConfig:
    holdout_days: int = 7
    bootstrap_samples: int = 2000
    random_seed: int = 42
    min_impressions: int = 1000
    min_users: int = 200
    min_clicks: int = 50
    min_purchases: int = 30
    min_slice_impressions: int = 200
    min_slice_users: int = 50
    min_slice_positives: int = 20
    overall_ndcg_min_delta: float = 0.0
    overall_ndcg_ci_lower: float = -0.005
    slice_ndcg_min_delta: float = -0.02
    slice_ndcg_ci_lower: float = -0.03
    pr_auc_max_absolute_regression: float = 0.005
    probability_error_max_relative_regression: float = 0.02
    value_wape_max_relative_regression: float = 0.05
    slice_error_max_relative_regression: float = 0.05
    cold_user_max_interactions: int = 4
    cold_item_max_age_days: int = 30
    cold_item_max_exposures: int = 5


@dataclass(frozen=True)
class QualityGateResult:
    decision: GateDecision
    reasons: Tuple[str, ...]
    policy_version: str
    overall: Mapping[str, float | int | None]
    slices: Mapping[str, Mapping[str, object]]
    bootstrap: Mapping[str, float]


def split_fixed_holdout(
    rows: Sequence[RankingEvaluationRow],
    *,
    holdout_days: int,
    attribution_cutoff: float,
) -> tuple[list[RankingEvaluationRow], list[RankingEvaluationRow], HoldoutWindow]:
    if not rows:
        raise ValueError("ranking quality gate requires at least one row")
    if int(holdout_days) <= 0:
        raise ValueError("holdout_days must be positive")
    end_ts = max(float(row.as_of_ts) for row in rows)
    if end_ts > float(attribution_cutoff):
        raise ValueError("holdout contains labels newer than attribution cutoff")
    start_ts = end_ts - int(holdout_days) * 86400.0
    training = [row for row in rows if float(row.as_of_ts) < start_ts]
    holdout = [row for row in rows if start_ts <= float(row.as_of_ts) <= end_ts]
    return training, holdout, HoldoutWindow(start_ts=start_ts, end_ts=end_ts)


def split_training_examples_fixed_holdout(
    examples: Sequence[object],
    *,
    holdout_days: int,
    attribution_cutoff: float,
) -> tuple[list[object], list[object], HoldoutWindow]:
    """Split typed ranking examples without fitting on the release holdout."""
    if not examples:
        raise ValueError("ranking quality gate requires at least one example")
    if int(holdout_days) <= 0:
        raise ValueError("holdout_days must be positive")
    timestamps = [float(example.bundle.as_of_ts) for example in examples]
    end_ts = max(timestamps)
    if end_ts > float(attribution_cutoff):
        raise ValueError("holdout contains labels newer than attribution cutoff")
    start_ts = end_ts - int(holdout_days) * 86400.0
    training = [
        example for example in examples if float(example.bundle.as_of_ts) < start_ts
    ]
    holdout = [
        example
        for example in examples
        if start_ts <= float(example.bundle.as_of_ts) <= end_ts
    ]
    return training, holdout, HoldoutWindow(start_ts=start_ts, end_ts=end_ts)


def evaluation_rows_from_examples(
    training_examples: Sequence[object],
    holdout_examples: Sequence[object],
    *,
    feature_schema_version: str = "",
) -> list[RankingEvaluationRow]:
    """Build immutable evaluation rows using pre-holdout slice statistics."""
    exposures: Dict[str, int] = {}
    training_prices: list[float] = []
    for example in training_examples:
        product_id = str(example.bundle.candidate.product_id)
        exposures[product_id] = exposures.get(product_id, 0) + 1
        training_prices.append(
            float(example.bundle.product_metadata.get("price", 0.0) or 0.0)
        )
    ordered_products = sorted(
        exposures, key=lambda product: (-exposures[product], product)
    )
    head_count = (
        max(1, int(math.ceil(len(ordered_products) * 0.2))) if ordered_products else 0
    )
    head_products = set(ordered_products[:head_count])
    high_price_threshold = (
        float(np.quantile(training_prices, 0.9)) if training_prices else float("inf")
    )

    rows = []
    for example in holdout_examples:
        bundle = example.bundle
        facts = example.attribution
        product_id = str(bundle.candidate.product_id)
        metadata = dict(bundle.product_metadata)
        context = dict(bundle.context)
        content = getattr(example, "multimodal_content", None)
        candidate_embeddings = dict(
            getattr(example, "candidate_embeddings", None) or {}
        )
        price = float(metadata.get("price", 0.0) or 0.0)
        relevance = {
            "view": 1.0,
            "click": 2.0,
            "add_to_cart": 3.0,
            "purchase": 4.0 + math.log1p(float(facts.attributed_value or 0.0)),
        }[facts.attributed_action]
        audio = (
            getattr(content, "audio_features", None) if content is not None else None
        )
        modality_presence = {}
        if "trimodal" in str(feature_schema_version):
            modality_presence = {
                "visual": bool(content and content.visual_embedding),
                "ocr": bool(content and (content.ocr_tracks or content.extracted_text)),
                "asr": bool(audio and audio.audio_transcript),
                "candidate_image": bool(candidate_embeddings.get("image")),
                "candidate_text": bool(candidate_embeddings.get("text")),
                "candidate_two_tower": bool(candidate_embeddings.get("two_tower")),
            }
        rows.append(
            RankingEvaluationRow(
                observation_id=str(example.observation_id),
                impression_id=str(example.impression_id),
                user_id=str(bundle.user_features.user_id),
                product_id=product_id,
                as_of_ts=float(bundle.as_of_ts),
                relevance=relevance,
                clicked=bool(facts.attributed_click),
                purchased=bool(facts.attributed_purchase),
                business_value=float(facts.attributed_value or 0.0),
                total_interactions=int(bundle.user_features.total_interactions),
                price=price,
                product_created_at=float(
                    metadata.get("created_at", bundle.as_of_ts) or bundle.as_of_ts
                ),
                pre_holdout_exposures=int(exposures.get(product_id, 0)),
                modality_presence=modality_presence,
                is_long_tail=product_id not in head_products,
                is_high_price=price > high_price_threshold,
            )
        )
    return rows


def evaluate_quality_gate(
    rows: Sequence[RankingEvaluationRow],
    champion: Mapping[str, RankingPrediction],
    challenger: Mapping[str, RankingPrediction],
    config: RankingQualityGateConfig,
) -> QualityGateResult:
    invalid_reason = _prediction_validation_reason(rows, champion, challenger)
    if invalid_reason:
        return QualityGateResult(
            decision=GateDecision.FAILED,
            reasons=(invalid_reason,),
            policy_version=QUALITY_GATE_POLICY_VERSION,
            overall={},
            slices={},
            bootstrap={},
        )

    evidence_reasons = _evidence_reasons(rows, config, slice_name=None)
    overall = _comparison_metrics(rows, champion, challenger)
    bootstrap = _cluster_bootstrap_ndcg(rows, champion, challenger, config)
    overall = {**overall, **bootstrap}
    slice_results: Dict[str, Mapping[str, object]] = {}
    insufficient_slices: list[str] = []
    failed_slices: list[str] = []
    for name, selected, applicable in _quality_slices(rows, config):
        if not applicable:
            slice_results[name] = {"status": "not_applicable"}
            continue
        slice_evidence = _evidence_reasons(selected, config, slice_name=name)
        if slice_evidence:
            slice_results[name] = {
                "status": GateDecision.INSUFFICIENT_EVIDENCE.value,
                "reasons": slice_evidence,
            }
            insufficient_slices.append(name)
            continue
        metrics = _comparison_metrics(selected, champion, challenger)
        ci = _cluster_bootstrap_ndcg(selected, champion, challenger, config)
        metrics = {**metrics, **ci}
        failures = _metric_failures(metrics, config, slice_name=name)
        slice_results[name] = {
            "status": GateDecision.FAILED.value
            if failures
            else GateDecision.PASSED.value,
            "reasons": failures,
            "metrics": metrics,
        }
        if failures:
            failed_slices.append(name)

    if evidence_reasons or insufficient_slices:
        reasons = tuple(evidence_reasons) + tuple(
            f"slice_{name}" for name in insufficient_slices
        )
        decision = GateDecision.INSUFFICIENT_EVIDENCE
    else:
        failures = _metric_failures(overall, config, slice_name=None)
        failures.extend(f"slice_{name}" for name in failed_slices)
        reasons = tuple(failures)
        decision = GateDecision.FAILED if failures else GateDecision.PASSED
    return QualityGateResult(
        decision=decision,
        reasons=reasons,
        policy_version=QUALITY_GATE_POLICY_VERSION,
        overall=overall,
        slices=slice_results,
        bootstrap=bootstrap,
    )


def _prediction_validation_reason(
    rows: Sequence[RankingEvaluationRow],
    champion: Mapping[str, RankingPrediction],
    challenger: Mapping[str, RankingPrediction],
) -> str | None:
    expected = {row.observation_id for row in rows}
    if set(champion) != expected or set(challenger) != expected:
        return "prediction_coverage_mismatch"
    for prediction in tuple(champion.values()) + tuple(challenger.values()):
        values = (
            prediction.serving_score,
            prediction.ctr,
            prediction.ctcvr,
            prediction.predicted_value,
        )
        if not all(math.isfinite(float(value)) for value in values):
            return "non_finite_prediction"
    return None


def _evidence_reasons(
    rows: Sequence[RankingEvaluationRow],
    config: RankingQualityGateConfig,
    *,
    slice_name: str | None,
) -> list[str]:
    prefix = f"slice_{slice_name}" if slice_name else "overall"
    impressions = len({row.impression_id for row in rows})
    users = len({row.user_id for row in rows})
    positives = sum(row.clicked or row.purchased for row in rows)
    clicks = sum(row.clicked for row in rows)
    purchases = sum(row.purchased for row in rows)
    if slice_name is not None:
        reasons = []
        if impressions < config.min_slice_impressions:
            reasons.append(f"{prefix}_impressions")
        if users < config.min_slice_users:
            reasons.append(f"{prefix}_users")
        if positives < config.min_slice_positives:
            reasons.append(f"{prefix}_positives")
        return reasons
    reasons = []
    if impressions < config.min_impressions:
        reasons.append("overall_impressions")
    if users < config.min_users:
        reasons.append("overall_users")
    if clicks < config.min_clicks:
        reasons.append("overall_clicks")
    if purchases < config.min_purchases:
        reasons.append("overall_purchases")
    return reasons


def _quality_slices(
    rows: Sequence[RankingEvaluationRow], config: RankingQualityGateConfig
):
    selectors = {
        "cold_user": lambda row: row.total_interactions
        <= config.cold_user_max_interactions,
        "cold_item": lambda row: (
            (row.as_of_ts - row.product_created_at)
            <= config.cold_item_max_age_days * 86400.0
            or row.pre_holdout_exposures <= config.cold_item_max_exposures
        ),
        "long_tail": lambda row: row.is_long_tail,
        "high_price": lambda row: row.is_high_price,
    }
    for name, selector in selectors.items():
        selected = [row for row in rows if selector(row)]
        yield name, selected, bool(selected)
    for modality in (
        "visual",
        "ocr",
        "asr",
        "candidate_image",
        "candidate_text",
        "candidate_two_tower",
    ):
        applicable = any(modality in row.modality_presence for row in rows)
        selected = [
            row
            for row in rows
            if modality in row.modality_presence and not row.modality_presence[modality]
        ]
        yield f"missing_{modality}", selected, applicable and bool(selected)


def _comparison_metrics(rows, champion, challenger) -> Dict[str, float | int | None]:
    champion_metrics = _model_metrics(rows, champion)
    challenger_metrics = _model_metrics(rows, challenger)

    def delta(name: str) -> float:
        return float(challenger_metrics[name]) - float(champion_metrics[name])

    def relative_regression(name: str, *, lower_is_better: bool) -> float:
        old = float(champion_metrics[name])
        new = float(challenger_metrics[name])
        denominator = max(abs(old), 1e-12)
        return (
            (new - old) / denominator if lower_is_better else (old - new) / denominator
        )

    return {
        "impressions": len({row.impression_id for row in rows}),
        "users": len({row.user_id for row in rows}),
        "ndcg_at_10_champion": champion_metrics["ndcg_at_10"],
        "ndcg_at_10_challenger": challenger_metrics["ndcg_at_10"],
        "ndcg_at_10_delta": delta("ndcg_at_10"),
        "ctr_pr_auc_delta": delta("ctr_pr_auc"),
        "ctcvr_pr_auc_delta": delta("ctcvr_pr_auc"),
        "ctr_log_loss_relative_regression": relative_regression(
            "ctr_log_loss", lower_is_better=True
        ),
        "ctcvr_log_loss_relative_regression": relative_regression(
            "ctcvr_log_loss", lower_is_better=True
        ),
        "ctr_brier_relative_regression": relative_regression(
            "ctr_brier", lower_is_better=True
        ),
        "ctcvr_brier_relative_regression": relative_regression(
            "ctcvr_brier", lower_is_better=True
        ),
        "value_wape_relative_regression": relative_regression(
            "value_wape", lower_is_better=True
        ),
        "value_mae_champion": champion_metrics["value_mae"],
        "value_mae_challenger": challenger_metrics["value_mae"],
    }


def _model_metrics(rows, predictions):
    clicked = np.asarray([float(row.clicked) for row in rows], dtype=np.float64)
    purchased = np.asarray([float(row.purchased) for row in rows], dtype=np.float64)
    ctr = np.clip(
        np.asarray([predictions[row.observation_id].ctr for row in rows]),
        1e-7,
        1 - 1e-7,
    )
    ctcvr = np.clip(
        np.asarray([predictions[row.observation_id].ctcvr for row in rows]),
        1e-7,
        1 - 1e-7,
    )
    purchase_rows = [row for row in rows if row.purchased]
    actual_values = np.asarray(
        [row.business_value for row in purchase_rows], dtype=np.float64
    )
    predicted_values = np.asarray(
        [predictions[row.observation_id].predicted_value for row in purchase_rows],
        dtype=np.float64,
    )
    value_errors = np.abs(actual_values - predicted_values)
    return {
        "ndcg_at_10": _mean_ndcg(rows, predictions),
        "ctr_pr_auc": float(average_precision_score(clicked, ctr)),
        "ctcvr_pr_auc": float(average_precision_score(purchased, ctcvr)),
        "ctr_log_loss": float(log_loss(clicked, ctr, labels=[0.0, 1.0])),
        "ctcvr_log_loss": float(log_loss(purchased, ctcvr, labels=[0.0, 1.0])),
        "ctr_brier": float(brier_score_loss(clicked, ctr)),
        "ctcvr_brier": float(brier_score_loss(purchased, ctcvr)),
        "value_wape": float(value_errors.sum() / max(actual_values.sum(), 1e-12)),
        "value_mae": float(value_errors.mean()) if len(value_errors) else 0.0,
    }


def _mean_ndcg(rows, predictions) -> float:
    grouped: Dict[str, list[RankingEvaluationRow]] = {}
    for row in rows:
        grouped.setdefault(row.impression_id, []).append(row)
    values = []
    for impression_rows in grouped.values():
        if len(impression_rows) < 2:
            continue
        relevance = [row.relevance for row in impression_rows]
        scores = [
            predictions[row.observation_id].serving_score for row in impression_rows
        ]
        values.append(float(ndcg_score([relevance], [scores], k=10)))
    return float(np.mean(values)) if values else 0.0


def _cluster_bootstrap_ndcg(rows, champion, challenger, config):
    rows_by_user: Dict[str, list[RankingEvaluationRow]] = {}
    for row in rows:
        rows_by_user.setdefault(row.user_id, []).append(row)
    users = sorted(rows_by_user)
    if not users:
        return {"ndcg_at_10_delta_ci_lower": 0.0, "ndcg_at_10_delta_ci_upper": 0.0}
    generator = np.random.default_rng(config.random_seed)
    deltas = []
    for _ in range(max(1, int(config.bootstrap_samples))):
        sampled_users = generator.choice(users, size=len(users), replace=True)
        sampled_rows = []
        for sample_index, user in enumerate(sampled_users):
            for row in rows_by_user[str(user)]:
                sampled_rows.append(
                    RankingEvaluationRow(
                        **{
                            **row.__dict__,
                            "impression_id": f"{sample_index}:{row.impression_id}",
                        }
                    )
                )
        deltas.append(
            _mean_ndcg(sampled_rows, challenger) - _mean_ndcg(sampled_rows, champion)
        )
    return {
        "ndcg_at_10_delta_ci_lower": float(np.quantile(deltas, 0.025)),
        "ndcg_at_10_delta_ci_upper": float(np.quantile(deltas, 0.975)),
    }


def _metric_failures(metrics, config, *, slice_name: str | None) -> list[str]:
    prefix = f"slice_{slice_name}" if slice_name else "overall"
    ndcg_min = (
        config.slice_ndcg_min_delta if slice_name else config.overall_ndcg_min_delta
    )
    ci_min = config.slice_ndcg_ci_lower if slice_name else config.overall_ndcg_ci_lower
    error_limit = (
        config.slice_error_max_relative_regression
        if slice_name
        else config.probability_error_max_relative_regression
    )
    failures = []
    if float(metrics["ndcg_at_10_delta"]) < ndcg_min:
        failures.append(f"{prefix}_ndcg_point_delta")
    if float(metrics["ndcg_at_10_delta_ci_lower"]) < ci_min:
        failures.append(f"{prefix}_ndcg_ci_lower")
    for name in ("ctr_pr_auc_delta", "ctcvr_pr_auc_delta"):
        if float(metrics[name]) < -config.pr_auc_max_absolute_regression:
            failures.append(f"{prefix}_{name}")
    for name in (
        "ctr_log_loss_relative_regression",
        "ctcvr_log_loss_relative_regression",
        "ctr_brier_relative_regression",
        "ctcvr_brier_relative_regression",
    ):
        if float(metrics[name]) > error_limit:
            failures.append(f"{prefix}_{name}")
    value_limit = (
        config.slice_error_max_relative_regression
        if slice_name
        else config.value_wape_max_relative_regression
    )
    if float(metrics["value_wape_relative_regression"]) > value_limit:
        failures.append(f"{prefix}_value_wape_relative_regression")
    return failures
