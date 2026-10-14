"""Deterministic offline quality gates for immutable ranking releases."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
import sqlite3
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from video_commerce.ml.probability_calibration import (
    expected_calibration_error,
    reliability_table,
)


QUALITY_GATE_POLICY_V1 = "ranking_quality_gate_v1"
QUALITY_GATE_POLICY_VERSION = "ranking_quality_gate_v2"
SUPPORTED_QUALITY_GATE_POLICY_VERSIONS = frozenset(
    {QUALITY_GATE_POLICY_V1, QUALITY_GATE_POLICY_VERSION}
)


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
    value_mask: bool = True
    category: str = ""
    retrieval_sources: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RankingPrediction:
    serving_score: float
    ctr: float
    ctcvr: float
    predicted_value: float


class RankingEvaluationSpool:
    """Compact disk record; multimodal training objects never enter gate state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.unlink(missing_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            "CREATE TABLE evaluation (observation_id TEXT PRIMARY KEY, "
            "row_json TEXT NOT NULL, champion_json TEXT NOT NULL, "
            "challenger_json TEXT NOT NULL)"
        )

    def add(
        self,
        row: RankingEvaluationRow,
        champion: RankingPrediction,
        challenger: RankingPrediction,
    ) -> None:
        from dataclasses import asdict

        self.connection.execute(
            "INSERT INTO evaluation VALUES (?,?,?,?)",
            (
                row.observation_id,
                json.dumps(asdict(row), sort_keys=True, separators=(",", ":")),
                json.dumps(asdict(champion), sort_keys=True, separators=(",", ":")),
                json.dumps(asdict(challenger), sort_keys=True, separators=(",", ":")),
            ),
        )

    def load(self):
        self.connection.commit()
        rows = []
        champion = {}
        challenger = {}
        for (
            observation_id,
            raw_row,
            raw_champion,
            raw_challenger,
        ) in self.connection.execute(
            "SELECT observation_id,row_json,champion_json,challenger_json "
            "FROM evaluation ORDER BY observation_id"
        ):
            row_payload = json.loads(raw_row)
            row_payload["modality_presence"] = dict(
                row_payload.get("modality_presence") or {}
            )
            row_payload["retrieval_sources"] = tuple(
                row_payload.get("retrieval_sources") or ()
            )
            rows.append(RankingEvaluationRow(**row_payload))
            champion[str(observation_id)] = RankingPrediction(
                **json.loads(raw_champion)
            )
            challenger[str(observation_id)] = RankingPrediction(
                **json.loads(raw_challenger)
            )
        return rows, champion, challenger

    def close(self) -> None:
        self.connection.close()


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
    ece_bins: int = 15
    ece_max_absolute_regression: float = 0.005
    slice_ece_max_absolute_regression: float = 0.01
    candidate_recall_max_absolute_regression: float = 0.005
    require_candidate_recall: bool = True
    candidate_recall_at_500_champion: float | None = None
    candidate_recall_at_500_challenger: float | None = None
    experiment_kind: str = "combined"
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
    value_target_policy: str = "legacy_preference",
    value_currency: str = "USD",
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
    from video_commerce.ml.ranking_training import RankingLabelBuilder

    label_builder = RankingLabelBuilder(
        target_policy=value_target_policy, value_currency=value_currency
    )
    for example in holdout_examples:
        bundle = example.bundle
        facts = example.attribution
        labels = label_builder.build(facts)
        product_id = str(bundle.candidate.product_id)
        metadata = dict(bundle.product_metadata)
        context = dict(bundle.context)
        content = getattr(example, "multimodal_content", None)
        candidate_embeddings = dict(
            getattr(example, "candidate_embeddings", None) or {}
        )
        price = float(metadata.get("price", 0.0) or 0.0)
        relevance = labels.relevance
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
                business_value=labels.business_value,
                value_mask=bool(labels.value_mask),
                category=str(metadata.get("category") or ""),
                retrieval_sources=tuple(sorted(bundle.candidate.retrieval_signals)),
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
    if config.experiment_kind not in {"features", "fusion", "calibration", "combined"}:
        raise ValueError("unsupported Phase C experiment kind")
    invalid_reason = _prediction_validation_reason(rows, champion, challenger)
    recall_values = (
        config.candidate_recall_at_500_champion,
        config.candidate_recall_at_500_challenger,
    )
    if any(
        value is not None
        and (not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0)
        for value in recall_values
    ):
        invalid_reason = "invalid_candidate_recall"
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
    if not rows:
        return QualityGateResult(
            decision=GateDecision.INSUFFICIENT_EVIDENCE,
            reasons=tuple(evidence_reasons),
            policy_version=QUALITY_GATE_POLICY_VERSION,
            overall={},
            slices={},
            bootstrap={},
        )
    overall = _comparison_metrics(rows, champion, challenger, ece_bins=config.ece_bins)
    if (
        config.candidate_recall_at_500_champion is not None
        and config.candidate_recall_at_500_challenger is not None
    ):
        champion_recall = float(config.candidate_recall_at_500_champion)
        challenger_recall = float(config.candidate_recall_at_500_challenger)
        overall.update(
            {
                "candidate_recall_at_500_champion": champion_recall,
                "candidate_recall_at_500_challenger": challenger_recall,
                "candidate_recall_at_500_absolute_regression": (
                    champion_recall - challenger_recall
                ),
            }
        )
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
        metrics = _comparison_metrics(
            selected, champion, challenger, ece_bins=config.ece_bins
        )
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
        if config.experiment_kind == "features":
            cold = dict(slice_results.get("cold_user", {}).get("metrics") or {})
            if (
                float(overall["ndcg_at_10_delta"]) < 0.005
                and float(cold.get("ndcg_at_10_delta", float("-inf"))) < 0.01
            ):
                failures.append("feature_challenger_minimum_improvement")
        elif config.experiment_kind == "fusion":
            recall_delta = -float(
                overall.get("candidate_recall_at_500_absolute_regression", 0.0)
            )
            if recall_delta < 0.005 and float(overall["ndcg_at_10_delta"]) < 0.005:
                failures.append("fusion_challenger_minimum_improvement")
        elif config.experiment_kind == "calibration":
            for head in ("ctr", "ctcvr"):
                old = float(overall[f"{head}_ece_champion"])
                new = float(overall[f"{head}_ece_challenger"])
                if old <= 0.0 or new > old * 0.9:
                    failures.append(f"calibration_{head}_minimum_improvement")
                if float(overall[f"{head}_brier_relative_regression"]) > 0.0:
                    failures.append(f"calibration_{head}_brier_regression")
            if float(overall["ndcg_at_10_delta"]) < 0.0:
                failures.append("calibration_ndcg_regression")
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
    if sum(row.purchased and row.value_mask for row in rows) < config.min_purchases:
        reasons.append("overall_value_labels")
    if config.require_candidate_recall and (
        config.candidate_recall_at_500_champion is None
        or config.candidate_recall_at_500_challenger is None
    ):
        reasons.append("candidate_recall_at_500")
    return reasons


def _quality_slices(
    rows: Sequence[RankingEvaluationRow], config: RankingQualityGateConfig
):
    selectors = {
        "cold_user": lambda row: row.total_interactions
        <= config.cold_user_max_interactions,
        "warm_user": lambda row: row.total_interactions
        > config.cold_user_max_interactions,
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
    for name, lower, upper in (
        ("price_low", 0.0, 50.0),
        ("price_mid", 50.0, 200.0),
        ("price_high", 200.0, float("inf")),
    ):
        selected = [row for row in rows if lower <= row.price < upper]
        yield name, selected, bool(selected)
    for category in sorted({row.category for row in rows if row.category}):
        yield f"category_{category}", [
            row for row in rows if row.category == category
        ], True
    for source in sorted({source for row in rows for source in row.retrieval_sources}):
        yield f"retrieval_{source}", [
            row for row in rows if source in row.retrieval_sources
        ], True
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


def _comparison_metrics(
    rows, champion, challenger, *, ece_bins: int = 15
) -> Dict[str, float | int | None]:
    champion_metrics = _model_metrics(rows, champion, ece_bins=ece_bins)
    challenger_metrics = _model_metrics(rows, challenger, ece_bins=ece_bins)

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
        "ctr_ece_champion": champion_metrics["ctr_ece"],
        "ctr_ece_challenger": challenger_metrics["ctr_ece"],
        "ctr_ece_absolute_regression": delta("ctr_ece"),
        "ctcvr_ece_champion": champion_metrics["ctcvr_ece"],
        "ctcvr_ece_challenger": challenger_metrics["ctcvr_ece"],
        "ctcvr_ece_absolute_regression": delta("ctcvr_ece"),
        "ctr_reliability_champion": champion_metrics["ctr_reliability"],
        "ctr_reliability_challenger": challenger_metrics["ctr_reliability"],
        "ctcvr_reliability_champion": champion_metrics["ctcvr_reliability"],
        "ctcvr_reliability_challenger": challenger_metrics["ctcvr_reliability"],
        "value_wape_relative_regression": relative_regression(
            "value_wape", lower_is_better=True
        ),
        "value_mae_champion": champion_metrics["value_mae"],
        "value_mae_challenger": challenger_metrics["value_mae"],
    }


def _model_metrics(rows, predictions, *, ece_bins: int = 15):
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
    purchase_rows = [row for row in rows if row.purchased and row.value_mask]
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
        "ctr_ece": expected_calibration_error(ctr, clicked, bins=ece_bins),
        "ctcvr_ece": expected_calibration_error(ctcvr, purchased, bins=ece_bins),
        "ctr_reliability": reliability_table(ctr, clicked, bins=ece_bins),
        "ctcvr_reliability": reliability_table(ctcvr, purchased, bins=ece_bins),
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
        ranked = sorted(
            impression_rows,
            key=lambda row: (
                -predictions[row.observation_id].serving_score,
                row.product_id,
            ),
        )[:10]
        ideal = sorted(
            impression_rows, key=lambda row: (-row.relevance, row.product_id)
        )[:10]

        def dcg(selected):
            return sum(
                float(row.relevance) / math.log2(position + 2)
                for position, row in enumerate(selected)
            )

        ideal_gain = dcg(ideal)
        values.append(dcg(ranked) / ideal_gain if ideal_gain > 0.0 else 0.0)
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
    ece_limit = (
        config.slice_ece_max_absolute_regression
        if slice_name
        else config.ece_max_absolute_regression
    )
    for name in ("ctr_ece_absolute_regression", "ctcvr_ece_absolute_regression"):
        if float(metrics[name]) > ece_limit:
            failures.append(f"{prefix}_{name}")
    if (
        slice_name is None
        and "candidate_recall_at_500_absolute_regression" in metrics
        and float(metrics["candidate_recall_at_500_absolute_regression"])
        > config.candidate_recall_max_absolute_regression
    ):
        failures.append("overall_candidate_recall_at_500_absolute_regression")
    value_limit = (
        config.slice_error_max_relative_regression
        if slice_name
        else config.value_wape_max_relative_regression
    )
    if float(metrics["value_wape_relative_regression"]) > value_limit:
        failures.append(f"{prefix}_value_wape_relative_regression")
    return failures
