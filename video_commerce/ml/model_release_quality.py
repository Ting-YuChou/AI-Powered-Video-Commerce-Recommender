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
from scipy import sparse
from video_commerce.ml.probability_calibration import reliability_table
from video_commerce.ml.ranking_training import VIEW_RELEVANCE


QUALITY_GATE_POLICY_V1 = "ranking_quality_gate_v1"
QUALITY_GATE_POLICY_V2 = "ranking_quality_gate_v2"
# v3: NDCG gains exclude view-only relevance and skip slates with no
# engagement, so all-view impressions no longer dilute the delta.
QUALITY_GATE_POLICY_V3 = "ranking_quality_gate_v3"
# v4: only pre-registered slices block the gate; category and retrieval-source
# slices are capped to the top-N by impressions plus "other" and are reported
# without blocking; blocking-slice NDCG CIs are Bonferroni-adjusted; adds the
# "retrain" experiment kind (ranking-only change, candidates unchanged).
QUALITY_GATE_POLICY_V4 = "ranking_quality_gate_v4"
# v5: NDCG is a three-way non-inferiority test on the bootstrap CI. A CI wholly
# below the margin fails, a CI straddling it is insufficient evidence
# (underpowered holdout), and the "point delta >= 0" check is dropped because
# an equal-quality challenger misses it half the time.
QUALITY_GATE_POLICY_V5 = "ranking_quality_gate_v5"
# v6: every gated metric (NDCG, PR-AUC, log loss, Brier, ECE, value WAPE) is a
# three-way test of its user-cluster bootstrap CI against its threshold, so a
# small holdout reports insufficient evidence instead of failing an
# equal-quality challenger on a noisy point estimate.
QUALITY_GATE_POLICY_VERSION = "ranking_quality_gate_v6"
SUPPORTED_QUALITY_GATE_POLICY_VERSIONS = frozenset(
    {
        QUALITY_GATE_POLICY_V1,
        QUALITY_GATE_POLICY_V2,
        QUALITY_GATE_POLICY_V3,
        QUALITY_GATE_POLICY_V4,
        QUALITY_GATE_POLICY_V5,
        QUALITY_GATE_POLICY_VERSION,
    }
)
# Policies that carry Phase C promotion evidence (prerequisites, ONNX parity,
# RRF fusion policy).
PHASE_C_QUALITY_GATE_POLICY_VERSIONS = frozenset(
    {
        QUALITY_GATE_POLICY_V2,
        QUALITY_GATE_POLICY_V3,
        QUALITY_GATE_POLICY_V4,
        QUALITY_GATE_POLICY_V5,
        QUALITY_GATE_POLICY_VERSION,
    }
)
EXPERIMENT_KINDS = frozenset({"features", "fusion", "calibration", "combined", "retrain"})
# Pre-registered slices whose missing evidence or regression blocks the gate.
# Every other slice is reported only, so a long-tail category cannot hold the
# whole release hostage and dozens of point thresholds do not multiply false
# failures.
BLOCKING_SLICES = frozenset(
    {"cold_user", "cold_item", "price_low", "price_mid", "price_high"}
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
    slice_top_n_categories: int = 10
    slice_top_n_retrieval_sources: int = 10


def default_gate_config(experiment_kind: str = "combined") -> RankingQualityGateConfig:
    """The frozen policy defaults for one experiment kind."""
    if experiment_kind not in EXPERIMENT_KINDS:
        raise ValueError("unsupported Phase C experiment kind")
    # A retrain keeps the candidate generator fixed, so there is no candidate
    # recall change to evidence.
    return RankingQualityGateConfig(
        experiment_kind=experiment_kind,
        require_candidate_recall=experiment_kind != "retrain",
    )


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
    purge_seconds: float = 0.0,
) -> tuple[list[RankingEvaluationRow], list[RankingEvaluationRow], HoldoutWindow]:
    if not rows:
        raise ValueError("ranking quality gate requires at least one row")
    if int(holdout_days) <= 0:
        raise ValueError("holdout_days must be positive")
    end_ts = max(float(row.as_of_ts) for row in rows)
    if end_ts > float(attribution_cutoff):
        raise ValueError("holdout contains labels newer than attribution cutoff")
    start_ts = end_ts - int(holdout_days) * 86400.0
    # Rows within one attribution window (plus lateness) of the holdout can be
    # labelled by the same events as holdout rows; purge them from training.
    training_end = start_ts - max(0.0, float(purge_seconds))
    training = [row for row in rows if float(row.as_of_ts) < training_end]
    holdout = [row for row in rows if start_ts <= float(row.as_of_ts) <= end_ts]
    return training, holdout, HoldoutWindow(start_ts=start_ts, end_ts=end_ts)


def split_training_examples_fixed_holdout(
    examples: Sequence[object],
    *,
    holdout_days: int,
    attribution_cutoff: float,
    purge_seconds: float = 0.0,
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
    training_end = start_ts - max(0.0, float(purge_seconds))
    training = [
        example
        for example in examples
        if float(example.bundle.as_of_ts) < training_end
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
    if config.experiment_kind not in EXPERIMENT_KINDS:
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
    columns = _gate_columns(rows, champion, challenger)
    statistics = _gate_statistics(columns, ece_bins=config.ece_bins)
    overall = _comparison_metrics(columns, statistics, ece_bins=config.ece_bins)
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
    # Overall CIs are needed even when evidence is short: a definite failure
    # takes precedence over insufficient evidence.
    bootstrap = _cluster_bootstrap(statistics, config)
    overall = {**overall, **bootstrap}
    slice_results: Dict[str, Mapping[str, object]] = {}
    insufficient_slices: list[str] = []
    slice_underpowered: list[str] = []
    failed_slices: list[str] = []
    slices = list(_quality_slices(rows, config))
    # Bonferroni across the blocking slices that are tested (not across the
    # metrics within a slice): each blocking slice's CIs use alpha / k.
    blocking_tests = max(
        1,
        sum(1 for name, _, applicable in slices if applicable and name in BLOCKING_SLICES),
    )
    for name, indices, applicable in slices:
        blocking = name in BLOCKING_SLICES
        if not applicable:
            slice_results[name] = {"status": "not_applicable", "blocking": blocking}
            continue
        selected = [rows[index] for index in indices]
        slice_evidence = _evidence_reasons(selected, config, slice_name=name)
        if slice_evidence:
            slice_results[name] = {
                "status": GateDecision.INSUFFICIENT_EVIDENCE.value,
                "reasons": slice_evidence,
                "blocking": blocking,
            }
            if blocking:
                insufficient_slices.append(name)
            continue
        slice_columns = columns.take(indices)
        slice_statistics = _gate_statistics(slice_columns, ece_bins=config.ece_bins)
        metrics = _comparison_metrics(
            slice_columns, slice_statistics, ece_bins=config.ece_bins
        )
        ci = _cluster_bootstrap(
            slice_statistics,
            config,
            alpha=0.05 / blocking_tests if blocking else 0.05,
        )
        metrics = {**metrics, **ci}
        metrics.update(_required_users_estimates(metrics, config, slice_name=name))
        failures = _metric_failures(metrics, config, slice_name=name)
        underpowered = (
            []
            if failures
            else _underpowered_reasons(metrics, config, slice_name=name)
        )
        if failures:
            status = GateDecision.FAILED.value
        elif underpowered:
            status = GateDecision.INSUFFICIENT_EVIDENCE.value
            failures = underpowered
        else:
            status = GateDecision.PASSED.value
        slice_results[name] = {
            "status": status,
            "reasons": failures,
            "metrics": metrics,
            "blocking": blocking,
        }
        if blocking and status == GateDecision.FAILED.value:
            failed_slices.append(name)
        elif blocking and underpowered:
            slice_underpowered.extend(underpowered)

    overall.update(_required_users_estimates(overall, config, slice_name=None))
    if _ndcg_underpowered(overall, config, slice_name=None):
        overall["ndcg_informative_impressions_required_estimate"] = (
            _required_informative_impressions(overall, config)
        )
    requirement_failures, requirement_underpowered = _experiment_requirements(
        overall, slice_results, config
    )
    # Definite failures (a CI wholly beyond its threshold, or a failed blocking
    # slice) take precedence over missing evidence and underpowered metrics.
    failures = (
        _metric_failures(overall, config, slice_name=None)
        + requirement_failures
        + [f"slice_{name}" for name in failed_slices]
    )
    underpowered = (
        _underpowered_reasons(overall, config, slice_name=None)
        + requirement_underpowered
        + slice_underpowered
    )
    if failures:
        reasons = tuple(failures)
        decision = GateDecision.FAILED
    elif evidence_reasons or insufficient_slices or underpowered:
        reasons = (
            tuple(evidence_reasons)
            + tuple(f"slice_{name}" for name in insufficient_slices)
            + tuple(underpowered)
        )
        decision = GateDecision.INSUFFICIENT_EVIDENCE
    else:
        reasons = ()
        decision = GateDecision.PASSED
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
    """Yield (name, row indices, applicable) for every reported slice."""

    def where(selector) -> list[int]:
        return [index for index, row in enumerate(rows) if selector(row)]

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
        selected = where(selector)
        yield name, selected, bool(selected)
    for name, lower, upper in (
        ("price_low", 0.0, 50.0),
        ("price_mid", 50.0, 200.0),
        ("price_high", 200.0, float("inf")),
    ):
        selected = where(lambda row: lower <= row.price < upper)
        yield name, selected, bool(selected)
    yield from _top_n_slices(
        rows,
        prefix="category",
        values_of=lambda row: (row.category,) if row.category else (),
        top_n=config.slice_top_n_categories,
    )
    yield from _top_n_slices(
        rows,
        prefix="retrieval",
        values_of=lambda row: tuple(row.retrieval_sources),
        top_n=config.slice_top_n_retrieval_sources,
    )
    for modality in (
        "visual",
        "ocr",
        "asr",
        "candidate_image",
        "candidate_text",
        "candidate_two_tower",
    ):
        applicable = any(modality in row.modality_presence for row in rows)
        selected = where(
            lambda row: modality in row.modality_presence
            and not row.modality_presence[modality]
        )
        yield f"missing_{modality}", selected, applicable and bool(selected)


def _top_n_slices(rows, *, prefix: str, values_of, top_n: int):
    """Top-N values by impression count, plus one "other" bucket for the rest."""
    impressions: Dict[str, set] = {}
    members: Dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        for value in {str(item) for item in values_of(row)}:
            impressions.setdefault(value, set()).add(row.impression_id)
            members.setdefault(value, []).append(index)
    ranked = sorted(impressions, key=lambda value: (-len(impressions[value]), value))
    kept = set(ranked[: max(0, int(top_n))])
    for value in sorted(kept):
        yield f"{prefix}_{value}", members[value], True
    if len(ranked) > len(kept):
        selected = [
            index
            for index, row in enumerate(rows)
            if any(str(item) not in kept for item in values_of(row))
        ]
        yield f"{prefix}_other", selected, bool(selected)


# Elements in one bootstrap work array (samples x users, or samples x PR-AUC
# segments). Peak memory stays bounded however many samples are drawn.
_BOOTSTRAP_CHUNK_ELEMENTS = 1 << 22
_PROBABILITY_HEADS = ("ctr", "ctcvr")
_CHAMPION, _CHALLENGER = 0, 1

# Per-user sum columns of _GateStatistics.dense.
_ROWS = 0
_SLATES = 1  # informative slates
_NDCG_SUM = (2, 3)
_LOG_LOSS_SUM = {"ctr": (4, 5), "ctcvr": (6, 7)}
_BRIER_SUM = {"ctr": (8, 9), "ctcvr": (10, 11)}
_VALUE_ACTUAL = 12
_VALUE_ERROR = (13, 14)
_VALUE_ROWS = 15


@dataclass(frozen=True)
class _GateColumns:
    """Columnar evaluation rows; predictions are (champion, challenger) rows."""

    user: np.ndarray  # codes in sorted user-id order
    impression: np.ndarray
    product_rank: np.ndarray  # lexicographic product-id rank: the NDCG tie break
    relevance: np.ndarray
    clicked: np.ndarray
    purchased: np.ndarray
    value_row: np.ndarray  # purchased with a usable value label
    business_value: np.ndarray
    serving_score: np.ndarray
    ctr: np.ndarray  # clipped to (0, 1) like the log loss inputs
    ctcvr: np.ndarray
    predicted_value: np.ndarray

    @property
    def users(self) -> int:
        return int(self.user.max()) + 1 if len(self.user) else 0

    def take(self, indices: np.ndarray) -> "_GateColumns":
        """A slice; user codes are recompacted in the same sorted order."""
        indices = np.asarray(indices, dtype=np.int64)
        user = np.unique(self.user[indices], return_inverse=True)[1].reshape(-1)
        return _GateColumns(
            user=user.astype(np.int64),
            impression=self.impression[indices],
            product_rank=self.product_rank[indices],
            relevance=self.relevance[indices],
            clicked=self.clicked[indices],
            purchased=self.purchased[indices],
            value_row=self.value_row[indices],
            business_value=self.business_value[indices],
            serving_score=self.serving_score[:, indices],
            ctr=self.ctr[:, indices],
            ctcvr=self.ctcvr[:, indices],
            predicted_value=self.predicted_value[:, indices],
        )


def _codes(values: Sequence[str]) -> np.ndarray:
    index = {value: code for code, value in enumerate(sorted(set(values)))}
    return np.fromiter((index[value] for value in values), np.int64, len(values))


def _gate_columns(rows, champion, challenger) -> _GateColumns:
    ids = [row.observation_id for row in rows]
    models = (champion, challenger)

    def predicted(field_name: str) -> np.ndarray:
        return np.array(
            [
                [float(getattr(model[key], field_name)) for key in ids]
                for model in models
            ],
            dtype=np.float64,
        ).reshape(2, len(ids))

    purchased = np.array([bool(row.purchased) for row in rows], dtype=bool)
    return _GateColumns(
        user=_codes([row.user_id for row in rows]),
        impression=_codes([row.impression_id for row in rows]),
        product_rank=_codes([row.product_id for row in rows]),
        relevance=np.array([float(row.relevance) for row in rows], dtype=np.float64),
        clicked=np.array([float(row.clicked) for row in rows], dtype=np.float64),
        purchased=purchased.astype(np.float64),
        value_row=purchased
        & np.array([bool(row.value_mask) for row in rows], dtype=bool),
        business_value=np.array(
            [float(row.business_value) for row in rows], dtype=np.float64
        ),
        serving_score=predicted("serving_score"),
        ctr=np.clip(predicted("ctr"), 1e-7, 1 - 1e-7),
        ctcvr=np.clip(predicted("ctcvr"), 1e-7, 1 - 1e-7),
        predicted_value=predicted("predicted_value"),
    )


def _dcg_at_10(slate, order_key, product_rank, gain, slates: int) -> np.ndarray:
    """DCG@10 per slate, ranked by descending key then product id."""
    order = np.lexsort((product_rank, -order_key, slate))
    ranked_slate = slate[order]
    starts = np.flatnonzero(np.r_[True, ranked_slate[1:] != ranked_slate[:-1]])
    position = np.arange(len(order)) - np.repeat(
        starts, np.diff(np.r_[starts, len(order)])
    )
    top = position < 10
    return np.bincount(
        ranked_slate[top],
        weights=gain[order][top] / np.log2(position[top] + 2.0),
        minlength=slates,
    )


def _slate_ndcg(columns: _GateColumns):
    """Per-slate NDCG@10 for both models, computed once.

    A slate is one user's rows of one impression. Graded gain is relevance
    above a plain view; only slates with 2+ rows and some engagement are
    informative, so all-view slates do not dilute the mean.
    """
    slate = np.unique(
        columns.impression * max(columns.users, 1) + columns.user,
        return_inverse=True,
    )[1].reshape(-1)
    slates = int(slate.max()) + 1 if len(slate) else 0
    gain = np.maximum(columns.relevance - VIEW_RELEVANCE, 0.0)
    size = np.bincount(slate, minlength=slates)
    engaged = np.bincount(slate, weights=gain > 0.0, minlength=slates) > 0.0
    informative = (size >= 2) & engaged
    slate_user = np.zeros(slates, dtype=np.int64)
    slate_user[slate] = columns.user
    ideal = _dcg_at_10(slate, columns.relevance, columns.product_rank, gain, slates)
    ndcg = np.zeros((2, slates))
    for model in (_CHAMPION, _CHALLENGER):
        dcg = _dcg_at_10(
            slate, columns.serving_score[model], columns.product_rank, gain, slates
        )
        np.divide(dcg, ideal, out=ndcg[model], where=informative)
    return slate_user, informative, ndcg


@dataclass(frozen=True)
class _RankedLabels:
    """One model head's rows cut at each score group holding a positive.

    Rows are ranked by descending score; tied scores form one threshold, as in
    sklearn. Segment k is the rows after positive group k-1 through positive
    group k, so cumulative segment weights give the weight above each
    threshold at which recall changes.
    """

    segment_rows: sparse.csr_matrix  # (segments, users) row counts
    segment_positives: sparse.csr_matrix  # (segments, users) positive counts


def _ranked_labels(scores, labels, user, users: int) -> _RankedLabels:
    order = np.argsort(-scores, kind="stable")
    ranked_scores = scores[order]
    starts = np.flatnonzero(np.r_[True, ranked_scores[1:] != ranked_scores[:-1]])
    ends = np.r_[starts[1:], len(order)]
    positions = np.flatnonzero(labels[order] > 0.0)
    positive_groups, positive_segment = np.unique(
        np.searchsorted(starts, positions, side="right") - 1, return_inverse=True
    )
    segment_ends = ends[positive_groups]
    covered = int(segment_ends[-1]) if len(segment_ends) else 0
    ranked_users = user[order]
    shape = (len(segment_ends), users)
    return _RankedLabels(
        segment_rows=sparse.csr_matrix(
            (
                np.ones(covered),
                (
                    np.searchsorted(segment_ends, np.arange(covered), side="right"),
                    ranked_users[:covered],
                ),
            ),
            shape=shape,
        ),
        segment_positives=sparse.csr_matrix(
            (
                np.ones(len(positions)),
                (positive_segment.reshape(-1), ranked_users[positions]),
            ),
            shape=shape,
        ),
    )


def _weighted_average_precision(ranked: _RankedLabels, weights) -> np.ndarray:
    """Average precision per row of user weights, as sklearn's step sum.

    AP = sum over positive score groups of recall gain * precision there.
    """
    if ranked.segment_rows.shape[0] == 0:
        return np.zeros(weights.shape[0])
    seen = np.cumsum(np.asarray(ranked.segment_rows @ weights.T), axis=0)
    gained = np.asarray(ranked.segment_positives @ weights.T)
    true_positives = np.cumsum(gained, axis=0)
    precision = np.divide(
        true_positives,
        seen,
        out=np.zeros_like(true_positives),
        where=seen > 0.0,
    )
    total = true_positives[-1]
    return np.divide(
        (gained * precision).sum(axis=0),
        total,
        out=np.zeros_like(total),
        where=total > 0.0,
    )


@dataclass(frozen=True)
class _GateStatistics:
    """Per-user sums from which every gated metric follows for any user weights.

    The champion and challenger share user weights, so each bootstrap sample
    evaluates both models on the same resampled users.
    """

    users: int
    dense: np.ndarray  # (users, 16); see the column constants above
    ece_gaps: sparse.csr_matrix  # (4 * bins, users): per-bin sum(p - y)
    ece_bins: int
    ranked: Mapping[Tuple[str, int], _RankedLabels]

    @property
    def width(self) -> int:
        """Widest per-sample work array: users, or PR-AUC segments."""
        return max(
            [self.users]
            + [ranked.segment_rows.shape[0] for ranked in self.ranked.values()]
        )


def _gate_statistics(columns: _GateColumns, *, ece_bins: int) -> _GateStatistics:
    users = columns.users
    user = columns.user

    def per_user(values) -> np.ndarray:
        return np.bincount(user, weights=values, minlength=users)

    slate_user, informative, ndcg = _slate_ndcg(columns)
    dense = np.zeros((users, 16))
    dense[:, _ROWS] = np.bincount(user, minlength=users)
    dense[:, _SLATES] = np.bincount(slate_user[informative], minlength=users)
    labels = {"ctr": columns.clicked, "ctcvr": columns.purchased}
    probabilities = {"ctr": columns.ctr, "ctcvr": columns.ctcvr}
    errors = np.abs(columns.business_value - columns.predicted_value)
    dense[:, _VALUE_ACTUAL] = per_user(columns.business_value * columns.value_row)
    dense[:, _VALUE_ROWS] = per_user(columns.value_row.astype(np.float64))
    ranked = {}
    gap_rows, gap_cols, gap_values = [], [], []
    bins = max(1, int(ece_bins))
    rows = len(user)
    for model in (_CHAMPION, _CHALLENGER):
        dense[:, _NDCG_SUM[model]] = np.bincount(
            slate_user[informative],
            weights=ndcg[model][informative],
            minlength=users,
        )
        dense[:, _VALUE_ERROR[model]] = per_user(errors[model] * columns.value_row)
        for head_index, head in enumerate(_PROBABILITY_HEADS):
            y = labels[head]
            p = probabilities[head][model]
            dense[:, _LOG_LOSS_SUM[head][model]] = per_user(
                -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
            )
            dense[:, _BRIER_SUM[head][model]] = per_user((p - y) ** 2)
            ranked[(head, model)] = _ranked_labels(p, y, user, users)
            # Equal-frequency bins are fixed on the evaluated rows (as in
            # reliability_table); resamples reweight users within them.
            order = np.argsort(p, kind="mergesort")
            used = min(bins, rows)
            sizes = np.full(used, rows // used)
            sizes[: rows % used] += 1
            row_bin = np.empty(rows, dtype=np.int64)
            row_bin[order] = np.repeat(np.arange(used), sizes)
            block = (head_index * 2 + model) * bins
            gap_rows.append(block + row_bin)
            gap_cols.append(user)
            gap_values.append(p - y)
    ece_gaps = sparse.csr_matrix(
        (
            np.concatenate(gap_values),
            (np.concatenate(gap_rows), np.concatenate(gap_cols)),
        ),
        shape=(4 * bins, users),
    )
    return _GateStatistics(
        users=users, dense=dense, ece_gaps=ece_gaps, ece_bins=bins, ranked=ranked
    )


def _model_statistics(statistics: _GateStatistics, weights) -> Dict[str, np.ndarray]:
    """Each model's metric values per row of user weights: name -> (2, samples)."""
    sums = weights @ statistics.dense
    rows = sums[:, _ROWS]
    gaps = np.abs(np.asarray(statistics.ece_gaps @ weights.T)).T
    bins = statistics.ece_bins

    def ratio(numerator, denominator) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )

    values: Dict[str, np.ndarray] = {}
    values["ndcg_at_10"] = np.stack(
        [ratio(sums[:, column], sums[:, _SLATES]) for column in _NDCG_SUM]
    )
    for head_index, head in enumerate(_PROBABILITY_HEADS):
        values[f"{head}_pr_auc"] = np.stack(
            [
                _weighted_average_precision(statistics.ranked[(head, model)], weights)
                for model in (_CHAMPION, _CHALLENGER)
            ]
        )
        values[f"{head}_log_loss"] = np.stack(
            [ratio(sums[:, column], rows) for column in _LOG_LOSS_SUM[head]]
        )
        values[f"{head}_brier"] = np.stack(
            [ratio(sums[:, column], rows) for column in _BRIER_SUM[head]]
        )
        values[f"{head}_ece"] = np.stack(
            [
                ratio(
                    gaps[
                        :,
                        (head_index * 2 + model) * bins : (head_index * 2 + model + 1)
                        * bins,
                    ].sum(axis=1),
                    rows,
                )
                for model in (_CHAMPION, _CHALLENGER)
            ]
        )
    actual = np.maximum(sums[:, _VALUE_ACTUAL], 1e-12)
    values["value_wape"] = np.stack([sums[:, column] / actual for column in _VALUE_ERROR])
    values["value_mae"] = np.stack(
        [ratio(sums[:, column], sums[:, _VALUE_ROWS]) for column in _VALUE_ERROR]
    )
    values["informative_slates"] = np.stack([sums[:, _SLATES], sums[:, _SLATES]])
    return values


def _relative_regression(values, *, lower_is_better: bool) -> np.ndarray:
    old, new = values[_CHAMPION], values[_CHALLENGER]
    denominator = np.maximum(np.abs(old), 1e-12)
    return (new - old) / denominator if lower_is_better else (old - new) / denominator


# Gated v6 metrics: (metric, reason stem, higher is better).
_GATED_METRICS = (
    ("ndcg_at_10_delta", "ndcg", True),
    ("ctr_pr_auc_delta", "ctr_pr_auc", True),
    ("ctcvr_pr_auc_delta", "ctcvr_pr_auc", True),
    ("ctr_log_loss_relative_regression", "ctr_log_loss", False),
    ("ctcvr_log_loss_relative_regression", "ctcvr_log_loss", False),
    ("ctr_brier_relative_regression", "ctr_brier", False),
    ("ctcvr_brier_relative_regression", "ctcvr_brier", False),
    ("ctr_ece_absolute_regression", "ctr_ece", False),
    ("ctcvr_ece_absolute_regression", "ctcvr_ece", False),
    ("value_wape_relative_regression", "value_wape", False),
)
# Bootstrapped for the calibration experiment's 10% ECE improvement check.
_BOOTSTRAP_METRICS = tuple(metric for metric, _, _ in _GATED_METRICS) + (
    "ctr_ece_relative_regression",
    "ctcvr_ece_relative_regression",
)


def _gated_metrics(values: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    def delta(name: str) -> np.ndarray:
        return values[name][_CHALLENGER] - values[name][_CHAMPION]

    gated = {"ndcg_at_10_delta": delta("ndcg_at_10")}
    for head in _PROBABILITY_HEADS:
        gated[f"{head}_pr_auc_delta"] = delta(f"{head}_pr_auc")
        for name in ("log_loss", "brier"):
            gated[f"{head}_{name}_relative_regression"] = _relative_regression(
                values[f"{head}_{name}"], lower_is_better=True
            )
        gated[f"{head}_ece_absolute_regression"] = delta(f"{head}_ece")
        gated[f"{head}_ece_relative_regression"] = _relative_regression(
            values[f"{head}_ece"], lower_is_better=True
        )
    gated["value_wape_relative_regression"] = _relative_regression(
        values["value_wape"], lower_is_better=True
    )
    return gated


def _comparison_metrics(
    columns: _GateColumns, statistics: _GateStatistics, *, ece_bins: int = 15
) -> Dict[str, object]:
    """Point estimates: the bootstrap statistics with every user weighted once."""
    unit = _model_statistics(statistics, np.ones((1, statistics.users)))
    gated = {name: float(value[0]) for name, value in _gated_metrics(unit).items()}
    values = {name: value[:, 0] for name, value in unit.items()}
    labels = {"ctr": columns.clicked, "ctcvr": columns.purchased}
    metrics: Dict[str, object] = {
        "impressions": int(len(np.unique(columns.impression))),
        "users": statistics.users,
        "ndcg_at_10_champion": float(values["ndcg_at_10"][_CHAMPION]),
        "ndcg_at_10_challenger": float(values["ndcg_at_10"][_CHALLENGER]),
        "ndcg_informative_impressions": int(values["informative_slates"][0]),
        # Per-head evidence: a head with no positives cannot be compared.
        "ctr_positives": int(columns.clicked.sum()),
        "ctr_negatives": int(len(columns.clicked) - columns.clicked.sum()),
        "ctcvr_positives": int(columns.purchased.sum()),
        "ctcvr_negatives": int(len(columns.purchased) - columns.purchased.sum()),
        "value_rows": int(columns.value_row.sum()),
    }
    metrics.update(gated)
    for head in _PROBABILITY_HEADS:
        probabilities = getattr(columns, head)
        for model, role in ((_CHAMPION, "champion"), (_CHALLENGER, "challenger")):
            metrics[f"{head}_ece_{role}"] = float(values[f"{head}_ece"][model])
            metrics[f"{head}_reliability_{role}"] = reliability_table(
                probabilities[model], labels[head], bins=ece_bins
            )
    metrics["value_mae_champion"] = float(values["value_mae"][_CHAMPION])
    metrics["value_mae_challenger"] = float(values["value_mae"][_CHALLENGER])
    return metrics


def _bootstrap_weights(generator, users: int, samples: int) -> np.ndarray:
    """Multinomial user multiplicities, one row per bootstrap sample.

    Each row consumes the generator exactly like v5's
    generator.choice(users, size=len(users)), so a seed reproduces the v5
    resamples.
    """
    weights = np.empty((samples, users))
    for sample in range(samples):
        weights[sample] = np.bincount(
            generator.integers(0, users, size=users), minlength=users
        )
    return weights


def _cluster_bootstrap(
    statistics: _GateStatistics,
    config: RankingQualityGateConfig,
    *,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Percentile CIs of every bootstrapped metric under user-cluster resampling."""
    if statistics.users == 0:
        return {
            "ci_alpha": alpha,
            **{
                f"{name}_ci_{bound}": 0.0
                for name in _BOOTSTRAP_METRICS
                for bound in ("lower", "upper", "half_width")
            },
        }
    samples = max(1, int(config.bootstrap_samples))
    generator = np.random.default_rng(config.random_seed)
    draws = {name: np.empty(samples) for name in _BOOTSTRAP_METRICS}
    step = max(1, _BOOTSTRAP_CHUNK_ELEMENTS // statistics.width)
    for begin in range(0, samples, step):
        weights = _bootstrap_weights(
            generator, statistics.users, min(step, samples - begin)
        )
        for name, value in _gated_metrics(
            _model_statistics(statistics, weights)
        ).items():
            draws[name][begin : begin + len(weights)] = value
    result = {"ci_alpha": alpha}
    for name, value in draws.items():
        lower = float(np.quantile(value, alpha / 2.0))
        upper = float(np.quantile(value, 1.0 - alpha / 2.0))
        result[f"{name}_ci_lower"] = lower
        result[f"{name}_ci_upper"] = upper
        result[f"{name}_ci_half_width"] = (upper - lower) / 2.0
    return result


def _cluster_bootstrap_rows(rows, champion, challenger, config, *, alpha=0.05):
    columns = _gate_columns(rows, champion, challenger)
    return _cluster_bootstrap(
        _gate_statistics(columns, ece_bins=config.ece_bins), config, alpha=alpha
    )


def _mean_ndcg(rows, predictions) -> float:
    """Mean NDCG@10 over informative slates for one model."""
    if not rows:
        return 0.0
    _, informative, ndcg = _slate_ndcg(_gate_columns(rows, predictions, predictions))
    return float(ndcg[_CHAMPION][informative].mean()) if informative.any() else 0.0


def _informative_impression_count(rows) -> int:
    if not rows:
        return 0
    blank = {row.observation_id: RankingPrediction(0.0, 0.5, 0.5, 0.0) for row in rows}
    return int(_slate_ndcg(_gate_columns(rows, blank, blank))[1].sum())


def _metric_threshold(metric: str, config, *, slice_name: str | None) -> float:
    sliced = slice_name is not None
    if metric == "ndcg_at_10_delta":
        return config.slice_ndcg_ci_lower if sliced else config.overall_ndcg_ci_lower
    if metric.endswith("_pr_auc_delta"):
        return -config.pr_auc_max_absolute_regression
    if metric.endswith("_ece_absolute_regression"):
        return (
            config.slice_ece_max_absolute_regression
            if sliced
            else config.ece_max_absolute_regression
        )
    if metric == "value_wape_relative_regression":
        return (
            config.slice_error_max_relative_regression
            if sliced
            else config.value_wape_max_relative_regression
        )
    return (
        config.slice_error_max_relative_regression
        if sliced
        else config.probability_error_max_relative_regression
    )


def _metric_has_evidence(metrics, metric: str, config, *, slice_name: str | None) -> bool:
    """Whether the rows carry any signal for the metric's head.

    Without it every resample scores both models identically (for example AP
    of 0 with no purchases), and a [0, 0] CI would pass with false certainty.
    """
    sliced = slice_name is not None
    if metric.startswith("ndcg_"):
        return int(metrics.get("ndcg_informative_impressions") or 0) > 0
    if metric.startswith("value_"):
        floor = config.min_slice_positives if sliced else config.min_purchases
        return int(metrics.get("value_rows") or 0) >= max(1, floor)
    head = "ctcvr" if metric.startswith("ctcvr_") else "ctr"
    if sliced:
        floor = config.min_slice_positives
    else:
        floor = config.min_purchases if head == "ctcvr" else config.min_clicks
    return (
        int(metrics.get(f"{head}_positives") or 0) >= max(1, floor)
        and int(metrics.get(f"{head}_negatives") or 0) > 0
    )


def _ci_decision(
    metrics, metric: str, higher_is_better: bool, threshold: float
) -> GateDecision:
    """Three-way CI test: wholly beyond the threshold fails, straddling it is
    insufficient evidence (underpowered holdout), otherwise it passes."""
    lower = float(metrics[f"{metric}_ci_lower"])
    upper = float(metrics[f"{metric}_ci_upper"])
    if higher_is_better:
        if upper < threshold:
            return GateDecision.FAILED
        if lower < threshold:
            return GateDecision.INSUFFICIENT_EVIDENCE
    else:
        if lower > threshold:
            return GateDecision.FAILED
        if upper > threshold:
            return GateDecision.INSUFFICIENT_EVIDENCE
    return GateDecision.PASSED


def _ci_decisions(metrics, config, *, slice_name: str | None):
    """(metric, decision, reason) per gated metric; reason is None on a pass."""
    prefix = f"slice_{slice_name}" if slice_name else "overall"
    for metric, stem, higher_is_better in _GATED_METRICS:
        if not _metric_has_evidence(metrics, metric, config, slice_name=slice_name):
            yield metric, GateDecision.INSUFFICIENT_EVIDENCE, f"{prefix}_{stem}_no_evidence"
            continue
        threshold = _metric_threshold(metric, config, slice_name=slice_name)
        decision = _ci_decision(metrics, metric, higher_is_better, threshold)
        if decision is GateDecision.FAILED:
            reason = (
                f"{prefix}_ndcg_regression" if stem == "ndcg" else f"{prefix}_{metric}"
            )
        elif decision is GateDecision.INSUFFICIENT_EVIDENCE:
            reason = f"{prefix}_{stem}_underpowered"
        else:
            reason = None
        yield metric, decision, reason


def _ndcg_minimum_improvement(metrics, config, *, slice_name: str | None):
    """A positive *_ndcg_min_delta is a superiority requirement on the CI."""
    minimum = (
        config.slice_ndcg_min_delta if slice_name else config.overall_ndcg_min_delta
    )
    if minimum <= 0.0:
        # Non-inferiority, already covered by the NDCG margin test.
        return GateDecision.PASSED
    return _evidenced_decision(
        metrics, "ndcg_at_10_delta", True, minimum, config, slice_name=slice_name
    )


def _evidenced_decision(
    metrics, metric, higher_is_better, threshold, config, *, slice_name
) -> GateDecision:
    if not metrics or not _metric_has_evidence(
        metrics, metric, config, slice_name=slice_name
    ):
        return GateDecision.INSUFFICIENT_EVIDENCE
    return _ci_decision(metrics, metric, higher_is_better, threshold)


def _ndcg_underpowered(metrics, config, *, slice_name: str | None) -> bool:
    """The NDCG CI straddles the non-inferiority margin."""
    return (
        _ci_decision(
            metrics,
            "ndcg_at_10_delta",
            True,
            _metric_threshold("ndcg_at_10_delta", config, slice_name=slice_name),
        )
        is GateDecision.INSUFFICIENT_EVIDENCE
    )


def _underpowered_reasons(metrics, config, *, slice_name: str | None) -> list[str]:
    """Gated metrics with no head evidence or a CI straddling the threshold."""
    prefix = f"slice_{slice_name}" if slice_name else "overall"
    reasons = [
        reason
        for _, decision, reason in _ci_decisions(metrics, config, slice_name=slice_name)
        if decision is GateDecision.INSUFFICIENT_EVIDENCE
    ]
    if (
        _ndcg_minimum_improvement(metrics, config, slice_name=slice_name)
        is GateDecision.INSUFFICIENT_EVIDENCE
    ):
        reasons.append(f"{prefix}_ndcg_minimum_improvement_underpowered")
    return reasons


def _metric_failures(metrics, config, *, slice_name: str | None) -> list[str]:
    """Definite failures: a CI wholly beyond its threshold."""
    prefix = f"slice_{slice_name}" if slice_name else "overall"
    failures = [
        reason
        for _, decision, reason in _ci_decisions(metrics, config, slice_name=slice_name)
        if decision is GateDecision.FAILED
    ]
    if (
        _ndcg_minimum_improvement(metrics, config, slice_name=slice_name)
        is GateDecision.FAILED
    ):
        failures.append(f"{prefix}_ndcg_minimum_improvement")
    # Candidate recall is external evidence with no holdout resample.
    if (
        slice_name is None
        and "candidate_recall_at_500_absolute_regression" in metrics
        and float(metrics["candidate_recall_at_500_absolute_regression"])
        > config.candidate_recall_max_absolute_regression
    ):
        failures.append("overall_candidate_recall_at_500_absolute_regression")
    return failures


def _any_arm(decisions) -> GateDecision:
    """An either-or requirement: one passing arm passes, all failing fails."""
    decisions = list(decisions)
    if GateDecision.PASSED in decisions:
        return GateDecision.PASSED
    if all(decision is GateDecision.FAILED for decision in decisions):
        return GateDecision.FAILED
    return GateDecision.INSUFFICIENT_EVIDENCE


def _experiment_requirements(overall, slice_results, config):
    """Per-kind superiority requirements, each a three-way CI decision.

    Returns (failures, underpowered) reason lists.
    """
    checks: list[tuple[str, GateDecision]] = []
    if config.experiment_kind == "features":
        cold = dict(slice_results.get("cold_user", {}).get("metrics") or {})
        checks.append(
            (
                "feature_challenger_minimum_improvement",
                _any_arm(
                    (
                        _evidenced_decision(
                            overall, "ndcg_at_10_delta", True, 0.005, config,
                            slice_name=None,
                        ),
                        _evidenced_decision(
                            cold, "ndcg_at_10_delta", True, 0.01, config,
                            slice_name="cold_user",
                        ),
                    )
                ),
            )
        )
    elif config.experiment_kind == "fusion":
        regression = overall.get("candidate_recall_at_500_absolute_regression")
        if regression is None:
            recall = GateDecision.INSUFFICIENT_EVIDENCE
        elif -float(regression) >= 0.005:
            recall = GateDecision.PASSED
        else:
            recall = GateDecision.FAILED
        checks.append(
            (
                "fusion_challenger_minimum_improvement",
                _any_arm(
                    (
                        recall,
                        _evidenced_decision(
                            overall, "ndcg_at_10_delta", True, 0.005, config,
                            slice_name=None,
                        ),
                    )
                ),
            )
        )
    elif config.experiment_kind == "calibration":
        for head in _PROBABILITY_HEADS:
            # ECE must improve by 10% and Brier must not regress.
            if float(overall[f"{head}_ece_champion"]) <= 0.0:
                improvement = GateDecision.FAILED
            else:
                improvement = _evidenced_decision(
                    overall, f"{head}_ece_relative_regression", False, -0.1, config,
                    slice_name=None,
                )
            checks.append((f"calibration_{head}_minimum_improvement", improvement))
            checks.append(
                (
                    f"calibration_{head}_brier_regression",
                    _evidenced_decision(
                        overall, f"{head}_brier_relative_regression", False, 0.0,
                        config, slice_name=None,
                    ),
                )
            )
        # Only a CI wholly below zero is a ranking regression; an
        # equal-order calibration misses "point delta >= 0" half the time.
        if float(overall["ndcg_at_10_delta_ci_upper"]) < 0.0:
            checks.append(("calibration_ndcg_regression", GateDecision.FAILED))
    failures = [name for name, decision in checks if decision is GateDecision.FAILED]
    underpowered = [
        f"{name}_underpowered"
        for name, decision in checks
        if decision is GateDecision.INSUFFICIENT_EVIDENCE
    ]
    return failures, underpowered


def _required_users_estimates(metrics, config, *, slice_name: str | None) -> dict:
    """Rough user count at which each underpowered metric would clear its
    threshold: the CI half-width shrinks with 1/sqrt(users), so
    users_required = users * (half_width / gap)^2, where gap is the distance
    from the point estimate to the threshold. None when the point estimate is
    already on the failing side."""
    users = int(metrics.get("users") or 0)
    estimates = {}
    for metric, stem, higher_is_better in _GATED_METRICS:
        threshold = _metric_threshold(metric, config, slice_name=slice_name)
        if (
            not _metric_has_evidence(metrics, metric, config, slice_name=slice_name)
            or _ci_decision(metrics, metric, higher_is_better, threshold)
            is not GateDecision.INSUFFICIENT_EVIDENCE
        ):
            continue
        point = float(metrics[metric])
        gap = point - threshold if higher_is_better else threshold - point
        half_width = float(metrics.get(f"{metric}_ci_half_width") or 0.0)
        estimates[f"{metric}_required_users_estimate"] = (
            int(math.ceil(users * (half_width / gap) ** 2))
            if gap > 0.0 and half_width > 0.0 and users > 0
            else None
        )
    return estimates


def _required_informative_impressions(metrics, config) -> int | None:
    """Rough holdout size at which the observed delta would clear the margin.

    The CI half-width shrinks with 1/sqrt(n), so n_required = n * (h / gap)^2
    where gap is the distance from the point delta to the margin.
    """
    gap = float(metrics["ndcg_at_10_delta"]) - config.overall_ndcg_ci_lower
    half_width = float(metrics.get("ndcg_at_10_delta_ci_half_width") or 0.0)
    informative = int(metrics.get("ndcg_informative_impressions") or 0)
    if gap <= 0.0 or half_width <= 0.0 or informative <= 0:
        return None
    return int(math.ceil(informative * (half_width / gap) ** 2))
