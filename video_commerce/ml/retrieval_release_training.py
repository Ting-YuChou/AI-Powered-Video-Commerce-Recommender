"""Offline Two-Tower training, evaluation, and governed release registration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Optional

import numpy as np

from video_commerce.ml.retrieval_evaluation import (
    RetrievalEvaluationConfig,
    RetrievalGateDecision,
    RetrievalQualityGateResult,
    evaluate_full_catalog,
    evaluate_retrieval_gate,
    exact_audit_ann_recall,
    score_ann_index,
    score_full_catalog_embeddings,
)
from video_commerce.ml.retrieval_training import RetrievalTrainingInputs


@dataclass(frozen=True)
class RetrievalReleaseTrainingOutcome:
    model_version: str
    release_id: str
    training_stats: Mapping[str, Any]
    popularity_metrics: Mapping[str, Any]
    challenger_metrics: Mapping[str, Any]
    ann_audit: Mapping[str, Any]
    gate: RetrievalQualityGateResult
    staged: bool


class RetrievalReleaseTrainingRunner:
    def __init__(
        self,
        *,
        recommendation_config: Any,
        artifact_manager: Any,
        system_store: Any,
        trainer_factory: Optional[Callable[[], Any]] = None,
        bundle_writer: Optional[Callable[[Any, str, str], Mapping[str, Any]]] = None,
        gate_config: Optional[RetrievalEvaluationConfig] = None,
    ) -> None:
        self.config = recommendation_config
        self.artifact_manager = artifact_manager
        self.system_store = system_store
        self.trainer_factory = trainer_factory or self._default_trainer
        self.bundle_writer = bundle_writer or write_two_tower_bundle
        self.gate_config = gate_config or _gate_config_from_recommendation(
            recommendation_config
        )

    async def run_inputs(
        self,
        inputs: RetrievalTrainingInputs,
        *,
        model_version: str,
        output_dir: str,
        lineage: Mapping[str, Any],
        champion_scores: Optional[Mapping[str, Mapping[str, float]]] = None,
        champion_release_id: Optional[str] = None,
    ) -> RetrievalReleaseTrainingOutcome:
        if not inputs.interactions:
            raise ValueError("retrieval PIT training has no positive interactions")
        seed = int(self.gate_config.random_seed)
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:
            pass
        trainer = self.trainer_factory()
        trainer.prepare(
            list(inputs.interactions),
            dict(inputs.product_metadata),
            dict(inputs.product_clip_embeddings),
            dict(inputs.user_features_map),
            external_negatives=list(inputs.external_negatives),
            as_of_ts=float(inputs.training_as_of_ts),
        )
        training_stats = dict(trainer.train())
        bundle = dict(self.bundle_writer(trainer, output_dir, model_version))
        payload = {
            **dict(lineage),
            "training_stats": training_stats,
            "holdout_start": inputs.holdout_window.start_ts,
            "holdout_end": inputs.holdout_window.end_ts,
        }
        record = await self.artifact_manager.persist_two_tower_artifacts(
            checkpoint_path=str(bundle["checkpoint_path"]),
            index_path=str(bundle["index_path"]),
            metadata_path=str(bundle["metadata_path"]),
            embedding_sidecar_path=(
                str(bundle["embedding_sidecar_path"])
                if bundle.get("embedding_sidecar_path")
                else None
            ),
            model_version=str(model_version),
            catalog_metadata={
                product_id: dict(metadata)
                for product_id, metadata in inputs.product_metadata.items()
            },
            payload=payload,
        )
        if record is None or not record.payload.get("model_release_id"):
            raise RuntimeError("retrieval artifact release was not registered")

        queries = inputs.holdout_queries
        exact_challenger_scores = score_full_catalog_embeddings(
            queries,
            bundle["item_embeddings"],
            encode_user=lambda query: trainer.encode_user(
                query.user_id,
                dict(query.user_features),
                current_time=query.as_of_ts,
            ),
        )
        if bundle.get("ann_index") is not None and bundle.get("index_map") is not None:
            challenger_scores = score_ann_index(
                queries,
                bundle["ann_index"],
                bundle["index_map"],
                encode_user=lambda query: trainer.encode_user(
                    query.user_id,
                    dict(query.user_features),
                    current_time=query.as_of_ts,
                ),
            )
        else:
            challenger_scores = exact_challenger_scores
        audit_queries = tuple(sorted(queries, key=lambda query: query.query_id)[:200])
        ann_audit = exact_audit_ann_recall(
            audit_queries,
            challenger_scores,
            exact_challenger_scores,
        )
        popularity_scores = _popularity_scores(inputs)
        comparison_scores = champion_scores or popularity_scores
        popularity_metrics = evaluate_full_catalog(queries, popularity_scores)
        champion_metrics = evaluate_full_catalog(queries, comparison_scores)
        challenger_metrics = evaluate_full_catalog(queries, challenger_scores)
        gate = evaluate_retrieval_gate(
            queries,
            comparison_scores,
            challenger_scores,
            self.gate_config,
        )
        await self.system_store.record_model_release_evaluation(
            release_id=str(record.payload["model_release_id"]),
            champion_release_id=champion_release_id,
            dataset_manifest_uri=str(lineage["manifest_uri"]),
            dataset_manifest_sha256=str(lineage["retrieval_pit_manifest_sha256"]),
            holdout_start=float(inputs.holdout_window.start_ts),
            holdout_end=float(inputs.holdout_window.end_ts),
            policy_version=str(lineage["quality_gate_policy_version"]),
            decision=gate.decision.value,
            metrics={
                "gate": dict(gate.overall),
                "popularity": popularity_metrics,
                "champion": champion_metrics,
                "challenger": challenger_metrics,
                "ann_exact_audit": ann_audit,
            },
            slice_metrics={key: dict(value) for key, value in gate.slices.items()},
            bootstrap=dict(gate.bootstrap),
            gate_config=asdict(self.gate_config),
        )
        staged = False
        if gate.decision is RetrievalGateDecision.PASSED:
            staged = bool(
                await self.system_store.stage_model_release(
                    release_id=str(record.payload["model_release_id"]),
                    actor="retrieval-quality-evaluator",
                    environment=str(self.config.retrieval_release_environment),
                )
            )
            if not staged:
                raise RuntimeError(
                    "validated retrieval release could not enter staging"
                )
        return RetrievalReleaseTrainingOutcome(
            model_version=str(model_version),
            release_id=str(record.payload["model_release_id"]),
            training_stats=training_stats,
            popularity_metrics=popularity_metrics,
            challenger_metrics=challenger_metrics,
            ann_audit=ann_audit,
            gate=gate,
            staged=staged,
        )

    def _default_trainer(self):
        from video_commerce.ml.two_tower import TwoTowerTrainer

        return TwoTowerTrainer(
            output_dim=self.config.tt_embedding_dim,
            temperature=self.config.tt_temperature,
            learning_rate=self.config.tt_learning_rate,
            batch_size=self.config.tt_batch_size,
            epochs=self.config.tt_epochs,
            num_hard_negatives=self.config.tt_num_hard_negatives,
            num_random_negatives=self.config.tt_num_random_negatives,
            hard_ratio_start=self.config.tt_hard_negative_ratio_start,
            hard_ratio_end=self.config.tt_hard_negative_ratio_end,
            hard_ratio_cap=self.config.tt_hard_negative_ratio_cap,
            impression_negative_ratio=self.config.tt_impression_negative_ratio,
            ranker_rejected_negative_ratio=self.config.tt_ranker_rejected_negative_ratio,
            hard_negative_weight=self.config.tt_hard_negative_weight,
            impression_negative_weight=self.config.tt_impression_negative_weight,
            ranker_rejected_negative_weight=self.config.tt_ranker_rejected_negative_weight,
            random_negative_weight=self.config.tt_random_negative_weight,
            enable_in_batch_negatives=self.config.tt_enable_in_batch_negatives,
            in_batch_loss_weight=self.config.tt_in_batch_loss_weight,
            enable_logq_correction=self.config.tt_logq_correction_enabled,
            user_hidden_dims=self.config.tt_user_hidden_dims,
            item_hidden_dims=self.config.tt_item_hidden_dims,
            architecture=self.config.tt_architecture,
            cross_layers=self.config.tt_cross_layers,
        )


def write_two_tower_bundle(
    trainer: Any, output_dir: str, model_version: str
) -> Mapping[str, Any]:
    import faiss
    from video_commerce.ml.cf_cold_start import save_item_embedding_sidecar

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target / f"{model_version}.pt"
    index_path = target / f"{model_version}.faiss"
    metadata_path = target / f"{model_version}.cf_meta.json"
    embedding_sidecar_path = target / f"{model_version}.cf_embeddings.npz"
    trainer.save_checkpoint(str(checkpoint_path))
    index, raw_index_map = trainer.build_item_index()
    faiss.write_index(index, str(index_path))
    product_index_map = {
        int(faiss_index): trainer.reverse_item_mapping[item_index]
        for faiss_index, item_index in raw_index_map.items()
    }
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "two_tower_index_v1",
                "model_version": str(model_version),
                "index_map": product_index_map,
                "embedding_dimension": trainer.output_dim,
                "index_type": type(index).__name__,
                "product_count": len(product_index_map),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    item_embeddings = trainer.get_item_embedding_map()
    save_item_embedding_sidecar(
        str(embedding_sidecar_path),
        embedding_map=item_embeddings,
        clip_available=trainer.get_item_clip_available_map(),
        item_features=trainer.get_item_side_feature_map(),
        model_version=str(model_version),
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "embedding_sidecar_path": str(embedding_sidecar_path),
        "item_embeddings": item_embeddings,
        "ann_index": index,
        "index_map": product_index_map,
    }


def _popularity_scores(inputs: RetrievalTrainingInputs):
    counts: dict[str, float] = {
        product_id: 0.0 for product_id in inputs.product_metadata
    }
    for interaction in inputs.interactions:
        product_id = str(interaction["product_id"])
        counts[product_id] = counts.get(product_id, 0.0) + 1.0
    return {
        query.query_id: {
            product_id: float(counts.get(product_id, 0.0))
            for product_id in query.eligible_product_ids
        }
        for query in inputs.holdout_queries
    }


def _gate_config_from_recommendation(config: Any) -> RetrievalEvaluationConfig:
    return RetrievalEvaluationConfig(
        min_queries=int(config.retrieval_gate_min_queries),
        min_users=int(config.retrieval_gate_min_users),
        min_relevant_labels=int(config.retrieval_gate_min_relevant_labels),
        min_slice_queries=int(config.retrieval_gate_min_slice_queries),
        min_slice_users=int(config.retrieval_gate_min_slice_users),
        min_slice_relevant_labels=int(config.retrieval_gate_min_slice_relevant_labels),
        bootstrap_samples=int(config.retrieval_gate_bootstrap_samples),
        random_seed=int(config.retrieval_gate_random_seed),
    )
