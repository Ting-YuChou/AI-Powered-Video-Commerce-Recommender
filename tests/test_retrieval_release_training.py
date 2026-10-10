import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from video_commerce.ml.retrieval_evaluation import (
    RetrievalEvaluationConfig,
    RetrievalEvaluationQuery,
    RetrievalGateDecision,
)
from video_commerce.ml.retrieval_pit_dataset import RetrievalHoldoutWindow
from video_commerce.ml.retrieval_release_training import RetrievalReleaseTrainingRunner
from video_commerce.ml.retrieval_training import RetrievalTrainingInputs


class FakeTrainer:
    def __init__(self):
        self.prepared = None

    def prepare(
        self,
        interactions,
        product_metadata,
        product_clip_embeddings,
        user_features_map,
        external_negatives=None,
        as_of_ts=None,
    ):
        self.prepared = {
            "interactions": interactions,
            "external_negatives": external_negatives,
            "as_of_ts": as_of_ts,
        }

    def train(self):
        return {"epoch_losses": [1.0], "num_samples": 1}

    def encode_user(self, user_id, user_features, current_time=None):
        return np.asarray([0.0, 1.0], dtype=np.float32)


class FakeArtifactManager:
    def __init__(self):
        self.payload = None

    async def persist_two_tower_artifacts(self, **kwargs):
        self.payload = kwargs["payload"]
        return SimpleNamespace(
            model_version=kwargs["model_version"],
            payload={**kwargs["payload"], "model_release_id": "release-1"},
        )


class FakeStore:
    def __init__(self):
        self.evaluation = None
        self.staged = None

    async def record_model_release_evaluation(self, **kwargs):
        self.evaluation = kwargs
        return {"evaluation_id": "evaluation-1"}

    async def stage_model_release(self, **kwargs):
        self.staged = kwargs
        return True


def test_pit_training_registers_evaluates_and_stages_only_after_gate_pass(tmp_path):
    query = RetrievalEvaluationQuery(
        query_id="q1",
        user_id="u1",
        relevant_product_ids=frozenset({"p2"}),
        eligible_product_ids=("p1", "p2"),
        seen_product_ids=frozenset(),
        as_of_ts=20.0,
        user_features={"total_interactions": 1},
    )
    inputs = RetrievalTrainingInputs(
        interactions=({"user_id": "u1", "product_id": "p1", "action": "click"},),
        external_negatives=(),
        product_metadata={"p1": {}, "p2": {}},
        product_clip_embeddings={
            "p1": np.asarray([1.0, 0.0]),
            "p2": np.asarray([0.0, 1.0]),
        },
        user_features_map={"u1": {"total_interactions": 1}},
        training_rows=(),
        holdout_rows=(),
        holdout_queries=(query,),
        holdout_window=RetrievalHoldoutWindow(10.0, 20.0),
        training_as_of_ts=9.0,
    )
    manager = FakeArtifactManager()
    store = FakeStore()

    def write_bundle(trainer, output_dir, model_version):
        paths = {}
        for name in ("checkpoint", "index", "metadata"):
            path = Path(output_dir) / name
            path.write_bytes(name.encode())
            paths[f"{name}_path"] = str(path)
        return {
            **paths,
            "item_embeddings": {
                "p1": np.asarray([1.0, 0.0]),
                "p2": np.asarray([0.0, 1.0]),
            },
        }

    runner = RetrievalReleaseTrainingRunner(
        recommendation_config=SimpleNamespace(
            retrieval_release_environment="production"
        ),
        artifact_manager=manager,
        system_store=store,
        trainer_factory=FakeTrainer,
        bundle_writer=write_bundle,
        gate_config=RetrievalEvaluationConfig(
            min_queries=1,
            min_users=1,
            min_relevant_labels=1,
            min_slice_queries=1,
            min_slice_users=1,
            min_slice_relevant_labels=1,
            bootstrap_samples=10,
        ),
    )
    lineage = {
        "manifest_uri": "manifest.json",
        "retrieval_pit_manifest_sha256": "a" * 64,
        "catalog_manifest_sha256": "b" * 64,
        "eligibility_policy_version": "retrieval_eligibility_v1",
        "label_policy_version": "retrieval_label_v1",
        "quality_gate_policy_version": "retrieval_quality_gate_v1",
        "embedding_dimension": 2,
    }

    outcome = asyncio.run(
        runner.run_inputs(
            inputs,
            model_version="tt-1",
            output_dir=str(tmp_path),
            lineage=lineage,
            champion_scores={"q1": {"p1": 1.0, "p2": 0.0}},
            champion_release_id="champion-1",
        )
    )

    assert outcome.gate.decision is RetrievalGateDecision.PASSED
    assert manager.payload["retrieval_pit_manifest_sha256"] == "a" * 64
    assert store.evaluation["decision"] == "passed"
    assert store.staged["release_id"] == "release-1"
