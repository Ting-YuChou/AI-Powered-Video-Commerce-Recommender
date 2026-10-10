from dataclasses import replace

import numpy as np

from video_commerce.ml.retrieval_evaluation import (
    RETRIEVAL_QUALITY_GATE_VERSION,
    RetrievalEvaluationConfig,
    RetrievalEvaluationQuery,
    RetrievalGateDecision,
    evaluate_full_catalog,
    evaluate_retrieval_gate,
    exact_audit_ann_recall,
    score_ann_index,
    score_full_catalog_embeddings,
)


def _query(query_id, user_id, relevant, *, cold_user=False, long_tail=()):
    return RetrievalEvaluationQuery(
        query_id=query_id,
        user_id=user_id,
        relevant_product_ids=frozenset(relevant),
        eligible_product_ids=("p1", "p2", "p3", "p4"),
        seen_product_ids=frozenset(),
        cold_user=cold_user,
        cold_item_product_ids=frozenset(),
        long_tail_product_ids=frozenset(long_tail),
        missing_modality_product_ids=frozenset(),
    )


def test_full_catalog_metrics_are_hand_checkable_and_deterministic():
    queries = [
        _query("q1", "u1", {"p1"}, long_tail={"p4"}),
        _query("q2", "u2", {"p2"}, long_tail={"p4"}),
    ]
    scores = {
        "q1": {"p1": 1.0, "p2": 0.8, "p3": 0.2, "p4": 0.1},
        "q2": {"p1": 1.0, "p2": 0.9, "p3": 0.2, "p4": 0.1},
    }

    result = evaluate_full_catalog(queries, scores, cutoffs=(1, 2, 4))

    assert result["recall_at_1"] == 0.5
    assert result["recall_at_2"] == 1.0
    assert result["recall_at_4"] == 1.0
    assert result["mrr"] == 0.75
    assert result["hit_rate_at_1"] == 0.5
    assert result["recommendation_catalog_coverage"] == 1.0
    assert result["eligible_item_encoding_coverage"] == 1.0
    assert result["long_tail_exposure_share"] == 0.25


def test_ties_are_broken_by_product_id_and_seen_items_are_filtered():
    query = replace(_query("q1", "u1", {"p2"}), seen_product_ids=frozenset({"p1"}))
    scores = {"q1": {"p1": 2.0, "p3": 1.0, "p2": 1.0, "p4": 0.0}}

    result = evaluate_full_catalog([query], scores, cutoffs=(1, 2))

    assert result["recall_at_1"] == 1.0
    assert result["ranked_product_ids"]["q1"][:2] == ["p2", "p3"]


def test_embedding_scorer_covers_the_pinned_catalog_without_sampling():
    query = _query("q1", "u1", {"p1"})
    item_embeddings = {
        "p1": np.asarray([1.0, 0.0]),
        "p2": np.asarray([0.0, 1.0]),
        "p3": np.asarray([0.5, 0.5]),
        "p4": np.asarray([-1.0, 0.0]),
    }

    scores = score_full_catalog_embeddings(
        [query],
        item_embeddings,
        encode_user=lambda current: np.asarray([1.0, 0.0]),
    )

    assert set(scores["q1"]) == set(query.eligible_product_ids)
    assert scores["q1"]["p1"] == 1.0


def _gate_config():
    return RetrievalEvaluationConfig(
        min_queries=2,
        min_users=2,
        min_relevant_labels=2,
        min_slice_queries=1,
        min_slice_users=1,
        min_slice_relevant_labels=1,
        bootstrap_samples=50,
        random_seed=42,
    )


def test_retrieval_gate_passes_non_regressing_challenger_reproducibly():
    queries = [
        _query("q1", "u1", {"p1"}, cold_user=True, long_tail={"p4"}),
        _query("q2", "u2", {"p2"}, long_tail={"p4"}),
    ]
    champion = {
        "q1": {"p1": 1.0, "p2": 0.5, "p3": 0.2, "p4": 0.1},
        "q2": {"p1": 1.0, "p2": 0.9, "p3": 0.2, "p4": 0.1},
    }
    challenger = {
        "q1": {"p1": 1.0, "p2": 0.5, "p3": 0.2, "p4": 0.1},
        "q2": {"p2": 1.0, "p1": 0.9, "p3": 0.2, "p4": 0.1},
    }

    first = evaluate_retrieval_gate(queries, champion, challenger, _gate_config())
    second = evaluate_retrieval_gate(queries, champion, challenger, _gate_config())

    assert first == second
    assert first.policy_version == RETRIEVAL_QUALITY_GATE_VERSION
    assert first.decision is RetrievalGateDecision.PASSED


def test_retrieval_gate_blocks_missing_evidence_and_non_finite_scores():
    queries = [_query("q1", "u1", {"p1"})]
    scores = {"q1": {"p1": 1.0, "p2": 0.0, "p3": 0.0, "p4": 0.0}}

    insufficient = evaluate_retrieval_gate(queries, scores, scores, _gate_config())
    assert insufficient.decision is RetrievalGateDecision.INSUFFICIENT_EVIDENCE

    bad = {"q1": dict(scores["q1"], p2=np.nan)}
    failed = evaluate_retrieval_gate(
        queries,
        scores,
        bad,
        replace(_gate_config(), min_queries=1, min_users=1, min_relevant_labels=1),
    )
    assert failed.decision is RetrievalGateDecision.FAILED
    assert failed.reasons == ("non_finite_score",)


def test_exact_audit_reports_ann_top_k_overlap():
    queries = [_query("q1", "u1", {"p1"})]
    exact = {"q1": {"p1": 0.9, "p2": 0.8, "p3": 0.7}}
    ann = {"q1": {"p1": 0.9, "p3": 0.7}}

    result = exact_audit_ann_recall(queries, ann, exact, cutoffs=(2,))

    assert result["ann_recall_at_2"] == 0.5
    assert result["audited_queries"] == 1


def test_ann_scorer_uses_complete_index_membership_and_filters_invalid_ids():
    class Index:
        ntotal = 4

        def search(self, vectors, k):
            assert vectors.shape == (1, 2)
            assert k == 4
            return (
                np.asarray([[0.9, 0.8, 0.7, -1.0]], dtype=np.float32),
                np.asarray([[0, 2, 1, -1]], dtype=np.int64),
            )

    query = _query("q1", "u1", {"p1"})
    scores = score_ann_index(
        [query],
        Index(),
        {0: "p1", 1: "p2", 2: "p3", 3: "p4"},
        encode_user=lambda current: np.asarray([1.0, 0.0]),
    )

    assert list(scores["q1"]) == ["p1", "p3", "p2"]
