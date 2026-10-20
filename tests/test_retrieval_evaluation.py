from dataclasses import replace

import faiss
import numpy as np
import pytest

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
        metric_type = faiss.METRIC_INNER_PRODUCT

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


def _hnsw_index(embeddings, metric):
    index = faiss.IndexHNSWFlat(embeddings.shape[1], 32, metric)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 50
    index.add(embeddings)
    return index


def test_real_hnsw_l2_and_inner_product_indexes_score_higher_is_better():
    rng = np.random.default_rng(11)
    dimension, item_count = 16, 300
    embeddings = rng.normal(size=(item_count, dimension)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    product_ids = tuple(f"p{index:03d}" for index in range(item_count))
    index_map = dict(enumerate(product_ids))
    user_vectors = {}
    queries = []
    for query_index in range(40):
        vector = rng.normal(size=dimension).astype(np.float32)
        vector /= np.linalg.norm(vector)
        user_id = f"u{query_index}"
        user_vectors[user_id] = vector
        nearest = np.argsort(-(embeddings @ vector))[:3]
        queries.append(
            RetrievalEvaluationQuery(
                query_id=f"q{query_index}",
                user_id=user_id,
                relevant_product_ids=frozenset(product_ids[i] for i in nearest),
                eligible_product_ids=product_ids,
                seen_product_ids=frozenset(),
            )
        )

    def encode_user(query):
        return user_vectors[query.user_id]

    exact = score_full_catalog_embeddings(
        queries, dict(zip(product_ids, embeddings)), encode_user=encode_user
    )
    # Legacy (pre two_tower_index_v2) champion built with the default L2 metric.
    champion = score_ann_index(
        queries,
        _hnsw_index(embeddings, faiss.METRIC_L2),
        index_map,
        encode_user=encode_user,
    )
    challenger = score_ann_index(
        queries,
        _hnsw_index(embeddings, faiss.METRIC_INNER_PRODUCT),
        index_map,
        encode_user=encode_user,
    )

    for ann in (champion, challenger):
        audit = exact_audit_ann_recall(queries, ann, exact, cutoffs=(50,))
        assert audit["ann_recall_at_50"] >= 0.95
        assert evaluate_full_catalog(queries, ann, cutoffs=(50,))["recall_at_50"] == (
            pytest.approx(1.0)
        )
    result = evaluate_retrieval_gate(
        queries,
        champion,
        challenger,
        replace(
            _gate_config(),
            min_queries=40,
            min_users=40,
            min_relevant_labels=40,
            min_slice_queries=10_000,
        ),
    )
    assert result.overall["recall_at_100_delta"] == pytest.approx(0.0, abs=0.01)
    assert result.overall["mrr_delta"] == pytest.approx(0.0, abs=0.01)


def test_ann_scorer_rejects_unknown_index_metric():
    class Index:
        ntotal = 1
        metric_type = 99

    with pytest.raises(ValueError, match="metric"):
        score_ann_index(
            [_query("q1", "u1", {"p1"})],
            Index(),
            {0: "p1"},
            encode_user=lambda current: np.asarray([1.0, 0.0]),
        )


def _reference_top_k(queries, item_embeddings, user_vectors, max_results):
    """Brute-force reference: score all eligible unseen items, sort, cut."""
    scores = {}
    for query in queries:
        user = np.asarray(user_vectors[query.query_id], dtype=np.float64)
        ranked = sorted(
            (
                (float(np.dot(user, item_embeddings[product_id])), product_id)
                for product_id in query.eligible_product_ids
                if product_id in item_embeddings
                and product_id not in query.seen_product_ids
            ),
            key=lambda item: (-item[0], item[1]),
        )[:max_results]
        scores[query.query_id] = {product_id: score for score, product_id in ranked}
    return scores


def _random_queries(rng, *, products, count, eligible):
    queries = []
    for index in range(count):
        relevant = set(rng.choice(products, size=rng.integers(1, 4), replace=False))
        seen = set(rng.choice(products, size=rng.integers(0, 6), replace=False))
        queries.append(
            RetrievalEvaluationQuery(
                query_id=f"q{index:03d}",
                user_id=f"u{index % 17}",
                relevant_product_ids=frozenset(relevant),
                eligible_product_ids=eligible,
                seen_product_ids=frozenset(seen),
                long_tail_product_ids=frozenset(products[::3]),
            )
        )
    return queries


def test_chunked_exact_scorer_matches_brute_force_with_ties_and_seen(monkeypatch):
    import video_commerce.ml.retrieval_evaluation as evaluation

    # Tiny blocks force several matrix chunks; integer embeddings force ties.
    monkeypatch.setattr(evaluation, "_SCORING_BLOCK_ELEMENTS", 50)
    rng = np.random.default_rng(7)
    products = [f"p{index:03d}" for index in range(40)]
    item_embeddings = {
        product_id: rng.integers(-2, 3, size=3).astype(np.float64)
        for product_id in products
        if product_id != "p005"  # unencoded items are skipped, not scored
    }
    queries = _random_queries(
        rng, products=products, count=30, eligible=tuple(products)
    )
    user_vectors = {
        query.query_id: rng.integers(-2, 3, size=3).astype(np.float64)
        for query in queries
    }

    for max_results in (1, 7, 200):
        scores = score_full_catalog_embeddings(
            queries,
            item_embeddings,
            encode_user=lambda query: user_vectors[query.query_id],
            max_results=max_results,
        )
        expected = _reference_top_k(queries, item_embeddings, user_vectors, max_results)
        assert {key: list(value.items()) for key, value in scores.items()} == {
            key: list(value.items()) for key, value in expected.items()
        }


def test_exact_scorer_rejects_non_finite_and_mismatched_embeddings():
    query = _query("q1", "u1", {"p1"})
    items = {product_id: np.asarray([1.0, 0.0]) for product_id in ("p1", "p2")}

    with pytest.raises(ValueError, match="not finite"):
        score_full_catalog_embeddings(
            [query], items, encode_user=lambda _query: np.asarray([np.nan, 0.0])
        )
    with pytest.raises(ValueError, match="dimensions differ"):
        score_full_catalog_embeddings(
            [query], items, encode_user=lambda _query: np.asarray([1.0, 0.0, 0.0])
        )


def _reference_evaluate(queries, scores, cutoffs):
    """The pre-optimization evaluate_full_catalog, kept as an oracle."""
    recalls = {cutoff: [] for cutoff in cutoffs}
    reciprocal_ranks = []
    all_eligible, encoded, exposed = set(), set(), set()
    tail_slots = total_slots = 0
    for query in queries:
        eligible = set(query.eligible_product_ids) - set(query.seen_product_ids)
        relevant = set(query.relevant_product_ids) & eligible
        if not relevant:
            continue
        all_eligible.update(eligible)
        query_scores = scores.get(query.query_id, {})
        encoded.update(product for product in eligible if product in query_scores)
        ranked = [
            product_id
            for product_id, _ in sorted(
                (
                    (product_id, float(query_scores[product_id]))
                    for product_id in eligible
                    if product_id in query_scores
                ),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        window = ranked[: max(cutoffs)]
        exposed.update(window)
        tail_slots += sum(product in query.long_tail_product_ids for product in window)
        total_slots += len(window)
        first = next(
            (rank for rank, product in enumerate(ranked, 1) if product in relevant),
            None,
        )
        reciprocal_ranks.append(0.0 if first is None else 1.0 / first)
        for cutoff in cutoffs:
            recalls[cutoff].append(len(set(ranked[:cutoff]) & relevant) / len(relevant))
    result = {
        "queries": len(reciprocal_ranks),
        "mrr": float(np.mean(reciprocal_ranks)),
        "eligible_item_encoding_coverage": len(encoded) / len(all_eligible),
        "recommendation_catalog_coverage": len(exposed) / len(all_eligible),
        "long_tail_exposure_share": tail_slots / total_slots,
    }
    for cutoff in cutoffs:
        result[f"recall_at_{cutoff}"] = float(np.mean(recalls[cutoff]))
    return result


def test_set_cached_evaluation_matches_the_reference_implementation():
    rng = np.random.default_rng(11)
    products = [f"p{index:03d}" for index in range(30)]
    shared = tuple(products)
    queries = _random_queries(rng, products=products, count=40, eligible=shared)
    # Mix in a second, distinct eligible tuple (and an equal-but-new tuple).
    queries += [
        replace(query, query_id=f"r{index}", eligible_product_ids=tuple(products[:20]))
        for index, query in enumerate(queries[:10])
    ]
    queries += [
        replace(query, query_id=f"s{index}", eligible_product_ids=tuple(products))
        for index, query in enumerate(queries[:5])
    ]
    scores = {
        query.query_id: {
            product_id: float(rng.integers(0, 5))
            for product_id in rng.choice(products, size=12, replace=False)
        }
        for query in queries
    }

    result = evaluate_full_catalog(queries, scores, cutoffs=(3, 5, 10))
    expected = _reference_evaluate(queries, scores, (3, 5, 10))

    for key, value in expected.items():
        assert result[key] == pytest.approx(value, abs=1e-12), key
    single_tuple = evaluate_full_catalog(queries[:40], scores, cutoffs=(3, 5, 10))
    assert single_tuple["recommendation_catalog_coverage"] == pytest.approx(
        _reference_evaluate(queries[:40], scores, (3, 5, 10))[
            "recommendation_catalog_coverage"
        ]
    )


def test_vectorized_bootstrap_matches_resampling_full_evaluation():
    from video_commerce.ml.retrieval_evaluation import _bootstrap_recall_delta

    rng = np.random.default_rng(3)
    products = [f"p{index:03d}" for index in range(150)]
    queries = _random_queries(
        rng, products=products, count=60, eligible=tuple(products)
    )
    champion = {
        query.query_id: {
            product_id: float(rng.random())
            for product_id in rng.choice(products, size=110, replace=False)
        }
        for query in queries
    }
    challenger = {
        query.query_id: {
            product_id: float(rng.random())
            for product_id in rng.choice(products, size=110, replace=False)
        }
        for query in queries
    }
    config = replace(_gate_config(), bootstrap_samples=40, random_seed=5)

    by_user = {}
    for query in queries:
        by_user.setdefault(query.user_id, []).append(query)
    users = sorted(by_user)
    generator = np.random.default_rng(config.random_seed)
    deltas = []
    for sample in range(config.bootstrap_samples):
        sampled = generator.choice(users, size=len(users), replace=True)
        sampled_queries, sampled_champion, sampled_challenger = [], {}, {}
        for cluster, user_id in enumerate(sampled):
            for query in by_user[str(user_id)]:
                query_id = f"{sample}:{cluster}:{query.query_id}"
                sampled_queries.append(replace(query, query_id=query_id))
                sampled_champion[query_id] = champion[query.query_id]
                sampled_challenger[query_id] = challenger[query.query_id]
        deltas.append(
            evaluate_full_catalog(sampled_queries, sampled_challenger)["recall_at_100"]
            - evaluate_full_catalog(sampled_queries, sampled_champion)["recall_at_100"]
        )

    result = _bootstrap_recall_delta(queries, champion, challenger, config)

    assert result["recall_at_100_delta_ci_lower"] == pytest.approx(
        float(np.quantile(deltas, 0.025)), abs=1e-12
    )
    assert result["recall_at_100_delta_ci_upper"] == pytest.approx(
        float(np.quantile(deltas, 0.975)), abs=1e-12
    )


def test_recall_at_40_is_reported_but_not_a_gate_blocker():
    queries = [
        _query("q1", "u1", {"p1"}, long_tail={"p4"}),
        _query("q2", "u2", {"p2"}, long_tail={"p4"}),
    ]
    scores = {
        "q1": {"p1": 1.0, "p2": 0.5, "p3": 0.2, "p4": 0.1},
        "q2": {"p2": 1.0, "p1": 0.9, "p3": 0.2, "p4": 0.1},
    }

    metrics = evaluate_full_catalog(queries, scores, include_ranked_product_ids=False)
    gate = evaluate_retrieval_gate(queries, scores, scores, _gate_config())

    assert metrics["recall_at_40"] == 1.0
    assert "ranked_product_ids" not in metrics
    assert gate.overall["recall_at_40_delta"] == 0.0
    assert not any("recall_at_40" in reason for reason in gate.reasons)


def test_ann_scorer_over_fetches_by_seen_count_and_drops_seen():
    class Index:
        ntotal = 4
        metric_type = faiss.METRIC_INNER_PRODUCT

        def search(self, vectors, k):
            assert k == 3  # max_results=2 plus one seen item
            return (
                np.asarray([[0.9, 0.8, 0.7]], dtype=np.float32),
                np.asarray([[0, 1, 2]], dtype=np.int64),
            )

    query = replace(_query("q1", "u1", {"p1"}), seen_product_ids=frozenset({"p2"}))
    scores = score_ann_index(
        [query],
        Index(),
        {0: "p1", 1: "p2", 2: "p3", 3: "p4"},
        encode_user=lambda current: np.asarray([1.0, 0.0]),
        max_results=2,
    )

    assert list(scores["q1"]) == ["p1", "p3"]
