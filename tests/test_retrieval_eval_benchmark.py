"""Opt-in retrieval evaluation benchmark (synthetic full catalog).

Run with:
    RETRIEVAL_EVAL_BENCHMARK=1 python -m pytest -s tests/test_retrieval_eval_benchmark.py

Sizes and bounds can be overridden with RETRIEVAL_EVAL_BENCHMARK_{ITEMS,QUERIES,
DIM,MAX_SECONDS,MAX_RSS_MB}.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from video_commerce.ml.retrieval_evaluation import (
    RetrievalEvaluationConfig,
    evaluate_full_catalog,
    evaluate_retrieval_gate,
    score_full_catalog_embeddings,
)
from video_commerce.ml.retrieval_pit_dataset import (
    RetrievalHoldoutWindow,
    RetrievalPitRow,
)
from video_commerce.ml.retrieval_release_training import _popularity_scores
from video_commerce.ml.retrieval_training import (
    RetrievalTrainingInputs,
    _evaluation_queries,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RETRIEVAL_EVAL_BENCHMARK") != "1",
    reason="set RETRIEVAL_EVAL_BENCHMARK=1 to run the retrieval eval benchmark",
)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def test_full_catalog_retrieval_eval_runs_in_bounded_time_and_memory():
    items = _env_int("RETRIEVAL_EVAL_BENCHMARK_ITEMS", 100_000)
    query_count = _env_int("RETRIEVAL_EVAL_BENCHMARK_QUERIES", 10_000)
    dim = _env_int("RETRIEVAL_EVAL_BENCHMARK_DIM", 64)
    max_seconds = float(os.getenv("RETRIEVAL_EVAL_BENCHMARK_MAX_SECONDS", "600"))
    max_rss_mb = float(os.getenv("RETRIEVAL_EVAL_BENCHMARK_MAX_RSS_MB", "6144"))
    rng = np.random.default_rng(2026)
    timings: dict[str, float] = {}
    started = time.perf_counter()

    product_ids = tuple(f"p{index:06d}" for index in range(items))
    day = 86400.0
    as_of_end = 400 * day
    created = rng.uniform(0.0, as_of_end, size=items)
    metadata = {
        product_id: {
            "created_at": float(created[index]),
            "modality_presence": {
                "visual": bool(index % 7),
                "ocr": bool(index % 3),
                "clip_embedding": True,
            },
        }
        for index, product_id in enumerate(product_ids)
    }
    item_matrix = rng.standard_normal((items, dim)).astype(np.float32)
    item_embeddings = {
        product_id: item_matrix[index] for index, product_id in enumerate(product_ids)
    }
    popular = rng.zipf(1.3, size=items * 2) % items
    training_rows = [
        SimpleNamespace(product_id=product_ids[index]) for index in popular
    ]
    users = query_count // 4
    holdout_rows = []
    for index in range(query_count):
        as_of_ts = float(rng.uniform(as_of_end - 7 * day, as_of_end))
        seen = tuple(product_ids[i] for i in rng.integers(0, items, size=20))
        for product_index in rng.integers(0, items, size=2):
            holdout_rows.append(
                RetrievalPitRow(
                    query_id=f"q{index:06d}",
                    user_id=f"u{index % users}",
                    product_id=product_ids[product_index],
                    label_source="viewed_impression",
                    label_type="click",
                    label_weight=2.0,
                    as_of_ts=as_of_ts,
                    label_event_time=as_of_ts,
                    label_available_at=as_of_ts,
                    feature_event_time=as_of_ts,
                    feature_available_at=as_of_ts,
                    catalog_generation_id="bench",
                    user_features={"total_interactions": index % 9},
                    seen_product_ids=seen,
                )
            )
    timings["synthesize_s"] = time.perf_counter() - started

    phase = time.perf_counter()
    queries = _evaluation_queries(
        training_rows=training_rows,
        holdout_rows=holdout_rows,
        product_metadata=metadata,
        eligible_product_ids=product_ids,
    )
    timings["build_queries_s"] = time.perf_counter() - phase
    assert len(queries) == query_count

    user_matrix = rng.standard_normal((query_count, dim)).astype(np.float32)
    user_rows = {query.query_id: index for index, query in enumerate(queries)}
    phase = time.perf_counter()
    exact = score_full_catalog_embeddings(
        queries,
        item_embeddings,
        encode_user=lambda query: user_matrix[user_rows[query.query_id]],
    )
    timings["exact_scoring_s"] = time.perf_counter() - phase
    assert all(len(value) == 200 for value in exact.values())

    phase = time.perf_counter()
    popularity = _popularity_scores(
        RetrievalTrainingInputs(
            interactions=tuple(
                {"product_id": row.product_id} for row in training_rows[: items // 2]
            ),
            external_negatives=(),
            product_metadata=metadata,
            product_clip_embeddings={},
            user_features_map={},
            training_rows=(),
            holdout_rows=(),
            holdout_queries=queries,
            holdout_window=RetrievalHoldoutWindow(as_of_end - 7 * day, as_of_end),
            training_as_of_ts=as_of_end - 7 * day,
        )
    )
    timings["popularity_s"] = time.perf_counter() - phase

    phase = time.perf_counter()
    metrics = evaluate_full_catalog(queries, exact, include_ranked_product_ids=False)
    timings["evaluate_full_catalog_s"] = time.perf_counter() - phase
    assert metrics["queries"] == query_count

    phase = time.perf_counter()
    gate = evaluate_retrieval_gate(
        queries, popularity, exact, RetrievalEvaluationConfig()
    )
    timings["gate_with_bootstrap_s"] = time.perf_counter() - phase
    assert "recall_at_100_delta_ci_lower" in gate.overall

    timings["total_s"] = time.perf_counter() - started
    report = {
        "items": items,
        "queries": query_count,
        "dim": dim,
        "bootstrap_samples": RetrievalEvaluationConfig().bootstrap_samples,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        **{key: round(value, 2) for key, value in timings.items()},
    }
    print("retrieval-eval-benchmark " + json.dumps(report, sort_keys=True))
    assert timings["total_s"] <= max_seconds
    assert report["peak_rss_mb"] <= max_rss_mb
