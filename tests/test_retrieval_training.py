import sys
import types
from dataclasses import replace

try:
    import faiss  # noqa: F401
except ModuleNotFoundError:
    fake_faiss = types.ModuleType("faiss")
    fake_faiss.Index = object
    sys.modules["faiss"] = fake_faiss

from video_commerce.ml.retrieval_pit_dataset import (
    RetrievalCatalog,
    RetrievalCatalogProduct,
    RetrievalItemFeature,
    RetrievalPitDataset,
    RetrievalPitManifest,
    RetrievalPitRow,
)
from video_commerce.ml.retrieval_training import build_retrieval_training_inputs


def _row(query_id, product_id, label_type, as_of_ts, *, source="viewed_impression"):
    return RetrievalPitRow(
        query_id=query_id,
        user_id="u1" if query_id != "q3" else "u2",
        product_id=product_id,
        label_source=source,
        label_type=label_type,
        label_weight=2.0 if label_type == "click" else 0.25,
        as_of_ts=as_of_ts,
        label_event_time=as_of_ts + 1,
        label_available_at=as_of_ts + 2,
        feature_event_time=as_of_ts - 1,
        feature_available_at=as_of_ts - 0.5,
        catalog_generation_id="catalog-1",
        user_features={
            "total_interactions": int(as_of_ts),
            "last_active": as_of_ts - 5,
        },
        seen_product_ids=("seen",),
    )


def _dataset():
    products = {
        product_id: RetrievalCatalogProduct(
            product_id=product_id,
            active=True,
            in_stock=True,
            metadata={"created_at": 1.0},
            modality_presence={"visual": product_id != "p3"},
        )
        for product_id in ("p1", "p2", "p3")
    }
    catalog = RetrievalCatalog(
        generation_id="catalog-1",
        effective_at=1.0,
        available_at=1.0,
        eligibility_policy_version="retrieval_eligibility_v1",
        products=products,
        item_features={
            "p1": RetrievalItemFeature("p1", (1.0, 0.0)),
            "p2": RetrievalItemFeature("p2", (0.0, 1.0)),
            "p3": RetrievalItemFeature("p3", None),
        },
        item_features_available_at=1.0,
        embedding_dimension=2,
        embedding_model_revision="clip-v1",
    )
    rows = (
        _row("q1", "p1", "click", 10.0),
        _row("q1", "p2", "viewed_no_positive", 10.0),
        _row("q2", "p3", "ranker_rejected", 20.0, source="ranker_rejected"),
        _row("q3", "p2", "purchase", 900_000.0),
    )
    manifest = RetrievalPitManifest(
        materialization_run_id="retrieval-pit-1",
        dataset_version="snapshot-1",
        attribution_cutoff=1_000_000.0,
        attribution_window_hours=168,
        allowed_lateness_hours=1,
        min_as_of_ts=10.0,
        max_as_of_ts=900_000.0,
        row_count=len(rows),
        label_policy_version="retrieval_label_v1",
        eligibility_policy_version="retrieval_eligibility_v1",
        catalog_generation_id="catalog-1",
        catalog_uri="catalog.json",
        catalog_sha256="a" * 64,
        manifest_uri="manifest.json",
        manifest_sha256="b" * 64,
    )
    return RetrievalPitDataset(manifest, catalog, rows)


def test_training_inputs_exclude_holdout_and_keep_negative_sources_separate():
    inputs = build_retrieval_training_inputs(_dataset(), holdout_days=7)

    assert [(row["user_id"], row["product_id"]) for row in inputs.interactions] == [
        ("u1", "p1")
    ]
    assert inputs.interactions[0]["as_of_ts"] == 10.0
    assert inputs.interactions[0]["user_features"]["total_interactions"] == 10
    assert {item["source"] for item in inputs.external_negatives} == {
        "impression_no_click",
        "ranker_rejected",
    }
    assert "p2" not in {row["product_id"] for row in inputs.interactions}
    assert inputs.product_clip_embeddings["p3"].tolist() == [0.0, 0.0]
    assert set(inputs.product_metadata) == {"p1", "p2", "p3"}


def test_holdout_queries_use_complete_catalog_and_mature_relevance():
    inputs = build_retrieval_training_inputs(_dataset(), holdout_days=7)

    assert len(inputs.holdout_queries) == 1
    query = inputs.holdout_queries[0]
    assert query.query_id == "q3"
    assert query.relevant_product_ids == frozenset({"p2"})
    assert query.eligible_product_ids == ("p1", "p2", "p3")
    assert query.seen_product_ids == frozenset({"seen"})


def test_ranker_rejected_experiment_modes_are_explicit():
    dataset = _dataset()
    disabled = build_retrieval_training_inputs(
        dataset, holdout_days=7, ranker_rejected_mode="disabled"
    )
    teacher_dataset = replace(
        dataset,
        rows=tuple(
            replace(row, ranker_score=-1.0)
            if row.label_source == "ranker_rejected"
            else row
            for row in dataset.rows
        ),
    )
    teacher = build_retrieval_training_inputs(
        teacher_dataset, holdout_days=7, ranker_rejected_mode="teacher_soft"
    )

    assert "ranker_rejected" not in {
        value["source"] for value in disabled.external_negatives
    }
    rejected = next(
        value
        for value in teacher.external_negatives
        if value["source"] == "ranker_rejected"
    )
    assert rejected["teacher_target"] < 0.5
