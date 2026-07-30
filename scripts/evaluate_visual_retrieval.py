"""Compare visual pooling variants against one fixed real product CLIP bundle."""

from __future__ import annotations

import argparse
import json

import numpy as np

from video_commerce.ml.visual_product_index import load_visual_product_index_bundle
from video_commerce.ml.visual_retrieval import evaluate_visual_retrieval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-index-manifest", required=True)
    parser.add_argument(
        "--queries",
        required=True,
        help="NPZ with mean, ranker_weights, attention and relevant_product_ids",
    )
    args = parser.parse_args()

    bundle = load_visual_product_index_bundle(args.product_index_manifest)
    product_ids = [
        bundle.product_index_map[index]
        for index in range(len(bundle.product_index_map))
    ]
    product_embeddings = np.vstack(
        [bundle.product_embeddings[product_id] for product_id in product_ids]
    )
    with np.load(args.queries, allow_pickle=False) as payload:
        relevant = [
            set(str(value).split(","))
            for value in payload["relevant_product_ids"].tolist()
        ]
        results = {}
        for name in ("mean", "ranker_weights", "attention"):
            metrics = evaluate_visual_retrieval(
                payload[name],
                relevant_product_ids=relevant,
                product_embeddings=product_embeddings,
                product_ids=product_ids,
                expected_catalog_size=int(bundle.manifest["expected_product_count"]),
            )
            results[name] = {
                "recall_at": metrics.recall_at,
                "mrr": metrics.mrr,
                "catalog_coverage": metrics.catalog_coverage,
            }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
