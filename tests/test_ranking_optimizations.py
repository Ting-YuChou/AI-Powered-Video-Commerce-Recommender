import numpy as np

from config import RankingConfig
from models import CandidateProduct
from ranking import RankingModel


def test_build_recommendations_constructs_only_top_k_in_score_order():
    ranking = RankingModel(RankingConfig())
    candidates = [
        (
            CandidateProduct(product_id=f"p{i}", combined_score=0.1, source="test"),
            {
                "title": f"Product {i}",
                "price": float(i),
                "category": "cat",
                "brand": "brand",
            },
        )
        for i in range(5)
    ]
    predictions = {
        "ctr": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "cvr": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "gmv": np.array([10, 20, 30, 40, 50]),
        "ranking_score": np.array([0.2, 0.9, 0.1, 0.8, 0.3]),
    }

    recommendations, _ = ranking.build_recommendations_from_predictions(
        candidates,
        predictions,
        k=2,
    )

    assert [item.product_id for item in recommendations] == ["p1", "p3"]
