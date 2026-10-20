from uuid import uuid4

import pytest

try:
    from pydantic.v1 import ValidationError
except ImportError:
    from pydantic import ValidationError

from video_commerce.common.models import (
    InteractionType,
    RecommendationRequest,
    UserInteractionRequest,
    ViewedImpressionRequest,
)


@pytest.mark.parametrize("field", ["user_id", "product_id"])
@pytest.mark.parametrize("value", ["", "x" * 256])
def test_interaction_request_rejects_out_of_bounds_ids(field, value):
    payload = {
        "user_id": "u1",
        "product_id": "p1",
        "action": InteractionType.CLICK,
        field: value,
    }

    with pytest.raises(ValidationError):
        UserInteractionRequest(**payload)


def test_interaction_request_accepts_max_length_ids():
    request = UserInteractionRequest(
        user_id="u" * 255, product_id="p" * 255, action=InteractionType.VIEW
    )

    assert len(request.user_id) == 255


@pytest.mark.parametrize("field", ["user_id", "content_id"])
@pytest.mark.parametrize("value", ["", "x" * 256])
def test_recommendation_request_rejects_out_of_bounds_ids(field, value):
    payload = {"user_id": "u1", "content_id": "c1", field: value}

    with pytest.raises(ValidationError):
        RecommendationRequest(**payload)


def test_recommendation_request_content_id_stays_optional():
    assert RecommendationRequest(user_id="u1").content_id is None


def test_viewed_impression_rejects_duplicate_item_event_ids():
    event_id = uuid4()

    with pytest.raises(ValidationError, match="duplicate item event_id"):
        ViewedImpressionRequest(
            impression_id=uuid4(),
            viewed_at=1_700_000_000.0,
            items=[
                {"event_id": event_id, "product_id": "p1", "position": 1},
                {"event_id": event_id, "product_id": "p2", "position": 2},
            ],
        )
