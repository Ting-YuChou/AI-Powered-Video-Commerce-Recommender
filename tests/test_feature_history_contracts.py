import hashlib
import json
from pathlib import Path

import pytest

from video_commerce.common.feature_history_contracts import (
    FEATURE_HISTORY_PAYLOAD_SCHEMA_VERSION,
    FeatureHistoryContractError,
    build_catalog_feature_event,
    build_feature_history_event,
    canonical_json,
    parse_feature_history_event,
    payload_sha256,
)
from video_commerce.ml.ranking_features import RANKING_LTR_FEATURE_DEFINITION_VERSION


def test_canonical_payload_hash_is_independent_of_mapping_order():
    first = {"price": 9.0, "nested": {"active": True, "tags": ["a", "b"]}}
    second = {"nested": {"tags": ["a", "b"], "active": True}, "price": 9.0}

    assert canonical_json(first) == canonical_json(second)
    assert payload_sha256(first) == payload_sha256(second)


def test_canonical_payload_uses_plain_decimal_for_cross_language_floats():
    assert canonical_json(
        {"epoch": 1783037653.5775127, "tiny": 1e-7, "whole": 42.0}
    ) == (
        '{"epoch":1783037653.5775127,"tiny":0.0000001,"whole":42}'
    )


def test_catalog_feature_event_has_deterministic_identity_and_lineage():
    event = build_catalog_feature_event(
        product_id="product-1",
        source_version="catalog-v42",
        event_time=100.0,
        available_at=105.0,
        payload={"price": 9.0, "active": True},
        request_id="request-1",
    )

    expected_event_id = hashlib.sha256(b"product-1\x00catalog-v42").hexdigest()
    assert event["event_id"] == expected_event_id
    assert event["source_event_id"] == expected_event_id
    assert event["event_type"] == "catalog_feature"
    assert event["payload_schema_version"] == FEATURE_HISTORY_PAYLOAD_SCHEMA_VERSION
    assert event["feature_definition_version"] == RANKING_LTR_FEATURE_DEFINITION_VERSION
    assert event["payload_hash"] == payload_sha256(event["payload"])


def test_history_contract_is_additive_but_rejects_missing_lineage():
    event = build_feature_history_event(
        event_type="feature_snapshot",
        entity_type="user",
        entity_id="user-1",
        event_time=100.0,
        available_at=101.0,
        source_event_id="source-1",
        source_version="flink-checkpoint-42",
        payload={"total_interactions": 3},
    )
    event["future_optional_field"] = {"safe": True}

    parsed = parse_feature_history_event(event)

    assert parsed.entity_id == "user-1"
    assert parsed.payload == {"total_interactions": 3}

    invalid = dict(event)
    invalid.pop("available_at")
    with pytest.raises(FeatureHistoryContractError, match="available_at"):
        parse_feature_history_event(invalid)


@pytest.mark.parametrize("field", ["entity_type", "entity_id", "source_event_id"])
def test_history_contract_rejects_blank_identifiers(field):
    kwargs = {
        "event_type": "feature_snapshot",
        "entity_type": "user",
        "entity_id": "user-1",
        "event_time": 100.0,
        "available_at": 101.0,
        "source_event_id": "source-1",
        "source_version": "v1",
        "payload": {},
    }
    kwargs[field] = " "

    with pytest.raises(FeatureHistoryContractError, match=field):
        build_feature_history_event(**kwargs)


def test_python_contract_parses_shared_java_fixture():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/feature_history_contract_v1.json").read_text(
            encoding="utf-8"
        )
    )

    record = parse_feature_history_event(fixture["event"])

    assert canonical_json(record.payload) == fixture["canonical_payload_json"]
    assert payload_sha256(record.payload) == fixture["event"]["payload_hash"]
