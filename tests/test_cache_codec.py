from cache_codec import CacheDecodeError, json_dumps, json_loads, pack_cache_payload, unpack_cache_payload


def test_msgpack_cache_round_trip():
    payload = {"user_id": "u1", "scores": [1, 2, 3]}

    encoded = pack_cache_payload("candidate_cache", payload)
    decoded = unpack_cache_payload(encoded, "candidate_cache")

    assert decoded == payload


def test_cache_kind_mismatch_raises():
    encoded = pack_cache_payload("recommendations_cache", {"ok": True})

    try:
        unpack_cache_payload(encoded, "candidate_cache")
    except CacheDecodeError as exc:
        assert "Unexpected cache payload kind" in str(exc)
    else:
        raise AssertionError("Expected CacheDecodeError")


def test_json_round_trip():
    payload = {"status": "pending", "metadata": {"filename": "video.mp4"}}

    encoded = json_dumps(payload)

    assert json_loads(encoded) == payload
