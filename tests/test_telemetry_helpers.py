from telemetry import inject_trace_headers, kafka_consumer_span


def test_telemetry_helpers_are_noop_without_required_context():
    headers = inject_trace_headers([("x-request-id", b"req-1")])

    assert ("x-request-id", b"req-1") in headers

    with kafka_consumer_span(topic="user-interactions", group_id="group", headers=headers):
        pass
