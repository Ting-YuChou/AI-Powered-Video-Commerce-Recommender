from observability import ObservabilityManager


def test_observability_manager_exposes_app_worker_and_dependency_metrics():
    manager = ObservabilityManager()

    manager.record_request("GET", "/health", 200, 0.01)
    manager.record_kafka_produce("user-interactions", "success")
    manager.record_kafka_consume("user-interactions", "video-commerce-group", "success", 0.02)
    manager.record_database_query("select", 0.003, "success")
    manager.record_worker_message("feature-worker", "user-interactions", "success", 0.04)
    manager.record_recommendation(
        result="success",
        cache_hit=True,
        serving_path="recommendation_cache",
        candidate_count=3,
        ranked_count=3,
    )
    manager.record_interaction_ingest("click", "accepted")

    payload = manager.prometheus_payload().decode("utf-8")

    assert "video_commerce_http_requests_total" in payload
    assert "video_commerce_kafka_messages_produced_total" in payload
    assert "video_commerce_kafka_messages_consumed_total" in payload
    assert "video_commerce_database_query_duration_seconds_bucket" in payload
    assert "video_commerce_worker_messages_processed_total" in payload
    assert "video_commerce_recommendation_requests_total" in payload
    assert "video_commerce_interactions_ingested_total" in payload
