from video_commerce.common.observability import ObservabilityManager


def test_observability_manager_exposes_app_worker_and_dependency_metrics():
    manager = ObservabilityManager()

    manager.record_request("GET", "/health", 200, 0.01)
    manager.record_kafka_produce("user-interactions", "success")
    manager.record_kafka_consume(
        "user-interactions", "video-commerce-group", "success", 0.02
    )
    manager.record_database_query("select", 0.003, "success")
    manager.record_worker_message(
        "feature-worker", "user-interactions", "success", 0.04
    )
    manager.record_asr_transcription("completed", 0.7)
    manager.record_recommendation(
        result="success",
        cache_hit=True,
        serving_path="recommendation_cache",
        candidate_count=3,
        ranked_count=3,
    )
    manager.set_ranking_queue_depth(2)
    manager.record_ranking_batch(
        request_count=4,
        candidate_count=120,
        queue_wait_seconds=0.003,
        path="microbatch",
    )
    manager.record_ranking_direct("queue_full")
    manager.set_ranking_runner_endpoint_available_connections("runner-1", 1)
    manager.set_ranking_runner_endpoint_state("runner-1", "active")
    manager.set_ranking_runner_endpoint_missing_refreshes("runner-1", 0)
    manager.record_ranking_runner_endpoint_removed("runner-old", "dns_missing")
    manager.record_ranking_runner_late_write("BrokenPipeError")
    manager.observe_ranking_runner_payload_bytes(payload_version="2", size_bytes=2048)
    manager.record_ranking_coordinator_client_error("timeout")
    manager.record_best_effort_task("cache_recommendations", "dropped_queue_full")
    manager.set_best_effort_queue_depth(3)
    manager.record_interaction_ingest("click", "accepted")
    manager.update_typed_pit_training_metrics(
        assembler_parity_ratio=1.0,
        label_reconciliation_ratio=1.0,
        current_state_calls=0,
        invalid_rows=0,
        value_mask_coverage=0.8,
    )
    manager.record_pit_orchestrator_run("completed")
    manager.set_pit_orchestrator_waiting_for_rows(True)
    manager.set_pit_trainer_waiting_for_manifest(True)
    manager.record_pit_duplicate_manifest_skip()
    manager.record_ranking_untrained_fallback()
    manager.update_pit_durable_state(
        last_success_timestamp=123.0,
        waiting_for_rows=False,
        run_in_progress=True,
        lease_expired=False,
    )

    payload = manager.prometheus_payload().decode("utf-8")

    assert "video_commerce_http_requests_total" in payload
    assert "video_commerce_kafka_messages_produced_total" in payload
    assert "video_commerce_kafka_messages_consumed_total" in payload
    assert "video_commerce_database_query_duration_seconds_bucket" in payload
    assert "video_commerce_worker_messages_processed_total" in payload
    assert "video_commerce_asr_transcriptions_total" in payload
    assert "video_commerce_asr_transcription_duration_seconds" in payload
    assert "video_commerce_recommendation_requests_total" in payload
    assert "video_commerce_ranking_batch_queue_depth" in payload
    assert "video_commerce_ranking_direct_total" in payload
    assert "video_commerce_ranking_runner_endpoint_available_connections" in payload
    assert "video_commerce_ranking_runner_endpoint_state" in payload
    assert "video_commerce_ranking_runner_endpoint_missing_refreshes" in payload
    assert "video_commerce_ranking_runner_endpoint_removed_total" in payload
    assert "video_commerce_ranking_runner_late_writes_total" in payload
    assert "video_commerce_ranking_runner_payload_bytes" in payload
    assert "video_commerce_ranking_coordinator_client_errors_total" in payload
    assert "video_commerce_best_effort_tasks_total" in payload
    assert "video_commerce_best_effort_queue_depth" in payload
    assert "video_commerce_interactions_ingested_total" in payload
    assert "pit_assembler_vector_parity_ratio" in payload
    assert "pit_label_reconciliation_ratio" in payload
    assert "pit_current_state_calls" in payload
    assert "pit_invalid_feature_or_label_rows" in payload
    assert "pit_value_mask_coverage_ratio" in payload
    assert 'pit_orchestrator_runs_total{status="completed"} 1.0' in payload
    assert "pit_orchestrator_last_success_timestamp" in payload
    assert "pit_orchestrator_waiting_for_rows 0.0" in payload
    assert "pit_trainer_waiting_for_manifest 1.0" in payload
    assert "pit_training_duplicate_manifest_skips_total 1.0" in payload
    assert "ranking_untrained_fallback_total 1.0" in payload
    assert "pit_orchestrator_last_success_timestamp 123.0" in payload
    assert "pit_orchestrator_run_in_progress 1.0" in payload
    assert "pit_orchestrator_lease_expired 0.0" in payload
