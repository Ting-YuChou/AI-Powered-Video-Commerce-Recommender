package com.videocommerce.flink;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class RetrievalPointInTimeJoinJobTest {
  @Test
  void retrievalTableMatchesTheTypedPythonContract() {
    String sql = RetrievalPointInTimeJoinJob.buildTrainingTableSql("video_commerce");

    assertTrue(sql.contains("retrieval_training_pit"));
    assertTrue(sql.contains("query_id STRING"));
    assertTrue(sql.contains("label_source STRING"));
    assertTrue(sql.contains("feature_available_at DOUBLE"));
    assertTrue(sql.contains("catalog_generation_id STRING"));
    assertTrue(sql.contains("user_features_json STRING"));
  }

  @Test
  void retrievalJoinUsesViewedAnchorsAndBothTemporalCutoffs() {
    String sql =
        RetrievalPointInTimeJoinJob.buildInsertSql(
            "video_commerce", 168, 1, 1_700_000_000.0, "retrieval-pit-1", "catalog-1");

    assertTrue(sql.contains("recommendation_view_history"));
    assertTrue(sql.contains("interaction_history"));
    assertTrue(sql.contains("user_feature_history"));
    assertTrue(sql.contains("item_feature_history"));
    assertTrue(sql.contains("event_time_epoch <= v.as_of_ts"));
    assertTrue(sql.contains("available_at_epoch <= v.as_of_ts"));
    assertTrue(sql.contains("168 * 3600"));
    assertTrue(sql.contains("viewed_no_positive"));
    assertTrue(sql.contains("organic_positive"));
    assertTrue(sql.contains("ranking_rejected_observation"));
    assertTrue(sql.contains("'ranker_rejected'"));
    assertTrue(sql.contains("catalog_generation_id"));
    assertTrue(sql.contains("h.source_version='catalog-1'"));
    assertFalse(sql.contains("recommendation_impression_history"));
  }

  @Test
  void immutableExportUsesRunAndAttemptGeneration() {
    String sql =
        RetrievalPointInTimeJoinJob.buildExportTableSql(
            "s3://features/retrieval", "retrieval-pit-1", 2);

    assertTrue(sql.contains("retrieval-pit-1/attempt-2"));
    assertTrue(sql.contains("'format'='parquet'"));
  }
}
