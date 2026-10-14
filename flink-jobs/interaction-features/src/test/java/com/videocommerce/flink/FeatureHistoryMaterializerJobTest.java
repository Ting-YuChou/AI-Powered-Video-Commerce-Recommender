package com.videocommerce.flink;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import java.util.LinkedHashMap;
import org.junit.jupiter.api.Test;

class FeatureHistoryMaterializerJobTest {
  @Test
  void recommendationSourceSignalsAndCategoricalSnapshotsSurviveHistoryMaterialization() throws Exception {
    Map<String, Object> signals = Map.of("two_tower", Map.of(
        "raw_score", 0.7, "source_rank", 1, "source_count", 10,
        "normalized_rank", 1.0, "reciprocal_rank", 1.0 / 61.0));
    Map<String, Object> item = Map.of(
        "product_id", "p1", "position", 1,
        "feature_bundle_hash", "a".repeat(64), "feature_definition_version", "ranking_ltr_v1",
        "feature_snapshot", Map.of("category", "shoes", "brand", "brand-a", "tags", List.of("new")),
        "scores", Map.of("retrieval_signals", signals, "fusion_policy_version", "candidate_rrf_v1"),
        "item_snapshot_complete", true);
    Map<String, Object> metadata = Map.of(
        "impression_id", "imp-1", "displayed_items", List.of(item),
        "rejected_candidate_items", List.of(),
        "feature_context", Map.of("device", "mobile", "surface", "home"),
        "user_feature_snapshot", Map.of("preferred_categories", List.of("shoes")),
        "ranking_feature_contract_version", "ranking_ltr_v3_categorical_sources");
    Map<String, Object> event = new LinkedHashMap<>();
    event.put("event_id", "e1"); event.put("event_type", "recommendation_served");
    event.put("user_id", "u1"); event.put("recommendations", List.of("p1"));
    event.put("response_time_ms", 12); event.put("metadata", metadata);
    event.put("source_event_id", "e1"); event.put("source_version", "model-v3");
    event.put("event_time", 100); event.put("available_at", 101);
    event.put("feature_definition_version", "ranking_ltr_v1");
    event.put("payload_schema_version", 1);
    event.put("payload_hash", FeatureHistoryContract.payloadHash(Map.of(
        "user_id", "u1", "recommendations", List.of("p1"), "response_time_ms", 12, "metadata", metadata)));
    List<FeatureHistoryMaterializerJob.LakeHistoryRow> rows = FeatureHistoryMaterializerJob.parseRawEvent(
        new FeatureHistoryMaterializerJob.RawKafkaEvent("recommendation-events", FeatureHistoryContract.canonicalJson(event)));
    assertEquals(1, rows.size());
    assertTrue(rows.get(0).candidateFeaturesJson.contains("reciprocal_rank"));
    assertTrue(rows.get(0).candidateFeaturesJson.contains("candidate_rrf_v1"));
    assertTrue(rows.get(0).itemSnapshotJson.contains("brand-a"));
    assertTrue(rows.get(0).userFeaturesJson.contains("preferred_categories"));
  }
  @Test
  void replayAlwaysUsesAnIsolatedNamespace() {
    assertEquals(
        "video_commerce_rebuild_run_42",
        FeatureHistoryMaterializerJob.resolveNamespace("video_commerce", "Run-42", true));
    assertThrows(
        IllegalArgumentException.class,
        () -> FeatureHistoryMaterializerJob.resolveNamespace("video_commerce", "", true));
  }

  @Test
  void dlqSinkParticipatesInCheckpointedStatementSet() {
    String sql =
        FeatureHistoryMaterializerJob.buildDlqTableSql(
            "kafka:9092", "dead-letter-events", 600000);
    assertTrue(sql.contains("'sink.delivery-guarantee'='exactly-once'"));
    assertTrue(sql.contains("'sink.transactional-id-prefix'='feature-history-dlq-'"));
    assertTrue(sql.contains("payload_json STRING"));
    assertTrue(sql.contains("'properties.transaction.timeout.ms'='600000'"));
  }

  @Test
  void userAndItemHistoryUseExplicitDeterministicEntityBuckets() throws Exception {
    String userSql =
        FeatureHistoryMaterializerJob.buildCreateTableSql(
            "video_commerce", "user_feature_history");
    String itemSql =
        FeatureHistoryMaterializerJob.buildCreateTableSql(
            "video_commerce", "item_feature_history");
    String interactionSql =
        FeatureHistoryMaterializerJob.buildCreateTableSql(
            "video_commerce", "interaction_history");

    assertTrue(userSql.contains("entity_bucket INT"));
    assertTrue(userSql.contains("PARTITIONED BY (event_date, entity_bucket)"));
    assertTrue(itemSql.contains("PARTITIONED BY (event_date, entity_bucket)"));
    assertTrue(interactionSql.contains("PARTITIONED BY (event_date)"));
    assertTrue(interactionSql.contains("'write.upsert.enabled'='false'"));
    assertEquals(
        FeatureHistoryMaterializerJob.entityBucket("user-42"),
        FeatureHistoryMaterializerJob.entityBucket("user-42"));
    assertTrue(FeatureHistoryMaterializerJob.entityBucket("user-42") >= 0);
    assertTrue(FeatureHistoryMaterializerJob.entityBucket("user-42") < 32);
  }

  @Test
  void featureContractRoutesCatalogIntoItemHistory() throws Exception {
    String raw =
        "{\"event_id\":\"e1\",\"event_type\":\"catalog_feature\","
            + "\"entity_type\":\"item\",\"entity_id\":\"p1\","
            + "\"event_time\":100,\"available_at\":105,\"source_event_id\":\"e1\","
            + "\"source_version\":\"catalog-v1\",\"feature_definition_version\":\"ranking_ltr_v1\","
            + "\"payload_schema_version\":1,"
            + "\"payload_hash\":\"89653e39ea989d1d70dc904c11411ba28804d29c22ec98d92f8213b75e7167c8\","
            + "\"payload\":{\"price\":10}}";
    List<FeatureHistoryMaterializerJob.LakeHistoryRow> rows =
        FeatureHistoryMaterializerJob.parseRawEvent(
            new FeatureHistoryMaterializerJob.RawKafkaEvent("catalog-feature-events", raw));

    assertEquals(1, rows.size());
    assertEquals("item_feature_history", rows.get(0).targetTable);
    assertEquals("p1", rows.get(0).productId);
  }

  @Test
  void recommendationViewedRoutesIntoDedicatedHistory() throws Exception {
    String raw =
        "{\"event_id\":\"view-1\",\"event_type\":\"recommendation_viewed\","
            + "\"impression_id\":\"imp-1\",\"user_id\":\"u1\",\"product_id\":\"p1\","
            + "\"position\":2,\"context\":{},\"event_time\":101,\"available_at\":102,"
            + "\"source_event_id\":\"view-1\",\"source_version\":\"client-view-v1\","
            + "\"feature_definition_version\":\"ranking_ltr_v1\",\"payload_schema_version\":1,"
            + "\"payload_hash\":\"af07b6c3cde94ec6181d246df40b5518e695c1e1c1dffd98f993ca4b1d11d982\"}";

    List<FeatureHistoryMaterializerJob.LakeHistoryRow> rows =
        FeatureHistoryMaterializerJob.parseRawEvent(
            new FeatureHistoryMaterializerJob.RawKafkaEvent("recommendation-events", raw));

    assertEquals(1, rows.size());
    assertEquals("recommendation_view_history", rows.get(0).targetTable);
    assertEquals("imp-1:p1", rows.get(0).observationId);
  }

  @Test
  void legacyOperationalFeatureUpdatesAreNotAcceptedAsHistorySnapshots() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            FeatureHistoryMaterializerJob.parseRawEvent(
                new FeatureHistoryMaterializerJob.RawKafkaEvent(
                    "feature-updates", "{\"event_type\":\"feature_update\"}")));
  }

  @Test
  void unknownHistoryFeatureTypeIsRejectedInsteadOfFallingIntoUserHistory() {
    String raw =
        "{\"event_id\":\"e1\",\"event_type\":\"content_feature\","
            + "\"entity_type\":\"content\",\"entity_id\":\"c1\","
            + "\"event_time\":100,\"available_at\":105,\"source_event_id\":\"e1\","
            + "\"source_version\":\"v1\",\"feature_definition_version\":\"ranking_ltr_v1\","
            + "\"payload_schema_version\":1,"
            + "\"payload_hash\":\"89653e39ea989d1d70dc904c11411ba28804d29c22ec98d92f8213b75e7167c8\","
            + "\"payload\":{\"price\":10}}";

    assertThrows(
        IllegalArgumentException.class,
        () ->
            FeatureHistoryMaterializerJob.parseRawEvent(
                new FeatureHistoryMaterializerJob.RawKafkaEvent(
                    "feature-history-snapshots", raw)));
  }

  @Test
  void dlqIdentityIsDeterministicForKafkaCoordinates() throws Exception {
    FeatureHistoryMaterializerJob.RawKafkaEvent raw =
        new FeatureHistoryMaterializerJob.RawKafkaEvent("feature-history-snapshots", "{}");
    raw.partition = 2;
    raw.offset = 42;
    raw.timestamp = 123_000L;

    String first =
        FeatureHistoryMaterializerJob.dlqPayload(raw, new IllegalArgumentException("bad"));
    String second =
        FeatureHistoryMaterializerJob.dlqPayload(raw, new IllegalArgumentException("bad"));

    assertEquals(first, second);
    assertTrue(first.contains("\"occurred_at\":123.0"));
    assertTrue(first.contains("\"event_id\""));
  }
}
