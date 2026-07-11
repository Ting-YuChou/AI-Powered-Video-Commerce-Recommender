package com.videocommerce.flink;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class InteractionFeatureJobTest {
  private static final ObjectMapper JSON = new ObjectMapper();

  @Test
  void javaContractParsesSharedPythonFixtureWithSameCanonicalHash() throws Exception {
    Map<String, Object> fixture =
        JSON.readValue(
            Files.readString(Path.of("../../tests/fixtures/feature_history_contract_v1.json")),
            new TypeReference<Map<String, Object>>() {});
    Map<String, Object> event = (Map<String, Object>) fixture.get("event");

    FeatureHistoryContract.Record record = FeatureHistoryContract.parse(event);

    assertEquals(
        fixture.get("canonical_payload_json"),
        FeatureHistoryContract.canonicalJson(record.payload));
    assertEquals(event.get("payload_hash"), FeatureHistoryContract.payloadHash(record.payload));
  }

  @Test
  void parserPrefersEventTimeAndContextCategory() throws Exception {
    String raw =
        "{"
            + "\"event_id\":\"e1\","
            + "\"schema_version\":1,"
            + "\"request_id\":\"req-1\","
            + "\"user_id\":\"u1\","
            + "\"product_id\":\"p1\","
            + "\"action\":\"click\","
            + "\"event_time\":1.5,"
            + "\"occurred_at\":2.5,"
            + "\"timestamp\":3.0,"
            + "\"context\":{\"product_category\":\"shoes\",\"session_length\":12.5}"
            + "}";

    InteractionFeatureJob.InteractionEvent event =
        InteractionFeatureJob.parseInteractionEvent(raw);

    assertEquals("e1", event.eventId);
    assertEquals("u1", event.userId);
    assertEquals("p1", event.productId);
    assertEquals("click", event.action);
    assertEquals("shoes", event.productCategory);
    assertEquals(1500L, event.eventTimeMillis);
    assertEquals(12.5, event.sessionLengthSeconds);
  }

  @Test
  void parserRejectsEventsWithoutRequiredIds() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            InteractionFeatureJob.parseInteractionEvent(
                "{\"event_id\":\"e1\",\"user_id\":\"u1\",\"action\":\"view\"}"));
  }

  @Test
  void recommendationParserBuildsImpressionAndItemRows() throws Exception {
    String raw =
        "{"
            + "\"event_id\":\"rec-event-1\","
            + "\"request_id\":\"req-1\","
            + "\"user_id\":\"u1\","
            + "\"timestamp\":10,"
            + "\"metadata\":{"
            + "\"impression_id\":\"imp-1\","
            + "\"session_id\":\"session-1\","
            + "\"content_id\":\"content-1\","
            + "\"model_version\":\"v1\","
            + "\"ranking_model_version\":\"rank-v1\","
            + "\"item_snapshot_scope\":\"returned_top_k\","
            + "\"context\":{\"surface\":\"home\"},"
            + "\"displayed_items\":[{"
            + "\"product_id\":\"p1\","
            + "\"position\":1,"
            + "\"candidate_source\":\"two_tower\","
            + "\"price\":12.5,"
            + "\"category\":\"Shoes\","
            + "\"brand\":\"A\","
            + "\"ranking_score\":0.91,"
            + "\"confidence_score\":0.8"
            + "}]"
            + "}"
            + "}";

    InteractionFeatureJob.RecommendationImpressionEvent event =
        InteractionFeatureJob.parseRecommendationEvent(raw);

    assertEquals("imp-1", event.impressionId);
    assertEquals("req-1", event.requestId);
    assertEquals("u1", event.userId);
    assertEquals("session-1", event.sessionId);
    assertEquals("content-1", event.contentId);
    assertEquals("v1", event.modelVersion);
    assertEquals("rank-v1", event.rankingModelVersion);
    assertEquals("returned_top_k", event.context.get("item_snapshot_scope"));
    assertEquals(10_000L, event.eventTimeMillis);

    List<InteractionFeatureJob.RecommendationImpressionItemRow> itemRows =
        event.itemRows;
    assertEquals(1, itemRows.size());
    InteractionFeatureJob.RecommendationImpressionItemRow item = itemRows.get(0);
    assertEquals("imp-1", item.impressionId);
    assertEquals("p1", item.productId);
    assertEquals(1, item.position);
    assertEquals("two_tower", item.source);
    assertEquals("two_tower", item.featureSnapshot.get("candidate_source"));
    assertEquals(12.5, item.featureSnapshot.get("price"));
    assertEquals("Shoes", item.featureSnapshot.get("category"));
    assertEquals("A", item.featureSnapshot.get("brand"));
    assertEquals(0.91, item.scores.get("ranking_score"));
    assertEquals(0.8, item.scores.get("confidence_score"));
  }

  @Test
  void recommendationParserRejectsEventsWithoutValidImpressionShape() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            InteractionFeatureJob.parseRecommendationEvent(
                "{"
                    + "\"event_id\":\"rec-event-1\","
                    + "\"user_id\":\"u1\","
                    + "\"metadata\":{\"displayed_items\":[{\"product_id\":\"p1\"}]}"
                    + "}"));

    assertThrows(
        IllegalArgumentException.class,
        () ->
            InteractionFeatureJob.parseRecommendationEvent(
                "{"
                    + "\"event_id\":\"rec-event-1\","
                    + "\"user_id\":\"u1\","
                    + "\"metadata\":{"
                    + "\"impression_id\":\"imp-1\","
                    + "\"displayed_items\":\"not-a-list\""
                    + "}"
                    + "}"));
  }

  @Test
  void recommendationParserSkipsMalformedDisplayedItemsAndNoopsEmptySlates()
      throws Exception {
    InteractionFeatureJob.RecommendationImpressionEvent event =
        InteractionFeatureJob.parseRecommendationEvent(
            "{"
                + "\"event_id\":\"rec-event-1\","
                + "\"user_id\":\"u1\","
                + "\"timestamp\":10,"
                + "\"metadata\":{"
                + "\"impression_id\":\"imp-1\","
                + "\"displayed_items\":["
                + "\"not-an-object\","
                + "{\"position\":2},"
                + "{\"product_id\":\"p1\",\"position\":3}"
                + "]"
                + "}"
                + "}");

    assertEquals(1, event.itemRows.size());
    assertEquals("p1", event.itemRows.get(0).productId);
    assertEquals(3, event.itemRows.get(0).position);

    assertNull(
        InteractionFeatureJob.parseRecommendationEvent(
            "{"
                + "\"event_id\":\"rec-event-2\","
                + "\"user_id\":\"u1\","
                + "\"timestamp\":10,"
                + "\"metadata\":{\"impression_id\":\"imp-2\",\"displayed_items\":[]}"
                + "}"));
    assertNull(
        InteractionFeatureJob.parseRecommendationEvent(
            "{"
                + "\"event_id\":\"rec-event-3\","
                + "\"user_id\":\"u1\","
                + "\"timestamp\":10,"
                + "\"metadata\":{"
                + "\"impression_id\":\"imp-3\","
                + "\"displayed_items\":[\"not-an-object\",{\"position\":1}]"
                + "}"
                + "}"));
  }

  @Test
  void recommendationParserUsesTimestampThenMetadataCreatedAtForCreatedAt()
      throws Exception {
    InteractionFeatureJob.RecommendationImpressionEvent timestampEvent =
        InteractionFeatureJob.parseRecommendationEvent(
            "{"
                + "\"event_id\":\"rec-event-1\","
                + "\"user_id\":\"u1\","
                + "\"timestamp\":10,"
                + "\"occurred_at\":20,"
                + "\"metadata\":{"
                + "\"created_at\":30,"
                + "\"impression_id\":\"imp-1\","
                + "\"displayed_items\":[{\"product_id\":\"p1\"}]"
                + "}"
                + "}");
    assertEquals(10_000L, timestampEvent.eventTimeMillis);

    InteractionFeatureJob.RecommendationImpressionEvent metadataCreatedAtEvent =
        InteractionFeatureJob.parseRecommendationEvent(
            "{"
                + "\"event_id\":\"rec-event-2\","
                + "\"user_id\":\"u1\","
                + "\"occurred_at\":20,"
                + "\"metadata\":{"
                + "\"created_at\":30,"
                + "\"impression_id\":\"imp-2\","
                + "\"displayed_items\":[{\"product_id\":\"p2\"}]"
                + "}"
                + "}");
    assertEquals(30_000L, metadataCreatedAtEvent.eventTimeMillis);
  }

  @Test
  void dlqPayloadIncludesSourceTopic() {
    InteractionFeatureJob.DlqEvent event =
        InteractionFeatureJob.DlqEvent.invalid(
            "{}", new IllegalArgumentException("bad event"), "recommendation-events");

    Map<String, Object> payload = event.payload();

    assertEquals("recommendation-events", payload.get("source_topic"));
    assertEquals("IllegalArgumentException", payload.get("error_type"));
  }

  @Test
  void pitCatalogUsesExplicitS3FileIoEndpointAndPathStyle() {
    String sql =
        PointInTimeFeatureJoinJob.buildRestCatalogSql(
            "feature_catalog",
            "http://iceberg-rest:8181",
            "s3://features/warehouse",
            "http://minio:9000");

    assertTrue(sql.contains("'io-impl'='org.apache.iceberg.aws.s3.S3FileIO'"));
    assertTrue(sql.contains("'s3.endpoint'='http://minio:9000'"));
    assertTrue(sql.contains("'s3.path-style-access'='true'"));
  }

  @Test
  void pointInTimeJoinSqlEnforcesEventAndAvailabilityCutsAndSevenDayAttribution() {
    String sql = PointInTimeFeatureJoinJob.buildPointInTimeInsertSql(
        "video_commerce", "ranking_ltr_v1", 168);

    assertEquals(true, sql.contains("u.event_time_epoch<=o.event_time_epoch"));
    assertEquals(true, sql.contains("u.available_at_epoch<=o.event_time_epoch"));
    assertEquals(true, sql.contains("i.event_time_epoch<=o.event_time_epoch"));
    assertEquals(true, sql.contains("i.available_at_epoch<=o.event_time_epoch"));
    assertEquals(true, sql.contains("i.event_time_epoch<=o.event_time_epoch+604800"));
    assertEquals(true, sql.contains("source_event_id DESC"));
    assertEquals(true, sql.contains("item_snapshot_complete"));
    assertEquals(true, sql.contains("CAST(1699391600.000 AS DOUBLE)"));
    assertEquals(true, sql.contains("i.available_at_epoch<=CAST(1700000000.000 AS DOUBLE)"));
    assertEquals(true, sql.contains("ranking_ltr_v1"));
    assertEquals(true, sql.startsWith("INSERT INTO"));
    assertFalse(sql.contains("(materialization_run_id,observation_id,"));
    assertEquals(false, sql.contains("INSERT OVERWRITE"));
    assertEquals(true, sql.contains("pit_feature_bundle_hash"));
    assertEquals(true, sql.contains("o.feature_bundle_hash"));
    assertTrue(sql.contains("'ranking_labels_v1'"));
    assertTrue(sql.contains("attributed_action"));
    assertTrue(sql.contains("attributed_value"));
    assertTrue(sql.contains("attributed_value_source"));
    assertTrue(sql.contains("JSON_VALUE(context_json,'$.purchase_value')"));
    assertTrue(sql.contains("\nWITH eligible_observations"));
    assertEquals(
        true,
        PointInTimeFeatureJoinJob.buildExistingRunSql("video_commerce", "run-1")
            .contains("ranking_training_pit_quarantine"));
  }

  @Test
  void pitSchemaEvolutionAddsTypedLabelColumnsAdditively() {
    String sql = PointInTimeFeatureJoinJob.buildAddColumnSql(
        "video_commerce", "attributed_value", "DOUBLE");

    assertEquals(
        "ALTER TABLE `video_commerce`.`ranking_training_pit` ADD `attributed_value` DOUBLE",
        sql);
    assertTrue(PointInTimeFeatureJoinJob.buildTrainingTableSql("video_commerce")
        .contains("label_definition_version STRING"));
    assertTrue(PointInTimeFeatureJoinJob.buildParquetExportTableSql("s3://bucket/pit", "run-1")
        .contains("attributed_action STRING"));
  }

  @Test
  void pitInsertAdaptsSelectOrderToExistingPhaseTwoSchema() {
    List<String> phaseTwoThenAddedColumns = List.of(
        "materialization_run_id", "observation_id", "impression_id", "user_id", "product_id",
        "action", "as_of_ts", "user_features_json", "product_metadata_json", "context_json",
        "candidate_features_json", "online_feature_bundle_hash", "feature_bundle_hash",
        "attributed_click", "attributed_purchase", "feature_definition_version",
        "materialized_at", "materialization_date", "attributed_action", "attributed_value",
        "attributed_value_source", "label_definition_version");

    String sql = PointInTimeFeatureJoinJob.buildPointInTimeInsertSql(
        "video_commerce", "ranking_ltr_v1", 168, 1, 1_700_000_000.0, "run-1",
        phaseTwoThenAddedColumns);

    assertTrue(sql.contains(
        "COALESCE(f.attributed_purchase,0),'ranking_ltr_v1',CURRENT_TIMESTAMP,CURRENT_DATE,"
            + "f.attributed_action,f.attributed_value,f.attributed_value_source,'ranking_labels_v1'"));
  }

  @Test
  void pitBundleHashReflectsFinalJoinedFeatures() throws Exception {
    PointInTimeFeatureJoinJob.PitFeatureBundleHash function =
        new PointInTimeFeatureJoinJob.PitFeatureBundleHash();

    String first =
        function.eval(
            50.0,
            "u1",
            "p1",
            "{\"total_interactions\":3}",
            "{\"price\":9.0}",
            "{}",
            "{\"combined_score\":0.5}",
            "ranking_ltr_v1");
    String changed =
        function.eval(
            50.0,
            "u1",
            "p1",
            "{\"total_interactions\":4}",
            "{\"price\":9.0}",
            "{}",
            "{\"combined_score\":0.5}",
            "ranking_ltr_v1");

    assertEquals(64, first.length());
    assertEquals(false, first.equals(changed));
  }

  @Test
  void userAccumulatorBuildsSnapshotAndChronologicalSequence() throws Exception {
    InteractionFeatureJob.UserAccumulator accumulator =
        new InteractionFeatureJob.UserAccumulator("u1");

    accumulator.apply(
        InteractionFeatureJob.parseInteractionEvent(
            "{\"event_id\":\"e2\",\"user_id\":\"u1\",\"product_id\":\"p2\",\"action\":\"click\","
                + "\"occurred_at\":2,\"context\":{\"product_category\":\"shoes\"}}"));
    accumulator.apply(
        InteractionFeatureJob.parseInteractionEvent(
            "{\"event_id\":\"e1\",\"user_id\":\"u1\",\"product_id\":\"p1\",\"action\":\"view\","
                + "\"occurred_at\":1,\"context\":{\"product_category\":\"shoes\"}}"));

    InteractionFeatureJob.UserFeatureSnapshot snapshot = accumulator.snapshot();

    assertEquals(2, snapshot.totalInteractions);
    assertEquals(1, snapshot.totalViews);
    assertEquals(1, snapshot.totalClicks);
    assertEquals(1.0, snapshot.clickThroughRate);
    assertEquals("p1", snapshot.sequence.get(0).productId);
    assertEquals("p2", snapshot.sequence.get(1).productId);
    assertEquals("e2", snapshot.sequenceToken().get("latest_event_id"));

    snapshot.availableAt = 3.0;
    Map<String, Object> kafkaEvent =
        JSON.readValue(
            new InteractionFeatureJob.FeatureUpdateJsonMapper().map(snapshot),
            new TypeReference<Map<String, Object>>() {});
    assertEquals(
        FeatureHistoryContract.canonicalJson(snapshot.userFeaturePayload()),
        FeatureHistoryContract.canonicalJson((Map<String, Object>) kafkaEvent.get("payload")));
    assertEquals("ranking_ltr_v1", kafkaEvent.get("feature_definition_version"));
    assertEquals(2.0, kafkaEvent.get("event_time"));
    assertEquals(3.0, kafkaEvent.get("available_at"));
  }

  @Test
  void windowCountsCalculatesCtrAndCvrInputs() {
    InteractionFeatureJob.WindowCounts counts = new InteractionFeatureJob.WindowCounts();
    counts.add("view");
    counts.add("view");
    counts.add("click");
    counts.add("purchase");

    assertEquals(2, counts.views);
    assertEquals(1, counts.clicks);
    assertEquals(1, counts.purchases);
    assertEquals(4, counts.totalEvents);
    assertEquals(0.5, InteractionFeatureJob.ratio(counts.clicks, counts.views));
    assertEquals(1.0, InteractionFeatureJob.ratio(counts.purchases, counts.clicks));
  }

  @Test
  void windowSnapshotPublishesTheSamePayloadUsedByRedis() throws Exception {
    InteractionFeatureJob.WindowFeatureSnapshot snapshot =
        new InteractionFeatureJob.WindowFeatureSnapshot();
    snapshot.entityType = "user";
    snapshot.entityId = "u1";
    snapshot.window = "5m";
    snapshot.views = 2;
    snapshot.clicks = 1;
    snapshot.totalEvents = 3;
    snapshot.clickThroughRate = 0.5;
    snapshot.windowStart = 100.0;
    snapshot.windowEnd = 400.0;
    snapshot.availableAt = 405.0;

    Map<String, Object> kafkaEvent =
        JSON.readValue(
            new InteractionFeatureJob.WindowFeatureUpdateJsonMapper().map(snapshot),
            new TypeReference<Map<String, Object>>() {});

    assertEquals(
        FeatureHistoryContract.canonicalJson(snapshot.payload()),
        FeatureHistoryContract.canonicalJson((Map<String, Object>) kafkaEvent.get("payload")));
    assertEquals("window_feature", kafkaEvent.get("event_type"));
    assertEquals(400.0, kafkaEvent.get("event_time"));
    assertEquals(405.0, kafkaEvent.get("available_at"));
  }
}
