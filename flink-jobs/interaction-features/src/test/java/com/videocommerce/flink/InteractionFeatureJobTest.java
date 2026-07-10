package com.videocommerce.flink;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class InteractionFeatureJobTest {
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
  void pointInTimeJoinSqlEnforcesEventAndAvailabilityCutsAndSevenDayAttribution() {
    String sql = PointInTimeFeatureJoinJob.buildPointInTimeInsertSql(
        "video_commerce", "ranking_ltr_v1", 168);

    assertEquals(true, sql.contains("u.event_time <= o.event_time"));
    assertEquals(true, sql.contains("u.available_at <= o.event_time"));
    assertEquals(true, sql.contains("i.event_time <= o.event_time"));
    assertEquals(true, sql.contains("i.available_at <= o.event_time"));
    assertEquals(true, sql.contains("INTERVAL '168' HOUR"));
    assertEquals(true, sql.contains("ranking_ltr_v1"));
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
}
