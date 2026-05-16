package com.videocommerce.flink;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.junit.jupiter.api.Test;

class InteractionFeatureJobTest {
  @Test
  void parserUsesOccurredAtAndContextCategory() throws Exception {
    String raw =
        "{"
            + "\"event_id\":\"e1\","
            + "\"schema_version\":1,"
            + "\"request_id\":\"req-1\","
            + "\"user_id\":\"u1\","
            + "\"product_id\":\"p1\","
            + "\"action\":\"click\","
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
    assertEquals(2500L, event.eventTimeMillis);
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
