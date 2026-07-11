package com.videocommerce.flink;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class FeatureHistoryMaterializerJobTest {
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
