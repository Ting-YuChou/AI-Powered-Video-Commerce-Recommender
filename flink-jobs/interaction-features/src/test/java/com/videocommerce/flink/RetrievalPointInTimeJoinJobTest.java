package com.videocommerce.flink;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;
import org.junit.jupiter.api.Test;

class RetrievalPointInTimeJoinJobTest {
  private static final ObjectMapper JSON = new ObjectMapper();
  // 168h attribution + 1h lateness: anchors must satisfy as_of <= CUTOFF - 608_400.
  private static final double CUTOFF = 1_000_000.0;

  @Test
  void retrievalTableMatchesTheTypedPythonContract() {
    String sql = RetrievalPointInTimeJoinJob.buildTrainingTableSql("video_commerce");

    assertTrue(sql.contains("retrieval_training_pit"));
    assertTrue(sql.contains("query_id STRING"));
    assertTrue(sql.contains("label_source STRING"));
    assertTrue(sql.contains("feature_available_at DOUBLE"));
    assertTrue(sql.contains("catalog_generation_id STRING"));
    assertTrue(sql.contains("user_features_json STRING"));
    assertTrue(sql.contains("seen_product_ids_json STRING"));
    assertTrue(sql.contains("ranker_score DOUBLE"));
  }

  @Test
  void retrievalJoinUsesViewedAnchorsAndBothTemporalCutoffs() {
    String sql =
        RetrievalPointInTimeJoinJob.buildInsertSql(
            "video_commerce", 168, 1, 1_700_000_000.0, "retrieval-pit-1", "catalog-1", 200);

    assertTrue(sql.startsWith("INSERT INTO `video_commerce`.`retrieval_training_pit`"));
    assertTrue(sql.contains("recommendation_view_history"));
    assertTrue(sql.contains("user_feature_history"));
    assertTrue(sql.contains("item_feature_history"));
    assertTrue(sql.contains("event_time_epoch <= v.as_of_ts"));
    assertTrue(sql.contains("available_at_epoch <= v.as_of_ts"));
    assertTrue(sql.contains("h.source_version='catalog-1'"));
    assertTrue(sql.contains("seen_rank <= 200"));
    assertFalse(sql.contains("'[]',CAST(NULL AS DOUBLE)"));
    assertFalse(sql.contains("recommendation_impression_history"));
  }

  @Test
  void zeroSeenHistoryLimitSkipsTheHistoryJoin() {
    String sql =
        RetrievalPointInTimeJoinJob.buildSelectSql(
            "video_commerce", 168, 1, CUTOFF, "run", "catalog-1", 0);

    assertFalse(sql.contains("seen_events"));
    assertTrue(sql.contains("u.user_features_json,'[]',v.ranker_score"));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            RetrievalPointInTimeJoinJob.buildSelectSql(
                "video_commerce", 168, 1, CUTOFF, "run", "catalog-1", -1));
  }

  @Test
  void seenHistoryLimitIsConfigurableAndDefaultsToTheServingReadLimit() {
    Map<String, String> environment = new HashMap<>();
    environment.put("FEATURE_LAKE_CATALOG_URI", "http://catalog");
    environment.put("FEATURE_LAKE_WAREHOUSE_URI", "s3://warehouse");
    environment.put("FEATURE_LAKE_S3_ENDPOINT", "http://s3");
    environment.put("FEATURE_LAKE_MATERIALIZATION_CUTOFF", "1000");
    environment.put("FEATURE_LAKE_MATERIALIZATION_RUN_ID", "run");
    environment.put("FEATURE_LAKE_RETRIEVAL_CATALOG_GENERATION_ID", "catalog-1");
    environment.put("FEATURE_LAKE_RETRIEVAL_PIT_EXPORT_URI", "s3://export");

    assertEquals(
        200,
        RetrievalPointInTimeJoinJob.Config.resolve(new String[0], environment).seenHistoryLimit);
    assertEquals(
        7,
        RetrievalPointInTimeJoinJob.Config.resolve(
                new String[] {"--seen-history-limit", "7"}, environment)
            .seenHistoryLimit);
  }

  @Test
  void executedJoinMatchesThePythonReaderContract() throws Exception {
    TableEnvironment tables =
        TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
    tables.executeSql("CREATE DATABASE ns");
    historyView(
        tables,
        "recommendation_view_history",
        List.of(
            history("obs-1").user("u1").product("p1").at(100_000).payload("{\"impression_id\":\"imp-1\"}"),
            // Immature: 400_000 + 608_400 > cutoff.
            history("obs-2").user("u1").product("p2").at(400_000).payload("{\"impression_id\":\"imp-2\"}"),
            // Not a member of the pinned catalog generation.
            history("obs-3").user("u1").product("p9").at(100_000).payload("{\"impression_id\":\"imp-3\"}")));
    historyView(
        tables,
        "ranking_observations",
        List.of(
            history("rej-2").type("ranking_rejected_observation").user("u1").product("p2").at(100_000)
                .observation("imp-1:p2").payload("{\"impression_id\":\"imp-1\"}")
                .candidateFeatures("{\"combined_score\":0.2,\"raw_ranking_score\":0.75,\"ranker_scored\":true}"),
            history("rej-3").type("ranking_rejected_observation").user("u1").product("p3").at(100_000)
                .observation("imp-1:p3").payload("{\"impression_id\":\"imp-1\"}")
                .candidateFeatures("{\"combined_score\":0.2,\"ranker_scored\":false}"),
            history("obs-1-ranked").type("ranking_observation").user("u1").product("p1").at(100_000)
                .observation("imp-1:p1").payload("{\"impression_id\":\"imp-1\"}")
                .candidateFeatures("{\"raw_ranking_score\":9.0}")));
    historyView(
        tables,
        "interaction_history",
        List.of(
            history("e-view-p5").action("view").user("u1").product("p5").at(90_000),
            history("e-click-p6").action("click").user("u1").product("p6").at(95_000)
                .context("{\"impression_id\":\"older\"}"),
            history("e-like-p5").action("like").user("u1").product("p5").at(97_000),
            // Happened before as_of but was not available yet.
            history("e-cart-p7").action("add_to_cart").user("u1").product("p7").at(99_000).available(150_000),
            // Same instant as the query: not strictly before it.
            history("e-like-p8").action("like").user("u1").product("p8").at(100_000),
            // Another user's history never leaks into u1's seen set.
            history("e-other").action("click").user("u2").product("p4").at(95_000),
            // Attributed positive for the viewed impression.
            history("e-click-p1").action("click").user("u1").product("p1").at(101_000).available(101_500)
                .context("{\"impression_id\":\"imp-1\"}"),
            // Mature organic positive.
            history("e-org-p3").action("click").user("u1").product("p3").at(200_000),
            // Immature organic positive (previously exported and failed the Python reader).
            history("e-org-p2").action("purchase").user("u1").product("p2").at(500_000)));
    historyView(
        tables,
        "user_feature_history",
        List.of(
            history("uf-1").entity("u1").at(50_000).payload("{\"total_interactions\":3}"),
            history("uf-future").entity("u1").at(150_000).payload("{\"total_interactions\":99}")));
    // The pinned generation is activated after the 100_000 queries but before the cutoff.
    historyView(
        tables,
        "item_feature_history",
        List.of(
            history("cat-p1").entity("p1").version("catalog-1").at(300_000),
            history("cat-p2").entity("p2").version("catalog-1").at(300_000),
            history("cat-p3").entity("p3").version("catalog-1").at(300_000),
            history("cat-p9").entity("p9").version("catalog-0").at(10_000)));
    tables.executeSql(
        "CREATE TEMPORARY VIEW ns.retrieval_training_pit AS SELECT "
            + "CAST(NULL AS STRING) materialization_run_id,CAST(NULL AS STRING) query_id,"
            + "CAST(NULL AS STRING) product_id,CAST(NULL AS STRING) label_source "
            + "FROM (VALUES (1)) AS t(x) WHERE x = 0");

    Map<String, Row> rows = new HashMap<>();
    try (CloseableIterator<Row> iterator =
        tables
            .executeSql(
                RetrievalPointInTimeJoinJob.buildSelectSql(
                    "ns", 168, 1, CUTOFF, "run-1", "catalog-1", 2))
            .collect()) {
      while (iterator.hasNext()) {
        Row row = iterator.next();
        rows.put(row.getField(1) + "|" + row.getField(3) + "|" + row.getField(4), row);
      }
    }

    assertEquals(
        Set.of(
            "imp-1|p1|viewed_impression",
            "imp-1|p2|ranker_rejected",
            "imp-1|p3|ranker_rejected",
            "e-org-p3|p3|organic_positive"),
        rows.keySet());

    Row viewed = rows.get("imp-1|p1|viewed_impression");
    assertEquals("click", viewed.getField(5));
    assertEquals("catalog-1", viewed.getField(12));
    assertEquals("{\"total_interactions\":3}", viewed.getField(13));
    // Two most recent events strictly before as_of and available by then: like p5, click p6.
    assertEquals(Set.of("p5", "p6"), seen(viewed));
    assertNull(viewed.getField(15));

    Row scored = rows.get("imp-1|p2|ranker_rejected");
    assertEquals(0.75, (Double) scored.getField(15), 1e-12);
    assertEquals(Set.of("p5", "p6"), seen(scored));
    assertNull(rows.get("imp-1|p3|ranker_rejected").getField(15));

    Row organic = rows.get("e-org-p3|p3|organic_positive");
    assertEquals("click", organic.getField(5));
    assertEquals(200_000.0, (Double) organic.getField(7), 1e-9);
    // Two most recent available events before 200_000: click p1 @101_000, like p8 @100_000.
    assertEquals(Set.of("p1", "p8"), seen(organic));
    // Python maturity rule: as_of + (window + lateness) * 3600 <= cutoff for every row.
    for (Row row : rows.values()) {
      assertTrue((Double) row.getField(7) + (168 + 1) * 3600.0 <= CUTOFF);
    }
  }

  @Test
  void immutableExportUsesRunAndAttemptGeneration() {
    String sql =
        RetrievalPointInTimeJoinJob.buildExportTableSql(
            "s3://features/retrieval", "retrieval-pit-1", 2);

    assertTrue(sql.contains("retrieval-pit-1/attempt-2"));
    assertTrue(sql.contains("'format'='parquet'"));
  }

  private static Set<String> seen(Row row) throws Exception {
    List<String> values =
        JSON.readValue((String) row.getField(14), new TypeReference<List<String>>() {});
    Set<String> unique = new HashSet<>(values);
    assertEquals(values.size(), unique.size(), "seen products must be distinct");
    return unique;
  }

  private static HistoryRow history(String eventId) {
    return new HistoryRow(eventId);
  }

  private static void historyView(TableEnvironment tables, String name, List<HistoryRow> rows) {
    List<String> values = new ArrayList<>();
    for (HistoryRow row : rows) {
      values.add(row.values());
    }
    tables.executeSql(
        "CREATE TEMPORARY VIEW ns."
            + name
            + " AS SELECT * FROM (VALUES "
            + String.join(",", values)
            + ") AS t(event_id,event_type,user_id,product_id,action,event_time_epoch,"
            + "available_at_epoch,source_version,entity_id,canonical_payload_json,context_json,"
            + "candidate_features_json,observation_id,event_date)");
  }

  private static final class HistoryRow {
    private final String eventId;
    private String eventType = "event";
    private String userId;
    private String productId;
    private String action;
    private double eventTime;
    private Double availableAt;
    private String sourceVersion = "v1";
    private String entityId;
    private String payload = "{}";
    private String context = "{}";
    private String candidateFeatures;
    private String observationId;

    private HistoryRow(String eventId) {
      this.eventId = eventId;
    }

    HistoryRow type(String value) {
      eventType = value;
      return this;
    }

    HistoryRow user(String value) {
      userId = value;
      return this;
    }

    HistoryRow product(String value) {
      productId = value;
      return this;
    }

    HistoryRow action(String value) {
      action = value;
      return this;
    }

    HistoryRow at(double value) {
      eventTime = value;
      return this;
    }

    HistoryRow available(double value) {
      availableAt = value;
      return this;
    }

    HistoryRow version(String value) {
      sourceVersion = value;
      return this;
    }

    HistoryRow entity(String value) {
      entityId = value;
      return this;
    }

    HistoryRow payload(String value) {
      payload = value;
      return this;
    }

    HistoryRow context(String value) {
      context = value;
      return this;
    }

    HistoryRow candidateFeatures(String value) {
      candidateFeatures = value;
      return this;
    }

    HistoryRow observation(String value) {
      observationId = value;
      return this;
    }

    String values() {
      return "("
          + String.join(
              ",",
              text(eventId),
              text(eventType),
              text(userId),
              text(productId),
              text(action),
              number(eventTime),
              number(availableAt == null ? eventTime : availableAt),
              text(sourceVersion),
              text(entityId),
              text(payload),
              text(context),
              text(candidateFeatures),
              text(observationId == null ? eventId : observationId),
              "DATE '2023-11-14'")
          + ")";
    }

    private static String text(String value) {
      return value == null
          ? "CAST(NULL AS STRING)"
          : "CAST('" + value.replace("'", "''") + "' AS STRING)";
    }

    private static String number(double value) {
      return "CAST(" + value + " AS DOUBLE)";
    }
  }
}
