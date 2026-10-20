package com.videocommerce.flink;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.api.TableResult;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;
import org.junit.jupiter.api.Test;

/** Executes the real ranking PIT SQL on in-memory history tables to pin label semantics. */
class PointInTimeAttributionSqlTest {
  private static final String NS = "video_commerce";
  private static final String VERSION = "ranking_ltr_v1";
  private static final double CUTOFF = 2_000_000.0;
  private static final long MAX_AGE = 604_800L;
  private static final String SCORES =
      "{\"collaborative_score\":0.1,\"combined_score\":0.4,"
          + "\"content_similarity_score\":0.2,\"popularity_score\":0.3}";
  private static final List<String> COLUMNS =
      List.of(
          "observation_id", "impression_id", "user_id", "product_id", "attributed_action",
          "attributed_click", "attributed_purchase", "attributed_value",
          "online_feature_bundle_hash", "label_definition_version");

  @Test
  void threeImpressionsOfOneProductAndOnePurchaseYieldExactlyOnePositive() throws Exception {
    TableEnvironment tables = environment();
    register(
        tables,
        List.of(
            observation("imp-1", "u1", "p1", 1_000.0, "src-1"),
            // An append-only replay of the same displayed item must not duplicate the row.
            observation("imp-1", "u1", "p1", 1_000.0, "src-1-replay"),
            observation("imp-2", "u1", "p1", 2_000.0, "src-2"),
            observation("imp-3", "u1", "p1", 3_000.0, "src-3"),
            // Never displayed: no bundle hash, but clicked later through the same slate.
            rejected("imp-1", "u1", "p9", 1_000.0, "src-r")),
        List.of(view("imp-1:p1", 1_005.0), view("imp-2:p1", 2_005.0), view("imp-3:p1", 3_005.0)),
        List.of(
            interaction("click", "u1", "p1", "imp-3", 3_100.0, 3_100.0, "i-1"),
            interaction("purchase", "u1", "p1", "imp-3", 3_500.0, 3_500.0, "i-2"),
            interaction("click", "u1", "p9", "imp-1", 1_100.0, 1_100.0, "i-3"),
            // Organic purchase with no impression context is attributed to nothing.
            interaction("purchase", "u1", "p1", null, 3_600.0, 3_600.0, "i-4")),
        List.of(userFeature("u1", 500.0)));

    Map<String, Map<String, Object>> rows = trainingRows(tables);

    assertEquals(List.of("imp-1:p1", "imp-2:p1", "imp-3:p1"), new ArrayList<>(rows.keySet()));
    assertEquals(1, rows.values().stream().filter(row -> intValue(row, "attributed_click") == 1).count());
    assertEquals(1, rows.values().stream().filter(row -> intValue(row, "attributed_purchase") == 1).count());
    assertEquals("purchase", rows.get("imp-3:p1").get("attributed_action"));
    assertEquals(1, intValue(rows.get("imp-3:p1"), "attributed_purchase"));
    assertEquals("view", rows.get("imp-1:p1").get("attributed_action"));
    assertEquals("view", rows.get("imp-2:p1").get("attributed_action"));
    for (Map<String, Object> row : rows.values()) {
      assertEquals(64, String.valueOf(row.get("online_feature_bundle_hash")).length());
      assertEquals(
          PointInTimeFeatureJoinJob.LABEL_DEFINITION_VERSION, row.get("label_definition_version"));
    }
  }

  @Test
  void feedbackLowerBoundUsesServerAvailabilityNotClientClock() throws Exception {
    TableEnvironment tables = environment();
    register(
        tables,
        List.of(
            observation("imp-1", "u1", "p1", 1_000.0, "src-1"),
            observation("imp-2", "u1", "p2", 1_000.0, "src-2")),
        // The client clock runs 10 seconds behind the server for both feedback events.
        List.of(view("imp-1:p1", 990.0, 1_002.0), view("imp-2:p2", 990.0, 1_002.0)),
        List.of(
            interaction("click", "u1", "p1", "imp-1", 995.0, 1_010.0, "i-1"),
            // Received before the impression was served: cannot be caused by it.
            interaction("click", "u1", "p2", "imp-2", 995.0, 999.0, "i-2")),
        List.of(userFeature("u1", 500.0)));

    Map<String, Map<String, Object>> rows = trainingRows(tables);

    assertEquals(1, intValue(rows.get("imp-1:p1"), "attributed_click"));
    assertEquals(0, intValue(rows.get("imp-2:p2"), "attributed_click"));
  }

  @Test
  void userFeaturesOlderThanMaxAgeAreQuarantinedNotJoined() throws Exception {
    TableEnvironment tables = environment();
    double observedAt = 1_000_000.0;
    register(
        tables,
        List.of(
            observation("imp-fresh", "fresh", "p1", observedAt, "src-1"),
            observation("imp-stale", "stale", "p1", observedAt, "src-2")),
        List.of(view("imp-fresh:p1", observedAt + 1), view("imp-stale:p1", observedAt + 1)),
        List.of(),
        List.of(
            userFeature("fresh", observedAt - MAX_AGE + 1),
            userFeature("stale", observedAt - MAX_AGE - 1)));

    Map<String, Map<String, Object>> rows = trainingRows(tables);
    assertEquals(List.of("imp-fresh:p1"), new ArrayList<>(rows.keySet()));

    List<Row> quarantined =
        collect(
            tables,
            PointInTimeFeatureJoinJob.buildQuarantineSelectSql(
                NS, VERSION, 168, 1, CUTOFF, "run-1", MAX_AGE));
    assertEquals(1, quarantined.size());
    assertEquals("imp-stale:p1", quarantined.get(0).getField(1));
    assertEquals("missing_user_feature", quarantined.get(0).getField(2));
    assertTrue(quarantined.stream().noneMatch(row -> "imp-fresh:p1".equals(row.getField(1))));
  }

  private static TableEnvironment environment() {
    TableEnvironment tables =
        TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
    tables.getConfig().set("parallelism.default", "1");
    tables.createTemporarySystemFunction(
        "pit_feature_bundle_hash", PointInTimeFeatureJoinJob.PitFeatureBundleHash.class);
    tables.executeSql("CREATE DATABASE IF NOT EXISTS `" + NS + "`");
    return tables;
  }

  private static Map<String, Map<String, Object>> trainingRows(TableEnvironment tables)
      throws Exception {
    String sql =
        PointInTimeFeatureJoinJob.buildPointInTimeSelectSql(
            NS, VERSION, 168, 1, CUTOFF, "run-1", COLUMNS, MAX_AGE);
    Map<String, Map<String, Object>> rows = new LinkedHashMap<>();
    List<Row> collected = collect(tables, sql);
    collected.sort((left, right) -> String.valueOf(left.getField(0)).compareTo(String.valueOf(right.getField(0))));
    for (Row row : collected) {
      Map<String, Object> values = new LinkedHashMap<>();
      for (int index = 0; index < COLUMNS.size(); index++) {
        values.put(COLUMNS.get(index), row.getField(index));
      }
      Object previous = rows.put(String.valueOf(row.getField(0)), values);
      assertEquals(null, previous, "duplicate observation " + row.getField(0));
    }
    return rows;
  }

  private static List<Row> collect(TableEnvironment tables, String sql) throws Exception {
    TableResult result = tables.executeSql(sql);
    List<Row> rows = new ArrayList<>();
    try (CloseableIterator<Row> iterator = result.collect()) {
      iterator.forEachRemaining(rows::add);
    }
    return rows;
  }

  private static int intValue(Map<String, Object> row, String column) {
    return ((Number) row.get(column)).intValue();
  }

  private static void register(
      TableEnvironment tables,
      List<String> observations,
      List<String> views,
      List<String> interactions,
      List<String> userFeatures) {
    view(
        tables,
        "ranking_observations",
        "observation_id,user_id,product_id,event_type,feature_definition_version,"
            + "event_time_epoch,available_at_epoch,source_event_id,canonical_payload_json,"
            + "context_json,candidate_features_json,item_snapshot_json,feature_bundle_hash",
        observations);
    view(
        tables,
        "recommendation_view_history",
        "observation_id,event_time_epoch,available_at_epoch",
        views);
    view(
        tables,
        "interaction_history",
        "user_id,product_id,action,context_json,event_time_epoch,available_at_epoch,"
            + "source_event_id",
        interactions);
    view(
        tables,
        "user_feature_history",
        "user_id,feature_definition_version,event_time_epoch,available_at_epoch,"
            + "source_event_id,canonical_payload_json",
        userFeatures);
    view(
        tables,
        "item_feature_history",
        "product_id,feature_definition_version,event_time_epoch,available_at_epoch,"
            + "source_event_id,canonical_payload_json",
        List.of());
  }

  private static void view(TableEnvironment tables, String name, String columns, List<String> rows) {
    String body;
    if (rows.isEmpty()) {
      // A typed empty relation: one placeholder row filtered out.
      int arity = columns.split(",").length;
      List<String> nulls = new ArrayList<>();
      for (int index = 0; index < arity; index++) {
        nulls.add(isEpoch(columns, index) ? dbl(null) : str(null));
      }
      body = "SELECT * FROM (VALUES (" + String.join(",", nulls) + ")) AS t(" + columns + ") WHERE 1=0";
    } else {
      body = "SELECT * FROM (VALUES " + String.join(",", rows) + ") AS t(" + columns + ")";
    }
    tables.executeSql("CREATE TEMPORARY VIEW `" + NS + "`.`" + name + "` AS " + body);
  }

  private static boolean isEpoch(String columns, int index) {
    return columns.split(",")[index].endsWith("_epoch");
  }

  private static String observation(
      String impressionId, String userId, String productId, double eventTime, String sourceId) {
    String payload =
        "{\"impression_id\":\"" + impressionId + "\",\"item_snapshot_complete\":true,"
            + "\"product_id\":\"" + productId + "\",\"user_id\":\"" + userId + "\"}";
    return row(
        str(impressionId + ":" + productId), str(userId), str(productId),
        str("ranking_observation"), str(VERSION), dbl(eventTime), dbl(eventTime),
        str(sourceId), str(payload), str("{}"), str(SCORES), str("{\"price\":9.0}"),
        str("a".repeat(64)));
  }

  private static String rejected(
      String impressionId, String userId, String productId, double eventTime, String sourceId) {
    String payload =
        "{\"impression_id\":\"" + impressionId + "\",\"product_id\":\"" + productId + "\"}";
    return row(
        str(impressionId + ":" + productId), str(userId), str(productId),
        str("ranking_rejected_observation"), str(VERSION), dbl(eventTime), dbl(eventTime),
        str(sourceId), str(payload), str("{}"), str(SCORES), str(null), str(null));
  }

  private static String view(String observationId, double availableAt) {
    return view(observationId, availableAt, availableAt);
  }

  private static String view(String observationId, double eventTime, double availableAt) {
    return row(str(observationId), dbl(eventTime), dbl(availableAt));
  }

  private static String interaction(
      String action,
      String userId,
      String productId,
      String impressionId,
      double eventTime,
      double availableAt,
      String sourceId) {
    String context =
        impressionId == null
            ? "{\"price\":20.0}"
            : "{\"impression_id\":\"" + impressionId + "\",\"price\":20.0}";
    return row(
        str(userId), str(productId), str(action), str(context), dbl(eventTime),
        dbl(availableAt), str(sourceId));
  }

  private static String userFeature(String userId, double eventTime) {
    return row(
        str(userId), str(VERSION), dbl(eventTime), dbl(eventTime), str("uf-" + userId + eventTime),
        str("{\"total_interactions\":3}"));
  }

  private static String row(String... values) {
    return "(" + String.join(",", values) + ")";
  }

  private static String str(String value) {
    return value == null ? "CAST(NULL AS STRING)" : "CAST('" + value.replace("'", "''") + "' AS STRING)";
  }

  private static String dbl(Double value) {
    return value == null ? "CAST(NULL AS DOUBLE)" : "CAST(" + value + " AS DOUBLE)";
  }
}
