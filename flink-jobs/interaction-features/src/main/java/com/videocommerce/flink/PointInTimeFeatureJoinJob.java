package com.videocommerce.flink;

import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;

/**
 * Batch materialization of ranking LTR examples from versioned Iceberg feature tables.
 *
 * <p>The source tables are populated by the streaming interaction/catalog materializers:
 * ranking_observations, user_feature_history, item_feature_history, and interaction_history.
 * This job deliberately refuses to join a version created after the observation's event time.
 */
public final class PointInTimeFeatureJoinJob {
  private PointInTimeFeatureJoinJob() {}

  public static void main(String[] args) throws Exception {
    String catalog = requiredEnv("FEATURE_LAKE_CATALOG_NAME", "feature_catalog");
    String catalogUri = requiredEnv("FEATURE_LAKE_CATALOG_URI", null);
    String warehouseUri = requiredEnv("FEATURE_LAKE_WAREHOUSE_URI", null);
    String namespace = requiredEnv("FEATURE_LAKE_NAMESPACE", "video_commerce");
    String featureDefinitionVersion = requiredEnv(
        "FEATURE_LAKE_FEATURE_DEFINITION_VERSION", "ranking_ltr_v1");
    int attributionWindowHours = Integer.parseInt(requiredEnv(
        "FEATURE_LAKE_ATTRIBUTION_WINDOW_HOURS", "168"));

    TableEnvironment tableEnvironment = TableEnvironment.create(
        EnvironmentSettings.newInstance().inBatchMode().build());
    tableEnvironment.executeSql(buildRestCatalogSql(catalog, catalogUri, warehouseUri));
    tableEnvironment.executeSql("USE CATALOG `" + catalog + "`");
    tableEnvironment.executeSql("CREATE DATABASE IF NOT EXISTS `" + namespace + "`");
    tableEnvironment.executeSql(buildPointInTimeInsertSql(
        namespace, featureDefinitionVersion, attributionWindowHours)).await();
  }

  static String buildRestCatalogSql(String catalog, String catalogUri, String warehouseUri) {
    return String.format(
        "CREATE CATALOG `%s` WITH ("
            + "'type' = 'iceberg', "
            + "'catalog-type' = 'rest', "
            + "'uri' = '%s', "
            + "'warehouse' = '%s')",
        catalog, catalogUri, warehouseUri);
  }

  static String buildPointInTimeInsertSql(
      String namespace, String featureDefinitionVersion, int attributionWindowHours) {
    if (attributionWindowHours <= 0) {
      throw new IllegalArgumentException("attributionWindowHours must be positive");
    }
    String version = featureDefinitionVersion.replace("'", "''");
    return String.format(
        "INSERT OVERWRITE `%s`.`ranking_training_pit`\n"
            + "WITH user_ranked AS (\n"
            + "  SELECT o.observation_id, u.feature_json,\n"
            + "    ROW_NUMBER() OVER (PARTITION BY o.observation_id "
            + "ORDER BY u.event_time DESC, u.available_at DESC) AS feature_rank\n"
            + "  FROM `%s`.`ranking_observations` o\n"
            + "  LEFT JOIN `%s`.`user_feature_history` u ON u.user_id = o.user_id\n"
            + "    AND u.feature_definition_version = '%s'\n"
            + "    AND u.event_time <= o.event_time\n"
            + "    AND u.available_at <= o.event_time\n"
            + "), item_ranked AS (\n"
            + "  SELECT o.observation_id, i.feature_json,\n"
            + "    ROW_NUMBER() OVER (PARTITION BY o.observation_id "
            + "ORDER BY i.event_time DESC, i.available_at DESC) AS feature_rank\n"
            + "  FROM `%s`.`ranking_observations` o\n"
            + "  LEFT JOIN `%s`.`item_feature_history` i ON i.product_id = o.product_id\n"
            + "    AND i.feature_definition_version = '%s'\n"
            + "    AND i.event_time <= o.event_time\n"
            + "    AND i.available_at <= o.event_time\n"
            + "), feedback AS (\n"
            + "  SELECT o.observation_id,\n"
            + "    MAX(CASE WHEN i.action IN ('click', 'add_to_cart', 'purchase') THEN 1 ELSE 0 END) AS attributed_click,\n"
            + "    MAX(CASE WHEN i.action = 'purchase' THEN 1 ELSE 0 END) AS attributed_purchase\n"
            + "  FROM `%s`.`ranking_observations` o\n"
            + "  LEFT JOIN `%s`.`interaction_history` i ON i.user_id = o.user_id\n"
            + "    AND i.product_id = o.product_id\n"
            + "    AND i.event_time >= o.event_time\n"
            + "    AND i.event_time <= o.event_time + INTERVAL '%d' HOUR\n"
            + "    AND i.available_at <= o.event_time + INTERVAL '%d' HOUR\n"
            + "  GROUP BY o.observation_id\n"
            + ")\n"
            + "SELECT o.observation_id, o.user_id, o.product_id, o.event_time AS as_of_ts,\n"
            + "  u.feature_json AS user_features_json, it.feature_json AS product_metadata_json,\n"
            + "  o.context_json, f.attributed_click, f.attributed_purchase, '%s' AS feature_definition_version\n"
            + "FROM `%s`.`ranking_observations` o\n"
            + "LEFT JOIN user_ranked u ON u.observation_id = o.observation_id AND u.feature_rank = 1\n"
            + "LEFT JOIN item_ranked it ON it.observation_id = o.observation_id AND it.feature_rank = 1\n"
            + "LEFT JOIN feedback f ON f.observation_id = o.observation_id\n"
            + "WHERE o.feature_definition_version = '%s'",
        namespace,
        namespace, namespace, version,
        namespace, namespace, version,
        namespace, namespace, attributionWindowHours, attributionWindowHours,
        version, namespace, version);
  }

  private static String requiredEnv(String name, String defaultValue) {
    String value = System.getenv(name);
    if (value == null || value.trim().isEmpty()) {
      value = defaultValue;
    }
    if (value == null || value.trim().isEmpty()) {
      throw new IllegalArgumentException(name + " must be configured");
    }
    return value.trim();
  }
}
