package com.videocommerce.flink;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.LinkedHashMap;
import java.util.Map;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;
import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.functions.ScalarFunction;

/** Batch PIT join, Iceberg snapshot materialization, and immutable Parquet shard export. */
public final class PointInTimeFeatureJoinJob {
  private static final ObjectMapper JSON = new ObjectMapper();
  private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {};

  private PointInTimeFeatureJoinJob() {}

  public static void main(String[] args) throws Exception {
    String catalog = requiredEnv("FEATURE_LAKE_CATALOG_NAME", "feature_catalog");
    String catalogUri = requiredEnv("FEATURE_LAKE_CATALOG_URI", null);
    String warehouseUri = requiredEnv("FEATURE_LAKE_WAREHOUSE_URI", null);
    String s3Endpoint = requiredEnv("FEATURE_LAKE_S3_ENDPOINT", null);
    String namespace = requiredEnv("FEATURE_LAKE_NAMESPACE", "video_commerce");
    String featureDefinitionVersion =
        requiredEnv("FEATURE_LAKE_FEATURE_DEFINITION_VERSION", "ranking_ltr_v1");
    int attributionWindowHours =
        Integer.parseInt(requiredEnv("FEATURE_LAKE_ATTRIBUTION_WINDOW_HOURS", "168"));
    int allowedLatenessHours =
        Integer.parseInt(requiredEnv("FEATURE_LAKE_ALLOWED_LATENESS_HOURS", "1"));
    double materializationCutoff =
        Double.parseDouble(requiredEnv("FEATURE_LAKE_MATERIALIZATION_CUTOFF", null));
    String runId = requiredEnv("FEATURE_LAKE_MATERIALIZATION_RUN_ID", null);
    String exportPrefix = requiredEnv("FEATURE_LAKE_PIT_EXPORT_URI", null);

    TableEnvironment tables =
        TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
    tables.createTemporarySystemFunction("pit_feature_bundle_hash", PitFeatureBundleHash.class);
    tables.executeSql(buildRestCatalogSql(catalog, catalogUri, warehouseUri, s3Endpoint));
    tables.executeSql("USE CATALOG `" + catalog + "`");
    tables.executeSql("CREATE DATABASE IF NOT EXISTS `" + namespace + "`");
    tables.executeSql(buildTrainingTableSql(namespace));
    tables.executeSql(buildQuarantineTableSql(namespace));
    rejectExistingRun(tables, namespace, runId);
    tables
        .executeSql(
            buildQuarantineInsertSql(
                namespace,
                featureDefinitionVersion,
                attributionWindowHours,
                allowedLatenessHours,
                materializationCutoff,
                runId))
        .await();
    tables
        .executeSql(
            buildPointInTimeInsertSql(
                namespace,
                featureDefinitionVersion,
                attributionWindowHours,
                allowedLatenessHours,
                materializationCutoff,
                runId))
        .await();
    tables.executeSql(buildParquetExportTableSql(exportPrefix, runId));
    tables.executeSql(buildParquetExportInsertSql(namespace, runId)).await();
  }

  static String buildRestCatalogSql(
      String catalog, String catalogUri, String warehouseUri, String s3Endpoint) {
    return String.format(
        "CREATE CATALOG `%s` WITH ('type'='iceberg','catalog-type'='rest','uri'='%s',"
            + "'warehouse'='%s','io-impl'='org.apache.iceberg.aws.s3.S3FileIO',"
            + "'s3.endpoint'='%s','s3.path-style-access'='true')",
        identifier(catalog),
        literal(catalogUri),
        literal(warehouseUri),
        literal(s3Endpoint));
  }

  static String buildTrainingTableSql(String namespace) {
    return String.format(
        "CREATE TABLE IF NOT EXISTS `%s`.`ranking_training_pit` ("
            + "materialization_run_id STRING,observation_id STRING,impression_id STRING,"
            + "user_id STRING,product_id STRING,"
            + "action STRING,as_of_ts DOUBLE,user_features_json STRING,product_metadata_json STRING,"
            + "context_json STRING,candidate_features_json STRING,online_feature_bundle_hash STRING,"
            + "feature_bundle_hash STRING,"
            + "attributed_click INT,attributed_purchase INT,feature_definition_version STRING,"
            + "materialized_at TIMESTAMP_LTZ(3),materialization_date DATE) "
            + "PARTITIONED BY (materialization_date) WITH ('format-version'='2')",
        identifier(namespace));
  }

  static String buildQuarantineTableSql(String namespace) {
    return String.format(
        "CREATE TABLE IF NOT EXISTS `%s`.`ranking_training_pit_quarantine` ("
            + "materialization_run_id STRING,observation_id STRING,reason STRING,"
            + "feature_definition_version STRING,as_of_ts DOUBLE,quarantined_at TIMESTAMP_LTZ(3),"
            + "materialization_date DATE) PARTITIONED BY (materialization_date) "
            + "WITH ('format-version'='2')",
        identifier(namespace));
  }

  static String buildExistingRunSql(String namespace, String runId) {
    return String.format(
        "SELECT SUM(row_count) FROM ("
            + "SELECT COUNT(*) row_count FROM `%s`.`ranking_training_pit` "
            + "WHERE materialization_run_id='%s' UNION ALL "
            + "SELECT COUNT(*) row_count FROM `%s`.`ranking_training_pit_quarantine` "
            + "WHERE materialization_run_id='%s')",
        identifier(namespace), literal(runId), identifier(namespace), literal(runId));
  }

  private static void rejectExistingRun(
      TableEnvironment tables, String namespace, String runId) throws Exception {
    try (CloseableIterator<Row> rows =
        tables.executeSql(buildExistingRunSql(namespace, runId)).collect()) {
      long existing = rows.hasNext() ? ((Number) rows.next().getField(0)).longValue() : 0L;
      if (existing > 0L) {
        throw new IllegalStateException(
            "materialization_run_id already exists and is immutable: " + runId);
      }
    }
  }

  static String buildPointInTimeInsertSql(
      String namespace, String version, int attributionWindowHours) {
    return buildPointInTimeInsertSql(
        namespace, version, attributionWindowHours, 1, 1_700_000_000.0, "test-run");
  }

  static String buildPointInTimeInsertSql(
      String namespace,
      String featureDefinitionVersion,
      int attributionWindowHours,
      int allowedLatenessHours,
      double materializationCutoff,
      String runId) {
    validateWindows(attributionWindowHours, allowedLatenessHours);
    String version = literal(featureDefinitionVersion);
    String run = literal(runId);
    String eligible =
        eligibleCtes(
            namespace,
            version,
            attributionWindowHours,
            allowedLatenessHours,
            materializationCutoff);
    return String.format(
        "INSERT INTO `%s`.`ranking_training_pit`\n%s\n"
            + "SELECT '%s',o.observation_id,JSON_VALUE(o.canonical_payload_json,'$.impression_id'),"
            + "o.user_id,o.product_id,"
            + "CASE WHEN f.attributed_purchase=1 THEN 'purchase' "
            + "WHEN f.attributed_click=1 THEN 'click' ELSE 'view' END,o.event_time_epoch,"
            + "u.canonical_payload_json,"
            + "CASE WHEN JSON_VALUE(o.canonical_payload_json,'$.item_snapshot_complete')='true' "
            + "THEN o.item_snapshot_json ELSE it.canonical_payload_json END,"
            + "o.context_json,o.candidate_features_json,o.feature_bundle_hash,"
            + "pit_feature_bundle_hash(o.event_time_epoch,o.user_id,o.product_id,"
            + "u.canonical_payload_json,CASE WHEN "
            + "JSON_VALUE(o.canonical_payload_json,'$.item_snapshot_complete')='true' "
            + "THEN o.item_snapshot_json ELSE it.canonical_payload_json END,"
            + "o.context_json,o.candidate_features_json,'%s'),"
            + "COALESCE(f.attributed_click,0),COALESCE(f.attributed_purchase,0),'%s',"
            + "CURRENT_TIMESTAMP,CURRENT_DATE\n"
            + "FROM eligible_observations o\n"
            + "LEFT JOIN user_ranked u ON u.observation_id=o.observation_id AND u.feature_rank=1\n"
            + "LEFT JOIN item_ranked it ON it.observation_id=o.observation_id AND it.feature_rank=1\n"
            + "LEFT JOIN feedback f ON f.observation_id=o.observation_id\n"
            + "WHERE u.canonical_payload_json IS NOT NULL AND ("
            + "JSON_VALUE(o.canonical_payload_json,'$.item_snapshot_complete')='true' "
            + "OR it.canonical_payload_json IS NOT NULL)",
        identifier(namespace), eligible, run, version, version);
  }

  static String buildQuarantineInsertSql(
      String namespace,
      String featureDefinitionVersion,
      int attributionWindowHours,
      int allowedLatenessHours,
      double materializationCutoff,
      String runId) {
    validateWindows(attributionWindowHours, allowedLatenessHours);
    String version = literal(featureDefinitionVersion);
    String eligible =
        eligibleCtes(
            namespace,
            version,
            attributionWindowHours,
            allowedLatenessHours,
            materializationCutoff);
    return String.format(
        "INSERT INTO `%s`.`ranking_training_pit_quarantine`\n%s\n"
            + "SELECT '%s',o.observation_id,CASE "
            + "WHEN u.canonical_payload_json IS NULL THEN 'missing_user_feature' "
            + "ELSE 'missing_complete_v1_item_snapshot' END,'%s',o.event_time_epoch,"
            + "CURRENT_TIMESTAMP,CURRENT_DATE FROM eligible_observations o "
            + "LEFT JOIN user_ranked u ON u.observation_id=o.observation_id AND u.feature_rank=1 "
            + "LEFT JOIN item_ranked it ON it.observation_id=o.observation_id AND it.feature_rank=1 "
            + "WHERE u.canonical_payload_json IS NULL OR ("
            + "COALESCE(JSON_VALUE(o.canonical_payload_json,'$.item_snapshot_complete'),'false')<>'true' "
            + "AND it.canonical_payload_json IS NULL)",
        identifier(namespace), eligible, literal(runId), version);
  }

  private static String eligibleCtes(
      String namespace,
      String version,
      int attributionHours,
      int latenessHours,
      double materializationCutoff) {
    String ns = identifier(namespace);
    long attributionSeconds = attributionHours * 3600L;
    long finalizedSeconds = (attributionHours + latenessHours) * 3600L;
    double finalizedCutoff = materializationCutoff - finalizedSeconds;
    return String.format(
        "WITH eligible_observations AS (\n"
            + " SELECT * FROM `%s`.`ranking_observations` o WHERE o.feature_definition_version='%s' "
            + "AND o.event_time_epoch <= CAST(%.3f AS DOUBLE)\n"
            + "),user_ranked AS (\n"
            + " SELECT o.observation_id,u.canonical_payload_json,ROW_NUMBER() OVER ("
            + "PARTITION BY o.observation_id ORDER BY u.event_time_epoch DESC,"
            + "u.available_at_epoch DESC,u.source_event_id DESC) feature_rank "
            + "FROM eligible_observations o LEFT JOIN `%s`.`user_feature_history` u "
            + "ON u.user_id=o.user_id AND u.feature_definition_version='%s' "
            + "AND u.event_time_epoch<=o.event_time_epoch AND u.available_at_epoch<=o.event_time_epoch\n"
            + "),item_ranked AS (\n"
            + " SELECT o.observation_id,i.canonical_payload_json,ROW_NUMBER() OVER ("
            + "PARTITION BY o.observation_id ORDER BY i.event_time_epoch DESC,"
            + "i.available_at_epoch DESC,i.source_event_id DESC) feature_rank "
            + "FROM eligible_observations o LEFT JOIN `%s`.`item_feature_history` i "
            + "ON i.product_id=o.product_id AND i.feature_definition_version='%s' "
            + "AND i.event_time_epoch<=o.event_time_epoch AND i.available_at_epoch<=o.event_time_epoch\n"
            + "),feedback AS (\n"
            + " SELECT o.observation_id,MAX(CASE WHEN i.action IN ('click','add_to_cart','purchase') "
            + "THEN 1 ELSE 0 END) attributed_click,MAX(CASE WHEN i.action='purchase' THEN 1 ELSE 0 END) "
            + "attributed_purchase FROM eligible_observations o LEFT JOIN `%s`.`interaction_history` i "
            + "ON i.user_id=o.user_id AND i.product_id=o.product_id "
            + "AND i.event_time_epoch>=o.event_time_epoch "
            + "AND i.event_time_epoch<=o.event_time_epoch+%d "
            + "AND i.available_at_epoch<=CAST(%.3f AS DOUBLE) GROUP BY o.observation_id\n"
            + ")",
        ns,
        version,
        finalizedCutoff,
        ns,
        version,
        ns,
        version,
        ns,
        attributionSeconds,
        materializationCutoff);
  }

  static String buildParquetExportTableSql(String exportPrefix, String runId) {
    return String.format(
        "CREATE TEMPORARY TABLE pit_parquet_export (observation_id STRING,impression_id STRING,"
            + "user_id STRING,"
            + "product_id STRING,action STRING,as_of_ts DOUBLE,user_features_json STRING,"
            + "product_metadata_json STRING,context_json STRING,candidate_features_json STRING,"
            + "online_feature_bundle_hash STRING,feature_bundle_hash STRING,"
            + "attributed_click INT,attributed_purchase INT,"
            + "feature_definition_version STRING) WITH ('connector'='filesystem','path'='%s',"
            + "'format'='parquet','sink.rolling-policy.file-size'='128mb')",
        literal(exportPrefix) + "/runs/" + literal(runId) + "/shards");
  }

  static String buildParquetExportInsertSql(String namespace, String runId) {
    return String.format(
        "INSERT INTO pit_parquet_export SELECT observation_id,impression_id,user_id,product_id,"
            + "action,as_of_ts,"
            + "user_features_json,product_metadata_json,context_json,candidate_features_json,"
            + "online_feature_bundle_hash,feature_bundle_hash,attributed_click,attributed_purchase,"
            + "feature_definition_version "
            + "FROM `%s`.`ranking_training_pit` WHERE materialization_run_id='%s'",
        identifier(namespace), literal(runId));
  }

  private static void validateWindows(int attributionHours, int latenessHours) {
    if (attributionHours <= 0 || latenessHours < 0) {
      throw new IllegalArgumentException("attribution must be positive and lateness non-negative");
    }
  }

  /** Canonical hash of the final offline bundle after PIT feature substitution. */
  public static final class PitFeatureBundleHash extends ScalarFunction {
    public String eval(
        Double asOfTs,
        String userId,
        String productId,
        String userFeaturesJson,
        String productMetadataJson,
        String contextJson,
        String candidateFeaturesJson,
        String featureDefinitionVersion)
        throws Exception {
      Map<String, Object> bundle = new LinkedHashMap<>();
      bundle.put("as_of_ts", asOfTs);
      bundle.put("candidate_features", parseMap(candidateFeaturesJson));
      bundle.put("context", parseMap(contextJson));
      bundle.put("feature_definition_version", featureDefinitionVersion);
      bundle.put("product_id", productId);
      bundle.put("product_metadata", parseMap(productMetadataJson));
      bundle.put("user_features", parseMap(userFeaturesJson));
      bundle.put("user_id", userId);
      return FeatureHistoryContract.payloadHash(bundle);
    }

    private static Map<String, Object> parseMap(String value) throws Exception {
      if (value == null || value.trim().isEmpty()) {
        throw new IllegalArgumentException("PIT feature bundle JSON must not be blank");
      }
      return JSON.readValue(value, MAP_TYPE);
    }
  }

  private static String identifier(String value) {
    String normalized = value == null ? "" : value.trim();
    if (!normalized.matches("[A-Za-z_][A-Za-z0-9_]*")) {
      throw new IllegalArgumentException("invalid SQL identifier " + value);
    }
    return normalized;
  }

  private static String literal(String value) {
    return value.replace("'", "''");
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
