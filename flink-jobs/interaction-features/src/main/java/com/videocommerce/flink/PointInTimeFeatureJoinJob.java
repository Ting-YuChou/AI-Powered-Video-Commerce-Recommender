package com.videocommerce.flink;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;
import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.functions.ScalarFunction;

/** Batch PIT join, Iceberg snapshot materialization, and immutable Parquet shard export. */
public final class PointInTimeFeatureJoinJob {
  static final String LABEL_DEFINITION_VERSION = "ranking_labels_v1";
  private static final ObjectMapper JSON = new ObjectMapper();
  private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {};

  private PointInTimeFeatureJoinJob() {}

  public static void main(String[] args) throws Exception {
    JobConfig config = JobConfig.resolve(args, System.getenv());

    TableEnvironment tables =
        TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
    tables.createTemporarySystemFunction("pit_feature_bundle_hash", PitFeatureBundleHash.class);
    tables.executeSql(
        buildRestCatalogSql(
            config.catalog, config.catalogUri, config.warehouseUri, config.s3Endpoint));
    tables.executeSql("USE CATALOG `" + config.catalog + "`");
    tables.executeSql("CREATE DATABASE IF NOT EXISTS `" + config.namespace + "`");
    tables.executeSql(buildTrainingTableSql(config.namespace));
    evolveTrainingTable(tables, config.namespace);
    List<String> trainingColumns =
        tables.from("`" + identifier(config.namespace) + "`.`ranking_training_pit`")
            .getResolvedSchema()
            .getColumnNames();
    tables.executeSql(buildQuarantineTableSql(config.namespace));
    if (existingRunCount(tables, config.namespace, "ranking_training_pit", config.runId) == 0L) {
      tables
          .executeSql(
              buildPointInTimeInsertSql(
                  config.namespace,
                  config.featureDefinitionVersion,
                  config.attributionWindowHours,
                  config.allowedLatenessHours,
                  config.materializationCutoff,
                  config.runId,
                  trainingColumns))
          .await();
    }
    if (existingRunCount(
            tables, config.namespace, "ranking_training_pit_quarantine", config.runId)
        == 0L) {
      tables
          .executeSql(
              buildQuarantineInsertSql(
                  config.namespace,
                  config.featureDefinitionVersion,
                  config.attributionWindowHours,
                  config.allowedLatenessHours,
                  config.materializationCutoff,
                  config.runId))
          .await();
    }
    tables.executeSql(
        buildParquetExportTableSql(config.exportPrefix, config.runId, config.exportAttempt));
    tables.executeSql(buildParquetExportInsertSql(config.namespace, config.runId)).await();
  }

  static final class JobConfig {
    private static final Set<String> SUPPORTED_OPTIONS = Set.of(
        "--catalog-name",
        "--catalog-uri",
        "--warehouse-uri",
        "--s3-endpoint",
        "--feature-definition-version",
        "--attribution-window-hours",
        "--allowed-lateness-hours",
        "--materialization-run-id",
        "--materialization-cutoff",
        "--namespace",
        "--export-uri",
        "--export-attempt");
    final String catalog;
    final String catalogUri;
    final String warehouseUri;
    final String s3Endpoint;
    final String namespace;
    final String featureDefinitionVersion;
    final int attributionWindowHours;
    final int allowedLatenessHours;
    final double materializationCutoff;
    final String runId;
    final String exportPrefix;
    final int exportAttempt;

    private JobConfig(
        String catalog,
        String catalogUri,
        String warehouseUri,
        String s3Endpoint,
        String namespace,
        String featureDefinitionVersion,
        int attributionWindowHours,
        int allowedLatenessHours,
        double materializationCutoff,
        String runId,
        String exportPrefix,
        int exportAttempt) {
      this.catalog = catalog;
      this.catalogUri = catalogUri;
      this.warehouseUri = warehouseUri;
      this.s3Endpoint = s3Endpoint;
      this.namespace = namespace;
      this.featureDefinitionVersion = featureDefinitionVersion;
      this.attributionWindowHours = attributionWindowHours;
      this.allowedLatenessHours = allowedLatenessHours;
      this.materializationCutoff = materializationCutoff;
      this.runId = runId;
      this.exportPrefix = exportPrefix;
      this.exportAttempt = exportAttempt;
    }

    static JobConfig resolve(String[] args, Map<String, String> environment) {
      Map<String, String> options = new LinkedHashMap<>();
      for (int index = 0; index < args.length; index += 2) {
        if (index + 1 >= args.length || !args[index].startsWith("--")) {
          throw new IllegalArgumentException("PIT job arguments must be --name value pairs");
        }
        if (!SUPPORTED_OPTIONS.contains(args[index])) {
          throw new IllegalArgumentException("Unsupported PIT job argument " + args[index]);
        }
        options.put(args[index], args[index + 1]);
      }
      return new JobConfig(
          option(options, "--catalog-name", environment, "FEATURE_LAKE_CATALOG_NAME", "feature_catalog"),
          option(options, "--catalog-uri", environment, "FEATURE_LAKE_CATALOG_URI", null),
          option(options, "--warehouse-uri", environment, "FEATURE_LAKE_WAREHOUSE_URI", null),
          option(options, "--s3-endpoint", environment, "FEATURE_LAKE_S3_ENDPOINT", null),
          option(options, "--namespace", environment, "FEATURE_LAKE_NAMESPACE", "video_commerce"),
          option(options, "--feature-definition-version", environment, "FEATURE_LAKE_FEATURE_DEFINITION_VERSION", "ranking_ltr_v1"),
          Integer.parseInt(option(options, "--attribution-window-hours", environment, "FEATURE_LAKE_ATTRIBUTION_WINDOW_HOURS", "168")),
          Integer.parseInt(option(options, "--allowed-lateness-hours", environment, "FEATURE_LAKE_ALLOWED_LATENESS_HOURS", "1")),
          Double.parseDouble(option(options, "--materialization-cutoff", environment, "FEATURE_LAKE_MATERIALIZATION_CUTOFF", null)),
          option(options, "--materialization-run-id", environment, "FEATURE_LAKE_MATERIALIZATION_RUN_ID", null),
          option(options, "--export-uri", environment, "FEATURE_LAKE_PIT_EXPORT_URI", null),
          Integer.parseInt(
              option(
                  options,
                  "--export-attempt",
                  environment,
                  "FEATURE_LAKE_EXPORT_ATTEMPT",
                  "1")));
    }

    private static String option(
        Map<String, String> options,
        String option,
        Map<String, String> environment,
        String environmentName,
        String defaultValue) {
      String value = options.get(option);
      return value == null || value.trim().isEmpty()
          ? required(environment, environmentName, defaultValue)
          : value.trim();
    }

    private static String required(
        Map<String, String> environment, String name, String defaultValue) {
      String value = environment.get(name);
      if (value == null || value.trim().isEmpty()) {
        value = defaultValue;
      }
      if (value == null || value.trim().isEmpty()) {
        throw new IllegalArgumentException(name + " must be configured");
      }
      return value.trim();
    }
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
            + "attributed_action STRING,attributed_click INT,attributed_purchase INT,"
            + "attributed_value DOUBLE,attributed_value_source STRING,"
            + "feature_definition_version STRING,label_definition_version STRING,"
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

  static String buildAddColumnSql(String namespace, String column, String type) {
    return String.format(
        "ALTER TABLE `%s`.`ranking_training_pit` ADD `%s` %s",
        identifier(namespace), identifier(column), sqlType(type));
  }

  private static void evolveTrainingTable(TableEnvironment tables, String namespace) {
    List<String> existing =
        tables.from("`" + identifier(namespace) + "`.`ranking_training_pit`")
            .getResolvedSchema()
            .getColumnNames();
    Map<String, String> additions = new LinkedHashMap<>();
    additions.put("attributed_action", "STRING");
    additions.put("attributed_value", "DOUBLE");
    additions.put("attributed_value_source", "STRING");
    additions.put("label_definition_version", "STRING");
    for (Map.Entry<String, String> addition : additions.entrySet()) {
      if (!existing.contains(addition.getKey())) {
        tables.executeSql(
            buildAddColumnSql(namespace, addition.getKey(), addition.getValue()));
      }
    }
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

  static String buildExistingTableRunSql(String namespace, String tableName, String runId) {
    if (!Set.of("ranking_training_pit", "ranking_training_pit_quarantine").contains(tableName)) {
      throw new IllegalArgumentException("unsupported PIT run table " + tableName);
    }
    return String.format(
        "SELECT COUNT(*) FROM `%s`.`%s` WHERE materialization_run_id='%s'",
        identifier(namespace), tableName, literal(runId));
  }

  private static long existingRunCount(
      TableEnvironment tables, String namespace, String tableName, String runId) throws Exception {
    try (CloseableIterator<Row> rows =
        tables.executeSql(buildExistingTableRunSql(namespace, tableName, runId)).collect()) {
      return rows.hasNext() ? ((Number) rows.next().getField(0)).longValue() : 0L;
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
    return buildPointInTimeInsertSql(
        namespace,
        featureDefinitionVersion,
        attributionWindowHours,
        allowedLatenessHours,
        materializationCutoff,
        runId,
        List.of(
            "materialization_run_id", "observation_id", "impression_id", "user_id", "product_id",
            "action", "as_of_ts", "user_features_json", "product_metadata_json", "context_json",
            "candidate_features_json", "online_feature_bundle_hash", "feature_bundle_hash",
            "attributed_action", "attributed_click", "attributed_purchase", "attributed_value",
            "attributed_value_source", "feature_definition_version", "label_definition_version",
            "materialized_at", "materialization_date"));
  }

  static String buildPointInTimeInsertSql(
      String namespace,
      String featureDefinitionVersion,
      int attributionWindowHours,
      int allowedLatenessHours,
      double materializationCutoff,
      String runId,
      List<String> targetColumns) {
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
    String selectList = buildPitSelectList(targetColumns, run, version);
    return String.format(
        "INSERT INTO `%s`.`ranking_training_pit`\n%s\n"
            + "SELECT %s\n"
            + "FROM eligible_observations o\n"
            + "LEFT JOIN user_ranked u ON u.observation_id=o.observation_id AND u.feature_rank=1\n"
            + "LEFT JOIN item_ranked it ON it.observation_id=o.observation_id AND it.feature_rank=1\n"
            + "LEFT JOIN feedback f ON f.observation_id=o.observation_id\n"
            + "WHERE u.canonical_payload_json IS NOT NULL "
            + "AND JSON_VALUE(o.canonical_payload_json,'$.impression_id') IS NOT NULL "
            + "AND o.context_json IS NOT NULL "
            + "AND JSON_VALUE(o.candidate_features_json,'$.collaborative_score') IS NOT NULL "
            + "AND JSON_VALUE(o.candidate_features_json,'$.content_similarity_score') IS NOT NULL "
            + "AND JSON_VALUE(o.candidate_features_json,'$.popularity_score') IS NOT NULL "
            + "AND JSON_VALUE(o.candidate_features_json,'$.combined_score') IS NOT NULL AND ("
            + "JSON_VALUE(o.canonical_payload_json,'$.item_snapshot_complete')='true' "
            + "OR it.canonical_payload_json IS NOT NULL)",
        identifier(namespace), eligible, selectList);
  }

  private static String buildPitSelectList(
      List<String> targetColumns, String run, String version) {
    List<String> expressions = new ArrayList<>();
    for (String column : targetColumns) {
      switch (column) {
        case "materialization_run_id": expressions.add("'" + run + "'"); break;
        case "observation_id": expressions.add("o.observation_id"); break;
        case "impression_id": expressions.add("JSON_VALUE(o.canonical_payload_json,'$.impression_id')"); break;
        case "user_id": expressions.add("o.user_id"); break;
        case "product_id": expressions.add("o.product_id"); break;
        case "action":
        case "attributed_action": expressions.add("f.attributed_action"); break;
        case "as_of_ts": expressions.add("o.event_time_epoch"); break;
        case "user_features_json": expressions.add("u.canonical_payload_json"); break;
        case "product_metadata_json":
          expressions.add("CASE WHEN JSON_VALUE(o.canonical_payload_json,'$.item_snapshot_complete')='true' THEN o.item_snapshot_json ELSE it.canonical_payload_json END");
          break;
        case "context_json": expressions.add("o.context_json"); break;
        case "candidate_features_json": expressions.add("o.candidate_features_json"); break;
        case "online_feature_bundle_hash": expressions.add("o.feature_bundle_hash"); break;
        case "feature_bundle_hash":
          expressions.add("pit_feature_bundle_hash(o.event_time_epoch,o.user_id,o.product_id,u.canonical_payload_json,CASE WHEN JSON_VALUE(o.canonical_payload_json,'$.item_snapshot_complete')='true' THEN o.item_snapshot_json ELSE it.canonical_payload_json END,o.context_json,o.candidate_features_json,'" + version + "')");
          break;
        case "attributed_click": expressions.add("COALESCE(f.attributed_click,0)"); break;
        case "attributed_purchase": expressions.add("COALESCE(f.attributed_purchase,0)"); break;
        case "attributed_value": expressions.add("f.attributed_value"); break;
        case "attributed_value_source": expressions.add("f.attributed_value_source"); break;
        case "feature_definition_version": expressions.add("'" + version + "'"); break;
        case "label_definition_version": expressions.add("'" + LABEL_DEFINITION_VERSION + "'"); break;
        case "materialized_at": expressions.add("CURRENT_TIMESTAMP"); break;
        case "materialization_date": expressions.add("CURRENT_DATE"); break;
        default: throw new IllegalStateException("unsupported ranking_training_pit column " + column);
      }
    }
    return String.join(",", expressions);
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
            + "WHEN JSON_VALUE(o.canonical_payload_json,'$.impression_id') IS NULL "
            + "THEN 'missing_impression_id' "
            + "WHEN o.context_json IS NULL THEN 'missing_context' "
            + "WHEN o.candidate_features_json IS NULL OR "
            + "JSON_VALUE(o.candidate_features_json,'$.collaborative_score') IS NULL OR "
            + "JSON_VALUE(o.candidate_features_json,'$.content_similarity_score') IS NULL OR "
            + "JSON_VALUE(o.candidate_features_json,'$.popularity_score') IS NULL OR "
            + "JSON_VALUE(o.candidate_features_json,'$.combined_score') IS NULL "
            + "THEN 'missing_candidate_scores' "
            + "WHEN u.canonical_payload_json IS NULL THEN 'missing_user_feature' "
            + "ELSE 'missing_complete_v1_item_snapshot' END,'%s',o.event_time_epoch,"
            + "CURRENT_TIMESTAMP,CURRENT_DATE FROM eligible_observations o "
            + "LEFT JOIN user_ranked u ON u.observation_id=o.observation_id AND u.feature_rank=1 "
            + "LEFT JOIN item_ranked it ON it.observation_id=o.observation_id AND it.feature_rank=1 "
            + "WHERE JSON_VALUE(o.canonical_payload_json,'$.impression_id') IS NULL "
            + "OR o.context_json IS NULL OR o.candidate_features_json IS NULL OR "
            + "JSON_VALUE(o.candidate_features_json,'$.collaborative_score') IS NULL OR "
            + "JSON_VALUE(o.candidate_features_json,'$.content_similarity_score') IS NULL OR "
            + "JSON_VALUE(o.candidate_features_json,'$.popularity_score') IS NULL OR "
            + "JSON_VALUE(o.candidate_features_json,'$.combined_score') IS NULL OR "
            + "u.canonical_payload_json IS NULL OR ("
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
            + "),feedback_events AS (\n"
            + " SELECT o.observation_id,i.action,i.context_json,i.event_time_epoch,"
            + "i.available_at_epoch,i.source_event_id FROM eligible_observations o "
            + "LEFT JOIN `%s`.`interaction_history` i "
            + "ON i.user_id=o.user_id AND i.product_id=o.product_id "
            + "AND i.event_time_epoch>=o.event_time_epoch "
            + "AND i.event_time_epoch<=o.event_time_epoch+%d "
            + "AND i.available_at_epoch<=CAST(%.3f AS DOUBLE)\n"
            + "),feedback_aggregated AS (\n"
            + " SELECT observation_id,CASE MAX(CASE action WHEN 'purchase' THEN 3 "
            + "WHEN 'add_to_cart' THEN 2 WHEN 'click' THEN 1 ELSE 0 END) "
            + "WHEN 3 THEN 'purchase' WHEN 2 THEN 'add_to_cart' WHEN 1 THEN 'click' ELSE 'view' END "
            + "attributed_action,MAX(CASE WHEN action IN ('click','add_to_cart','purchase') "
            + "THEN 1 ELSE 0 END) attributed_click,MAX(CASE WHEN action='purchase' THEN 1 ELSE 0 END) "
            + "attributed_purchase FROM feedback_events GROUP BY observation_id\n"
            + "),purchase_ranked AS (\n"
            + " SELECT observation_id,COALESCE("
            + "TRY_CAST(JSON_VALUE(context_json,'$.margin') AS DOUBLE),"
            + "TRY_CAST(JSON_VALUE(context_json,'$.profit') AS DOUBLE),"
            + "TRY_CAST(JSON_VALUE(context_json,'$.gross_margin') AS DOUBLE),"
            + "TRY_CAST(JSON_VALUE(context_json,'$.value') AS DOUBLE),"
            + "TRY_CAST(JSON_VALUE(context_json,'$.gmv') AS DOUBLE),"
            + "TRY_CAST(JSON_VALUE(context_json,'$.purchase_value') AS DOUBLE),"
            + "TRY_CAST(JSON_VALUE(context_json,'$.price') AS DOUBLE)) attributed_value,"
            + "CASE WHEN JSON_VALUE(context_json,'$.margin') IS NOT NULL THEN 'margin' "
            + "WHEN JSON_VALUE(context_json,'$.profit') IS NOT NULL THEN 'profit' "
            + "WHEN JSON_VALUE(context_json,'$.gross_margin') IS NOT NULL THEN 'gross_margin' "
            + "WHEN JSON_VALUE(context_json,'$.value') IS NOT NULL THEN 'value' "
            + "WHEN JSON_VALUE(context_json,'$.gmv') IS NOT NULL THEN 'gmv' "
            + "WHEN JSON_VALUE(context_json,'$.purchase_value') IS NOT NULL THEN 'purchase_value' "
            + "WHEN JSON_VALUE(context_json,'$.price') IS NOT NULL THEN 'price' END "
            + "attributed_value_source,ROW_NUMBER() OVER (PARTITION BY observation_id "
            + "ORDER BY event_time_epoch DESC,available_at_epoch DESC,source_event_id DESC) purchase_rank "
            + "FROM feedback_events i WHERE action='purchase'\n"
            + "),feedback AS (\n"
            + " SELECT a.observation_id,a.attributed_action,a.attributed_click,a.attributed_purchase,"
            + "p.attributed_value,p.attributed_value_source FROM feedback_aggregated a "
            + "LEFT JOIN purchase_ranked p ON p.observation_id=a.observation_id AND p.purchase_rank=1\n"
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
    return buildParquetExportTableSql(exportPrefix, runId, 1);
  }

  static String buildParquetExportTableSql(
      String exportPrefix, String runId, int exportAttempt) {
    if (exportAttempt < 1) {
      throw new IllegalArgumentException("export attempt must be positive");
    }
    return String.format(
        "CREATE TEMPORARY TABLE pit_parquet_export (observation_id STRING,impression_id STRING,"
            + "user_id STRING,"
            + "product_id STRING,action STRING,as_of_ts DOUBLE,user_features_json STRING,"
            + "product_metadata_json STRING,context_json STRING,candidate_features_json STRING,"
            + "online_feature_bundle_hash STRING,feature_bundle_hash STRING,"
            + "attributed_action STRING,attributed_click INT,attributed_purchase INT,"
            + "attributed_value DOUBLE,attributed_value_source STRING,"
            + "feature_definition_version STRING,label_definition_version STRING) "
            + "WITH ('connector'='filesystem','path'='%s',"
            + "'format'='parquet','sink.rolling-policy.file-size'='128mb')",
        literal(exportPrefix)
            + "/runs/"
            + literal(runId)
            + "/attempts/"
            + exportAttempt
            + "/shards");
  }

  static String buildParquetExportInsertSql(String namespace, String runId) {
    return String.format(
        "INSERT INTO pit_parquet_export SELECT observation_id,impression_id,user_id,product_id,"
            + "action,as_of_ts,"
            + "user_features_json,product_metadata_json,context_json,candidate_features_json,"
            + "online_feature_bundle_hash,feature_bundle_hash,attributed_action,"
            + "attributed_click,attributed_purchase,attributed_value,attributed_value_source,"
            + "feature_definition_version,label_definition_version "
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

  private static String sqlType(String value) {
    String normalized = value == null ? "" : value.trim().toUpperCase();
    if (!normalized.matches("(STRING|DOUBLE|INT|BIGINT)")) {
      throw new IllegalArgumentException("invalid SQL type " + value);
    }
    return normalized;
  }

  private static String literal(String value) {
    return value.replace("'", "''");
  }

}
