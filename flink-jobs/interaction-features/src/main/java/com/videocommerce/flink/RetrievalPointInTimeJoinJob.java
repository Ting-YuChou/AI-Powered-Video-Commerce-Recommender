package com.videocommerce.flink;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;

/** Materializes mature viewed-impression queries for immutable Two-Tower training. */
public final class RetrievalPointInTimeJoinJob {
  private RetrievalPointInTimeJoinJob() {}

  public static void main(String[] args) throws Exception {
    Config config = Config.resolve(args, System.getenv());
    TableEnvironment tables =
        TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());
    tables.executeSql(
        FeatureHistoryMaterializerJob.buildRestCatalogSql(
            config.catalog,
            config.catalogUri,
            config.warehouseUri,
            config.s3Endpoint));
    tables.executeSql("USE CATALOG `" + identifier(config.catalog) + "`");
    tables.executeSql("CREATE DATABASE IF NOT EXISTS `" + identifier(config.namespace) + "`");
    tables.executeSql(buildTrainingTableSql(config.namespace));
    tables
        .executeSql(
            buildInsertSql(
                config.namespace,
                config.attributionWindowHours,
                config.allowedLatenessHours,
                config.materializationCutoff,
                config.runId,
                config.catalogGenerationId))
        .await();
    tables.executeSql(
        buildExportTableSql(config.exportUri, config.runId, config.exportAttempt));
    tables
        .executeSql(
            "INSERT INTO retrieval_pit_export SELECT query_id,user_id,product_id,label_source,"
                + "label_type,label_weight,as_of_ts,label_event_time,label_available_at,"
                + "feature_event_time,feature_available_at,catalog_generation_id,"
                + "user_features_json,seen_product_ids_json,ranker_score "
                + "FROM `"
                + identifier(config.namespace)
                + "`.`retrieval_training_pit` WHERE materialization_run_id='"
                + literal(config.runId)
                + "'")
        .await();
  }

  static String buildTrainingTableSql(String namespace) {
    return String.format(
        "CREATE TABLE IF NOT EXISTS `%s`.`retrieval_training_pit` ("
            + "materialization_run_id STRING,query_id STRING,user_id STRING,product_id STRING,"
            + "label_source STRING,label_type STRING,label_weight DOUBLE,as_of_ts DOUBLE,"
            + "label_event_time DOUBLE,label_available_at DOUBLE,feature_event_time DOUBLE,"
            + "feature_available_at DOUBLE,catalog_generation_id STRING,user_features_json STRING,"
            + "seen_product_ids_json STRING,ranker_score DOUBLE,materialized_at TIMESTAMP_LTZ(3),"
            + "materialization_date DATE) PARTITIONED BY (materialization_date) "
            + "WITH ('format-version'='2','write.upsert.enabled'='false')",
        identifier(namespace));
  }

  static String buildInsertSql(
      String namespace,
      int attributionWindowHours,
      int allowedLatenessHours,
      double materializationCutoff,
      String runId,
      String catalogGenerationId) {
    if (attributionWindowHours <= 0 || allowedLatenessHours < 0) {
      throw new IllegalArgumentException("retrieval PIT windows are invalid");
    }
    return String.format(
        "INSERT INTO `%1$s`.`retrieval_training_pit`\n"
            + "WITH viewed AS (\n"
            + " SELECT observation_id,COALESCE(JSON_VALUE(canonical_payload_json,'$.impression_id'),observation_id) query_id,"
            + "user_id,product_id,event_time_epoch as_of_ts,event_date,'viewed_impression' label_source,"
            + "CAST(NULL AS STRING) fixed_label_type,CAST(NULL AS DOUBLE) fixed_label_event_time,"
            + "CAST(NULL AS DOUBLE) fixed_label_available_at\n"
            + " FROM `%1$s`.`recommendation_view_history`\n"
            + " WHERE event_time_epoch + %2$d * 3600 + %3$d * 3600 <= %4$f"
            + " AND available_at_epoch <= %4$f\n"
            + " UNION ALL SELECT observation_id,"
            + "COALESCE(JSON_VALUE(canonical_payload_json,'$.impression_id'),observation_id),"
            + "user_id,product_id,event_time_epoch,event_date,'ranker_rejected','ranker_rejected',"
            + "event_time_epoch,available_at_epoch FROM `%1$s`.`ranking_observations`"
            + " WHERE event_type='ranking_rejected_observation'"
            + " AND event_time_epoch + %2$d * 3600 + %3$d * 3600 <= %4$f"
            + " AND available_at_epoch <= %4$f\n"
            + " UNION ALL SELECT event_id,event_id,user_id,product_id,event_time_epoch,event_date,"
            + "'organic_positive',action,event_time_epoch,available_at_epoch"
            + " FROM `%1$s`.`interaction_history` WHERE action IN ('click','add_to_cart','purchase')"
            + " AND JSON_VALUE(context_json,'$.impression_id') IS NULL"
            + " AND event_time_epoch <= %4$f AND available_at_epoch <= %4$f\n),\n"
            + "user_candidates AS (\n"
            + " SELECT v.observation_id,h.canonical_payload_json user_features_json,"
            + "h.event_time_epoch feature_event_time,h.available_at_epoch feature_available_at,"
            + "ROW_NUMBER() OVER (PARTITION BY v.observation_id ORDER BY h.event_time_epoch DESC,h.available_at_epoch DESC,h.event_id DESC) feature_rank\n"
            + " FROM viewed v JOIN `%1$s`.`user_feature_history` h ON h.entity_id=v.user_id"
            + " AND h.event_time_epoch <= v.as_of_ts AND h.available_at_epoch <= v.as_of_ts\n),\n"
            + "catalog_candidates AS (\n"
            + " SELECT v.observation_id,h.source_version catalog_generation_id,"
            + "ROW_NUMBER() OVER (PARTITION BY v.observation_id ORDER BY h.event_time_epoch DESC,h.available_at_epoch DESC,h.event_id DESC) catalog_rank\n"
            + " FROM viewed v JOIN `%1$s`.`item_feature_history` h ON h.entity_id=v.product_id"
            + " AND h.source_version='%6$s'"
            + " AND h.event_time_epoch <= v.as_of_ts AND h.available_at_epoch <= v.as_of_ts\n),\n"
            + "feedback_candidates AS (\n"
            + " SELECT v.observation_id,i.action,i.event_time_epoch label_event_time,"
            + "i.available_at_epoch label_available_at,ROW_NUMBER() OVER (PARTITION BY v.observation_id "
            + "ORDER BY CASE i.action WHEN 'purchase' THEN 3 WHEN 'add_to_cart' THEN 2 ELSE 1 END DESC,"
            + "i.event_time_epoch ASC,i.event_id ASC) feedback_rank\n"
            + " FROM viewed v JOIN `%1$s`.`interaction_history` i ON i.user_id=v.user_id AND i.product_id=v.product_id"
            + " AND v.label_source='viewed_impression'"
            + " AND i.action IN ('click','add_to_cart','purchase')"
            + " AND i.event_time_epoch >= v.as_of_ts AND i.event_time_epoch <= v.as_of_ts + %2$d * 3600"
            + " AND i.available_at_epoch <= %4$f\n),\n"
            + "deduplicated AS (SELECT * FROM `%1$s`.`retrieval_training_pit` WHERE materialization_run_id='%5$s')\n"
            + "SELECT '%5$s',v.query_id,v.user_id,v.product_id,v.label_source,"
            + "COALESCE(v.fixed_label_type,f.action,'viewed_no_positive'),"
            + "CASE COALESCE(v.fixed_label_type,f.action,'viewed_no_positive') WHEN 'purchase' THEN 5.0 WHEN 'add_to_cart' THEN 3.0 WHEN 'click' THEN 2.0 WHEN 'ranker_rejected' THEN 0.15 ELSE 0.25 END,"
            + "v.as_of_ts,COALESCE(v.fixed_label_event_time,f.label_event_time,v.as_of_ts + %2$d * 3600),"
            + "COALESCE(v.fixed_label_available_at,f.label_available_at,%4$f),u.feature_event_time,u.feature_available_at,"
            + "c.catalog_generation_id,u.user_features_json,'[]',CAST(NULL AS DOUBLE),CURRENT_TIMESTAMP,v.event_date\n"
            + "FROM viewed v JOIN user_candidates u ON u.observation_id=v.observation_id AND u.feature_rank=1"
            + " JOIN catalog_candidates c ON c.observation_id=v.observation_id AND c.catalog_rank=1"
            + " LEFT JOIN feedback_candidates f ON f.observation_id=v.observation_id AND f.feedback_rank=1"
            + " LEFT JOIN deduplicated d ON d.query_id=v.query_id AND d.product_id=v.product_id AND d.label_source=v.label_source"
            + " WHERE d.query_id IS NULL",
        identifier(namespace),
        attributionWindowHours,
        allowedLatenessHours,
        materializationCutoff,
        literal(runId),
        literal(catalogGenerationId));
  }

  static String buildExportTableSql(String exportUri, String runId, int attempt) {
    if (attempt <= 0) {
      throw new IllegalArgumentException("retrieval export attempt must be positive");
    }
    String path =
        exportUri.replaceAll("/+$", "")
            + "/"
            + runId
            + "/attempt-"
            + attempt;
    return String.format(
        "CREATE TEMPORARY TABLE retrieval_pit_export (query_id STRING,user_id STRING,product_id STRING,"
            + "label_source STRING,label_type STRING,label_weight DOUBLE,as_of_ts DOUBLE,"
            + "label_event_time DOUBLE,label_available_at DOUBLE,feature_event_time DOUBLE,"
            + "feature_available_at DOUBLE,catalog_generation_id STRING,user_features_json STRING,"
            + "seen_product_ids_json STRING,ranker_score DOUBLE) WITH "
            + "('connector'='filesystem','path'='%s','format'='parquet')",
        literal(path));
  }

  private static String identifier(String value) {
    return value.replace("`", "``");
  }

  private static String literal(String value) {
    return value.replace("'", "''");
  }

  static final class Config {
    private static final Set<String> OPTIONS =
        Set.of(
            "--catalog-name",
            "--catalog-uri",
            "--warehouse-uri",
            "--s3-endpoint",
            "--namespace",
            "--attribution-window-hours",
            "--allowed-lateness-hours",
            "--materialization-cutoff",
            "--materialization-run-id",
            "--catalog-generation-id",
            "--export-uri",
            "--export-attempt");
    String catalog;
    String catalogUri;
    String warehouseUri;
    String s3Endpoint;
    String namespace;
    int attributionWindowHours;
    int allowedLatenessHours;
    double materializationCutoff;
    String runId;
    String catalogGenerationId;
    String exportUri;
    int exportAttempt;

    static Config resolve(String[] args, Map<String, String> environment) {
      Map<String, String> values = new LinkedHashMap<>();
      for (int index = 0; index < args.length; index += 2) {
        if (index + 1 >= args.length || !OPTIONS.contains(args[index])) {
          throw new IllegalArgumentException("unsupported retrieval PIT argument");
        }
        values.put(args[index], args[index + 1]);
      }
      Config config = new Config();
      config.catalog = value(values, "--catalog-name", environment, "FEATURE_LAKE_CATALOG_NAME", "feature_catalog");
      config.catalogUri = value(values, "--catalog-uri", environment, "FEATURE_LAKE_CATALOG_URI", null);
      config.warehouseUri = value(values, "--warehouse-uri", environment, "FEATURE_LAKE_WAREHOUSE_URI", null);
      config.s3Endpoint = value(values, "--s3-endpoint", environment, "FEATURE_LAKE_S3_ENDPOINT", null);
      config.namespace = value(values, "--namespace", environment, "FEATURE_LAKE_NAMESPACE", "video_commerce");
      config.attributionWindowHours = Integer.parseInt(value(values, "--attribution-window-hours", environment, "FEATURE_LAKE_ATTRIBUTION_WINDOW_HOURS", "168"));
      config.allowedLatenessHours = Integer.parseInt(value(values, "--allowed-lateness-hours", environment, "FEATURE_LAKE_ALLOWED_LATENESS_HOURS", "1"));
      config.materializationCutoff = Double.parseDouble(value(values, "--materialization-cutoff", environment, "FEATURE_LAKE_MATERIALIZATION_CUTOFF", null));
      config.runId = value(values, "--materialization-run-id", environment, "FEATURE_LAKE_MATERIALIZATION_RUN_ID", null);
      config.catalogGenerationId = value(values, "--catalog-generation-id", environment, "FEATURE_LAKE_RETRIEVAL_CATALOG_GENERATION_ID", null);
      config.exportUri = value(values, "--export-uri", environment, "FEATURE_LAKE_RETRIEVAL_PIT_EXPORT_URI", null);
      config.exportAttempt = Integer.parseInt(value(values, "--export-attempt", environment, "FEATURE_LAKE_EXPORT_ATTEMPT", "1"));
      return config;
    }

    private static String value(
        Map<String, String> args,
        String option,
        Map<String, String> environment,
        String environmentName,
        String defaultValue) {
      String value = args.get(option);
      if (value == null || value.isBlank()) {
        value = environment.get(environmentName);
      }
      if (value == null || value.isBlank()) {
        value = defaultValue;
      }
      if (value == null || value.isBlank()) {
        throw new IllegalArgumentException(environmentName + " must be configured");
      }
      return value.trim();
    }
  }
}
