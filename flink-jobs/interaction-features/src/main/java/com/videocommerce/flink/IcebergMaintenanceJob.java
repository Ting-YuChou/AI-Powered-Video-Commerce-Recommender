package com.videocommerce.flink;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.apache.flink.core.execution.JobClient;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.Table;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.flink.CatalogLoader;
import org.apache.iceberg.flink.TableLoader;
import org.apache.iceberg.flink.maintenance.api.DeleteOrphanFiles;
import org.apache.iceberg.flink.maintenance.api.TableMaintenance;

/** Bounded standalone Iceberg maintenance execution for the approved feature-lake tables. */
public final class IcebergMaintenanceJob {
  private static final Set<String> ALLOWED_TABLES =
      Set.of(
          "interaction_history",
          "recommendation_view_history",
          "ranking_observations",
          "user_feature_history",
          "item_feature_history",
          "window_feature_history",
          "ranking_training_pit",
          "ranking_training_pit_quarantine",
          "retrieval_training_pit");

  private IcebergMaintenanceJob() {}

  static final class Policy {
    final Duration maxSnapshotAge;
    final int retainLast;
    final Duration orphanMinAge;
    final int deleteBatchSize;

    Policy(
        Duration maxSnapshotAge,
        int retainLast,
        Duration orphanMinAge,
        int deleteBatchSize) {
      this.maxSnapshotAge = maxSnapshotAge;
      this.retainLast = retainLast;
      this.orphanMinAge = orphanMinAge;
      this.deleteBatchSize = deleteBatchSize;
    }

    static Policy defaults() {
      return new Policy(Duration.ofDays(30), 20, Duration.ofDays(7), 1000);
    }

    static Policy fromArguments(String[] args) {
      return new Policy(
          Duration.ofDays(argumentInt(args, "--snapshot-retention-days", envInt("ICEBERG_MAINTENANCE_SNAPSHOT_RETENTION_DAYS", 30))),
          argumentInt(args, "--retain-last-snapshots", envInt("ICEBERG_MAINTENANCE_RETAIN_LAST_SNAPSHOTS", 20)),
          Duration.ofDays(argumentInt(args, "--orphan-min-age-days", envInt("ICEBERG_MAINTENANCE_ORPHAN_MIN_AGE_DAYS", 7))),
          argumentInt(args, "--delete-batch-size", envInt("ICEBERG_MAINTENANCE_DELETE_BATCH_SIZE", 1000)));
    }
  }

  public static void main(String[] args) throws Exception {
    String tableName = validateTable(requiredArgument(args, "--table"));
    String namespace = argument(args, "--namespace", env("FEATURE_LAKE_NAMESPACE", "video_commerce"));
    Policy policy = Policy.fromArguments(args);

    Map<String, String> properties = new HashMap<>();
    properties.put("uri", argumentOrRequiredEnv(args, "--catalog-uri", "FEATURE_LAKE_CATALOG_URI"));
    properties.put("warehouse", argumentOrRequiredEnv(args, "--warehouse-uri", "FEATURE_LAKE_WAREHOUSE_URI"));
    properties.put("io-impl", "org.apache.iceberg.aws.s3.S3FileIO");
    putIfPresent(properties, "s3.endpoint", System.getenv("FEATURE_LAKE_S3_ENDPOINT"));
    putIfPresent(properties, "s3.access-key-id", System.getenv("OBJECT_STORAGE_ACCESS_KEY_ID"));
    putIfPresent(
        properties, "s3.secret-access-key", System.getenv("OBJECT_STORAGE_SECRET_ACCESS_KEY"));
    properties.put("s3.path-style-access", "true");

    CatalogLoader catalog =
        CatalogLoader.rest(
            env("FEATURE_LAKE_CATALOG_NAME", "feature_catalog"),
            new Configuration(),
            properties);
    TableLoader tableLoader =
        TableLoader.fromCatalog(catalog, TableIdentifier.of(namespace, tableName));
    expirePlannedSnapshots(
        catalog,
        namespace,
        tableName,
        Long.parseLong(requiredArgument(args, "--expected-head-snapshot")),
        arguments(args, "--expire-snapshot"));
    StreamExecutionEnvironment execution =
        StreamExecutionEnvironment.getExecutionEnvironment();
    execution.setParallelism(envInt("ICEBERG_MAINTENANCE_PARALLELISM", 2));
    execution.enableCheckpointing(60_000L);

    TableMaintenance.forTable(execution, tableLoader)
        .uidSuffix("video-commerce-" + tableName)
        .rateLimit(Duration.ofHours(1))
        .lockCheckDelay(Duration.ofSeconds(10))
        .add(
            DeleteOrphanFiles.builder()
                .scheduleOnInterval(Duration.ofSeconds(1))
                .minAge(policy.orphanMinAge)
                .deleteBatchSize(policy.deleteBatchSize)
                .prefixMismatchMode(
                    org.apache.iceberg.actions.DeleteOrphanFiles.PrefixMismatchMode.ERROR))
        .append();

    JobClient client =
        execution.executeAsync("video-commerce-iceberg-maintenance-" + tableName);
    long runSeconds = argumentInt(args, "--run-seconds", envInt("ICEBERG_MAINTENANCE_RUN_SECONDS", 300));
    Thread.sleep(TimeUnit.SECONDS.toMillis(runSeconds));
    client.cancel().get(60, TimeUnit.SECONDS);
  }

  private static void expirePlannedSnapshots(
      CatalogLoader loader,
      String namespace,
      String tableName,
      long expectedHeadSnapshot,
      java.util.List<String> values) {
    Catalog catalog = loader.loadCatalog();
    Table table = catalog.loadTable(TableIdentifier.of(namespace, tableName));
    validateHeadSnapshot(table.currentSnapshot().snapshotId(), expectedHeadSnapshot);
    if (values.isEmpty()) {
      return;
    }
    org.apache.iceberg.ExpireSnapshots expiration = table.expireSnapshots();
    for (String value : values) {
      expiration.expireSnapshotId(Long.parseLong(value));
    }
    expiration.cleanExpiredFiles(true).commit();
  }

  static void validateHeadSnapshot(long actual, long expected) {
    if (actual != expected) {
      throw new IllegalStateException(
          "Iceberg table head changed after planning: expected " + expected + ", found " + actual);
    }
  }

  static String validateTable(String table) {
    String normalized = table == null ? "" : table.trim();
    if (!ALLOWED_TABLES.contains(normalized)) {
      throw new IllegalArgumentException("Iceberg table is not in the maintenance allowlist");
    }
    return normalized;
  }

  private static String requiredArgument(String[] args, String name) {
    for (int index = 0; index < args.length - 1; index++) {
      if (name.equals(args[index])) {
        return args[index + 1];
      }
    }
    throw new IllegalArgumentException(name + " is required");
  }

  private static String argument(String[] args, String name, String fallback) {
    for (int index = 0; index < args.length - 1; index++) {
      if (name.equals(args[index])) {
        return args[index + 1];
      }
    }
    return fallback;
  }

  private static String argumentOrRequiredEnv(String[] args, String name, String envName) {
    String value = argument(args, name, "");
    return value.isBlank() ? requiredEnv(envName) : value;
  }

  private static java.util.List<String> arguments(String[] args, String name) {
    java.util.List<String> values = new java.util.ArrayList<>();
    for (int index = 0; index < args.length - 1; index++) {
      if (name.equals(args[index])) {
        values.add(args[index + 1]);
      }
    }
    return values;
  }

  private static int argumentInt(String[] args, String name, int fallback) {
    int value = Integer.parseInt(argument(args, name, Integer.toString(fallback)));
    if (value < 1) {
      throw new IllegalArgumentException(name + " must be positive");
    }
    return value;
  }

  private static int envInt(String name, int defaultValue) {
    int value = Integer.parseInt(env(name, Integer.toString(defaultValue)));
    if (value < 1) {
      throw new IllegalArgumentException(name + " must be positive");
    }
    return value;
  }

  private static String requiredEnv(String name) {
    String value = System.getenv(name);
    if (value == null || value.trim().isEmpty()) {
      throw new IllegalArgumentException(name + " is required");
    }
    return value.trim();
  }

  private static String env(String name, String defaultValue) {
    String value = System.getenv(name);
    return value == null || value.trim().isEmpty() ? defaultValue : value.trim();
  }

  private static void putIfPresent(Map<String, String> values, String key, String value) {
    if (value != null && !value.trim().isEmpty()) {
      values.put(key, value.trim());
    }
  }
}
