package com.videocommerce.flink;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.metrics.Counter;
import org.apache.flink.api.common.state.StateTtlConfig;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.connector.kafka.source.reader.deserializer.KafkaRecordDeserializationSchema;
import org.apache.flink.contrib.streaming.state.EmbeddedRocksDBStateBackend;
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.CheckpointConfig;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.StatementSet;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.OffsetResetStrategy;

/** Independent Kafka to Iceberg append-only history materializer. */
public final class FeatureHistoryMaterializerJob {
  static final String PRODUCTION_GROUP_ID = "feature-history-materializer-v1";
  static final List<String> TABLES =
      Arrays.asList(
          "interaction_history",
          "ranking_observations",
          "user_feature_history",
          "item_feature_history",
          "window_feature_history");
  private static final ObjectMapper JSON = new ObjectMapper();
  private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {};
  private static final OutputTag<String> DLQ =
      new OutputTag<>("feature-history-materializer-dlq", TypeInformation.of(String.class));

  private FeatureHistoryMaterializerJob() {}

  public static void main(String[] args) throws Exception {
    Config config = Config.fromEnv();
    StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
    configureEnvironment(env, config);
    StreamTableEnvironment tableEnvironment =
        StreamTableEnvironment.create(
            env, EnvironmentSettings.newInstance().inStreamingMode().build());
    tableEnvironment
        .getConfig()
        .getConfiguration()
        .setString("pipeline.name", "video-commerce-feature-history-materializer");
    tableEnvironment.executeSql(
        buildRestCatalogSql(
            config.catalogName,
            config.catalogUri,
            config.warehouseUri,
            config.s3Endpoint));
    tableEnvironment.executeSql("USE CATALOG `" + config.catalogName + "`");
    tableEnvironment.executeSql("CREATE DATABASE IF NOT EXISTS `" + config.namespace + "`");
    for (String table : TABLES) {
      tableEnvironment.executeSql(buildCreateTableSql(config.namespace, table));
    }

    KafkaSource<RawKafkaEvent> source =
        KafkaSource.<RawKafkaEvent>builder()
            .setBootstrapServers(config.kafkaBootstrapServers)
            .setTopics(config.topics)
            .setGroupId(config.consumerGroupId)
            .setStartingOffsets(
                OffsetsInitializer.committedOffsets(OffsetResetStrategy.EARLIEST))
            .setDeserializer(new RawKafkaEventDeserializer())
            .build();

    SingleOutputStreamOperator<LakeHistoryRow> parsed =
        env.fromSource(source, WatermarkStrategy.noWatermarks(), "feature-history-kafka-source")
            .process(new ParseHistoryEventFunction())
            .returns(LakeHistoryRow.class);
    tableEnvironment.executeSql(
        buildDlqTableSql(
            config.kafkaBootstrapServers,
            config.deadLetterTopic,
            config.kafkaTransactionTimeoutMs));
    tableEnvironment.createTemporaryView(
        "invalid_history",
        tableEnvironment.fromDataStream(parsed.getSideOutput(DLQ)).as("payload_json"));

    SingleOutputStreamOperator<LakeHistoryRow> deduped =
        parsed
            .keyBy(row -> row.targetTable + ":" + row.sourceEventId)
            .process(new SourceEventDeduplicator(config.dedupStateTtlDays))
            .returns(LakeHistoryRow.class)
            .name("feature-history-source-event-dedupe");
    tableEnvironment.createTemporaryView(
        "materialized_history", tableEnvironment.fromDataStream(deduped));

    StatementSet statementSet = tableEnvironment.createStatementSet();
    for (String table : TABLES) {
      statementSet.addInsertSql(buildInsertSql(config.namespace, table));
    }
    statementSet.addInsertSql(
        "INSERT INTO feature_history_dlq SELECT payload_json FROM invalid_history");
    statementSet.execute();
  }

  static void configureEnvironment(StreamExecutionEnvironment env, Config config) {
    env.enableCheckpointing(30_000L, CheckpointingMode.EXACTLY_ONCE);
    env.setStateBackend(new EmbeddedRocksDBStateBackend(true));
    CheckpointConfig checkpoints = env.getCheckpointConfig();
    checkpoints.setCheckpointTimeout(600_000L);
    checkpoints.setMinPauseBetweenCheckpoints(5_000L);
    checkpoints.setExternalizedCheckpointCleanup(
        CheckpointConfig.ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION);
    checkpoints.setCheckpointStorage(config.checkpointDir);
  }

  static String buildRestCatalogSql(
      String catalog,
      String uri,
      String warehouse,
      String endpoint) {
    return String.format(
        "CREATE CATALOG `%s` WITH ("
            + "'type'='iceberg','catalog-type'='rest','uri'='%s','warehouse'='%s',"
            + "'io-impl'='org.apache.iceberg.aws.s3.S3FileIO','s3.endpoint'='%s',"
            + "'s3.path-style-access'='true')",
        sql(catalog), sql(uri), sql(warehouse), sql(endpoint));
  }

  static String buildCreateTableSql(String namespace, String table) {
    if (!TABLES.contains(table)) {
      throw new IllegalArgumentException("unsupported history table " + table);
    }
    String partitions =
        ("user_feature_history".equals(table) || "item_feature_history".equals(table))
            ? "event_date, entity_bucket"
            : "event_date";
    return String.format(
        "CREATE TABLE IF NOT EXISTS `%s`.`%s` ("
            + "source_topic STRING,event_id STRING,source_event_id STRING,event_type STRING,"
            + "source_partition INT,source_offset BIGINT,request_id STRING,"
            + "entity_type STRING,entity_id STRING,entity_bucket INT,user_id STRING,product_id STRING,action STRING,"
            + "event_time TIMESTAMP_LTZ(3),available_at TIMESTAMP_LTZ(3),"
            + "event_time_epoch DOUBLE,available_at_epoch DOUBLE,source_version STRING,"
            + "feature_definition_version STRING,payload_schema_version INT,payload_hash STRING,"
            + "backfill_run_id STRING,"
            + "canonical_payload_json STRING,context_json STRING,candidate_features_json STRING,"
            + "item_snapshot_json STRING,user_features_json STRING,feature_bundle_hash STRING,"
            + "observation_id STRING,event_date DATE) PARTITIONED BY (%s) "
            + "WITH ('format-version'='2','write.upsert.enabled'='false')",
        sql(namespace), table, partitions);
  }

  static String buildDlqTableSql(
      String bootstrapServers, String topic, int transactionTimeoutMs) {
    return String.format(
        "CREATE TEMPORARY TABLE feature_history_dlq (payload_json STRING) WITH ("
            + "'connector'='kafka','topic'='%s','properties.bootstrap.servers'='%s',"
            + "'properties.transaction.timeout.ms'='%d',"
            + "'format'='raw','sink.delivery-guarantee'='exactly-once',"
            + "'sink.transactional-id-prefix'='feature-history-dlq-')",
        sql(topic), sql(bootstrapServers), transactionTimeoutMs);
  }

  static String buildInsertSql(String namespace, String table) {
    return String.format(
        "INSERT INTO `%s`.`%s` SELECT sourceTopic,eventId,sourceEventId,eventType,sourcePartition,"
            + "sourceOffset,requestId,entityType,"
            + "entityId,entityBucket,userId,productId,action,TO_TIMESTAMP_LTZ(CAST(eventTime * 1000 AS BIGINT),3),"
            + "TO_TIMESTAMP_LTZ(CAST(availableAt * 1000 AS BIGINT),3),eventTime,availableAt,sourceVersion,"
            + "featureDefinitionVersion,payloadSchemaVersion,payloadHash,backfillRunId,"
            + "canonicalPayloadJson,"
            + "contextJson,candidateFeaturesJson,itemSnapshotJson,userFeaturesJson,"
            + "featureBundleHash,observationId,CAST(eventDate AS DATE) FROM materialized_history "
            + "WHERE targetTable='%s'",
        sql(namespace), table, table);
  }

  static String resolveNamespace(String baseNamespace, String replayRunId, boolean allowOffsetReset) {
    String base = required("FEATURE_LAKE_NAMESPACE", baseNamespace);
    String runId = replayRunId == null ? "" : replayRunId.trim();
    if (runId.isEmpty()) {
      if (allowOffsetReset) {
        throw new IllegalArgumentException(
            "production namespace forbids offset reset; configure FEATURE_LAKE_REPLAY_RUN_ID");
      }
      return base;
    }
    String safeRunId = runId.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9_]", "_");
    if (safeRunId.isBlank()) {
      throw new IllegalArgumentException("FEATURE_LAKE_REPLAY_RUN_ID is invalid");
    }
    return base + "_rebuild_" + safeRunId;
  }

  static List<LakeHistoryRow> parseRawEvent(RawKafkaEvent raw) throws Exception {
    Map<String, Object> event = JSON.readValue(raw.rawValue, MAP_TYPE);
    List<LakeHistoryRow> rows;
    if (raw.topic.endsWith("user-interactions")
        || raw.topic.endsWith("user-interactions-backfill")) {
      rows = Collections.singletonList(parseInteraction(raw.topic, event));
    } else if (raw.topic.endsWith("recommendation-events")
        || raw.topic.endsWith("recommendation-events-backfill")) {
      rows = parseRecommendation(raw.topic, event);
    } else if (raw.topic.endsWith("feature-history-snapshots")
        || raw.topic.endsWith("feature-updates-backfill")) {
      rows = Collections.singletonList(parseFeature(raw.topic, event));
    } else if (raw.topic.endsWith("catalog-feature-events")
        || raw.topic.endsWith("catalog-feature-events-backfill")) {
      rows = Collections.singletonList(parseFeature(raw.topic, event));
    } else {
      throw new IllegalArgumentException("unsupported source topic " + raw.topic);
    }
    for (LakeHistoryRow row : rows) {
      row.sourcePartition = raw.partition;
      row.sourceOffset = raw.offset;
      row.requestId = nullable(event.get("request_id"));
    }
    return rows;
  }

  private static LakeHistoryRow parseInteraction(String topic, Map<String, Object> event)
      throws Exception {
    String eventId = required("event_id", event.get("event_id"));
    String userId = required("user_id", event.get("user_id"));
    String productId = required("product_id", event.get("product_id"));
    String action = required("action", event.get("action"));
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("user_id", userId);
    payload.put("product_id", productId);
    payload.put("action", action);
    payload.put("context", safeMap(event.get("context")));
    String payloadHash = required("payload_hash", event.get("payload_hash"));
    if (!payloadHash.equals(FeatureHistoryContract.payloadHash(payload))) {
      throw new IllegalArgumentException("payload_hash does not match interaction payload");
    }
    LakeHistoryRow row = baseRow(topic, event, payload);
    row.targetTable = "interaction_history";
    row.eventId = eventId;
    row.entityType = "user";
    row.entityId = userId;
    row.entityBucket = entityBucket(userId);
    row.userId = userId;
    row.productId = productId;
    row.action = action;
    row.contextJson = FeatureHistoryContract.canonicalJson(safeMap(event.get("context")));
    return row;
  }

  private static List<LakeHistoryRow> parseRecommendation(
      String topic, Map<String, Object> event) throws Exception {
    requireFixedContract(event);
    Map<String, Object> metadata = safeMap(event.get("metadata"));
    String impressionId = required("impression_id", metadata.get("impression_id"));
    String userId = required("user_id", event.get("user_id"));
    double eventTime = number("event_time", event.get("event_time"));
    double availableAt = number("available_at", event.get("available_at"));
    String sourceEvent = required("source_event_id", event.get("source_event_id"));
    String sourceVersion = required("source_version", event.get("source_version"));
    String definition =
        required("feature_definition_version", event.get("feature_definition_version"));
    int schemaVersion = integer("payload_schema_version", event.get("payload_schema_version"));
    Map<String, Object> context = safeMap(metadata.get("feature_context"));
    Map<String, Object> userFeatures = safeMap(metadata.get("user_feature_snapshot"));
    List<Object> displayedItems = safeList(metadata.get("displayed_items"));
    Map<String, Object> sourcePayload = new LinkedHashMap<>();
    sourcePayload.put("user_id", userId);
    sourcePayload.put("recommendations", safeList(event.get("recommendations")));
    sourcePayload.put("response_time_ms", event.get("response_time_ms"));
    sourcePayload.put("metadata", metadata);
    String sourcePayloadHash = required("payload_hash", event.get("payload_hash"));
    if (!sourcePayloadHash.equals(FeatureHistoryContract.payloadHash(sourcePayload))) {
      throw new IllegalArgumentException("payload_hash does not match recommendation payload");
    }
    List<LakeHistoryRow> rows = new ArrayList<>();
    for (Object rawItem : displayedItems) {
      Map<String, Object> item = safeMap(rawItem);
      String productId = required("displayed_items.product_id", item.get("product_id"));
      String bundleHash = required("displayed_items.feature_bundle_hash", item.get("feature_bundle_hash"));
      String itemDefinition =
          required(
              "displayed_items.feature_definition_version",
              item.get("feature_definition_version"));
      if (!definition.equals(itemDefinition)) {
        throw new IllegalArgumentException(
            "displayed item feature definition version does not match event");
      }
      String observationId = impressionId + ":" + productId;
      String observationSourceId =
          FeatureHistoryContract.deterministicId(
              sourceEvent, observationId, String.valueOf(item.getOrDefault("position", 0)));
      Map<String, Object> payload = new LinkedHashMap<>();
      payload.put("impression_id", impressionId);
      payload.put("user_id", userId);
      payload.put("product_id", productId);
      payload.put("position", item.get("position"));
      payload.put("context", context);
      payload.put("user_features", userFeatures);
      payload.put("item_snapshot", safeMap(item.get("feature_snapshot")));
      payload.put("candidate_features", safeMap(item.get("scores")));
      payload.put("item_snapshot_complete", item.get("item_snapshot_complete"));
      payload.put("feature_bundle_hash", bundleHash);
      LakeHistoryRow row = new LakeHistoryRow();
      row.targetTable = "ranking_observations";
      row.sourceTopic = topic;
      row.eventId = observationSourceId;
      row.sourceEventId = observationSourceId;
      row.eventType = "ranking_observation";
      row.entityType = "item";
      row.entityId = productId;
      row.entityBucket = entityBucket(productId);
      row.userId = userId;
      row.productId = productId;
      row.eventTime = eventTime;
      row.availableAt = availableAt;
      row.sourceVersion = sourceVersion;
      row.featureDefinitionVersion = definition;
      row.payloadSchemaVersion = schemaVersion;
      row.payloadHash = FeatureHistoryContract.payloadHash(payload);
      row.canonicalPayloadJson = FeatureHistoryContract.canonicalJson(payload);
      row.contextJson = FeatureHistoryContract.canonicalJson(context);
      row.candidateFeaturesJson =
          FeatureHistoryContract.canonicalJson(safeMap(item.get("scores")));
      row.itemSnapshotJson =
          FeatureHistoryContract.canonicalJson(safeMap(item.get("feature_snapshot")));
      row.userFeaturesJson = FeatureHistoryContract.canonicalJson(userFeatures);
      row.featureBundleHash = bundleHash;
      row.observationId = observationId;
      row.eventDate = eventDate(eventTime);
      rows.add(row);
    }
    if (rows.isEmpty()) {
      throw new IllegalArgumentException("recommendation event has no displayed items");
    }
    return rows;
  }

  private static LakeHistoryRow parseFeature(String topic, Map<String, Object> event)
      throws Exception {
    FeatureHistoryContract.Record record = FeatureHistoryContract.parse(event);
    if (!"ranking_ltr_v1".equals(record.featureDefinitionVersion)) {
      throw new IllegalArgumentException("unsupported feature_definition_version");
    }
    LakeHistoryRow row = new LakeHistoryRow();
    if ("catalog_feature".equals(record.eventType) && "item".equals(record.entityType)) {
      row.targetTable = "item_feature_history";
    } else if ("window_feature".equals(record.eventType)
        && "user".equals(record.entityType)) {
      row.targetTable = "window_feature_history";
    } else if ("user_feature".equals(record.eventType)
        && "user".equals(record.entityType)) {
      row.targetTable = "user_feature_history";
    } else {
      throw new IllegalArgumentException(
          "unsupported feature history event/entity combination: "
              + record.eventType
              + "/"
              + record.entityType);
    }
    row.sourceTopic = topic;
    row.eventId = record.eventId;
    row.sourceEventId = record.sourceEventId;
    row.eventType = record.eventType;
    row.entityType = record.entityType;
    row.entityId = record.entityId;
    row.entityBucket = entityBucket(record.entityId);
    row.userId = "user".equals(record.entityType) ? record.entityId : null;
    row.productId = "item".equals(record.entityType) ? record.entityId : null;
    row.eventTime = record.eventTime;
    row.availableAt = record.availableAt;
    row.sourceVersion = record.sourceVersion;
    row.featureDefinitionVersion = record.featureDefinitionVersion;
    row.payloadSchemaVersion = record.payloadSchemaVersion;
    row.payloadHash = record.payloadHash;
    row.backfillRunId = nullable(event.get("backfill_run_id"));
    row.canonicalPayloadJson = FeatureHistoryContract.canonicalJson(record.payload);
    row.eventDate = eventDate(record.eventTime);
    return row;
  }

  private static LakeHistoryRow baseRow(
      String topic, Map<String, Object> event, Map<String, Object> payload) throws Exception {
    requireFixedContract(event);
    LakeHistoryRow row = new LakeHistoryRow();
    row.sourceTopic = topic;
    row.eventId = required("event_id", event.get("event_id"));
    row.sourceEventId = required("source_event_id", event.get("source_event_id"));
    row.eventType = required("event_type", event.get("event_type"));
    row.eventTime = number("event_time", event.get("event_time"));
    row.availableAt = number("available_at", event.get("available_at"));
    row.sourceVersion = required("source_version", event.get("source_version"));
    row.featureDefinitionVersion =
        required("feature_definition_version", event.get("feature_definition_version"));
    row.payloadSchemaVersion =
        integer("payload_schema_version", event.get("payload_schema_version"));
    row.payloadHash = required("payload_hash", event.get("payload_hash"));
    row.backfillRunId = nullable(event.get("backfill_run_id"));
    row.canonicalPayloadJson = FeatureHistoryContract.canonicalJson(payload);
    row.eventDate = eventDate(row.eventTime);
    return row;
  }

  static String dlqPayload(RawKafkaEvent raw, Exception error) throws Exception {
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("schema_version", 1);
    String dlqEventId =
        FeatureHistoryContract.deterministicId(
            raw.topic, String.valueOf(raw.partition), String.valueOf(raw.offset));
    payload.put("event_id", dlqEventId);
    payload.put("event_type", "feature_history_materialization_error");
    payload.put("source_event_id", dlqEventId);
    payload.put("source_topic", raw.topic);
    payload.put("source_partition", raw.partition);
    payload.put("source_offset", raw.offset);
    payload.put("source_value", raw.rawValue);
    try {
      Map<String, Object> source = JSON.readValue(raw.rawValue, MAP_TYPE);
      payload.put("backfill_run_id", nullable(source.get("backfill_run_id")));
      payload.put("failed_source_event_id", nullable(source.get("source_event_id")));
    } catch (Exception ignored) {
      payload.put("backfill_run_id", null);
      payload.put("failed_source_event_id", null);
    }
    payload.put("error_type", error.getClass().getSimpleName());
    payload.put("error_message", error.getMessage());
    payload.put("occurred_at", raw.timestamp / 1000.0);
    return JSON.writeValueAsString(payload);
  }

  private static void requireFixedContract(Map<String, Object> event) {
    if (integer("payload_schema_version", event.get("payload_schema_version")) != 1) {
      throw new IllegalArgumentException("unsupported payload_schema_version");
    }
    if (!"ranking_ltr_v1".equals(
        required("feature_definition_version", event.get("feature_definition_version")))) {
      throw new IllegalArgumentException("unsupported feature_definition_version");
    }
  }

  private static String eventDate(double seconds) {
    return Instant.ofEpochMilli((long) (seconds * 1000.0))
        .atZone(ZoneOffset.UTC)
        .toLocalDate()
        .toString();
  }

  static int entityBucket(String entityId) throws Exception {
    String digest = FeatureHistoryContract.deterministicId(required("entity_id", entityId));
    return (int) (Long.parseUnsignedLong(digest.substring(0, 8), 16) % 32L);
  }

  @SuppressWarnings("unchecked")
  private static Map<String, Object> safeMap(Object value) {
    return value instanceof Map
        ? new LinkedHashMap<>((Map<String, Object>) value)
        : new LinkedHashMap<>();
  }

  @SuppressWarnings("unchecked")
  private static List<Object> safeList(Object value) {
    if (!(value instanceof List)) {
      throw new IllegalArgumentException("displayed_items must be a list");
    }
    return (List<Object>) value;
  }

  private static String required(String name, Object value) {
    String normalized = value == null ? "" : String.valueOf(value).trim();
    if (normalized.isEmpty()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return normalized;
  }

  private static String nullable(Object value) {
    if (value == null) {
      return null;
    }
    String normalized = String.valueOf(value).trim();
    return normalized.isEmpty() ? null : normalized;
  }

  private static double number(String name, Object value) {
    if (!(value instanceof Number) || !Double.isFinite(((Number) value).doubleValue())) {
      throw new IllegalArgumentException(name + " must be a finite number");
    }
    return ((Number) value).doubleValue();
  }

  private static int integer(String name, Object value) {
    if (!(value instanceof Number)) {
      throw new IllegalArgumentException(name + " must be numeric");
    }
    return ((Number) value).intValue();
  }

  private static String sql(String value) {
    return value.replace("'", "''");
  }

  static final class ParseHistoryEventFunction
      extends ProcessFunction<RawKafkaEvent, LakeHistoryRow> {
    private transient Counter records;
    private transient Counter dlq;
    private transient Map<String, Counter> recordsByType;
    private transient Map<String, AtomicLong> lagByType;

    @Override
    public void open(Configuration parameters) {
      records = getRuntimeContext().getMetricGroup().counter("feature_lake_records_total");
      dlq = getRuntimeContext().getMetricGroup().counter("feature_lake_dlq_total");
      recordsByType = new HashMap<>();
      lagByType = new HashMap<>();
    }

    @Override
    public void processElement(
        RawKafkaEvent raw, Context context, Collector<LakeHistoryRow> output) throws Exception {
      try {
        for (LakeHistoryRow row : parseRawEvent(raw)) {
          records.inc();
          recordsByType
              .computeIfAbsent(
                  row.eventType,
                  type ->
                      getRuntimeContext()
                          .getMetricGroup()
                          .addGroup("record_type", type)
                          .counter("feature_lake_records_total"))
              .inc();
          AtomicLong lag =
              lagByType.computeIfAbsent(
                  row.eventType,
                  type -> {
                    AtomicLong value = new AtomicLong();
                    getRuntimeContext()
                        .getMetricGroup()
                        .addGroup("record_type", type)
                        .gauge(
                            "feature_lake_materialization_lag_seconds",
                            () -> value.get() / 1000.0);
                    return value;
                  });
          lag.set(
              Math.max(0L, System.currentTimeMillis() - (long) (row.availableAt * 1000.0)));
          output.collect(row);
        }
      } catch (Exception error) {
        dlq.inc();
        context.output(DLQ, dlqPayload(raw, error));
      }
    }
  }

  static final class SourceEventDeduplicator
      extends KeyedProcessFunction<String, LakeHistoryRow, LakeHistoryRow> {
    private final int ttlDays;
    private transient ValueState<Boolean> seen;

    SourceEventDeduplicator(int ttlDays) {
      this.ttlDays = ttlDays;
    }

    @Override
    public void open(Configuration parameters) {
      ValueStateDescriptor<Boolean> descriptor =
          new ValueStateDescriptor<>("materialized-source-event", Boolean.class);
      descriptor.enableTimeToLive(
          StateTtlConfig.newBuilder(org.apache.flink.api.common.time.Time.days(ttlDays))
              .setUpdateType(StateTtlConfig.UpdateType.OnCreateAndWrite)
              .setStateVisibility(StateTtlConfig.StateVisibility.NeverReturnExpired)
              .build());
      seen = getRuntimeContext().getState(descriptor);
    }

    @Override
    public void processElement(
        LakeHistoryRow row, Context context, Collector<LakeHistoryRow> output) throws Exception {
      if (seen.value() == null) {
        seen.update(Boolean.TRUE);
        output.collect(row);
      }
    }
  }

  static final class RawKafkaEventDeserializer
      implements KafkaRecordDeserializationSchema<RawKafkaEvent> {
    @Override
    public void deserialize(ConsumerRecord<byte[], byte[]> record, Collector<RawKafkaEvent> out) {
      RawKafkaEvent event = new RawKafkaEvent();
      event.topic = record.topic();
      event.partition = record.partition();
      event.offset = record.offset();
      event.timestamp = Math.max(0L, record.timestamp());
      event.rawValue =
          record.value() == null ? "" : new String(record.value(), StandardCharsets.UTF_8);
      out.collect(event);
    }

    @Override
    public TypeInformation<RawKafkaEvent> getProducedType() {
      return TypeInformation.of(RawKafkaEvent.class);
    }
  }

  public static final class RawKafkaEvent implements Serializable {
    public String topic;
    public int partition;
    public long offset;
    public long timestamp;
    public String rawValue;

    public RawKafkaEvent() {}

    RawKafkaEvent(String topic, String rawValue) {
      this.topic = topic;
      this.rawValue = rawValue;
    }
  }

  public static final class LakeHistoryRow implements Serializable {
    public String targetTable;
    public String sourceTopic;
    public String eventId;
    public String sourceEventId;
    public String eventType;
    public int sourcePartition;
    public long sourceOffset;
    public String requestId;
    public String entityType;
    public String entityId;
    public int entityBucket;
    public String userId;
    public String productId;
    public String action;
    public double eventTime;
    public double availableAt;
    public String sourceVersion;
    public String featureDefinitionVersion;
    public int payloadSchemaVersion;
    public String payloadHash;
    public String backfillRunId;
    public String canonicalPayloadJson;
    public String contextJson;
    public String candidateFeaturesJson;
    public String itemSnapshotJson;
    public String userFeaturesJson;
    public String featureBundleHash;
    public String observationId;
    public String eventDate;
  }

  static final class Config {
    String kafkaBootstrapServers;
    List<String> topics;
    String deadLetterTopic;
    int kafkaTransactionTimeoutMs;
    String consumerGroupId;
    String checkpointDir;
    int dedupStateTtlDays;
    String catalogName;
    String catalogUri;
    String warehouseUri;
    String namespace;
    String s3Endpoint;

    static Config fromEnv() {
      Config config = new Config();
      String replayRunId = env("FEATURE_LAKE_REPLAY_RUN_ID", "");
      boolean allowOffsetReset = Boolean.parseBoolean(env("FEATURE_LAKE_ALLOW_OFFSET_RESET", "false"));
      config.kafkaBootstrapServers = required("KAFKA_BOOTSTRAP_SERVERS", env("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"));
      config.topics =
          new ArrayList<>(
              Arrays.asList(
                  env("KAFKA_USER_INTERACTIONS_TOPIC", "user-interactions"),
                  env("KAFKA_RECOMMENDATION_EVENTS_TOPIC", "recommendation-events"),
                  env("KAFKA_FEATURE_UPDATES_TOPIC", "feature-updates"),
                  env("KAFKA_CATALOG_FEATURE_EVENTS_TOPIC", "catalog-feature-events")));
      if (Boolean.parseBoolean(env("FEATURE_HISTORY_INCLUDE_BACKFILL_TOPICS", "false"))) {
        config.topics.add(env("KAFKA_USER_INTERACTIONS_BACKFILL_TOPIC", "user-interactions-backfill"));
        config.topics.add(env("KAFKA_RECOMMENDATION_EVENTS_BACKFILL_TOPIC", "recommendation-events-backfill"));
        config.topics.add(env("KAFKA_FEATURE_UPDATES_BACKFILL_TOPIC", "feature-updates-backfill"));
        config.topics.add(env("KAFKA_CATALOG_FEATURE_EVENTS_BACKFILL_TOPIC", "catalog-feature-events-backfill"));
      }
      config.deadLetterTopic = env("KAFKA_DEAD_LETTER_TOPIC", "dead-letter-events");
      config.kafkaTransactionTimeoutMs =
          Integer.parseInt(env("FLINK_KAFKA_TRANSACTION_TIMEOUT_MS", "600000"));
      config.consumerGroupId =
          replayRunId.isBlank()
              ? PRODUCTION_GROUP_ID
              : PRODUCTION_GROUP_ID + "-rebuild-" + replayRunId.replaceAll("[^A-Za-z0-9_-]", "_");
      config.checkpointDir =
          required(
              "FEATURE_HISTORY_CHECKPOINT_DIR",
              env("FEATURE_HISTORY_CHECKPOINT_DIR", "file:///flink/checkpoints/feature-history-materializer"));
      config.dedupStateTtlDays = Integer.parseInt(env("FEATURE_HISTORY_DEDUP_STATE_TTL_DAYS", "400"));
      config.catalogName = env("FEATURE_LAKE_CATALOG_NAME", "feature_catalog");
      config.catalogUri = required("FEATURE_LAKE_CATALOG_URI", env("FEATURE_LAKE_CATALOG_URI", ""));
      config.warehouseUri = required("FEATURE_LAKE_WAREHOUSE_URI", env("FEATURE_LAKE_WAREHOUSE_URI", ""));
      config.namespace =
          resolveNamespace(env("FEATURE_LAKE_NAMESPACE", "video_commerce"), replayRunId, allowOffsetReset);
      config.s3Endpoint = required("FEATURE_LAKE_S3_ENDPOINT", env("FEATURE_LAKE_S3_ENDPOINT", ""));
      if (config.dedupStateTtlDays <= 0) {
        throw new IllegalArgumentException("FEATURE_HISTORY_DEDUP_STATE_TTL_DAYS must be positive");
      }
      if (config.kafkaTransactionTimeoutMs <= 0) {
        throw new IllegalArgumentException("FLINK_KAFKA_TRANSACTION_TIMEOUT_MS must be positive");
      }
      return config;
    }
  }

  private static String env(String name, String defaultValue) {
    String value = System.getenv(name);
    return value == null || value.trim().isEmpty() ? defaultValue : value.trim();
  }
}
