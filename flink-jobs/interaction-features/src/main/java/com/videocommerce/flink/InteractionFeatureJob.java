package com.videocommerce.flink;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.lettuce.core.RedisClient;
import io.lettuce.core.RedisURI;
import io.lettuce.core.ScriptOutputType;
import io.lettuce.core.codec.ByteArrayCodec;
import io.lettuce.core.api.StatefulRedisConnection;
import io.lettuce.core.api.sync.RedisCommands;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.sql.Timestamp;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.metrics.Counter;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.api.common.state.StateTtlConfig;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeHint;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.base.DeliveryGuarantee;
import org.apache.flink.connector.jdbc.JdbcConnectionOptions;
import org.apache.flink.connector.jdbc.JdbcExecutionOptions;
import org.apache.flink.connector.jdbc.JdbcSink;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.contrib.streaming.state.EmbeddedRocksDBStateBackend;
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.CheckpointConfig;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;
import org.apache.flink.streaming.api.functions.windowing.ProcessWindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.SlidingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.kafka.clients.consumer.OffsetResetStrategy;
import org.msgpack.jackson.dataformat.MessagePackFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class InteractionFeatureJob {
  private static final ObjectMapper JSON = new ObjectMapper();
  private static final ObjectMapper MSGPACK = new ObjectMapper(new MessagePackFactory());
  private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {};
  private static final Logger LOG = LoggerFactory.getLogger(InteractionFeatureJob.class);

  private static final OutputTag<DlqEvent> DLQ_TAG =
      new OutputTag<>("dead-letter-events", TypeInformation.of(DlqEvent.class));
  private static final String USER_INTERACTIONS_SOURCE_TOPIC = "user-interactions";
  private static final String RECOMMENDATION_EVENTS_SOURCE_TOPIC = "recommendation-events";

  public static void main(String[] args) throws Exception {
    JobConfig config = JobConfig.fromEnv();
    StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
    configureEnvironment(env, config);

    KafkaSource<String> source =
        KafkaSource.<String>builder()
            .setBootstrapServers(config.kafkaBootstrapServers)
            .setTopics(config.userInteractionsTopic)
            .setGroupId(config.consumerGroupId)
            .setStartingOffsets(
                OffsetsInitializer.committedOffsets(OffsetResetStrategy.EARLIEST))
            .setValueOnlyDeserializer(new SimpleStringSchema())
            .build();

    SingleOutputStreamOperator<InteractionEvent> parsed =
        env.fromSource(source, WatermarkStrategy.noWatermarks(), "kafka-user-interactions")
            .process(new ParseInteractionEventFunction(config.userInteractionsTopic))
            .returns(InteractionEvent.class);

    parsed
        .getSideOutput(DLQ_TAG)
        .map(new DlqJsonMapper())
        .sinkTo(buildKafkaSink(config.kafkaBootstrapServers, config.deadLetterTopic, null, config))
        .name("invalid-event-dlq");

    KafkaSource<String> recommendationSource =
        KafkaSource.<String>builder()
            .setBootstrapServers(config.kafkaBootstrapServers)
            .setTopics(config.recommendationEventsTopic)
            .setGroupId(config.consumerGroupId)
            .setStartingOffsets(
                OffsetsInitializer.committedOffsets(OffsetResetStrategy.EARLIEST))
            .setValueOnlyDeserializer(new SimpleStringSchema())
            .build();

    SingleOutputStreamOperator<RecommendationImpressionEvent> recommendationEvents =
        env.fromSource(
                recommendationSource,
                WatermarkStrategy.noWatermarks(),
                "kafka-recommendation-events")
            .process(new ParseRecommendationEventFunction(config.recommendationEventsTopic))
            .returns(RecommendationImpressionEvent.class);

    recommendationEvents
        .getSideOutput(DLQ_TAG)
        .map(new DlqJsonMapper())
        .sinkTo(buildKafkaSink(config.kafkaBootstrapServers, config.deadLetterTopic, null, config))
        .name("invalid-recommendation-event-dlq");

    DataStream<RecommendationImpressionEvent> servedRecommendationEvents =
        recommendationEvents.filter(event -> !event.viewed).name("served-recommendation-events");

    servedRecommendationEvents
        .addSink(buildRecommendationImpressionSink(config))
        .name("postgres-recommendation-impressions");

    servedRecommendationEvents
        .flatMap(new RecommendationImpressionItemFanout())
        .returns(RecommendationImpressionItemRow.class)
        .addSink(buildRecommendationImpressionItemSink(config))
        .name("postgres-recommendation-impression-items");

    recommendationEvents
        .filter(event -> event.viewed)
        .flatMap(new RecommendationImpressionItemFanout())
        .returns(RecommendationImpressionItemRow.class)
        .addSink(buildRecommendationImpressionViewSink(config))
        .name("postgres-recommendation-impression-views");

    DataStream<InteractionEvent> events =
        parsed.assignTimestampsAndWatermarks(
            WatermarkStrategy.<InteractionEvent>forBoundedOutOfOrderness(
                    Duration.ofSeconds(config.watermarkOutOfOrdernessSeconds))
                .withIdleness(Duration.ofSeconds(config.watermarkIdlenessSeconds))
                .withTimestampAssigner((event, previousTimestamp) -> event.eventTimeMillis));

    DataStream<InteractionEvent> deduped =
        events
            .keyBy((KeySelector<InteractionEvent, String>) event -> event.eventId)
            .process(new EventIdDeduplicator(config.dedupStateTtlDays))
            .name("dedupe-by-event-id")
            .setUidHash("baf1e4c622ccdfb35723ebd10e622fa5")
            .returns(InteractionEvent.class);

    deduped
        .addSink(buildPostgresSink(config))
        .name("postgres-interaction-events");

    DataStream<UserFeatureSnapshot> userFeatures =
        deduped
            .keyBy((KeySelector<InteractionEvent, String>) event -> event.userId)
            .process(new UserFeatureAggregator(config.featureStateEpoch))
            .name("user-feature-snapshots")
            .setUidHash("8347a046395918dded45c1e79a2e6b85")
            .returns(UserFeatureSnapshot.class);

    userFeatures
        .addSink(new UserFeatureRedisSink(config.redisConfig(), config.featureStateEpoch))
        .name("redis-user-feature-snapshots");

    userFeatures
        .map(new FeatureUpdateJsonMapper())
        .sinkTo(
            buildKafkaSink(
                config.kafkaBootstrapServers,
                config.featureUpdatesTopic,
                "interaction-features-updates",
                config))
        .name("feature-update-events");

    DataStream<EntityEvent> entityEvents =
        deduped.flatMap(new EntityEventFanout()).returns(EntityEvent.class);
    WindowStreams window5m =
        buildWindow(entityEvents, "5m", Time.minutes(5), Time.minutes(1), config);
    WindowStreams window1h =
        buildWindow(entityEvents, "1h", Time.hours(1), Time.minutes(5), config);
    WindowStreams window24h =
        buildWindow(entityEvents, "24h", Time.hours(24), Time.hours(1), config);
    DataStream<WindowFeatureSnapshot> windowFeatures =
        window5m.snapshots.union(window1h.snapshots).union(window24h.snapshots);
    window5m.lateEvents
        .union(window1h.lateEvents)
        .union(window24h.lateEvents)
        .map(event -> JSON.writeValueAsString(event.payload()))
        .returns(String.class)
        .sinkTo(
            buildKafkaSink(
                config.kafkaBootstrapServers, config.lateInteractionsTopic, null, config))
        .name("late-interaction-quarantine");

    windowFeatures
        .addSink(new WindowFeatureRedisSink(config.redisConfig()))
        .name("redis-realtime-window-features");

    windowFeatures
        .map(new WindowFeatureUpdateJsonMapper())
        .sinkTo(
            buildKafkaSink(
                config.kafkaBootstrapServers,
                config.featureUpdatesTopic,
                "interaction-window-feature-updates",
                config))
        .name("window-feature-update-events");

    env.execute("video-commerce-interaction-features");
  }

  private static void configureEnvironment(StreamExecutionEnvironment env, JobConfig config)
      throws Exception {
    env.enableCheckpointing(30_000L, CheckpointingMode.EXACTLY_ONCE);
    env.setStateBackend(new EmbeddedRocksDBStateBackend(true));
    CheckpointConfig checkpointConfig = env.getCheckpointConfig();
    checkpointConfig.setCheckpointTimeout(600_000L);
    checkpointConfig.setMinPauseBetweenCheckpoints(5_000L);
    checkpointConfig.setTolerableCheckpointFailureNumber(3);
    checkpointConfig.setExternalizedCheckpointCleanup(
        CheckpointConfig.ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION);
    if (!config.checkpointDir.isBlank()) {
      checkpointConfig.setCheckpointStorage(config.checkpointDir);
    }
    // Bounded restarts: a deterministic failure must end in FAILED (and fire
    // FlinkInteractionFeatureJobMissing) instead of an endless restart loop.
    Configuration restart = new Configuration();
    restart.setString("restart-strategy.type", "failure-rate");
    restart.setString("restart-strategy.failure-rate.max-failures-per-interval", "10");
    restart.setString("restart-strategy.failure-rate.failure-rate-interval", "10 min");
    restart.setString("restart-strategy.failure-rate.delay", "10 s");
    env.configure(restart);
  }

  private static WindowStreams buildWindow(
      DataStream<EntityEvent> events,
      String windowName,
      Time size,
      Time slide,
      JobConfig config) {
    OutputTag<EntityEvent> lateTag =
        new OutputTag<>("late-" + windowName, TypeInformation.of(EntityEvent.class));
    SingleOutputStreamOperator<WindowFeatureSnapshot> snapshots =
        events
        .keyBy((KeySelector<EntityEvent, EntityKey>) event -> event.key)
        .window(SlidingEventTimeWindows.of(size, slide))
        .allowedLateness(Time.seconds(config.allowedLatenessSeconds))
        .sideOutputLateData(lateTag)
        .aggregate(new WindowCountsAggregate(), new WindowSnapshotFunction(windowName))
        .returns(WindowFeatureSnapshot.class)
        .name("window-" + windowName)
        .setUidHash(windowUidHash(windowName));
    DataStream<LateInteractionEvent> lateEvents =
        snapshots
            .getSideOutput(lateTag)
            .process(new LateInteractionMapper(windowName))
            .returns(LateInteractionEvent.class)
            .name("quarantine-late-" + windowName)
            .uid("quarantine-late-" + windowName);
    return new WindowStreams(snapshots, lateEvents);
  }

  static String windowUidHash(String windowName) {
    if ("5m".equals(windowName)) return "cf0eeab4e18fc06c05154ab68936a8a3";
    if ("1h".equals(windowName)) return "afd5ffb424d61a18dbb90f74ee3ad4b0";
    if ("24h".equals(windowName)) return "95a9dff8bd5d7246318a578d3eb26e6e";
    throw new IllegalArgumentException("Unknown window name: " + windowName);
  }

  private static org.apache.flink.streaming.api.functions.sink.SinkFunction<InteractionEvent>
      buildPostgresSink(JobConfig config) {
    String sql =
        "INSERT INTO interaction_events "
            + "(event_id, schema_version, request_id, user_id, product_id, action, context, occurred_at) "
            + "VALUES (?, ?, ?, ?, ?, ?, CAST(? AS json), ?) "
            + "ON CONFLICT DO NOTHING";
    return JdbcSink.sink(
        sql,
        (statement, event) -> {
          statement.setString(1, event.eventId);
          statement.setInt(2, event.schemaVersion);
          statement.setString(3, event.requestId);
          statement.setString(4, event.userId);
          statement.setString(5, event.productId);
          statement.setString(6, event.action);
          statement.setString(7, event.contextJson);
          statement.setTimestamp(8, Timestamp.from(Instant.ofEpochMilli(event.eventTimeMillis)));
        },
        JdbcExecutionOptions.builder()
            .withBatchSize(config.jdbcBatchSize)
            .withBatchIntervalMs(config.jdbcBatchIntervalMs)
            .withMaxRetries(config.jdbcMaxRetries)
            .build(),
        new JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
            .withUrl(config.postgresJdbcUrl)
            .withDriverName("org.postgresql.Driver")
            .withUsername(config.postgresUser)
            .withPassword(config.postgresPassword)
            .build());
  }

  private static org.apache.flink.streaming.api.functions.sink.SinkFunction<RecommendationImpressionEvent>
      buildRecommendationImpressionSink(JobConfig config) {
    String sql =
        "INSERT INTO recommendation_impressions "
            + "(impression_id, request_id, user_id, session_id, content_id, model_version, "
            + "ranking_model_version, context, created_at) "
            + "VALUES (?, ?, ?, ?, ?, ?, ?, CAST(? AS json), ?) "
            + "ON CONFLICT (impression_id) DO UPDATE SET "
            + "request_id = EXCLUDED.request_id, "
            + "user_id = EXCLUDED.user_id, "
            + "session_id = EXCLUDED.session_id, "
            + "content_id = EXCLUDED.content_id, "
            + "model_version = EXCLUDED.model_version, "
            + "ranking_model_version = EXCLUDED.ranking_model_version, "
            + "context = EXCLUDED.context, "
            + "created_at = EXCLUDED.created_at";
    return JdbcSink.sink(
        sql,
        (statement, event) -> {
          statement.setString(1, event.impressionId);
          statement.setString(2, event.requestId);
          statement.setString(3, event.userId);
          statement.setString(4, event.sessionId);
          statement.setString(5, event.contentId);
          statement.setString(6, event.modelVersion);
          statement.setString(7, event.rankingModelVersion);
          statement.setString(8, jsonString(event.context));
          statement.setTimestamp(9, Timestamp.from(Instant.ofEpochMilli(event.eventTimeMillis)));
        },
        JdbcExecutionOptions.builder()
            .withBatchSize(config.jdbcBatchSize)
            .withBatchIntervalMs(config.jdbcBatchIntervalMs)
            .withMaxRetries(config.jdbcMaxRetries)
            .build(),
        new JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
            .withUrl(config.postgresJdbcUrl)
            .withDriverName("org.postgresql.Driver")
            .withUsername(config.postgresUser)
            .withPassword(config.postgresPassword)
            .build());
  }

  private static org.apache.flink.streaming.api.functions.sink.SinkFunction<RecommendationImpressionItemRow>
      buildRecommendationImpressionItemSink(JobConfig config) {
    String sql =
        "INSERT INTO recommendation_impression_items "
            + "(impression_id, product_id, position, source, feature_snapshot, scores, created_at) "
            + "VALUES (?, ?, ?, ?, CAST(? AS json), CAST(? AS json), ?) "
            + "ON CONFLICT (impression_id, product_id) DO UPDATE SET "
            + "position = EXCLUDED.position, "
            + "source = EXCLUDED.source, "
            + "feature_snapshot = EXCLUDED.feature_snapshot, "
            + "scores = EXCLUDED.scores, "
            + "created_at = EXCLUDED.created_at";
    return JdbcSink.sink(
        sql,
        (statement, item) -> {
          statement.setString(1, item.impressionId);
          statement.setString(2, item.productId);
          statement.setInt(3, item.position);
          statement.setString(4, item.source);
          statement.setString(5, jsonString(item.featureSnapshot));
          statement.setString(6, jsonString(item.scores));
          statement.setTimestamp(7, Timestamp.from(Instant.ofEpochMilli(item.eventTimeMillis)));
        },
        JdbcExecutionOptions.builder()
            .withBatchSize(config.jdbcBatchSize)
            .withBatchIntervalMs(config.jdbcBatchIntervalMs)
            .withMaxRetries(config.jdbcMaxRetries)
            .build(),
        new JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
            .withUrl(config.postgresJdbcUrl)
            .withDriverName("org.postgresql.Driver")
            .withUsername(config.postgresUser)
            .withPassword(config.postgresPassword)
            .build());
  }

  private static org.apache.flink.streaming.api.functions.sink.SinkFunction<RecommendationImpressionItemRow>
      buildRecommendationImpressionViewSink(JobConfig config) {
    String sql =
        "INSERT INTO recommendation_impression_views "
            + "(event_id, impression_id, product_id, position, context, viewed_at) "
            + "VALUES (?, ?, ?, ?, CAST(? AS json), ?) "
            // No conflict target: the table has both an event_id PK and a unique
            // (impression_id, product_id) index; either duplicate must be a no-op,
            // not a sink failure that replays forever.
            + "ON CONFLICT DO NOTHING";
    return JdbcSink.sink(
        sql,
        (statement, item) -> {
          statement.setString(1, item.eventId);
          statement.setString(2, item.impressionId);
          statement.setString(3, item.productId);
          statement.setInt(4, item.position);
          statement.setString(5, jsonString(item.featureSnapshot));
          statement.setTimestamp(6, Timestamp.from(Instant.ofEpochMilli(item.eventTimeMillis)));
        },
        JdbcExecutionOptions.builder()
            .withBatchSize(config.jdbcBatchSize)
            .withBatchIntervalMs(config.jdbcBatchIntervalMs)
            .withMaxRetries(config.jdbcMaxRetries)
            .build(),
        new JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
            .withUrl(config.postgresJdbcUrl)
            .withDriverName("org.postgresql.Driver")
            .withUsername(config.postgresUser)
            .withPassword(config.postgresPassword)
            .build());
  }

  private static KafkaSink<String> buildKafkaSink(
      String bootstrapServers, String topic, String transactionalIdPrefix, JobConfig config) {
    var builder =
        KafkaSink.<String>builder()
            .setBootstrapServers(bootstrapServers)
            .setRecordSerializer(
                KafkaRecordSerializationSchema.builder()
                    .setTopic(topic)
                    .setValueSerializationSchema(new SimpleStringSchema())
                    .build());
    if (transactionalIdPrefix != null && !transactionalIdPrefix.isBlank()) {
      builder
          .setDeliveryGuarantee(DeliveryGuarantee.EXACTLY_ONCE)
          .setTransactionalIdPrefix(transactionalIdPrefix + "-" + UUID.randomUUID() + "-")
          .setProperty(
              "transaction.timeout.ms", Integer.toString(config.kafkaTransactionTimeoutMs));
    } else {
      builder.setDeliveryGuarantee(DeliveryGuarantee.AT_LEAST_ONCE);
    }
    return builder.build();
  }

  static final class ParseInteractionEventFunction
      extends ProcessFunction<String, InteractionEvent> {
    private final String sourceTopic;

    ParseInteractionEventFunction() {
      this(USER_INTERACTIONS_SOURCE_TOPIC);
    }

    ParseInteractionEventFunction(String sourceTopic) {
      this.sourceTopic = sourceTopic;
    }

    @Override
    public void processElement(String raw, Context context, Collector<InteractionEvent> out) {
      try {
        out.collect(parseInteractionEvent(raw));
      } catch (Exception exc) {
        context.output(DLQ_TAG, DlqEvent.invalid(raw, exc, sourceTopic));
      }
    }
  }

  static final class ParseRecommendationEventFunction
      extends ProcessFunction<String, RecommendationImpressionEvent> {
    private final String sourceTopic;

    ParseRecommendationEventFunction() {
      this(RECOMMENDATION_EVENTS_SOURCE_TOPIC);
    }

    ParseRecommendationEventFunction(String sourceTopic) {
      this.sourceTopic = sourceTopic;
    }

    @Override
    public void processElement(
        String raw, Context context, Collector<RecommendationImpressionEvent> out) {
      try {
        RecommendationImpressionEvent event = parseRecommendationEvent(raw);
        if (event != null) {
          out.collect(event);
        }
      } catch (Exception exc) {
        context.output(DLQ_TAG, DlqEvent.invalid(raw, exc, sourceTopic));
      }
    }
  }

  // Postgres column widths (migrations/postgres, system_store.py). Oversized values
  // must fail at parse time so they go to the DLQ instead of failing a JDBC batch,
  // which would restart the job and replay the same record forever.
  static final int ID_COLUMN_LENGTH = 64;
  static final int ENTITY_COLUMN_LENGTH = 255;

  static void requireMaxLength(String field, String value, int maxLength) {
    if (value != null && value.length() > maxLength) {
      throw new IllegalArgumentException(
          field + " exceeds " + maxLength + " characters (" + value.length() + ")");
    }
  }

  static InteractionEvent parseInteractionEvent(String raw) throws Exception {
    Map<String, Object> payload = JSON.readValue(raw, MAP_TYPE);
    String eventId = stringValue(payload.get("event_id"));
    String userId = stringValue(payload.get("user_id"));
    String productId = stringValue(payload.get("product_id"));
    String action = stringValue(payload.get("action"));
    if (eventId.isBlank() || userId.isBlank() || productId.isBlank() || action.isBlank()) {
      throw new IllegalArgumentException("interaction event missing event_id/user_id/product_id/action");
    }
    requireMaxLength("event_id", eventId, ID_COLUMN_LENGTH);
    requireMaxLength("user_id", userId, ENTITY_COLUMN_LENGTH);
    requireMaxLength("product_id", productId, ENTITY_COLUMN_LENGTH);
    requireMaxLength("action", action, ID_COLUMN_LENGTH);

    Object contextObject = payload.get("context");
    Map<String, Object> eventContext =
        contextObject instanceof Map
            ? new HashMap<>((Map<String, Object>) contextObject)
            : new HashMap<>();
    String contextJson = JSON.writeValueAsString(eventContext);

    double occurredAt = numericTimestamp(payload.get("event_time"));
    if (occurredAt <= 0) {
      occurredAt = numericTimestamp(payload.get("occurred_at"));
    }
    if (occurredAt <= 0) {
      occurredAt = numericTimestamp(payload.get("timestamp"));
    }
    if (occurredAt <= 0) {
      occurredAt = Instant.now().toEpochMilli();
    }
    long eventTimeMillis = timestampMillis(occurredAt);

    InteractionEvent event = new InteractionEvent();
    event.eventId = eventId;
    event.schemaVersion = intValue(payload.get("schema_version"), 1);
    event.requestId = nullableString(payload.get("request_id"));
    requireMaxLength("request_id", event.requestId, ID_COLUMN_LENGTH);
    event.userId = userId;
    event.productId = productId;
    event.action = action;
    event.context = eventContext;
    event.contextJson = contextJson;
    event.eventTimeMillis = eventTimeMillis;
    event.occurredAtSeconds = eventTimeMillis / 1000.0;
    event.timestampSeconds = numericTimestamp(payload.get("event_time"));
    if (event.timestampSeconds <= 0) {
      event.timestampSeconds = numericTimestamp(payload.get("timestamp"));
    }
    if (event.timestampSeconds <= 0) {
      event.timestampSeconds = event.occurredAtSeconds;
    }
    event.productCategory = stringValue(eventContext.get("product_category"));
    event.sessionLengthSeconds = optionalDouble(eventContext.get("session_length"));
    return event;
  }

  static RecommendationImpressionEvent parseRecommendationEvent(String raw) throws Exception {
    Map<String, Object> payload = JSON.readValue(raw, MAP_TYPE);
    String eventType = stringValue(payload.get("event_type"));
    if ("recommendation_viewed".equals(eventType)) {
      String eventId = stringValue(payload.get("event_id"));
      String impressionId = stringValue(payload.get("impression_id"));
      String productId = stringValue(payload.get("product_id"));
      if (eventId.isBlank() || impressionId.isBlank() || productId.isBlank()) {
        throw new IllegalArgumentException(
            "recommendation view missing event_id/impression_id/product_id");
      }
      requireMaxLength("event_id", eventId, ID_COLUMN_LENGTH);
      requireMaxLength("impression_id", impressionId, ID_COLUMN_LENGTH);
      requireMaxLength("product_id", productId, ENTITY_COLUMN_LENGTH);
      RecommendationImpressionEvent event = new RecommendationImpressionEvent();
      event.viewed = true;
      event.impressionId = impressionId;
      event.userId = stringValue(payload.get("user_id"));
      event.eventTimeMillis = recommendationEventTimeMillis(payload, safeMap(payload.get("context")));
      RecommendationImpressionItemRow item = new RecommendationImpressionItemRow();
      item.eventId = eventId;
      item.impressionId = impressionId;
      item.productId = productId;
      item.position = intValue(payload.get("position"), 1);
      item.featureSnapshot = safeMap(payload.get("context"));
      item.eventTimeMillis = event.eventTimeMillis;
      event.itemRows.add(item);
      return event;
    }
    Map<String, Object> metadata = safeMap(payload.get("metadata"));
    String impressionId =
        nullableString(firstNonNull(metadata.get("impression_id"), payload.get("impression_id")));
    String userId = nullableString(firstNonNull(payload.get("user_id"), metadata.get("user_id")));
    Object rawDisplayedItems =
        metadata.containsKey("displayed_items")
            ? metadata.get("displayed_items")
            : payload.get("displayed_items");
    List<Object> displayedItems = safeList(rawDisplayedItems);
    if (impressionId == null || userId == null || displayedItems == null) {
      throw new IllegalArgumentException(
          "recommendation event missing impression_id/user_id/displayed_items");
    }

    long eventTimeMillis = recommendationEventTimeMillis(payload, metadata);
    Map<String, Object> context =
        metadata.containsKey("context")
            ? safeMap(metadata.get("context"))
            : safeMap(metadata.get("request_context"));
    Object contentId = metadata.get("content_id");
    if (contentId != null) {
      context.putIfAbsent("content_id", contentId);
    }
    Object itemSnapshotScope = metadata.get("item_snapshot_scope");
    if (itemSnapshotScope != null) {
      context.put("item_snapshot_scope", stringValue(itemSnapshotScope));
    }

    RecommendationImpressionEvent event = new RecommendationImpressionEvent();
    event.impressionId = impressionId;
    event.requestId = nullableString(firstNonNull(payload.get("request_id"), metadata.get("request_id")));
    event.userId = userId;
    event.sessionId = nullableString(firstNonNull(metadata.get("session_id"), context.get("session_id")));
    event.contentId = nullableString(contentId);
    event.modelVersion = nullableString(metadata.get("model_version"));
    event.rankingModelVersion = nullableString(metadata.get("ranking_model_version"));
    event.context = context;
    event.eventTimeMillis = eventTimeMillis;
    event.itemRows = new ArrayList<>();
    requireMaxLength("impression_id", event.impressionId, ID_COLUMN_LENGTH);
    requireMaxLength("request_id", event.requestId, ID_COLUMN_LENGTH);
    requireMaxLength("user_id", event.userId, ENTITY_COLUMN_LENGTH);
    requireMaxLength("session_id", event.sessionId, ENTITY_COLUMN_LENGTH);
    requireMaxLength("content_id", event.contentId, ENTITY_COLUMN_LENGTH);
    requireMaxLength("model_version", event.modelVersion, ENTITY_COLUMN_LENGTH);
    requireMaxLength("ranking_model_version", event.rankingModelVersion, ENTITY_COLUMN_LENGTH);

    for (Object rawItem : displayedItems) {
      if (!(rawItem instanceof Map)) {
        continue;
      }
      Map<String, Object> item = safeMap(rawItem);
      String productId = nullableString(item.get("product_id"));
      if (productId == null) {
        continue;
      }
      Object rawSource = firstNonNull(item.get("source"), item.get("candidate_source"));
      String source = nullableString(rawSource);
      requireMaxLength("product_id", productId, ENTITY_COLUMN_LENGTH);
      requireMaxLength("source", source, ENTITY_COLUMN_LENGTH);

      RecommendationImpressionItemRow itemRow = new RecommendationImpressionItemRow();
      itemRow.impressionId = impressionId;
      itemRow.productId = productId;
      itemRow.position = intValue(item.get("position"), 0);
      itemRow.source = source;
      itemRow.featureSnapshot = featureSnapshotFromItem(item);
      itemRow.scores = scoreSnapshotFromItem(item);
      itemRow.eventTimeMillis = eventTimeMillis;
      event.itemRows.add(itemRow);
    }

    if (event.itemRows.isEmpty()) {
      return null;
    }
    return event;
  }

  static final class EventIdDeduplicator
      extends KeyedProcessFunction<String, InteractionEvent, InteractionEvent> {
    private final int ttlDays;
    private transient ValueState<Boolean> seen;

    EventIdDeduplicator(int ttlDays) {
      this.ttlDays = ttlDays;
    }

    @Override
    public void open(Configuration parameters) {
      ValueStateDescriptor<Boolean> descriptor =
          new ValueStateDescriptor<>("seen-event-id", Boolean.class);
      descriptor.enableTimeToLive(
          StateTtlConfig.newBuilder(org.apache.flink.api.common.time.Time.days(ttlDays))
              .setUpdateType(StateTtlConfig.UpdateType.OnCreateAndWrite)
              .setStateVisibility(StateTtlConfig.StateVisibility.NeverReturnExpired)
              .build());
      seen = getRuntimeContext().getState(descriptor);
    }

    @Override
    public void processElement(InteractionEvent event, Context context, Collector<InteractionEvent> out)
        throws Exception {
      if (seen.value() == null) {
        seen.update(Boolean.TRUE);
        out.collect(event);
      }
    }
  }

  static final class UserFeatureAggregator
      extends KeyedProcessFunction<String, InteractionEvent, UserFeatureSnapshot> {
    private final long stateEpoch;
    private transient ValueState<UserAccumulator> accumulatorState;
    private transient Counter activeKeys;
    private transient AtomicLong estimatedStateBytesPerUser;

    UserFeatureAggregator(long stateEpoch) {
      this.stateEpoch = stateEpoch;
    }

    @Override
    public void open(Configuration parameters) {
      ValueStateDescriptor<UserAccumulator> descriptor =
          new ValueStateDescriptor<>(
              "user-feature-accumulator", TypeInformation.of(UserAccumulator.class));
      accumulatorState = getRuntimeContext().getState(descriptor);
      var metrics = getRuntimeContext().getMetricGroup().addGroup("user_feature_state");
      activeKeys = metrics.counter("active_keys");
      estimatedStateBytesPerUser = new AtomicLong();
      metrics.gauge("estimated_bytes_per_user", estimatedStateBytesPerUser::get);
    }

    @Override
    public void processElement(
        InteractionEvent event, Context context, Collector<UserFeatureSnapshot> out) throws Exception {
      UserAccumulator accumulator = accumulatorState.value();
      if (accumulator == null) {
        accumulator = new UserAccumulator(event.userId);
        activeKeys.inc();
      }
      accumulator.apply(event);
      estimatedStateBytesPerUser.set(accumulator.estimatedBytes());
      accumulatorState.update(accumulator);
      UserFeatureSnapshot snapshot = accumulator.snapshot();
      snapshot.stateEpoch = stateEpoch;
      snapshot.snapshotDigest = snapshotDigest(snapshot);
      snapshot.availableAt = Instant.now().toEpochMilli() / 1000.0;
      out.collect(snapshot);
    }
  }

  static final class EntityEventFanout implements FlatMapFunction<InteractionEvent, EntityEvent> {
    @Override
    public void flatMap(InteractionEvent event, Collector<EntityEvent> out) {
      out.collect(EntityEvent.from(event, "user", event.userId));
      out.collect(EntityEvent.from(event, "product", event.productId));
      if (!event.productCategory.isBlank()) {
        out.collect(EntityEvent.from(event, "category", event.productCategory));
      }
    }
  }

  static final class WindowStreams {
    final DataStream<WindowFeatureSnapshot> snapshots;
    final DataStream<LateInteractionEvent> lateEvents;

    WindowStreams(
        DataStream<WindowFeatureSnapshot> snapshots,
        DataStream<LateInteractionEvent> lateEvents) {
      this.snapshots = snapshots;
      this.lateEvents = lateEvents;
    }
  }

  static final class LateInteractionMapper
      extends ProcessFunction<EntityEvent, LateInteractionEvent> {
    private final String windowName;
    private transient Counter lateEvents;
    private transient Counter latePurchases;
    private transient AtomicLong watermarkLagSeconds;

    LateInteractionMapper(String windowName) {
      this.windowName = windowName;
    }

    @Override
    public void open(Configuration parameters) {
      var metrics = getRuntimeContext().getMetricGroup().addGroup("late_interactions");
      lateEvents = metrics.counter(windowName + "_total");
      latePurchases = metrics.counter(windowName + "_purchase_total");
      watermarkLagSeconds = new AtomicLong();
      metrics.gauge(windowName + "_watermark_lag_seconds", watermarkLagSeconds::get);
    }

    @Override
    public void processElement(
        EntityEvent event, Context context, Collector<LateInteractionEvent> out) {
      long watermark = context.timerService().currentWatermark();
      LateInteractionEvent late = LateInteractionEvent.from(event, windowName, watermark);
      lateEvents.inc();
      if ("purchase".equals(event.action)) latePurchases.inc();
      watermarkLagSeconds.set((long) late.latenessSeconds);
      out.collect(late);
    }
  }

  static final class RecommendationImpressionItemFanout
      implements FlatMapFunction<RecommendationImpressionEvent, RecommendationImpressionItemRow> {
    @Override
    public void flatMap(
        RecommendationImpressionEvent event, Collector<RecommendationImpressionItemRow> out) {
      for (RecommendationImpressionItemRow item : event.itemRows) {
        out.collect(item);
      }
    }
  }

  static final class WindowCountsAggregate
      implements AggregateFunction<EntityEvent, WindowCounts, WindowCounts> {
    @Override
    public WindowCounts createAccumulator() {
      return new WindowCounts();
    }

    @Override
    public WindowCounts add(EntityEvent event, WindowCounts counts) {
      counts.add(event.action);
      return counts;
    }

    @Override
    public WindowCounts getResult(WindowCounts counts) {
      return counts;
    }

    @Override
    public WindowCounts merge(WindowCounts left, WindowCounts right) {
      left.views += right.views;
      left.clicks += right.clicks;
      left.addToCarts += right.addToCarts;
      left.purchases += right.purchases;
      left.totalEvents += right.totalEvents;
      return left;
    }
  }

  static final class WindowSnapshotFunction
      extends ProcessWindowFunction<WindowCounts, WindowFeatureSnapshot, EntityKey, TimeWindow> {
    private final String windowName;

    WindowSnapshotFunction(String windowName) {
      this.windowName = windowName;
    }

    @Override
    public void process(
        EntityKey key,
        Context context,
        Iterable<WindowCounts> elements,
        Collector<WindowFeatureSnapshot> out) {
      WindowCounts counts = elements.iterator().next();
      WindowFeatureSnapshot snapshot = new WindowFeatureSnapshot();
      snapshot.entityType = key.entityType;
      snapshot.entityId = key.entityId;
      snapshot.window = windowName;
      snapshot.views = counts.views;
      snapshot.clicks = counts.clicks;
      snapshot.addToCarts = counts.addToCarts;
      snapshot.purchases = counts.purchases;
      snapshot.totalEvents = counts.totalEvents;
      snapshot.clickThroughRate = ratio(counts.clicks, counts.views);
      snapshot.conversionRate = ratio(counts.purchases, counts.clicks);
      snapshot.windowStart = context.window().getStart() / 1000.0;
      snapshot.windowEnd = context.window().getEnd() / 1000.0;
      snapshot.availableAt = Instant.now().toEpochMilli() / 1000.0;
      out.collect(snapshot);
    }
  }

  enum EpochOwnerDecision {
    CLAIM,
    CONTINUE,
    REFUSE
  }

  /**
   * Decides whether this job may write user snapshots under {@code epoch}.
   *
   * <p>Generations live only in job state, so a job that starts from empty state under an epoch
   * another job already wrote would have every write judged stale, then hit digest conflicts
   * once its generations catch up. Such a start is refused; it must restore from a
   * checkpoint/savepoint or bump FLINK_FEATURE_STATE_EPOCH.
   */
  static EpochOwnerDecision decideEpochOwner(
      boolean restored, long epoch, Long ownerEpoch, String ownerJobId, String jobId) {
    if (ownerEpoch == null || ownerEpoch < epoch) {
      return EpochOwnerDecision.CLAIM;
    }
    if (ownerEpoch > epoch) {
      return EpochOwnerDecision.REFUSE;
    }
    if (jobId.equals(ownerJobId)) {
      return EpochOwnerDecision.CONTINUE;
    }
    return restored ? EpochOwnerDecision.CLAIM : EpochOwnerDecision.REFUSE;
  }

  static final class UserFeatureRedisSink extends BaseRedisSink<UserFeatureSnapshot>
      implements CheckpointedFunction {
    static final String EPOCH_OWNER_KEY = "flink:feature_state_owner";
    private static final String ATOMIC_COMMIT_LUA =
        "local epoch=tonumber(ARGV[1]); local generation=tonumber(ARGV[2]); "
            + "if not epoch or not generation then return 'invalid' end; "
            + "local oldEpoch=tonumber(redis.call('HGET',KEYS[5],'epoch') or '-1'); "
            + "local oldGeneration=tonumber(redis.call('HGET',KEYS[5],'generation') or '-1'); "
            + "local oldDigest=redis.call('HGET',KEYS[5],'digest'); "
            + "if epoch<oldEpoch or (epoch==oldEpoch and generation<oldGeneration) then return 'stale' end; "
            + "if epoch==oldEpoch and generation==oldGeneration and oldDigest and oldDigest~=ARGV[3] then return 'conflict' end; "
            + "local duplicate=(epoch==oldEpoch and generation==oldGeneration and oldDigest==ARGV[3]); "
            + "redis.call('SETEX',KEYS[1],ARGV[5],ARGV[4]); "
            + "if ARGV[6]~='' then redis.call('ZADD',KEYS[2],ARGV[7],ARGV[6]); redis.call('ZREMRANGEBYRANK',KEYS[2],0,-1001); redis.call('EXPIRE',KEYS[2],ARGV[8]); end; "
            + "if ARGV[9]=='1' then redis.call('ZADD',KEYS[3],ARGV[7],ARGV[6]); redis.call('ZREMRANGEBYRANK',KEYS[3],0,-201); redis.call('EXPIRE',KEYS[3],ARGV[8]); end; "
            + "redis.call('SETEX',KEYS[4],ARGV[8],ARGV[10]); "
            + "redis.call('HSET',KEYS[5],'epoch',ARGV[1],'generation',ARGV[2],'digest',ARGV[3]); redis.call('EXPIRE',KEYS[5],ARGV[8]); "
            + "if duplicate then return 'duplicate' else return 'applied' end";

    private final long stateEpoch;
    private transient boolean restored;

    UserFeatureRedisSink(RedisSinkConfig config, long stateEpoch) {
      super(config);
      this.stateEpoch = stateEpoch;
    }

    @Override
    public void initializeState(FunctionInitializationContext context) {
      restored = context.isRestored();
    }

    @Override
    public void snapshotState(FunctionSnapshotContext context) {}

    @Override
    public void open(Configuration parameters) {
      super.open(parameters);
      String jobId = getRuntimeContext().getJobInfo().getJobId().toHexString();
      byte[] ownerKey = bytes(config.key(EPOCH_OWNER_KEY));
      byte[] rawEpoch = commands.hget(ownerKey, bytes("epoch"));
      byte[] rawJobId = commands.hget(ownerKey, bytes("job_id"));
      Long ownerEpoch =
          rawEpoch == null ? null : Long.parseLong(new String(rawEpoch, StandardCharsets.UTF_8));
      String ownerJobId = rawJobId == null ? null : new String(rawJobId, StandardCharsets.UTF_8);
      EpochOwnerDecision decision =
          decideEpochOwner(restored, stateEpoch, ownerEpoch, ownerJobId, jobId);
      if (decision == EpochOwnerDecision.REFUSE) {
        throw new IllegalStateException(
            "Refusing to write user features: Redis epoch owner is epoch="
                + ownerEpoch
                + " job="
                + ownerJobId
                + ", this job "
                + jobId
                + " runs epoch="
                + stateEpoch
                + " without restored state. Restore from the latest checkpoint/savepoint, "
                + "or bump FLINK_FEATURE_STATE_EPOCH to intentionally discard state "
                + "(docs/operations/flink-realtime-recovery.md).");
      }
      if (decision == EpochOwnerDecision.CLAIM) {
        Map<byte[], byte[]> owner = new HashMap<>();
        owner.put(bytes("epoch"), bytes(Long.toString(stateEpoch)));
        owner.put(bytes("job_id"), bytes(jobId));
        owner.put(bytes("claimed_at"), bytes(Long.toString(Instant.now().getEpochSecond())));
        commands.hset(ownerKey, owner);
      }
    }

    @Override
    public void invoke(UserFeatureSnapshot snapshot, Context context) throws Exception {
      long started = System.nanoTime();
      byte[] featurePayload = packCachePayload("user_features", snapshot.userFeaturePayload());
      String sequenceKey = config.key("uiz:" + snapshot.userId);
      String tokenKey = config.key("ust:200:" + snapshot.userId);
      SequenceEvent delta = snapshot.sequenceDelta;
      String deltaPayload = delta == null ? "" : JSON.writeValueAsString(delta.payload());
      String action = delta == null ? null : dinAction(delta.action);
      String actionKey =
          config.key("uiza:" + (action == null ? "none" : action) + ":" + snapshot.userId);
      byte[] result;
      try {
        result = commands.eval(
              bytes(ATOMIC_COMMIT_LUA),
              ScriptOutputType.VALUE,
              new byte[][] {
                bytes(config.key("uf:" + snapshot.userId)),
                bytes(sequenceKey),
                bytes(actionKey),
                bytes(tokenKey),
                bytes(config.key("ufm:" + snapshot.userId))
              },
              bytes(Long.toString(snapshot.stateEpoch)),
              bytes(Long.toString(snapshot.generation)),
              bytes(snapshot.snapshotDigest),
              featurePayload,
              bytes(Integer.toString(config.userFeaturesTtlSeconds)),
              bytes(deltaPayload),
              bytes(Double.toString(delta == null ? 0.0 : delta.occurredAtSeconds)),
              bytes(Integer.toString(config.sequenceTtlSeconds)),
              bytes(action == null ? "0" : "1"),
              bytes(JSON.writeValueAsString(snapshot.sequenceToken())));
        networkRoundTrips.inc();
        sequenceDeltas.inc(delta == null ? 0 : 1);
      } catch (Exception exc) {
        failures.inc();
        throw exc;
      } finally {
        lastWriteLatencyMicros.set((System.nanoTime() - started) / 1_000L);
      }
      String status = result == null ? "" : new String(result, StandardCharsets.UTF_8);
      recordStatus(status);
      if ("invalid".equals(status)) {
        throw new IllegalStateException("Redis user snapshot commit failed: " + status);
      }
      if ("conflict".equals(status)) {
        // Counted by recordStatus (FlinkRedisSnapshotConflict alert). Skipping keeps the
        // existing Redis snapshot; throwing would replay the same record forever.
        LOG.error(
            "Redis user snapshot conflict skipped user={} epoch={} generation={}",
            snapshot.userId,
            snapshot.stateEpoch,
            snapshot.generation);
      }
    }
  }

  static final class WindowFeatureRedisSink extends BaseRedisSink<WindowFeatureSnapshot> {
    private static final String ATOMIC_WINDOW_COMMIT_LUA =
        "local windowEnd=tonumber(ARGV[1]); local revision=tonumber(ARGV[2]); "
            + "local oldEnd=tonumber(redis.call('HGET',KEYS[2],'window_end') or '-1'); "
            + "local oldRevision=tonumber(redis.call('HGET',KEYS[2],'revision') or '-1'); "
            + "local oldDigest=redis.call('HGET',KEYS[2],'digest'); "
            + "if windowEnd<oldEnd or (windowEnd==oldEnd and revision<oldRevision) then return 'stale' end; "
            + "if windowEnd==oldEnd and revision==oldRevision and oldDigest and oldDigest~=ARGV[3] then return 'conflict' end; "
            + "local duplicate=(windowEnd==oldEnd and revision==oldRevision and oldDigest==ARGV[3]); "
            + "redis.call('SETEX',KEYS[1],ARGV[5],ARGV[4]); "
            + "redis.call('HSET',KEYS[2],'window_end',ARGV[1],'revision',ARGV[2],'digest',ARGV[3]); redis.call('EXPIRE',KEYS[2],ARGV[5]); "
            + "if duplicate then return 'duplicate' else return 'applied' end";

    WindowFeatureRedisSink(RedisSinkConfig config) {
      super(config);
    }

    @Override
    public void invoke(WindowFeatureSnapshot snapshot, Context context) throws Exception {
      long started = System.nanoTime();
      String key =
          config.key(
              "rtwf:"
                  + snapshot.entityType
                  + ":"
                  + snapshot.entityId
                  + ":"
                  + snapshot.window);
      String metadataKey =
          config.key(
              "rtwfm:"
                  + snapshot.entityType
                  + ":"
                  + snapshot.entityId
                  + ":"
                  + snapshot.window);
      String payload = JSON.writeValueAsString(snapshot.payload());
      String digest = FeatureHistoryContract.payloadHash(snapshot.payload());
      byte[] result;
      try {
        result = commands.eval(
              bytes(ATOMIC_WINDOW_COMMIT_LUA),
              ScriptOutputType.VALUE,
              new byte[][] {bytes(key), bytes(metadataKey)},
              bytes(Double.toString(snapshot.windowEnd)),
              bytes(Long.toString(snapshot.totalEvents)),
              bytes(digest),
              bytes(payload),
              bytes(Integer.toString(config.windowFeatureTtlSeconds)));
        networkRoundTrips.inc();
      } catch (Exception exc) {
        failures.inc();
        throw exc;
      } finally {
        lastWriteLatencyMicros.set((System.nanoTime() - started) / 1_000L);
      }
      String status = result == null ? "" : new String(result, StandardCharsets.UTF_8);
      recordStatus(status);
      if ("conflict".equals(status)) {
        LOG.error(
            "Redis window snapshot conflict skipped key={} window_end={} revision={}",
            key,
            snapshot.windowEnd,
            snapshot.totalEvents);
      }
    }
  }

  abstract static class BaseRedisSink<T> extends RichSinkFunction<T> {
    protected final RedisSinkConfig config;
    protected transient RedisClient client;
    protected transient StatefulRedisConnection<byte[], byte[]> connection;
    protected transient RedisCommands<byte[], byte[]> commands;
    protected transient Counter applied;
    protected transient Counter duplicates;
    protected transient Counter stale;
    protected transient Counter conflicts;
    protected transient Counter failures;
    protected transient Counter networkRoundTrips;
    protected transient Counter sequenceDeltas;
    protected transient AtomicLong lastWriteLatencyMicros;

    BaseRedisSink(RedisSinkConfig config) {
      this.config = config;
    }

    @Override
    public void open(Configuration parameters) {
      RedisURI.Builder builder =
          RedisURI.builder()
              .withHost(config.host)
              .withPort(config.port)
              .withDatabase(config.db)
              .withTimeout(Duration.ofSeconds(5));
      if (!config.password.isBlank()) {
        builder.withPassword(config.password.toCharArray());
      }
      client = RedisClient.create(builder.build());
      connection = client.connect(ByteArrayCodec.INSTANCE);
      commands = connection.sync();
      var metrics = getRuntimeContext().getMetricGroup().addGroup("redis_snapshot_sink");
      applied = metrics.counter("applied");
      duplicates = metrics.counter("duplicate");
      stale = metrics.counter("stale");
      conflicts = metrics.counter("conflict");
      failures = metrics.counter("failure");
      networkRoundTrips = metrics.counter("network_round_trips");
      sequenceDeltas = metrics.counter("sequence_deltas");
      lastWriteLatencyMicros = new AtomicLong();
      metrics.gauge("last_write_latency_micros", lastWriteLatencyMicros::get);
    }

    protected void recordStatus(String status) {
      if ("applied".equals(status)) applied.inc();
      else if ("duplicate".equals(status)) duplicates.inc();
      else if ("stale".equals(status)) stale.inc();
      else if ("conflict".equals(status)) conflicts.inc();
      else failures.inc();
    }

    @Override
    public void close() {
      if (connection != null) {
        connection.close();
      }
      if (client != null) {
        client.shutdown();
      }
    }
  }

  static final class FeatureUpdateJsonMapper implements MapFunction<UserFeatureSnapshot, String> {
    @Override
    public String map(UserFeatureSnapshot snapshot) throws Exception {
      Map<String, Object> payload = snapshot.userFeaturePayload();
      String sourceEventId =
          FeatureHistoryContract.deterministicId(
              "user_feature",
              snapshot.userId,
              snapshot.sourceEventId,
              FeatureHistoryContract.payloadHash(payload));
      Map<String, Object> event =
          FeatureHistoryContract.build(
              "user_feature",
              "user",
              snapshot.userId,
              snapshot.eventTime,
              snapshot.availableAt,
              sourceEventId,
              "interaction-features-v1",
              payload,
              null,
              sourceEventId);
      event.put("upstream_source_event_id", snapshot.sourceEventId);
      return JSON.writeValueAsString(event);
    }
  }

  static final class WindowFeatureUpdateJsonMapper
      implements MapFunction<WindowFeatureSnapshot, String> {
    @Override
    public String map(WindowFeatureSnapshot snapshot) throws Exception {
      Map<String, Object> payload = snapshot.payload();
      String sourceEventId =
          FeatureHistoryContract.deterministicId(
              "window_feature",
              snapshot.entityType,
              snapshot.entityId,
              snapshot.window,
              Double.toString(snapshot.windowEnd),
              FeatureHistoryContract.payloadHash(payload));
      Map<String, Object> event =
          FeatureHistoryContract.build(
              "window_feature",
              snapshot.entityType,
              snapshot.entityId,
              snapshot.windowEnd,
              snapshot.availableAt,
              sourceEventId,
              "interaction-features-v1",
              payload,
              null,
              sourceEventId);
      return JSON.writeValueAsString(event);
    }
  }

  static final class DlqJsonMapper implements MapFunction<DlqEvent, String> {
    @Override
    public String map(DlqEvent event) throws Exception {
      return JSON.writeValueAsString(event.payload());
    }
  }

  static byte[] packCachePayload(String kind, Object payload) throws Exception {
    Map<String, Object> envelope = new HashMap<>();
    envelope.put("schema_version", 1);
    envelope.put("kind", kind);
    envelope.put("payload", payload);
    return MSGPACK.writeValueAsBytes(envelope);
  }

  static byte[] bytes(String value) {
    return value.getBytes(StandardCharsets.UTF_8);
  }

  static String snapshotDigest(UserFeatureSnapshot snapshot) {
    try {
      Map<String, Object> canonical = new HashMap<>();
      canonical.put("state_epoch", snapshot.stateEpoch);
      canonical.put("generation", snapshot.generation);
      canonical.put("feature", snapshot.userFeaturePayload());
      canonical.put("sequence_token", snapshot.sequenceToken());
      canonical.put(
          "sequence_delta",
          snapshot.sequenceDelta == null ? null : snapshot.sequenceDelta.payload());
      byte[] digest =
          MessageDigest.getInstance("SHA-256")
              .digest(FeatureHistoryContract.canonicalJson(canonical).getBytes(StandardCharsets.UTF_8));
      StringBuilder result = new StringBuilder(digest.length * 2);
      for (byte value : digest) {
        result.append(String.format("%02x", value & 0xff));
      }
      return result.toString();
    } catch (Exception exc) {
      throw new IllegalStateException("Unable to hash user snapshot", exc);
    }
  }

  static String deterministicId(String... parts) {
    try {
      return FeatureHistoryContract.deterministicId(parts);
    } catch (Exception exc) {
      throw new IllegalStateException("Unable to build deterministic event id", exc);
    }
  }

  static String payloadHash(Map<String, Object> payload) {
    try {
      return FeatureHistoryContract.payloadHash(payload);
    } catch (Exception exc) {
      throw new IllegalStateException("Unable to hash event payload", exc);
    }
  }

  static double ratio(long numerator, long denominator) {
    if (denominator <= 0) {
      return 0.0;
    }
    return Math.max(0.0, Math.min(1.0, ((double) numerator) / denominator));
  }

  static String dinAction(String action) {
    if ("click".equals(action)) {
      return "click";
    }
    if ("add_to_cart".equals(action) || "cart".equals(action)) {
      return "cart";
    }
    if ("purchase".equals(action)) {
      return "purchase";
    }
    return null;
  }

  static String stringValue(Object value) {
    return value == null ? "" : String.valueOf(value);
  }

  static String nullableString(Object value) {
    String text = stringValue(value);
    return text.isBlank() ? null : text;
  }

  static int intValue(Object value, int defaultValue) {
    if (value instanceof Number) {
      return ((Number) value).intValue();
    }
    try {
      return Integer.parseInt(stringValue(value));
    } catch (Exception ignored) {
      return defaultValue;
    }
  }

  static double numericTimestamp(Object value) {
    if (value instanceof Number) {
      return ((Number) value).doubleValue();
    }
    String text = stringValue(value);
    if (text.isBlank()) {
      return 0.0;
    }
    try {
      return Double.parseDouble(text);
    } catch (NumberFormatException ignored) {
      try {
        return Instant.parse(text).toEpochMilli();
      } catch (Exception ignoredAgain) {
        return 0.0;
      }
    }
  }

  static long timestampMillis(double timestamp) {
    return timestamp < 10_000_000_000.0 ? Math.round(timestamp * 1000.0) : Math.round(timestamp);
  }

  static Double optionalDouble(Object value) {
    if (value instanceof Number) {
      return ((Number) value).doubleValue();
    }
    try {
      String text = stringValue(value);
      return text.isBlank() ? null : Double.parseDouble(text);
    } catch (Exception ignored) {
      return null;
    }
  }

  static Object firstNonNull(Object first, Object second) {
    return first != null ? first : second;
  }

  static long recommendationEventTimeMillis(
      Map<String, Object> payload, Map<String, Object> metadata) {
    double timestamp = numericTimestamp(payload.get("timestamp"));
    if (timestamp <= 0) {
      timestamp = numericTimestamp(metadata.get("created_at"));
    }
    if (timestamp <= 0) {
      timestamp = Instant.now().toEpochMilli();
    }
    return timestampMillis(timestamp);
  }

  static String jsonString(Object value) {
    try {
      return JSON.writeValueAsString(value == null ? Collections.emptyMap() : value);
    } catch (Exception exc) {
      throw new IllegalArgumentException("Unable to serialize JSON value", exc);
    }
  }

  @SuppressWarnings("unchecked")
  static List<Object> safeList(Object value) {
    if (value instanceof List) {
      return (List<Object>) value;
    }
    return null;
  }

  static Map<String, Object> safeMap(Object value) {
    Map<String, Object> result = new HashMap<>();
    if (!(value instanceof Map)) {
      return result;
    }
    Map<?, ?> source = (Map<?, ?>) value;
    for (Map.Entry<?, ?> entry : source.entrySet()) {
      if (entry.getKey() != null) {
        result.put(String.valueOf(entry.getKey()), entry.getValue());
      }
    }
    return result;
  }

  static Map<String, Object> scoreSnapshotFromItem(Map<String, Object> item) {
    Map<String, Object> scores = safeMap(item.get("scores"));
    for (String key :
        List.of(
            "collaborative_score",
            "content_similarity_score",
            "popularity_score",
            "combined_score",
            "ranking_score",
            "confidence_score")) {
      if (item.containsKey(key) && !scores.containsKey(key)) {
        scores.put(key, item.get(key));
      }
    }
    return scores;
  }

  static Map<String, Object> featureSnapshotFromItem(Map<String, Object> item) {
    Map<String, Object> featureSnapshot = safeMap(item.get("feature_snapshot"));
    for (String key : List.of("price", "category", "brand")) {
      if (item.containsKey(key) && !featureSnapshot.containsKey(key)) {
        featureSnapshot.put(key, item.get(key));
      }
    }
    Object source = firstNonNull(item.get("source"), item.get("candidate_source"));
    if (source != null && !featureSnapshot.containsKey("candidate_source")) {
      featureSnapshot.put("candidate_source", source);
    }
    return featureSnapshot;
  }

  public static final class JobConfig implements Serializable {
    public String kafkaBootstrapServers;
    public String userInteractionsTopic;
    public String recommendationEventsTopic;
    public String featureUpdatesTopic;
    public String deadLetterTopic;
    public String consumerGroupId;
    public String checkpointDir;
    public int watermarkOutOfOrdernessSeconds;
    public int watermarkIdlenessSeconds;
    public int allowedLatenessSeconds;
    public int dedupStateTtlDays;
    public int replayHorizonDays;
    public long featureStateEpoch;
    public String lateInteractionsTopic;
    public String outputNamespace;
    public String redisHost;
    public int redisPort;
    public int redisDb;
    public String redisPassword;
    static final int DEFAULT_USER_FEATURES_TTL_SECONDS = 604_800;
    public int userFeaturesTtlSeconds = DEFAULT_USER_FEATURES_TTL_SECONDS;
    public int sequenceTtlSeconds;
    public int windowFeatureTtlSeconds;
    public String postgresJdbcUrl;
    public String postgresUser;
    public String postgresPassword;
    public int jdbcBatchSize;
    public long jdbcBatchIntervalMs;
    public int jdbcMaxRetries;
    public int kafkaTransactionTimeoutMs;

    static JobConfig fromEnv() {
      JobConfig config = new JobConfig();
      config.kafkaBootstrapServers = env("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092");
      config.userInteractionsTopic = env("KAFKA_USER_INTERACTIONS_TOPIC", "user-interactions");
      config.recommendationEventsTopic =
          env("KAFKA_RECOMMENDATION_EVENTS_TOPIC", "recommendation-events");
      // Immutable user/window snapshots for the feature-history materializer. The legacy
      // Python "feature-updates" topic is excluded from history materialization.
      config.featureUpdatesTopic =
          env(
              "KAFKA_FEATURE_HISTORY_SNAPSHOTS_TOPIC",
              env("KAFKA_FEATURE_UPDATES_TOPIC", "feature-history-snapshots"));
      config.deadLetterTopic = env("KAFKA_DEAD_LETTER_TOPIC", "dead-letter-events");
      config.consumerGroupId = env("FLINK_INTERACTION_FEATURE_CONSUMER_GROUP", "video-commerce-flink-feature-v1");
      config.checkpointDir = env("FLINK_CHECKPOINT_DIR", "file:///flink/checkpoints/runtime");
      config.watermarkOutOfOrdernessSeconds = envInt("FLINK_WATERMARK_OUT_OF_ORDERNESS_SECONDS", 300);
      config.watermarkIdlenessSeconds = envInt("FLINK_WATERMARK_IDLENESS_SECONDS", 60);
      config.allowedLatenessSeconds = envInt("FLINK_ALLOWED_LATENESS_SECONDS", 600);
      config.dedupStateTtlDays = envInt("FLINK_EVENT_DEDUP_TTL_DAYS", 8);
      config.replayHorizonDays = envInt("FLINK_REPLAY_HORIZON_DAYS", 7);
      config.featureStateEpoch = envLong("FLINK_FEATURE_STATE_EPOCH", 1L);
      config.lateInteractionsTopic =
          env("KAFKA_LATE_INTERACTIONS_TOPIC", "late-interaction-events");
      config.outputNamespace = env("FLINK_FEATURE_OUTPUT_NAMESPACE", "official");
      config.redisHost = env("REDIS_HOST", "redis");
      config.redisPort = envInt("REDIS_PORT", 6379);
      config.redisDb = envInt("REDIS_DB", 0);
      config.redisPassword = env("REDIS_PASSWORD", "");
      // Official uf: keys share the 7-day feature max-age contract with the PIT join
      // (FEATURE_LAKE_FEATURE_MAX_AGE_SECONDS); the Python cache TTL no longer applies.
      config.userFeaturesTtlSeconds =
          envInt("FLINK_USER_FEATURES_TTL_SECONDS", DEFAULT_USER_FEATURES_TTL_SECONDS);
      config.sequenceTtlSeconds = envInt("FLINK_SEQUENCE_TTL_SECONDS", 2_592_000);
      config.windowFeatureTtlSeconds = envInt("FLINK_WINDOW_FEATURE_TTL_SECONDS", 86_400);
      config.postgresJdbcUrl = env("POSTGRES_JDBC_URL", "jdbc:postgresql://postgres:5432/video_commerce");
      config.postgresUser = env("POSTGRES_USER", "video_commerce");
      config.postgresPassword = env("POSTGRES_PASSWORD", "video_commerce");
      config.jdbcBatchSize = envInt("FLINK_JDBC_BATCH_SIZE", 500);
      config.jdbcBatchIntervalMs = envLong("FLINK_JDBC_BATCH_INTERVAL_MS", 2000L);
      config.jdbcMaxRetries = envInt("FLINK_JDBC_MAX_RETRIES", 5);
      config.kafkaTransactionTimeoutMs = envInt("FLINK_KAFKA_TRANSACTION_TIMEOUT_MS", 600_000);
      config.validate();
      validateProductionStateEpoch(
          System.getenv("ENVIRONMENT"), System.getenv("FLINK_FEATURE_STATE_EPOCH"));
      return config;
    }

    static void validateProductionStateEpoch(String environment, String configuredEpoch) {
      if ("production".equalsIgnoreCase(environment)
          && (configuredEpoch == null || configuredEpoch.isBlank())) {
        throw new IllegalArgumentException(
            "Production requires explicit FLINK_FEATURE_STATE_EPOCH");
      }
    }

    void validate() {
      if (userFeaturesTtlSeconds < 1) {
        throw new IllegalArgumentException("FLINK_USER_FEATURES_TTL_SECONDS must be positive");
      }
      if (featureStateEpoch < 1) {
        throw new IllegalArgumentException("FLINK_FEATURE_STATE_EPOCH must be at least 1");
      }
      if (watermarkIdlenessSeconds < 1) {
        throw new IllegalArgumentException("FLINK_WATERMARK_IDLENESS_SECONDS must be positive");
      }
      if (replayHorizonDays < 1 || dedupStateTtlDays < replayHorizonDays + 1) {
        throw new IllegalArgumentException(
            "FLINK_EVENT_DEDUP_TTL_DAYS must be at least FLINK_REPLAY_HORIZON_DAYS + 1");
      }
    }

    RedisSinkConfig redisConfig() {
      String prefix = "shadow".equalsIgnoreCase(outputNamespace) ? "flink:shadow:" : "";
      return new RedisSinkConfig(
          redisHost,
          redisPort,
          redisDb,
          redisPassword,
          prefix,
          userFeaturesTtlSeconds,
          sequenceTtlSeconds,
          windowFeatureTtlSeconds);
    }
  }

  static String env(String name, String defaultValue) {
    String value = System.getenv(name);
    return value == null || value.isBlank() ? defaultValue : value;
  }

  static int envInt(String name, int defaultValue) {
    try {
      return Integer.parseInt(env(name, String.valueOf(defaultValue)));
    } catch (NumberFormatException exc) {
      return defaultValue;
    }
  }

  static long envLong(String name, long defaultValue) {
    try {
      return Long.parseLong(env(name, String.valueOf(defaultValue)));
    } catch (NumberFormatException exc) {
      return defaultValue;
    }
  }

  public static final class RedisSinkConfig implements Serializable {
    public final String host;
    public final int port;
    public final int db;
    public final String password;
    public final String keyPrefix;
    public final int userFeaturesTtlSeconds;
    public final int sequenceTtlSeconds;
    public final int windowFeatureTtlSeconds;

    RedisSinkConfig(
        String host,
        int port,
        int db,
        String password,
        String keyPrefix,
        int userFeaturesTtlSeconds,
        int sequenceTtlSeconds,
        int windowFeatureTtlSeconds) {
      this.host = host;
      this.port = port;
      this.db = db;
      this.password = password;
      this.keyPrefix = keyPrefix;
      this.userFeaturesTtlSeconds = userFeaturesTtlSeconds;
      this.sequenceTtlSeconds = sequenceTtlSeconds;
      this.windowFeatureTtlSeconds = windowFeatureTtlSeconds;
    }

    String key(String suffix) {
      return keyPrefix + suffix;
    }
  }

  public static final class InteractionEvent implements Serializable {
    public String eventId;
    public int schemaVersion;
    public String requestId;
    public String userId;
    public String productId;
    public String action;
    public Map<String, Object> context;
    public String contextJson;
    public String productCategory;
    public Double sessionLengthSeconds;
    public long eventTimeMillis;
    public double occurredAtSeconds;
    public double timestampSeconds;
  }

  public static final class RecommendationImpressionEvent implements Serializable {
    public boolean viewed;
    public String impressionId;
    public String requestId;
    public String userId;
    public String sessionId;
    public String contentId;
    public String modelVersion;
    public String rankingModelVersion;
    public Map<String, Object> context = new HashMap<>();
    public long eventTimeMillis;
    public List<RecommendationImpressionItemRow> itemRows = new ArrayList<>();
  }

  public static final class RecommendationImpressionItemRow implements Serializable {
    public String eventId;
    public String impressionId;
    public String productId;
    public int position;
    public String source;
    public Map<String, Object> featureSnapshot = new HashMap<>();
    public Map<String, Object> scores = new HashMap<>();
    public long eventTimeMillis;
  }

  public static final class UserAccumulator implements Serializable {
    public String userId;
    public long totalInteractions;
    public long totalViews;
    public long totalClicks;
    public long totalAddToCarts;
    public long totalPurchases;
    public double sessionLengthSum;
    public long sessionLengthCount;
    public double lastActive;
    public LinkedHashSet<String> preferredCategories = new LinkedHashSet<>();
    public List<SequenceEvent> sequence = new ArrayList<>();
    public double featureEventTime;
    public String featureSourceEventId;
    public long generation;
    public SequenceEvent lastSequenceDelta;

    public UserAccumulator() {}

    UserAccumulator(String userId) {
      this.userId = userId;
    }

    void apply(InteractionEvent event) {
      generation += 1;
      lastSequenceDelta = null;
      totalInteractions += 1;
      lastActive = Math.max(lastActive, event.occurredAtSeconds);
      double sourceEventTime = event.eventTimeMillis / 1000.0;
      if (sourceEventTime > featureEventTime
          || (sourceEventTime == featureEventTime
              && (featureSourceEventId == null
                  || event.eventId.compareTo(featureSourceEventId) > 0))) {
        featureEventTime = sourceEventTime;
        featureSourceEventId = event.eventId;
      }
      String action = event.action.toLowerCase(Locale.ROOT);
      if ("view".equals(action)) {
        totalViews += 1;
      } else if ("click".equals(action)) {
        totalClicks += 1;
      } else if ("add_to_cart".equals(action)) {
        totalAddToCarts += 1;
      } else if ("purchase".equals(action)) {
        totalPurchases += 1;
      }
      if (event.sessionLengthSeconds != null) {
        sessionLengthSum += event.sessionLengthSeconds;
        sessionLengthCount += 1;
      }
      if (!event.productCategory.isBlank() && preferredCategories.size() < 10) {
        preferredCategories.add(event.productCategory);
      }
      if (isPositiveAction(action)) {
        lastSequenceDelta = SequenceEvent.from(event);
        sequence.add(lastSequenceDelta);
        sequence.sort(Comparator.comparingDouble((SequenceEvent item) -> item.occurredAtSeconds)
            .thenComparing(item -> item.eventId == null ? "" : item.eventId));
        if (sequence.size() > 1000) {
          sequence = new ArrayList<>(sequence.subList(sequence.size() - 1000, sequence.size()));
        }
      }
    }

    UserFeatureSnapshot snapshot() {
      UserFeatureSnapshot snapshot = new UserFeatureSnapshot();
      snapshot.userId = userId;
      snapshot.totalInteractions = totalInteractions;
      snapshot.avgSessionLength =
          sessionLengthCount > 0 ? sessionLengthSum / sessionLengthCount : 0.0;
      snapshot.preferredCategories = new ArrayList<>(preferredCategories);
      snapshot.priceSensitivity = 0.5;
      snapshot.clickThroughRate = ratio(totalClicks, Math.max(totalViews, 1));
      snapshot.conversionRate = ratio(totalPurchases, Math.max(totalClicks, 1));
      snapshot.lastActive = lastActive;
      snapshot.totalViews = totalViews;
      snapshot.totalClicks = totalClicks;
      snapshot.totalAddToCarts = totalAddToCarts;
      snapshot.totalPurchases = totalPurchases;
      snapshot.sequence = new ArrayList<>(sequence);
      snapshot.eventTime = featureEventTime;
      snapshot.sourceEventId = featureSourceEventId;
      snapshot.generation = generation;
      snapshot.sequenceDelta = lastSequenceDelta;
      return snapshot;
    }

    long estimatedBytes() {
      long categories = preferredCategories.stream().mapToLong(value -> value.length() * 2L).sum();
      long events = sequence.size() * 256L;
      return 160L + categories + events;
    }
  }

  public static final class UserFeatureSnapshot implements Serializable {
    public String userId;
    public long totalInteractions;
    public double avgSessionLength;
    public List<String> preferredCategories = new ArrayList<>();
    public double priceSensitivity;
    public double clickThroughRate;
    public double conversionRate;
    public double lastActive;
    public long totalViews;
    public long totalClicks;
    public long totalAddToCarts;
    public long totalPurchases;
    public List<SequenceEvent> sequence = new ArrayList<>();
    public double eventTime;
    public double availableAt;
    public String sourceEventId;
    public long stateEpoch;
    public long generation;
    public String snapshotDigest;
    public SequenceEvent sequenceDelta;

    Map<String, Object> userFeaturePayload() {
      Map<String, Object> payload = new HashMap<>();
      payload.put("user_id", userId);
      payload.put("total_interactions", totalInteractions);
      payload.put("avg_session_length", avgSessionLength);
      payload.put("preferred_categories", preferredCategories);
      payload.put("price_sensitivity", priceSensitivity);
      payload.put("click_through_rate", clickThroughRate);
      payload.put("conversion_rate", conversionRate);
      payload.put("last_active", lastActive);
      payload.put("demographics", Collections.emptyMap());
      return payload;
    }

    Map<String, Long> actionCounts() {
      Map<String, Long> counts = new HashMap<>();
      counts.put("view", totalViews);
      counts.put("click", totalClicks);
      counts.put("add_to_cart", totalAddToCarts);
      counts.put("purchase", totalPurchases);
      return counts;
    }

    Map<String, Object> sequenceToken() {
      List<SequenceEvent> recent =
          sequence.size() > 200 ? sequence.subList(sequence.size() - 200, sequence.size()) : sequence;
      Map<String, Object> token = new HashMap<>();
      token.put("length", recent.size());
      token.put("state_epoch", stateEpoch);
      token.put("generation", generation);
      token.put("snapshot_digest", snapshotDigest);
      if (recent.isEmpty()) {
        token.put("latest_event_id", null);
        token.put("latest_occurred_at", 0);
        token.put("latest_product_id", null);
        token.put("latest_action", null);
      } else {
        SequenceEvent latest = recent.get(recent.size() - 1);
        token.put("latest_event_id", latest.eventId);
        token.put("latest_occurred_at", latest.occurredAtSeconds);
        token.put("latest_product_id", latest.productId);
        token.put("latest_action", latest.action);
      }
      return token;
    }
  }

  public static final class SequenceEvent implements Serializable {
    public String userId;
    public String productId;
    public String action;
    public double timestampSeconds;
    public double occurredAtSeconds;
    public String eventId;
    public int schemaVersion;
    public Map<String, Object> context = new HashMap<>();

    static SequenceEvent from(InteractionEvent event) {
      SequenceEvent item = new SequenceEvent();
      item.userId = event.userId;
      item.productId = event.productId;
      item.action = event.action;
      item.timestampSeconds = event.timestampSeconds;
      item.occurredAtSeconds = event.occurredAtSeconds;
      item.eventId = event.eventId;
      item.schemaVersion = event.schemaVersion;
      item.context = event.context;
      return item;
    }

    Map<String, Object> payload() {
      Map<String, Object> payload = new HashMap<>();
      payload.put("user_id", userId);
      payload.put("product_id", productId);
      payload.put("action", action);
      payload.put("timestamp", timestampSeconds);
      payload.put("occurred_at", occurredAtSeconds);
      payload.put("event_id", eventId);
      payload.put("schema_version", schemaVersion);
      payload.put("context", context);
      return payload;
    }
  }

  public static boolean isPositiveAction(String action) {
    return "view".equals(action)
        || "click".equals(action)
        || "add_to_cart".equals(action)
        || "purchase".equals(action);
  }

  public static final class EntityEvent implements Serializable {
    public EntityKey key;
    public String action;
    public String eventId;
    public String userId;
    public String productId;
    public long eventTimeMillis;

    static EntityEvent from(InteractionEvent event, String entityType, String entityId) {
      EntityEvent entityEvent = new EntityEvent();
      entityEvent.key = new EntityKey(entityType, entityId);
      entityEvent.action = event.action;
      entityEvent.eventId = event.eventId;
      entityEvent.userId = event.userId;
      entityEvent.productId = event.productId;
      entityEvent.eventTimeMillis = event.eventTimeMillis;
      return entityEvent;
    }
  }

  public static final class LateInteractionEvent implements Serializable {
    public String quarantineEventId;
    public String sourceEventId;
    public String userId;
    public String productId;
    public String action;
    public String entityType;
    public String entityId;
    public String window;
    public double eventTime;
    public double currentWatermark;
    public double latenessSeconds;

    static LateInteractionEvent from(EntityEvent event, String window, long watermarkMillis) {
      LateInteractionEvent result = new LateInteractionEvent();
      result.sourceEventId = event.eventId;
      result.userId = event.userId;
      result.productId = event.productId;
      result.action = event.action;
      result.entityType = event.key.entityType;
      result.entityId = event.key.entityId;
      result.window = window;
      result.eventTime = event.eventTimeMillis / 1000.0;
      result.currentWatermark = watermarkMillis / 1000.0;
      result.latenessSeconds = Math.max(0.0, result.currentWatermark - result.eventTime);
      result.quarantineEventId =
          deterministicId(
              "late-interaction-v1",
              result.sourceEventId,
              result.entityType,
              result.entityId,
              result.window);
      return result;
    }

    Map<String, Object> payload() {
      Map<String, Object> payload = new HashMap<>();
      payload.put("schema_version", 1);
      payload.put("event_type", "late_interaction");
      payload.put("event_id", quarantineEventId);
      payload.put("source_event_id", sourceEventId);
      payload.put("user_id", userId);
      payload.put("product_id", productId);
      payload.put("action", action);
      payload.put("entity_type", entityType);
      payload.put("entity_id", entityId);
      payload.put("window", window);
      payload.put("event_time", eventTime);
      payload.put("current_watermark", currentWatermark);
      payload.put("lateness_seconds", latenessSeconds);
      return payload;
    }
  }

  public static final class EntityKey implements Serializable {
    public String entityType;
    public String entityId;

    public EntityKey() {}

    EntityKey(String entityType, String entityId) {
      this.entityType = entityType;
      this.entityId = entityId;
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof EntityKey)) {
        return false;
      }
      EntityKey otherKey = (EntityKey) other;
      return Objects.equals(entityType, otherKey.entityType)
          && Objects.equals(entityId, otherKey.entityId);
    }

    @Override
    public int hashCode() {
      return Objects.hash(entityType, entityId);
    }
  }

  public static final class WindowCounts implements Serializable {
    public long views;
    public long clicks;
    public long addToCarts;
    public long purchases;
    public long totalEvents;

    void add(String rawAction) {
      String action = rawAction == null ? "" : rawAction.toLowerCase(Locale.ROOT);
      totalEvents += 1;
      if ("view".equals(action)) {
        views += 1;
      } else if ("click".equals(action)) {
        clicks += 1;
      } else if ("add_to_cart".equals(action)) {
        addToCarts += 1;
      } else if ("purchase".equals(action)) {
        purchases += 1;
      }
    }
  }

  public static final class WindowFeatureSnapshot implements Serializable {
    public String entityType;
    public String entityId;
    public String window;
    public long views;
    public long clicks;
    public long addToCarts;
    public long purchases;
    public long totalEvents;
    public double clickThroughRate;
    public double conversionRate;
    public double windowStart;
    public double windowEnd;
    public double availableAt;

    Map<String, Object> payload() {
      Map<String, Object> payload = new HashMap<>();
      payload.put("schema_version", 1);
      payload.put("entity_type", entityType);
      payload.put("entity_id", entityId);
      payload.put("window", window);
      payload.put("views", views);
      payload.put("clicks", clicks);
      payload.put("add_to_cart", addToCarts);
      payload.put("purchases", purchases);
      payload.put("total_events", totalEvents);
      payload.put("click_through_rate", clickThroughRate);
      payload.put("conversion_rate", conversionRate);
      payload.put("window_start", windowStart);
      payload.put("window_end", windowEnd);
      return payload;
    }
  }

  public static final class DlqEvent implements Serializable {
    public String rawValue;
    public String errorType;
    public String errorMessage;
    public double occurredAt;
    public String sourceTopic;

    static DlqEvent invalid(String rawValue, Exception exc) {
      return invalid(rawValue, exc, USER_INTERACTIONS_SOURCE_TOPIC);
    }

    static DlqEvent invalid(String rawValue, Exception exc, String sourceTopic) {
      DlqEvent event = new DlqEvent();
      event.rawValue = rawValue;
      event.errorType = exc.getClass().getSimpleName();
      event.errorMessage = exc.getMessage();
      event.occurredAt = Instant.now().toEpochMilli() / 1000.0;
      event.sourceTopic =
          sourceTopic == null || sourceTopic.isBlank()
              ? USER_INTERACTIONS_SOURCE_TOPIC
              : sourceTopic;
      return event;
    }

    Map<String, Object> payload() {
      Map<String, Object> payload = new HashMap<>();
      Map<String, Object> sourcePayload = new HashMap<>();
      sourcePayload.put("source_value", rawValue);
      payload.put("schema_version", 2);
      payload.put(
          "event_id",
          deterministicId(
              "flink-dlq-v2",
              sourceTopic,
              payloadHash(sourcePayload)));
      payload.put("event_type", "dead_letter");
      payload.put("source_topic", sourceTopic);
      payload.put("source_value", rawValue);
      payload.put("source_partition", null);
      payload.put("source_offset", null);
      payload.put("source_headers", Collections.emptyList());
      payload.put("error_type", errorType);
      payload.put("error_message", errorMessage);
      payload.put("occurred_at", occurredAt);
      return payload;
    }
  }
}
