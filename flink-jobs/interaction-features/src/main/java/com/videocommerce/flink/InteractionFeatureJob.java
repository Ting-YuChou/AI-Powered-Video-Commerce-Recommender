package com.videocommerce.flink;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.lettuce.core.RedisClient;
import io.lettuce.core.RedisURI;
import io.lettuce.core.codec.ByteArrayCodec;
import io.lettuce.core.api.StatefulRedisConnection;
import io.lettuce.core.api.sync.RedisCommands;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
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
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
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

public class InteractionFeatureJob {
  private static final ObjectMapper JSON = new ObjectMapper();
  private static final ObjectMapper MSGPACK = new ObjectMapper(new MessagePackFactory());
  private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {};
  private static final OutputTag<DlqEvent> DLQ_TAG =
      new OutputTag<>("dead-letter-events", TypeInformation.of(DlqEvent.class));

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
            .process(new ParseInteractionEventFunction())
            .returns(InteractionEvent.class);

    parsed
        .getSideOutput(DLQ_TAG)
        .map(new DlqJsonMapper())
        .sinkTo(buildKafkaSink(config.kafkaBootstrapServers, config.deadLetterTopic, null, config))
        .name("invalid-event-dlq");

    DataStream<InteractionEvent> events =
        parsed.assignTimestampsAndWatermarks(
            WatermarkStrategy.<InteractionEvent>forBoundedOutOfOrderness(
                    Duration.ofSeconds(config.watermarkOutOfOrdernessSeconds))
                .withTimestampAssigner((event, previousTimestamp) -> event.eventTimeMillis));

    DataStream<InteractionEvent> deduped =
        events
            .keyBy((KeySelector<InteractionEvent, String>) event -> event.eventId)
            .process(new EventIdDeduplicator(config.dedupStateTtlDays))
            .name("dedupe-by-event-id")
            .returns(InteractionEvent.class);

    deduped
        .addSink(buildPostgresSink(config))
        .name("postgres-interaction-events");

    DataStream<UserFeatureSnapshot> userFeatures =
        deduped
            .keyBy((KeySelector<InteractionEvent, String>) event -> event.userId)
            .process(new UserFeatureAggregator())
            .name("user-feature-snapshots")
            .returns(UserFeatureSnapshot.class);

    userFeatures
        .addSink(new UserFeatureRedisSink(config.redisConfig()))
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
    DataStream<WindowFeatureSnapshot> windowFeatures =
        buildWindow(entityEvents, "5m", Time.minutes(5), Time.minutes(1), config)
            .union(buildWindow(entityEvents, "1h", Time.hours(1), Time.minutes(5), config))
            .union(buildWindow(entityEvents, "24h", Time.hours(24), Time.hours(1), config));

    windowFeatures
        .addSink(new WindowFeatureRedisSink(config.redisConfig()))
        .name("redis-realtime-window-features");

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
  }

  private static DataStream<WindowFeatureSnapshot> buildWindow(
      DataStream<EntityEvent> events,
      String windowName,
      Time size,
      Time slide,
      JobConfig config) {
    return events
        .keyBy((KeySelector<EntityEvent, EntityKey>) event -> event.key)
        .window(SlidingEventTimeWindows.of(size, slide))
        .allowedLateness(Time.seconds(config.allowedLatenessSeconds))
        .aggregate(new WindowCountsAggregate(), new WindowSnapshotFunction(windowName))
        .returns(WindowFeatureSnapshot.class)
        .name("window-" + windowName);
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
    @Override
    public void processElement(String raw, Context context, Collector<InteractionEvent> out) {
      try {
        out.collect(parseInteractionEvent(raw));
      } catch (Exception exc) {
        context.output(DLQ_TAG, DlqEvent.invalid(raw, exc));
      }
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

    Object contextObject = payload.get("context");
    Map<String, Object> eventContext =
        contextObject instanceof Map
            ? new HashMap<>((Map<String, Object>) contextObject)
            : new HashMap<>();
    String contextJson = JSON.writeValueAsString(eventContext);

    double occurredAt = numericTimestamp(payload.get("occurred_at"));
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
    event.userId = userId;
    event.productId = productId;
    event.action = action;
    event.context = eventContext;
    event.contextJson = contextJson;
    event.eventTimeMillis = eventTimeMillis;
    event.occurredAtSeconds = eventTimeMillis / 1000.0;
    event.timestampSeconds = numericTimestamp(payload.get("timestamp"));
    if (event.timestampSeconds <= 0) {
      event.timestampSeconds = event.occurredAtSeconds;
    }
    event.productCategory = stringValue(eventContext.get("product_category"));
    event.sessionLengthSeconds = optionalDouble(eventContext.get("session_length"));
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
    private transient ValueState<UserAccumulator> accumulatorState;

    @Override
    public void open(Configuration parameters) {
      ValueStateDescriptor<UserAccumulator> descriptor =
          new ValueStateDescriptor<>(
              "user-feature-accumulator", TypeInformation.of(UserAccumulator.class));
      accumulatorState = getRuntimeContext().getState(descriptor);
    }

    @Override
    public void processElement(
        InteractionEvent event, Context context, Collector<UserFeatureSnapshot> out) throws Exception {
      UserAccumulator accumulator = accumulatorState.value();
      if (accumulator == null) {
        accumulator = new UserAccumulator(event.userId);
      }
      accumulator.apply(event);
      accumulatorState.update(accumulator);
      out.collect(accumulator.snapshot());
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
      out.collect(snapshot);
    }
  }

  static final class UserFeatureRedisSink extends BaseRedisSink<UserFeatureSnapshot> {
    UserFeatureRedisSink(RedisSinkConfig config) {
      super(config);
    }

    @Override
    public void invoke(UserFeatureSnapshot snapshot, Context context) throws Exception {
      byte[] featurePayload = packCachePayload("user_features", snapshot.userFeaturePayload());
      commands.setex(bytes(config.key("uf:" + snapshot.userId)), config.userFeaturesTtlSeconds, featurePayload);

      String sequenceKey = config.key("uiz:" + snapshot.userId);
      for (SequenceEvent event : snapshot.sequence) {
        commands.zadd(
            bytes(sequenceKey),
            event.occurredAtSeconds,
            bytes(JSON.writeValueAsString(event.payload())));
      }
      commands.zremrangebyrank(bytes(sequenceKey), 0, -1001);
      commands.expire(bytes(sequenceKey), config.sequenceTtlSeconds);

      String tokenKey = config.key("ust:200:" + snapshot.userId);
      commands.setex(
          bytes(tokenKey),
          config.sequenceTtlSeconds,
          bytes(JSON.writeValueAsString(snapshot.sequenceToken())));
    }
  }

  static final class WindowFeatureRedisSink extends BaseRedisSink<WindowFeatureSnapshot> {
    WindowFeatureRedisSink(RedisSinkConfig config) {
      super(config);
    }

    @Override
    public void invoke(WindowFeatureSnapshot snapshot, Context context) throws Exception {
      String key =
          config.key(
              "rtwf:"
                  + snapshot.entityType
                  + ":"
                  + snapshot.entityId
                  + ":"
                  + snapshot.window);
      commands.setex(
          bytes(key),
          config.windowFeatureTtlSeconds,
          bytes(JSON.writeValueAsString(snapshot.payload())));
    }
  }

  abstract static class BaseRedisSink<T> extends RichSinkFunction<T> {
    protected final RedisSinkConfig config;
    protected transient RedisClient client;
    protected transient StatefulRedisConnection<byte[], byte[]> connection;
    protected transient RedisCommands<byte[], byte[]> commands;

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
      Map<String, Object> updates = new HashMap<>();
      updates.put("interactions_processed", snapshot.totalInteractions);
      updates.put("action_counts", snapshot.actionCounts());
      updates.put("preferred_categories", snapshot.preferredCategories);
      updates.put("updated_at", snapshot.lastActive);

      Map<String, Object> event = new HashMap<>();
      event.put("schema_version", 1);
      event.put("event_id", "feature-update-" + snapshot.userId + "-" + snapshot.totalInteractions);
      event.put("event_type", "feature_update");
      event.put("entity_type", "user");
      event.put("entity_id", snapshot.userId);
      event.put("updates", updates);
      event.put("timestamp", Instant.now().toEpochMilli() / 1000.0);
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

  static double ratio(long numerator, long denominator) {
    if (denominator <= 0) {
      return 0.0;
    }
    return Math.max(0.0, Math.min(1.0, ((double) numerator) / denominator));
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

  public static final class JobConfig implements Serializable {
    public String kafkaBootstrapServers;
    public String userInteractionsTopic;
    public String featureUpdatesTopic;
    public String deadLetterTopic;
    public String consumerGroupId;
    public String checkpointDir;
    public int watermarkOutOfOrdernessSeconds;
    public int allowedLatenessSeconds;
    public int dedupStateTtlDays;
    public String outputNamespace;
    public String redisHost;
    public int redisPort;
    public int redisDb;
    public String redisPassword;
    public int userFeaturesTtlSeconds;
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
      config.featureUpdatesTopic = env("KAFKA_FEATURE_UPDATES_TOPIC", "feature-updates");
      config.deadLetterTopic = env("KAFKA_DEAD_LETTER_TOPIC", "dead-letter-events");
      config.consumerGroupId = env("FLINK_INTERACTION_FEATURE_CONSUMER_GROUP", "video-commerce-flink-feature-v1");
      config.checkpointDir = env("FLINK_CHECKPOINT_DIR", "file:///flink/checkpoints/runtime");
      config.watermarkOutOfOrdernessSeconds = envInt("FLINK_WATERMARK_OUT_OF_ORDERNESS_SECONDS", 300);
      config.allowedLatenessSeconds = envInt("FLINK_ALLOWED_LATENESS_SECONDS", 600);
      config.dedupStateTtlDays = envInt("FLINK_EVENT_DEDUP_TTL_DAYS", 7);
      config.outputNamespace = env("FLINK_FEATURE_OUTPUT_NAMESPACE", "official");
      config.redisHost = env("REDIS_HOST", "redis");
      config.redisPort = envInt("REDIS_PORT", 6379);
      config.redisDb = envInt("REDIS_DB", 0);
      config.redisPassword = env("REDIS_PASSWORD", "");
      config.userFeaturesTtlSeconds = envInt("CACHE_USER_FEATURES_TTL", 1800);
      config.sequenceTtlSeconds = envInt("FLINK_SEQUENCE_TTL_SECONDS", 2_592_000);
      config.windowFeatureTtlSeconds = envInt("FLINK_WINDOW_FEATURE_TTL_SECONDS", 86_400);
      config.postgresJdbcUrl = env("POSTGRES_JDBC_URL", "jdbc:postgresql://postgres:5432/video_commerce");
      config.postgresUser = env("POSTGRES_USER", "video_commerce");
      config.postgresPassword = env("POSTGRES_PASSWORD", "video_commerce");
      config.jdbcBatchSize = envInt("FLINK_JDBC_BATCH_SIZE", 500);
      config.jdbcBatchIntervalMs = envLong("FLINK_JDBC_BATCH_INTERVAL_MS", 2000L);
      config.jdbcMaxRetries = envInt("FLINK_JDBC_MAX_RETRIES", 5);
      config.kafkaTransactionTimeoutMs = envInt("FLINK_KAFKA_TRANSACTION_TIMEOUT_MS", 600_000);
      return config;
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

    public UserAccumulator() {}

    UserAccumulator(String userId) {
      this.userId = userId;
    }

    void apply(InteractionEvent event) {
      totalInteractions += 1;
      lastActive = Math.max(lastActive, event.occurredAtSeconds);
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
        sequence.add(SequenceEvent.from(event));
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
      return snapshot;
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

    static EntityEvent from(InteractionEvent event, String entityType, String entityId) {
      EntityEvent entityEvent = new EntityEvent();
      entityEvent.key = new EntityKey(entityType, entityId);
      entityEvent.action = event.action;
      return entityEvent;
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

    static DlqEvent invalid(String rawValue, Exception exc) {
      DlqEvent event = new DlqEvent();
      event.rawValue = rawValue;
      event.errorType = exc.getClass().getSimpleName();
      event.errorMessage = exc.getMessage();
      event.occurredAt = Instant.now().toEpochMilli() / 1000.0;
      return event;
    }

    Map<String, Object> payload() {
      Map<String, Object> payload = new HashMap<>();
      payload.put("schema_version", 1);
      payload.put("event_id", UUID.randomUUID().toString());
      payload.put("event_type", "dead_letter");
      payload.put("source_topic", "user-interactions");
      payload.put("source_value", rawValue);
      payload.put("error_type", errorType);
      payload.put("error_message", errorMessage);
      payload.put("occurred_at", occurredAt);
      return payload;
    }
  }
}
