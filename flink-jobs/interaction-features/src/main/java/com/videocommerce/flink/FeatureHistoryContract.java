package com.videocommerce.flink;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.module.SimpleModule;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.LinkedHashMap;
import java.util.Map;

/** Shared additive contract for append-only feature history events. */
public final class FeatureHistoryContract {
  public static final int PAYLOAD_SCHEMA_VERSION = 1;
  public static final String FEATURE_DEFINITION_VERSION = "ranking_ltr_v1";
  private static final ObjectMapper CANONICAL_JSON = canonicalMapper();

  private FeatureHistoryContract() {}

  private static ObjectMapper canonicalMapper() {
    SimpleModule numbers = new SimpleModule();
    numbers.addSerializer(Double.class, new PlainDoubleSerializer());
    numbers.addSerializer(Float.class, new PlainFloatSerializer());
    return new ObjectMapper()
        .registerModule(numbers)
        .configure(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS, true);
  }

  private static final class PlainDoubleSerializer extends JsonSerializer<Double> {
    @Override
    public void serialize(Double value, JsonGenerator generator, SerializerProvider provider)
        throws IOException {
      if (value == null || !Double.isFinite(value)) {
        throw new IOException("floating-point values must be finite");
      }
      generator.writeNumber(BigDecimal.valueOf(value).stripTrailingZeros().toPlainString());
    }
  }

  private static final class PlainFloatSerializer extends JsonSerializer<Float> {
    @Override
    public void serialize(Float value, JsonGenerator generator, SerializerProvider provider)
        throws IOException {
      if (value == null || !Float.isFinite(value)) {
        throw new IOException("floating-point values must be finite");
      }
      generator.writeNumber(
          new BigDecimal(Float.toString(value)).stripTrailingZeros().toPlainString());
    }
  }

  public static Map<String, Object> build(
      String eventType,
      String entityType,
      String entityId,
      double eventTime,
      double availableAt,
      String sourceEventId,
      String sourceVersion,
      Map<String, Object> payload,
      String requestId,
      String eventId)
      throws Exception {
    Map<String, Object> immutablePayload = new LinkedHashMap<>(payload);
    Map<String, Object> event = new LinkedHashMap<>();
    event.put("event_id", required("event_id", eventId));
    event.put("event_type", required("event_type", eventType));
    event.put("entity_type", required("entity_type", entityType));
    event.put("entity_id", required("entity_id", entityId));
    event.put("event_time", finiteTimestamp("event_time", eventTime));
    event.put("available_at", finiteTimestamp("available_at", availableAt));
    event.put("source_event_id", required("source_event_id", sourceEventId));
    event.put("source_version", required("source_version", sourceVersion));
    event.put("feature_definition_version", FEATURE_DEFINITION_VERSION);
    event.put("payload_schema_version", PAYLOAD_SCHEMA_VERSION);
    event.put("payload_hash", payloadHash(immutablePayload));
    event.put("payload", immutablePayload);
    event.put("request_id", requestId);
    return event;
  }

  @SuppressWarnings("unchecked")
  public static Record parse(Map<String, Object> event) throws Exception {
    Object rawPayload = event.get("payload");
    if (!(rawPayload instanceof Map)) {
      throw new IllegalArgumentException("payload must be an object");
    }
    int payloadSchemaVersion = integer(event.get("payload_schema_version"));
    if (payloadSchemaVersion != PAYLOAD_SCHEMA_VERSION) {
      throw new IllegalArgumentException(
          "unsupported payload_schema_version " + payloadSchemaVersion);
    }
    Map<String, Object> payload = new LinkedHashMap<>((Map<String, Object>) rawPayload);
    String expectedHash = required("payload_hash", event.get("payload_hash"));
    if (!expectedHash.equals(payloadHash(payload))) {
      throw new IllegalArgumentException("payload_hash does not match canonical payload");
    }
    return new Record(
        required("event_id", event.get("event_id")),
        required("event_type", event.get("event_type")),
        required("entity_type", event.get("entity_type")),
        required("entity_id", event.get("entity_id")),
        finiteTimestamp("event_time", number(event.get("event_time"))),
        finiteTimestamp("available_at", number(event.get("available_at"))),
        required("source_event_id", event.get("source_event_id")),
        required("source_version", event.get("source_version")),
        required("feature_definition_version", event.get("feature_definition_version")),
        payloadSchemaVersion,
        expectedHash,
        payload,
        nullable(event.get("request_id")));
  }

  public static String canonicalJson(Map<String, Object> payload) throws Exception {
    return CANONICAL_JSON.writeValueAsString(payload);
  }

  public static String payloadHash(Map<String, Object> payload) throws Exception {
    return sha256(canonicalJson(payload));
  }

  public static String deterministicId(String... parts) throws Exception {
    return sha256(String.join("\u0000", parts));
  }

  private static String sha256(String value) throws Exception {
    byte[] digest =
        MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8));
    StringBuilder encoded = new StringBuilder(digest.length * 2);
    for (byte item : digest) {
      encoded.append(String.format("%02x", item & 0xff));
    }
    return encoded.toString();
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

  private static double number(Object value) {
    if (!(value instanceof Number)) {
      throw new IllegalArgumentException("timestamp must be numeric");
    }
    return ((Number) value).doubleValue();
  }

  private static int integer(Object value) {
    if (!(value instanceof Number)) {
      throw new IllegalArgumentException("payload_schema_version must be numeric");
    }
    return ((Number) value).intValue();
  }

  private static double finiteTimestamp(String name, double value) {
    if (!Double.isFinite(value)) {
      throw new IllegalArgumentException(name + " must be finite");
    }
    return value;
  }

  public static final class Record {
    public final String eventId;
    public final String eventType;
    public final String entityType;
    public final String entityId;
    public final double eventTime;
    public final double availableAt;
    public final String sourceEventId;
    public final String sourceVersion;
    public final String featureDefinitionVersion;
    public final int payloadSchemaVersion;
    public final String payloadHash;
    public final Map<String, Object> payload;
    public final String requestId;

    Record(
        String eventId,
        String eventType,
        String entityType,
        String entityId,
        double eventTime,
        double availableAt,
        String sourceEventId,
        String sourceVersion,
        String featureDefinitionVersion,
        int payloadSchemaVersion,
        String payloadHash,
        Map<String, Object> payload,
        String requestId) {
      this.eventId = eventId;
      this.eventType = eventType;
      this.entityType = entityType;
      this.entityId = entityId;
      this.eventTime = eventTime;
      this.availableAt = availableAt;
      this.sourceEventId = sourceEventId;
      this.sourceVersion = sourceVersion;
      this.featureDefinitionVersion = featureDefinitionVersion;
      this.payloadSchemaVersion = payloadSchemaVersion;
      this.payloadHash = payloadHash;
      this.payload = payload;
      this.requestId = requestId;
    }
  }
}
