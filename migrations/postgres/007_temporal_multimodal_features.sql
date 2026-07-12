-- Durable frame-sequence and OCR-track features for temporal multimodal ranking.

CREATE TABLE IF NOT EXISTS content_feature_artifacts (
    content_id VARCHAR(64) PRIMARY KEY,
    schema_version VARCHAR(64) NOT NULL,
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_content_feature_artifacts_schema_version
ON content_feature_artifacts (schema_version);
