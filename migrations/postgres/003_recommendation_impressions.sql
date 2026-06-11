-- Recommendation impression/slate logging for LTR training.

CREATE TABLE IF NOT EXISTS recommendation_impressions (
    impression_id VARCHAR(64) PRIMARY KEY,
    request_id VARCHAR(64),
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    content_id VARCHAR(255),
    model_version VARCHAR(255),
    ranking_model_version VARCHAR(255),
    context JSON NOT NULL DEFAULT '{}'::json,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendation_impression_items (
    id SERIAL PRIMARY KEY,
    impression_id VARCHAR(64) NOT NULL,
    product_id VARCHAR(255) NOT NULL,
    position INTEGER NOT NULL,
    source VARCHAR(255),
    feature_snapshot JSON NOT NULL DEFAULT '{}'::json,
    scores JSON NOT NULL DEFAULT '{}'::json,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_recommendation_impressions_user_created
ON recommendation_impressions (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_recommendation_impressions_created_at
ON recommendation_impressions (created_at DESC);

CREATE INDEX IF NOT EXISTS ix_recommendation_impression_items_impression
ON recommendation_impression_items (impression_id);

CREATE INDEX IF NOT EXISTS ix_recommendation_impression_items_product_created
ON recommendation_impression_items (product_id, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS ux_recommendation_impression_items_impression_product
ON recommendation_impression_items (impression_id, product_id);
