CREATE EXTENSION IF NOT EXISTS vector;

-- Track B: synthetic vectors for database-mechanics measurements only.
CREATE TABLE IF NOT EXISTS benchmark_chunks (
    id BIGSERIAL PRIMARY KEY,
    source_id TEXT NOT NULL UNIQUE,
    scale_stage TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(content, ''))
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_benchmark_chunks_search_vector
ON benchmark_chunks USING GIN (search_vector);

-- Intentionally no vector index: this experiment establishes exact-search scaling.

CREATE TABLE IF NOT EXISTS benchmark_queries (
    query_id INTEGER PRIMARY KEY,
    query_text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL
);

CREATE TABLE IF NOT EXISTS scale_ingestion_runs (
    run_id TEXT PRIMARY KEY,
    target_chunks BIGINT NOT NULL,
    inserted_chunks BIGINT NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('running', 'complete', 'stopped', 'failed')),
    message TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

