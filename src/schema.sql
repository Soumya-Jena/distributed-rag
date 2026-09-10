CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,

    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,

    embedding VECTOR(384) NOT NULL,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS corpus_config (
    id SMALLINT PRIMARY KEY CHECK (id = 1),
    embedding_model TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    chunk_overlap INTEGER NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE chunks
ADD COLUMN IF NOT EXISTS search_vector TSVECTOR
GENERATED ALWAYS AS (
    to_tsvector('simple', COALESCE(content, ''))
) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_search_vector
ON chunks
USING GIN (search_vector);
