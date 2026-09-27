# Corpus-volume scalability hypothesis

## Research question

How do ingestion throughput, PostgreSQL storage, and retrieval latency change as the corpus grows while the query workload, vector dimensions, database configuration, and concurrency remain fixed?

## Hypotheses

1. Exact pgvector cosine search will grow roughly linearly because it must compare the query with every stored vector.
2. GIN-backed lexical search will grow more slowly than exact vector search for selective terms.
3. Hybrid latency will be dominated by its exact-vector stage; rank fusion itself will remain small.
4. Storage will grow approximately linearly with chunk count.
5. Semantic quality should remain stable only when added material is unrelated. Real, related distractors may displace relevant chunks, so that claim requires a genuine semantic corpus rather than synthetic vectors.

## Controlled variables

- PostgreSQL container and configuration
- 384 embedding dimensions
- seeded workload (`seed=42`)
- 50 fixed queries
- Top-K = 20
- batch size = 500
- concurrency = 1
- no vector ANN index
- two passes named `first_pass` and `warm_repeat` (not claimed to be a true cold-cache test)

## Independent variable

Corpus size: 10K, 50K, 100K, 250K, 500K, and 1M chunks, subject to the documented safety stop.

## Track boundary

- **Track A — semantic quality:** genuine text, real model embeddings, labelled questions. The repository's frozen real corpus is retained as a baseline. It is too small and narrow to support a defensible quality-vs-scale conclusion.
- **Track B — database mechanics:** deterministic synthetic 384-dimensional vectors and templated text. It supports latency, storage, and ingestion conclusions only. It must never be used for semantic-quality claims.

