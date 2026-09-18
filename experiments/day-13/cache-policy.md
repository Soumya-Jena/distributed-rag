# Cache policy

| Layer | Cached value | Key dependencies | TTL | Default |
|---|---|---|---:|---|
| Query embedding | Normalized float vector | Model, normalization, query hash | 24 h | Enabled |
| Query transformation | Three rewritten queries | Model, prompt version, query hash | 24 h | Enabled when transformation is used |
| Retrieval | Chunk identifiers, ranks, scores | Corpus version, retrieval/chunk config, query strategy, query hash | 1 h | Enabled |
| Response | Answer, chunk identifiers, metrics | Corpus version, complete pipeline/prompt/security config, query hash | 10 min | Disabled; experimental |

Only conservative whitespace normalization is used. Query case is preserved, and raw questions are never embedded in Redis keys. Retrieval entries deliberately exclude chunk text and hydrate current indexed content from PostgreSQL.

Response entries are admitted only when the strict answer reports `SUPPORTED` or `PARTIAL` and all citation labels are valid. Blocked, insufficient, uncited, invalidly cited, or failed generations are not cached. Response creation uses a best-effort Redis `SET NX` lock. Approximate or semantic response reuse is not implemented.

All Redis operations fail open. Cache unavailability logs a warning and performs the underlying computation.
