# PostgreSQL Full-Text Search Verification

The production `ragdb` schema was initialized and the frozen corpus was re-ingested idempotently.

## Schema checks

- Corpus configuration: MiniLM, 100-token chunks, 20-token overlap
- Stored chunks: 17
- Populated generated `search_vector` values: 17
- GIN index: `idx_chunks_search_vector`
- Table statistics refreshed with `ANALYZE chunks`

## Direct lexical checks

| Query | First result | Chunk | `ts_rank_cd` |
|---|---|---:|---:|
| `streaming replication` | postgres-replication | 0 | 0.3556 |
| `MVCC` | postgres-mvcc | 0 | 0.2000 |
| `VACUUM` | postgres-mvcc | 2 | 0.1000 |
| `consumer group` | kafka-overview | 1 | 0.1444 |

The tiny 17-row corpus may legitimately use a sequential scan despite the GIN index; index scaling is outside this experiment.
