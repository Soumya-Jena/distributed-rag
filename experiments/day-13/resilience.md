# Cache resilience and freshness checks

## Redis unavailable

Redis was stopped, a real hybrid retrieval was issued, and Redis was immediately restarted. The request returned five chunks while reporting `cache_available=False`. Read and write timeouts emitted warnings; PostgreSQL retrieval continued normally. Redis subsequently returned `PONG`.

## Idempotent ingestion

The first ingestion after adding content hashes populated hashes and advanced the corpus namespace from version 1 to 2. Re-running ingestion over the same four files skipped chunking and embedding for every document and kept version 2 unchanged.

## Version invalidation

A controlled namespace bump produced this sequence:

| Request | Corpus version | Retrieval cache |
|---|---:|---|
| First | 2 | MISS |
| Exact repeat | 2 | HIT |
| Same query after bump | 3 | MISS |

The old version-2 key remained in Redis and was harmless; it expires by TTL. The live retrieval keys had approximately 3,536 seconds remaining when inspected, consistent with the configured one-hour TTL.

## Output safety

The experimental response cache was exercised with a deterministic grounded fixture: MISS followed by HIT, with identical answer, citations, and chunk identifiers. This is an integration check, not a Qwen latency claim. The cache rejects the uncited short Qwen outputs observed in the preceding context experiment.
