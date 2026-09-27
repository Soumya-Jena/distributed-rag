# Query-plan and buffer analysis

Every tier includes raw `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` output under `plans/`.

At 500K chunks, the representative exact-vector plan remained a sequential scan followed by Top-N ordering. It completed in 135.629 ms and touched 16,231 shared-hit blocks plus 108,899 shared-read blocks. That is the expected baseline behavior without an ANN index: PostgreSQL must inspect the vector column across the relation.

The representative lexical query completed in 0.333 ms and touched 77 shared-hit plus 13 shared-read blocks. Its GIN index was selective for that sample. Aggregate lexical timing was non-monotonic across tiers because first-pass cache state and query selectivity materially affected it; the raw plans and both timing passes are retained rather than smoothing that behavior away.

`first_pass` means the first execution inside this benchmark process. It is not described as a true cold-cache result because PostgreSQL, Docker, and the operating system retained buffers from ingestion and prior tiers. `warm_repeat` is the immediately repeated fixed workload.

