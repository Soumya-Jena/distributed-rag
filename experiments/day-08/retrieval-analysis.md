# Retrieval and Rescue Analysis

## Query cohorts

The benchmark contains 50 answerable questions:

- 20 semantic paraphrases
- 15 exact lexical/identifier queries
- 15 mixed semantic and lexical queries

Four additional unsupported identifiers are retained as controls but excluded from relevance averages.

## Candidate rescue

| Metric | Result |
|---|---:|
| Vector Hit@20 | 1.000 |
| Lexical Hit@20 | 0.600 |
| Union candidate hit | 1.000 |
| Lexical rescues | 0 / 50 (0%) |
| Semantic rescues | 20 / 50 (40%) |

The corpus has only 17 chunks, so a vector candidate depth of 20 exhausts it and cannot miss a relevant source. Consequently, lexical candidate rescue cannot be demonstrated at this scale. PostgreSQL `simple` text search also applies AND semantics to natural-language query terms; it found none of the deliberate paraphrases, which explains all 20 semantic rescues.

## Rank-one rescue

Although candidate recall was saturated, lexical evidence corrected four vector rank-one errors through RRF:

| ID | Query | Correct source promoted by RRF |
|---|---|---|
| q121 | `WAL` | postgres-replication.md |
| q127 | `transaction ID wraparound` | postgres-mvcc.md |
| q132 | `replication factor` | kafka-overview.md |
| q148 | `replication factor follower leader` | kafka-overview.md |

That changes overall Hit@1 from 0.920 to 1.000 while preserving Hit@5 at 1.000.

## Query-type behavior

| Query type | Vector Hit@1 | Lexical Hit@20 | RRF Hit@1 |
|---|---:|---:|---:|
| Semantic | 1.000 | 0.000 | 1.000 |
| Lexical | 0.800 | 1.000 | 1.000 |
| Mixed | 0.933 | 1.000 | 1.000 |

The result matches the complementary-retriever hypothesis: vectors handle paraphrases, while full-text search anchors exact technical terminology.

## RRF sensitivity

RRF constants 10, 30, 60, and 100 all produced Hit@1, Hit@5, and MRR of 1.000. The conventional value 60 is retained because this small corpus provides no evidence for tuning it.
