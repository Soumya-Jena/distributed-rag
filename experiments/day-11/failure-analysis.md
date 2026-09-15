# Query-transformation failure analysis

## Original-query rank-one misses

- `qt012` asks how change records reach and are applied to a backup database. The original query ranked `postgres-mvcc.md` first; the relevant replication source appeared fourth.
- `qt020` describes replaying earlier stream records without Kafka terminology. The original query ranked `postgres-replication.md` first; the relevant Kafka source appeared fifth.
- `qt025` asks only why a commit became slower. The original query ranked `postgres-mvcc.md` first; the labeled replication source appeared third.
- `qt029` asks why a log fills disk. The original query ranked `kafka-overview.md` first; the labeled replication source appeared fourth.

These are ambiguity and vocabulary failures rather than complete retrieval failures: every labeled source was still inside the top five. Diverse query variants moved the labeled source to rank one in all four cases.

## Risks

- Rewrite-only can discard identifiers or narrow an intentionally broad question, so it is not eligible as the default.
- Vague queries may have several defensible interpretations. A benchmark with one labeled source can reward a rewrite for guessing the evaluator's intended domain.
- Multiple nearly identical rewrites can over-count one interpretation in cross-query RRF. Provenance is recorded in `found_by` and `query_ranks` so this can be audited.
- A rewrite can introduce terminology absent from the user's question. The 20-query model review records meaning preservation, identifier preservation, and hallucinated terminology separately.
- Candidate Hit@20 is saturated because the corpus contains only 17 chunks; it cannot demonstrate candidate-generation gains here.

## Mitigations

The ensemble keeps the original query, validates parser output, falls back to the original on generation failure, records per-query provenance, and reranks against the original question when reranking is enabled. Query transformation happens before the existing strict-grounding and layered-security boundary; transformed text never bypasses those controls.

## Observed model-transformation failures

- `qt042` changed a two-part `MVCC` and `VACUUM` question into generic snapshot-isolation and deadlock queries, losing both identifiers and much of the intent.
- `qt024` interpreted an ambiguous request to read the same messages again as an email-retrieval task instead of Kafka replay.
- `qt041` dropped `WAL` from one semantic variant, although the other variants and retained original still carried it.
- `qt031` produced an incomplete alternate query because generation stopped at the token limit.
- Four of 20 generations did not satisfy the strict three-line parser and safely reverted every variant to the original.
