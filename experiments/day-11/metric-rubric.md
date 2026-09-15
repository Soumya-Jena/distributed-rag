# Query-transformation metric rubric

- **Hit@1 / Hit@5**: whether a labeled relevant source appears in the first one or five final chunks.
- **MRR**: reciprocal rank of the first labeled relevant source in the final five.
- **Candidate Hit@20**: whether cross-query fusion contains a relevant source before the final top-five cut. With this four-document, 17-chunk corpus, the candidate pool can saturate; this metric is retained for compatibility but is not expected to discriminate strategies well.
- **Transformation rescue rate at K**: among questions where original-query Hit@K is zero, the share for which a transformed strategy reaches Hit@K. Both K=1 and K=5 are reported; a blank value means the original strategy had no misses to rescue.
- **Transformation regression rate at K**: among questions where original-query Hit@K is one, the share for which a transformed strategy loses that evidence at K. Both K=1 and K=5 are reported.
- **Exact-identifier preservation**: each uppercase acronym, five-digit code, or underscore-delimited identifier extracted from the original remains verbatim (case-insensitive) in a generated query. The original query is retained in all ensemble strategies even if a rewrite loses an identifier.
- **Meaning preserved**: manual review as Yes, Partial, or No.
- **Hallucinated terminology**: manual Yes/No judgment for technical terms that change or over-assume the user's intent.
- **Retrieval latency**: wall-clock hybrid retrieval across all query variants, excluding transformation generation. Transformation latency is reported separately by the inspection run.

The 50-query benchmark uses reviewed, frozen variants stored with the dataset so all retrieval strategies are deterministic. The separate 20-query Qwen inspection measures the configured transformer itself and records fallbacks, identifier loss, latency, and manual judgments.
