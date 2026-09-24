# Metric catalog

| Metric | Type | Labels | Purpose |
|---|---|---|---|
| `rag_requests_total` | counter | `status` | Request rate and failures |
| `rag_request_duration_seconds` | histogram | none | End-to-end latency |
| `rag_stage_duration_seconds` | histogram | `stage` | Pipeline bottlenecks |
| `rag_retrieval_candidates` | histogram | none | Retrieval breadth |
| `rag_final_context_chunks` | histogram | none | Context size |
| `rag_empty_retrieval_total` | counter | none | No-evidence frequency |
| `rag_tokens` | histogram | `kind` | Prompt/context/output volume |
| `rag_context_compression_ratio` | histogram | none | Context optimization |
| `rag_cache_operations_total` | counter | `cache`, `result` | Cache outcomes |
| `rag_cache_lookup_duration_seconds` | histogram | `cache` | Cache cost |
| `rag_evidence_status_total` | counter | `status` | Grounding outcomes |
| `rag_refusals_total` | counter | none | Abstention frequency |
| `rag_security_events_total` | counter | `decision` | Clean/flagged/allowed/blocked decisions |
| `rag_errors_total` | counter | `stage`, `error_type` | Bounded failures |
| `rag_observability_overhead_seconds` | histogram | none | Telemetry recording cost |

Queries, document IDs, source paths, user IDs, and trace IDs are deliberately excluded from metric labels to prevent unbounded cardinality. Trace IDs belong in traces and logs only.
