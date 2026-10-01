# Dependency classification

| Component | Class | Safe behavior when unavailable |
|---|---|---|
| Redis cache | Soft | Bypass cache and compute normally (`degraded_cache`). |
| Query transformation | Soft | Search the original user question (`degraded_query`). |
| Embedding/vector retrieval | Soft only when lexical retrieval works | Continue lexical-only (`degraded_lexical_only`). |
| PostgreSQL lexical retrieval | Soft only when vector retrieval works | Continue vector-only (`degraded_vector_only`). |
| Both retrieval branches | Hard | Return HTTP 503 (`unavailable_retrieval`); never generate without evidence. |
| Cross-encoder reranker | Soft | Preserve the first-stage ranking (`degraded_no_rerank`). |
| Context optimizer | Soft | Use the retrieved raw chunks (`degraded_raw_context`). |
| Local generation model | Hard | Return HTTP 503 (`unavailable_generation`). |
| Input/output security validation | Hard | Fail closed with HTTP 503 (`blocked_security`). |
| Metrics/traces/log enrichment | Soft | Return the valid application response; telemetry fails open. |

HTTP 200 means the application produced a usable normal or explicitly degraded answer. HTTP 503 means a hard dependency prevented a safe answer. HTTP 206 is not used because this API is not returning a partial byte range or a partially completed representation.
