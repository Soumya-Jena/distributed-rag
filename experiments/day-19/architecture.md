# Deployment architecture

```text
Internet / client
       |
       | :8000 (only production host binding)
       v
  +---------+       private HTTP        +--------------+
  | RAG API | ------------------------> | Model server |
  +----+----+    /internal/generate     +--------------+
       |                                     |
       | SQL                                 | model_cache volume
       v                                     v
  +------------+                         Hugging Face cache
  | PostgreSQL |
  +------------+
       ^
       | optional cache (fail-open)
  +----+--+
  | Redis |
  +-------+
```

The `backend` network is internal. The API and model server also join `egress` so they can download model assets when needed. Production publishes only API port 8000. `compose.dev.yaml` deliberately adds PostgreSQL and Redis to `egress` and publishes ports 5432, 6379, and model port 8010 for local tools.

The API owns orchestration, retrieval, context construction, answer validation, and the public request contract. The model server owns model loading, generation, readiness, and its single-slot compute bulkhead. The `/internal/generate` route is not a public API. Local CLI workflows retain the in-process generation adapter, while container deployment selects the HTTP adapter.

Migration and ingestion are one-shot tools under the `tools` profile. Observability and resilience infrastructure use separate opt-in profiles. Source code is baked into the production image; the development override is the only configuration that mounts source, and it does so read-only.
