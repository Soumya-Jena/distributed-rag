# Production packaging report

## Result

The RAG application is packaged as two independently health-checked services: a public orchestration API and a private model server. Both reuse the same 450,662,140-byte (about 429.8 MiB) multi-stage CPU image and run as `uid=10001(rag)`. Source code, secrets, and model weights have distinct lifecycles: code is immutable in the image, the PostgreSQL password is a mounted secret, and Hugging Face assets persist in `model_cache`.

The packaged smoke query returned HTTP 200, preserved its request ID, included a trace ID, and returned three sources from the persisted corpus. All readiness checks—PostgreSQL, embeddings, generation, and security—were true. An explicit migration completed successfully, and PostgreSQL retained all 17 chunks after container recreation.

| Scenario | API live | API ready | Query | Outcome |
|---|---:|---:|---:|---|
| Healthy | 200 | 200 | 200 | Normal answer with three sources |
| Model server stopped | 200 | 503 | 503 | Hard dependency contained; no unsupported answer |
| Model server restarted | 200 | 200 | 200 | Readiness recovered |
| Redis stopped | 200 | 200 | 200 | Cache loss failed open |

The deterministic fake-model boundary benchmark made 100 requests through a reused HTTP connection. Mean measured service-boundary overhead was 47.2793 ms, P50 was 47.7527 ms, and P95 was 50.4928 ms on Docker Desktop. The raw samples are in `generation-boundary.csv`; this is packaging overhead, not real-model inference latency.

The complete regression suite passed: 83 tests in 16.17 seconds, with one upstream Starlette/httpx deprecation warning.

## Observation

Separating generation makes the most expensive dependency independently observable and restartable. Liveness remains useful during a downstream outage, while readiness protects traffic from entering an instance that cannot safely answer. Moving the existing bulkhead to the model process also prevents multiple API workers from independently oversubscribing the same model server.

The internal-only production network prevented accidental host publication even when a development port mapping was present. The development override therefore has to add PostgreSQL and Redis to the egress network as well as publish their ports; after doing so, Docker exposed 5432 and 6379 as intended. The production configuration remains private.

Named volumes behaved as the persistence boundary: application and dependency containers could be recreated without losing the indexed corpus or cached model marker. Explicit migrations remove schema mutation from ordinary database connections and make startup ordering auditable.

## Limitations

- The container smoke and boundary benchmark use a deterministic fake generator. The real Qwen model contract is unchanged, but its cold download, memory use, and answer latency were not repeated in the container.
- Docker Desktop networking dominates the fake-model HTTP measurement; it should not be generalized to Linux hosts or real inference workloads.
- This is a single-host Compose deployment, not a multi-replica scheduler, load balancer, autoscaler, or zero-downtime rollout system.
- Compose resource limits are initial safety ceilings, not values derived from production traffic.
- The ingestion job expects a supplied read-only dataset mount; production object storage is outside this lab.
- The database migration is intentionally simple and additive. A larger schema requires versioned forward and rollback migrations.
- TLS, external secret management, image signing, vulnerability scanning, and registry publication remain deployment-platform responsibilities.

## Reproduction

See `runbook.md`. Machine-readable evidence is saved in `smoke-results.json`, `generation-boundary-summary.json`, `generation-boundary.csv`, and `failure-matrix.csv`.
