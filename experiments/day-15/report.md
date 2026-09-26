# Load, Concurrency, and Saturation Report

## Outcome

The project now has a reproducible load-testing boundary and workflow: FastAPI, realistic categorized workloads, Locust fixed/step/spike profiles, guarded capacity sweeps, raw CSV preservation, resource sampling, saturation summaries, and Grafana capacity panels.

The genuine local Qwen result is decisive: **the tested deployment did not sustain concurrency 1 under the experimental SLO**. The C1 run produced zero valid responses, 100% request failures, and the API process exited after the first full request remained in flight for roughly 118 seconds. Host memory reached 93%. Higher concurrency was not attempted because the predefined safety stop fired.

## Genuine RAG capacity result

| Workload | Concurrency | Goodput | P50 | P95 | P99 | Errors | SLO |
|---|---:|---:|---:|---:|---:|---:|---|
| Unique | 1 | 0.00 RPS | 2.0 s* | 118 s* | 118 s* | 100% | Fail |

`*` These are Locust timings for failed connections, not successful answer latency. The 118-second observation was the first connection ending when the API process exited; subsequent connection failures were fast. Maximum sustainable concurrency is therefore **0** for this hardware/configuration and SLO.

## Bottleneck evidence

- Qwen reported CPU/disk offloading at startup.
- Host memory peaked at 93%; aggregate host CPU peaked at 12.2% across 16 logical CPUs.
- PostgreSQL used at most 5.82% container CPU and about 38 MiB memory.
- Redis used at most 7.51% container CPU and about 9 MiB memory.
- The application process disappeared; the database, cache, and observability containers stayed running.
- A post-run database snapshot showed one active `ragdb` connection; peak connections were not available because the service exited during C1.

This is consistent with local generator/offload memory pressure as the first observed limit. It is not evidence that pgvector, lexical search, Redis, or the load generator saturated.

## Harness validation

To verify the tooling without risking repeated Qwen crashes, a deterministic service with one simulated compute slot was tested separately. These numbers are **not RAG capacity**.

| Mode | C | Goodput RPS | P50 | P95 | P99 | Error % |
|---|---:|---:|---:|---:|---:|---:|
| Unique/uncached | 1 | 1.33 | 200 ms | 240 ms | 240 ms | 0 |
| Unique/uncached | 2 | 2.60 | 200 ms | 420 ms | 420 ms | 0 |
| Unique/uncached | 5 | 4.48 | 400 ms | 730 ms | 780 ms | 0 |
| Repeated/cached | 1 | 1.63 | 14 ms | 240 ms | 240 ms | 0 |
| Repeated/cached | 2 | 3.14 | 14 ms | 18 ms | 220 ms | 0 |
| Repeated/cached | 5 | 7.70 | 15 ms | 18 ms | 28 ms | 0 |
| Repeated/cached | 10 | 12.22 | 14 ms | 22 ms | 38 ms | 0 |

The uncached path hit the relative P95 safety threshold at C5: throughput was flattening near the simulated 5 RPS compute ceiling while tail latency rose. The cached repeated set bypassed most simulated compute and scaled through C10. This validates the analysis pipeline and demonstrates why unique/repeat results must be separated, without presenting fake model performance.

## Verification

All 58 unit tests pass. API validation covers health/readiness, request/response schema, invalid input, and empty answers. The Locust runner, safety stop, manifests, summarizer, plotter, resource sampler, Prometheus metrics, and Grafana dashboard were exercised end to end.

## Recommendation

Do not add PostgreSQL pooling, pgvector tuning, or more application concurrency yet. First make one generator-capacity change—quantization/smaller model, additional memory/GPU, or a dedicated model server with an explicit bounded queue—then rerun C1 before any higher level. Add separate generation queue and execution timers when a queue/semaphore is introduced.
