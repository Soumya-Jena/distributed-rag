# Observability, Metrics, and Tracing Report

## Result

The RAG service now emits the three complementary observability signals:

- Prometheus metrics for request rate, errors, duration, pipeline stages, retrieval quality proxies, tokens, caches, grounding, and security decisions.
- OpenTelemetry traces rooted at `rag.request`, with nested cache, transformation, retrieval, embedding, vector, lexical, fusion, reranking, context, security, generation, and validation operations when those paths run.
- JSON logs sharing the active trace ID and a one-way, truncated SHA-256 query hash. Raw questions, chunks, prompts, and generated answers are not logged or attached to spans.

Prometheus, Grafana, and Jaeger are defined as pinned Docker services. Grafana provisions the data sources and a RED/quality dashboard automatically. The Python metrics endpoint is explicitly started once at application startup with `--metrics-server`, rather than once per answer.

## Overhead experiment

The deterministic 1,000-iteration microbenchmark measured:

| Mode | Mean | P50 | P95 | P99 |
|---|---:|---:|---:|---:|
| Disabled | 0.0119 ms | 0.0118 ms | 0.0119 ms | 0.0134 ms |
| Enabled | 0.0658 ms | 0.0652 ms | 0.0686 ms | 0.0757 ms |

The measured absolute overhead was **0.0539 ms per synthetic request**. Its percentage (454%) is intentionally not representative of a real RAG request because the baseline workload is only about 0.012 ms. Against database retrieval and local model generation measured in earlier experiments, 0.054 ms is negligible. The CSV retains both absolute and percentage values so this denominator effect is explicit.

## Verification

Nine final tests passed: five observability checks (bounded metric labels, shared nested trace identity, privacy-safe structured logging, bounded error classification, and the disable path) plus four security-scoring regressions. The sample trace contains six spans in one trace, and the Prometheus snapshot exposes the new metric families. Python compilation, Compose validation, live Prometheus scraping, Grafana provisioning, and OTLP delivery to Jaeger were also verified.

## Operational use

Start the infrastructure with `docker compose up -d prometheus grafana jaeger`, then run the RAG command with `--metrics-server`. Open Prometheus at port 9090, Grafana at port 3000 (default local credentials `admin`/`admin`), and Jaeger at port 16686. Set `OTEL_EXPORTER=otlp` to send application traces to Jaeger through port 4318.

## Limitations

- The overhead result isolates instrumentation cost; it is not a full model-inference benchmark.
- Histograms are process-local until scraped. A production deployment needs durable Prometheus storage and retention policy.
- Query hashes reduce accidental content exposure but are not anonymization for a small, guessable query set.
- Logs and traces still require access control, retention limits, and transport security in production.
- Metrics are designed to remain low-cardinality; per-user or per-document debugging belongs in access-controlled traces, not Prometheus labels.
