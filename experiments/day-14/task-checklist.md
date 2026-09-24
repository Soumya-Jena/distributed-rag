# Completion checklist

- [x] Freeze the existing RAG configuration.
- [x] Define RED, retrieval, token, cache, grounding, security, and bounded error metrics.
- [x] Expose a process-wide Prometheus endpoint on port 9108.
- [x] Add Prometheus configuration and pinned Docker service.
- [x] Add a provisioned Grafana dashboard and Prometheus/Jaeger data sources.
- [x] Add OpenTelemetry root and stage spans with exception recording.
- [x] Keep raw questions, chunks, prompts, and answers out of telemetry.
- [x] Add trace-correlated structured JSON logs using query hashes.
- [x] Add Jaeger OTLP ingestion and UI.
- [x] Add unit tests for label cardinality, trace correlation, privacy, and errors.
- [x] Capture a deterministic metric snapshot and trace sample.
- [x] Measure isolated telemetry overhead over 1,000 iterations.
- [x] Document PromQL queries, metric meanings, results, and limitations.
