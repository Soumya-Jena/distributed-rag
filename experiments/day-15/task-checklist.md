# Completion checklist

- [x] Freeze the existing RAG configuration and write the hypothesis.
- [x] Add a minimal `/health`, `/ready`, and `/query` HTTP boundary.
- [x] Build categorized unique and repeated-query workloads.
- [x] Add Locust fixed, stepped, and spike profiles with gradual ramping.
- [x] Define goodput, in-flight, HTTP-outcome, and aggregate-token metrics.
- [x] Add Grafana concurrency, tail-latency, error, goodput, and token panels.
- [x] Add guarded sweep automation and Locust CSV summarization.
- [x] Add host/container resource sampling and PostgreSQL connection diagnostics.
- [x] Define the experimental SLO and explicit safety stops.
- [x] Validate the harness with unique and repeat modes and plot their curves.
- [x] Run a genuine C1 Qwen check and capture resource/failure evidence.
- [x] Stop before C2 after C1 process termination and 100% failures.
- [x] Document why spike, stampede, and soak runs are unsafe after C1 failure.
- [x] Run the complete unit-test suite.
