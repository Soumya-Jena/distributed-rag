# Distributed RAG

An experimental lab for building, evaluating, and benchmarking retrieval-augmented generation systems with PostgreSQL, pgvector, full-text search, and local language models.

## Project structure

- `src/ingest.py` — chunk, embed, and store documents
- `src/retriever.py` — dense pgvector retrieval
- `src/lexical_retriever.py` — PostgreSQL full-text retrieval
- `src/fusion.py` — Reciprocal Rank Fusion (RRF)
- `src/hybrid_retriever.py` — vector + lexical candidate generation
- `src/reranker.py` — optional cross-encoder second stage
- `src/rag_service.py` — retrieval, prompting, and local generation
- `evaluation` — repeatable retrieval and RAG experiments
- `datasets` — source documents and evaluation questions
- `experiments` — versioned measurements, reports, and charts
- `tests` — unit tests for retrieval orchestration and fusion

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
docker compose up -d
python -m src.ingest datasets/raw
```

## Hybrid retrieval

The default retrieval path combines dense MiniLM embeddings with PostgreSQL full-text search over a generated `tsvector` column. RRF combines the two rankings without mixing incomparable cosine and `ts_rank_cd` scores. Cross-encoder reranking remains available but is disabled by default because it did not improve the current benchmark.

```powershell
python -m src.rag "How does PostgreSQL streaming replication work?"
python -m src.rag "What is MVCC?" --retrieval-mode hybrid --reranker
python -m src.rag "What is MVCC?" --grounding-mode strict
python -m src.rag "What is MVCC?" --grounding-mode baseline
python -m evaluation.inspect_hybrid
python -m evaluation.evaluate_hybrid --label hybrid
```

Configuration lives in `.env`; copy `.env.example` when setting up a new environment.

## Grounding evaluation

The selected strict evidence contract requires `SUPPORTED`, `PARTIAL`, or `INSUFFICIENT` status, forbids filling gaps from model memory, and requests claim-level citations. The baseline prompt remains available for reproducible A/B evaluation.

```powershell
python -m evaluation.evaluate_grounding --mode baseline --label baseline-grounding --batch-size 4 --max-new-tokens 32
python -m evaluation.evaluate_grounding --mode strict --label strict-grounding --batch-size 4 --max-new-tokens 32
python -m evaluation.score_grounding
python -m evaluation.plot_grounding_results
```

See `experiments/day-09/report.md` for the claim-level review and limitations.

## Security experiment

The security test uses a separate `ragdb_security` database and `datasets/security`; it does not ingest synthetic attacks into the normal corpus. Six synthetic poisoned documents include instruction override, prompt extraction, citation hijack, output hijack, social-engineering, and retrieval-poisoning cases. No real credentials are used.

```powershell
python -m evaluation.setup_security_db
$previousDatabaseUrl = $env:DATABASE_URL
$env:DATABASE_URL = "postgresql://rag:rag@localhost:5432/ragdb_security"
python -m src.ingest datasets/security
python -m evaluation.annotate_security_provenance
$env:DATABASE_URL = $previousDatabaseUrl
python -m evaluation.evaluate_security --mode all --batch-size 4 --max-new-tokens 32
python -m evaluation.recount_security_tokens
python -m evaluation.score_security
```

The evaluator always checks that it is connected to `ragdb_security`; it refuses to run on `ragdb`. It saves one raw-result CSV per mode and checkpoints after each model batch. `SECURITY_MODE=layered` enables source delimiting, injection flags, and an output guard for synthetic test markers. All three modes had 0% conditional marker/canary ASR in this small test, while the layered detector found six of nine poisoned chunks with no clean-chunk flags. Neither regex detection nor marker blocking proves the answer is factually grounded. See `experiments/day-10/report.md` for results and limitations.

## Query transformation

The query experiment compares original-only retrieval, rewrite-only retrieval, original plus one rewrite, and original plus three diverse variants. Each query still uses the selected hybrid retrieval path; a second RRF pass combines candidates across variants and records which query found each chunk. Optional reranking always compares candidates with the original question. The secure grounding and output controls remain downstream of retrieval.

```powershell
python -m evaluation.evaluate_multi_query
python -m evaluation.inspect_query_transform --limit 20 --batch-size 4 --max-new-tokens 64
python -m evaluation.plot_multi_query_results
python -m src.query_search "Why is the log filling the disk?" --strategy multi_query
python -m src.rag "Why is the log filling the disk?" --query-strategy multi_query
```

Multi-query retrieval improved Hit@1 from 0.92 to 1.00 on the frozen 50-question benchmark, but it remains opt-in because retrieval was about four times slower and the separate Qwen review averaged 67.9 seconds per transformation with imperfect meaning preservation. See `experiments/day-11/report.md` for the complete comparison and limitations.

## Context optimization

Post-retrieval context handling supports full chunks, conservative near-duplicate removal, query-focused extractive sentence selection, and a global whole-sentence token budget. Every optimized fragment retains its original document, chunk, and sentence provenance. Full context remains the default because the 50% extractive arm reduced mean context tokens by 41.2% but did not preserve answer correctness in the small generation sample.

```powershell
python -m evaluation.evaluate_context
python -m evaluation.evaluate_context_rag --max-new-tokens 20
python -m evaluation.recount_context_tokens
python -m evaluation.score_context_rag
python -m evaluation.plot_context_results
python -m src.rag "How does PostgreSQL streaming replication work?" --context-strategy extractive
python -m src.rag "How does PostgreSQL streaming replication work?" --context-strategy budgeted
```

Configuration is available through `CONTEXT_STRATEGY`, `CONTEXT_DEDUP_THRESHOLD`, `CONTEXT_KEEP_RATIO`, `CONTEXT_TOKEN_BUDGET`, and `CONTEXT_NEIGHBOR_WINDOW`. See `experiments/day-12/report.md` for the complete results and limitations.

## Caching and invalidation

Redis caches exact normalized query embeddings, optional query transformations, and identifier-only retrieval rankings. Cache keys include model, prompt, retrieval, and chunking fingerprints; retrieval and experimental response keys also include the monotonic corpus version. Re-ingesting identical content does not advance that version. Redis failures log a warning and compute normally.

```powershell
docker compose up -d redis postgres
python -m src.ingest datasets/raw
python -m evaluation.build_cache_workload
python -m evaluation.evaluate_cache --label cold --flush
python -m evaluation.evaluate_cache --label warm
python -m evaluation.evaluate_embedding_cache
python -m evaluation.summarize_cache
python -m evaluation.plot_cache_results
```

Final-response caching is intentionally disabled by default. Enable it only for a controlled run with `--response-cache`; uncited, invalidly cited, insufficient, blocked, and failed outputs are not admitted. See `experiments/day-13/report.md` for measured results and limitations.

## Observability

The online pipeline exposes low-cardinality Prometheus metrics, nested OpenTelemetry traces, and trace-correlated JSON logs. Raw questions and document contents are excluded. Start the monitoring services and the application endpoint with:

```powershell
docker compose up -d prometheus grafana jaeger
$env:OTEL_EXPORTER = "otlp"
python -m src.rag "How does PostgreSQL streaming replication work?" --metrics-server
```

Prometheus is available at `http://localhost:9090`, Grafana at `http://localhost:3000`, Jaeger at `http://localhost:16686`, and application metrics at `http://localhost:9108/metrics`. See `experiments/day-14/report.md` for the metric catalog, PromQL runbook, measured overhead, and limitations.

## Load and saturation testing

The load boundary exposes `/health`, `/ready`, and `/query`; it is intentionally smaller than a production API. Locust supports categorized unique/repeated workloads, fixed sweeps with safety stops, gradual step load, and spike recovery. Start the service and a guarded sweep with:

```powershell
python -m uvicorn src.load_test_api:app --host 0.0.0.0 --port 8000
python -m evaluation.run_load_sweep --mode unique --duration 5m
```

On the tested CPU/disk-offloaded environment, the genuine C1 Qwen run exited under 93% host-memory pressure and achieved zero valid goodput, so higher concurrency was intentionally not attempted. See `experiments/day-15/report.md` for the real result, separate synthetic harness validation, safety policy, and rerun procedure.

## Corpus-volume scalability

The corpus benchmark uses a guarded `ragdb_scale` database and keeps semantic quality separate from synthetic database mechanics. It streams seeded 384-dimensional vectors in bounded batches, benchmarks exact vector, GIN lexical, and hybrid retrieval with 50 fixed queries at concurrency 1, and saves storage, buffer, latency, and host-resource evidence.

```powershell
python -m evaluation.setup_scale_db
python -m evaluation.generate_scale_data --targets 10000
python -m evaluation.evaluate_corpus_scale --stage S1 --expected-chunks 10000
python -m evaluation.summarize_corpus_scale
python -m evaluation.plot_corpus_scale
```

The healthy measured ceiling was 500K chunks: exact-vector warm P95 was 164.1 ms and the relation occupied 1,051.4 MiB. The 1M tier was stopped before ingestion under the fixed host-memory gate. See `experiments/day-16/report.md` for the results and the boundary on semantic claims.

## Resilience and graceful degradation

The service now distinguishes soft dependency loss from hard safety boundaries. A vector/embedding failure falls back to lexical retrieval, a lexical failure falls back to vector retrieval, and reranker or context-optimizer failures preserve the earlier evidence. Loss of both retrieval paths, generation, or security validation returns an explicit HTTP 503 instead of producing an unsupported answer. Cache and observability failures remain fail-open.

```powershell
docker compose up -d postgres redis toxiproxy
python -m evaluation.setup_toxiproxy
python -m evaluation.run_resilience --network
python -m pytest tests/test_resilience.py -q
```

`FAULT_INJECTION_ENABLED` is accepted only when `ENVIRONMENT` is `development` or `test`. Toxiproxy uses separate test ports (`15432` for PostgreSQL and `16379` for Redis), and the runner restores healthy routes after each fault. See `experiments/day-18/report.md` for the failure matrix, measured timeouts, recovery result, and limitations.

## Containerized deployment

The deployment separates the public RAG API from the private generation model server. PostgreSQL, Redis, and the model endpoint stay on an internal network; only the API publishes a production port. The same non-root, multi-stage image is reused for the API, model server, migrations, and ingestion jobs.

Create `secrets/postgres_password.txt`, then start the core stack:

```powershell
docker compose up -d postgres redis model-server
docker compose --profile tools run --rm db-migrate
docker compose up -d rag-api
python -m evaluation.verify_deployment
```

For local development, the override publishes PostgreSQL, Redis, and the model server and mounts source code read-only:

```powershell
docker compose -f docker-compose.yml -f compose.dev.yaml up -d
```

Use `--profile tools` for migration and ingestion jobs, `--profile observability` for Prometheus/Grafana/Jaeger, and `--profile resilience` for Toxiproxy. Model weights live in the `model_cache` volume rather than the image. See `experiments/day-19/runbook.md` and `experiments/day-19/report.md` for operations, measured results, and limitations.
