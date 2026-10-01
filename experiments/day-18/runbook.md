# Resilience experiment runbook

Start the test dependencies and create the test-only proxy routes:

```powershell
docker compose up -d postgres redis toxiproxy
python -m evaluation.setup_toxiproxy
```

Run the deterministic and real-network scenarios:

```powershell
python -m evaluation.run_resilience --network
python -m pytest tests/test_resilience.py -q
```

The proxy connection strings are deliberately separate from normal configuration:

- PostgreSQL: `postgresql://rag:rag@localhost:15432/ragdb`
- Redis: `redis://localhost:16379/0`
- Toxiproxy API: `http://localhost:8474`

The runner restores both proxies and removes toxics in `finally` blocks. To recover manually, rerun `python -m evaluation.setup_toxiproxy`; it is idempotent. Verify normal connectivity with `docker compose ps` and a normal RAG query.

Fault injection is disabled by default and raises during construction if enabled outside `development` or `test`. Never route production traffic through these proxy ports.

Useful Prometheus signals:

```promql
sum by (mode, component) (rate(rag_resilience_events_total[5m]))
max by (dependency) (rag_circuit_state)
sum by (dependency) (rate(rag_bulkhead_rejections_total[5m]))
sum by (status) (rate(rag_http_requests_total[5m]))
```
