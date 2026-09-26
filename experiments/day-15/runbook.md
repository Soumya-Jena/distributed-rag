# Load-test runbook

## Start the real service

```powershell
docker compose up -d postgres redis prometheus grafana jaeger
$env:OTEL_EXPORTER = "otlp"
python -m uvicorn src.load_test_api:app --host 0.0.0.0 --port 8000
```

Wait for `GET http://localhost:8000/ready` to return `{"status":"ready"}`.

## Guarded capacity sweeps

```powershell
python -m evaluation.run_load_sweep --mode unique --duration 5m
python -m evaluation.run_load_sweep --mode repeat --duration 5m
python -m evaluation.summarize_load
python -m evaluation.plot_load_results
```

The runner gradually tests `1, 2, 5, 10, 15, 25, 50`, writes raw Locust CSV files, and stops on errors or P95 above three times baseline. Do not bypass the stop on a developer laptop.

## Resource sampling

Run alongside Locust:

```powershell
python -m evaluation.sample_load_resources --seconds 300 --interval 5 --pid <uvicorn-pid>
```

Inspect database connections with `load_tests/diagnostics.sql`. Grafana shows in-flight requests, goodput, P50/P95/P99, HTTP outcomes, stage P95, cache outcomes, and aggregate output tokens/s.

If the Locust process reaches high CPU, stop: that is load-generator saturation. Prefer a separate machine, or use Locust worker processes/FastHttpUser only after proving the generator—not the RAG service—is limiting the test. On one laptop, record the load-generator PID separately so its CPU is not mistaken for application CPU.

## Step and spike profiles

```powershell
$env:LOAD_MODE = "unique"
locust -f load_tests/step_shape.py --headless --host http://localhost:8000
locust -f load_tests/spike_shape.py --headless --host http://localhost:8000
```

Only run these after an individual sweep proves every included stage safe. The spike profile is 2 users, suddenly 20, then recovery at 2. Recovery time is the interval from spike end until latency and resources return near baseline.

## Cold-cache stampede protocol

Only after C10 is known safe:

```powershell
docker exec rag-redis redis-cli FLUSHDB
$env:LOAD_MODE = "repeat"
locust -f load_tests/locustfile.py --headless --host http://localhost:8000 -u 10 -r 10 -t 2m --csv experiments/day-15/stampede
```

Compare generation calls, cache operations, lock waits, and traces. A successful lock produces one expensive computation; followers reuse the result.

## Short soak protocol

After finding saturation, run 15–30 minutes at 70–80% of the safe concurrency. Watch process/host memory, connection counts, cache size, failures, and recovery. No soak is permitted when C1 fails.
