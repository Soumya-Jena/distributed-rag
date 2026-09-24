"""Measure isolated telemetry cost without loading generation models."""

import argparse
import csv
import statistics
import time
from pathlib import Path

from src.runtime_metrics import REQUESTS, REQUEST_DURATION, STAGE_DURATION
from src.tracing import observed_span


def workload(observed):
    started = time.perf_counter()
    if observed:
        with observed_span("rag.request"):
            with observed_span("retrieval"):
                sum(value * value for value in range(200))
            STAGE_DURATION.labels(stage="vector_search").observe(0.0001)
            REQUESTS.labels(status="success").inc()
            REQUEST_DURATION.observe(time.perf_counter() - started)
    else:
        sum(value * value for value in range(200))
    return (time.perf_counter() - started) * 1000


def percentile(values, percentile_value):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile_value)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("experiments/day-14/observability-overhead.csv"))
    args = parser.parse_args()
    for _ in range(100):
        workload(False)
        workload(True)
    rows = []
    for mode in ("disabled", "enabled"):
        values = [workload(mode == "enabled") for _ in range(args.iterations)]
        rows.append({
            "mode": mode, "iterations": args.iterations,
            "mean_ms": statistics.mean(values), "p50_ms": percentile(values, .50),
            "p95_ms": percentile(values, .95), "p99_ms": percentile(values, .99),
        })
    baseline = rows[0]["mean_ms"]
    rows[1]["overhead_ms"] = rows[1]["mean_ms"] - baseline
    rows[1]["overhead_percent"] = 100 * rows[1]["overhead_ms"] / baseline
    rows[0]["overhead_ms"] = rows[0]["overhead_percent"] = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
