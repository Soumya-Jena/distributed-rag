"""Measure the overhead added by the HTTP generation service boundary."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import httpx

from src.model_server.generator import FakeGenerator


MESSAGES = [{"role": "user", "content": "Answer from the supplied context."}]


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8010")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/day-19/generation-boundary.csv"),
    )
    args = parser.parse_args()

    local_generator = FakeGenerator()
    samples: list[dict[str, float | int | str]] = []
    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        for sample_id in range(1, args.requests + 1):
            started = time.perf_counter()
            local_generator.generate(MESSAGES, max_new_tokens=32)
            local_ms = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            response = client.post(
                "/internal/generate",
                json={"messages": MESSAGES, "max_new_tokens": 32},
            )
            response.raise_for_status()
            http_ms = (time.perf_counter() - started) * 1000
            samples.append({
                "sample": sample_id,
                "local_ms": round(local_ms, 4),
                "http_ms": round(http_ms, 4),
                "boundary_overhead_ms": round(http_ms - local_ms, 4),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=samples[0].keys())
        writer.writeheader()
        writer.writerows(samples)

    overhead = [float(row["boundary_overhead_ms"]) for row in samples]
    summary = {
        "requests": args.requests,
        "mean_boundary_overhead_ms": round(statistics.mean(overhead), 4),
        "p50_boundary_overhead_ms": round(percentile(overhead, 0.50), 4),
        "p95_boundary_overhead_ms": round(percentile(overhead, 0.95), 4),
    }
    summary_path = args.output.with_name("generation-boundary-summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
