"""Summarize per-layer cache hit rates and latency percentiles."""

import csv
from pathlib import Path
from statistics import mean

import numpy as np


OUTPUT = Path("experiments/day-13")


def percentile(rows, field, value):
    return float(np.percentile([float(row[field]) for row in rows], value))


def main():
    summaries = []
    for label in ("cold", "warm"):
        with (OUTPUT / f"{label}-cache-results.csv").open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        with (OUTPUT / f"embedding-{label}-results.csv").open(
            encoding="utf-8"
        ) as file:
            embedding_rows = list(csv.DictReader(file))
        summaries.append({
            "state": label,
            "requests": len(rows),
            "embedding_hit_rate": mean(
                int(row["cache_hit"]) for row in embedding_rows
            ),
            "retrieval_hit_rate": mean(
                int(row["retrieval_cache_hit"]) for row in rows
            ),
            "latency_p50_seconds": percentile(rows, "end_to_end_seconds", 50),
            "latency_p95_seconds": percentile(rows, "end_to_end_seconds", 95),
            "latency_p99_seconds": percentile(rows, "end_to_end_seconds", 99),
            "mean_embedding_compute_seconds": mean(
                float(row["compute_seconds"]) for row in embedding_rows
            ),
            "mean_embedding_end_to_end_seconds": mean(
                float(row["end_to_end_seconds"]) for row in embedding_rows
            ),
            "mean_retrieval_seconds": mean(
                float(row["retrieval_seconds"]) for row in rows
            ),
        })
    cold, warm = summaries
    for field in ("latency_p50_seconds", "latency_p95_seconds", "latency_p99_seconds"):
        warm[f"{field}_reduction"] = (
            (cold[field] - warm[field]) / cold[field] if cold[field] else 0.0
        )
    path = OUTPUT / "cache-comparison.csv"
    fields = list(cold.keys()) + [
        key for key in warm if key not in cold
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(*summaries, sep="\n")


if __name__ == "__main__":
    main()
