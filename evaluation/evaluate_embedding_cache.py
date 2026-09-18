"""Measure embedding-cache hits independently of retrieval-cache bypasses."""

import csv
import json
from pathlib import Path
from time import perf_counter

from src.cache import Cache
from src.config import EMBEDDING_MODEL
from src.embedding_service import EmbeddingService


DATASET = Path("datasets/evaluation/cache_workload.jsonl")
OUTPUT = Path("experiments/day-13")


def run(label, service, workload):
    rows = []
    for record in workload:
        started = perf_counter()
        service.encode_query(record["query"])
        elapsed = perf_counter() - started
        stats = service.last_cache_stats
        rows.append({
            **record,
            "cache_hit": int(stats.hit),
            "cache_available": int(stats.available),
            "cache_lookup_seconds": stats.cache_lookup_seconds,
            "compute_seconds": stats.compute_seconds,
            "end_to_end_seconds": elapsed,
        })
    path = OUTPUT / f"embedding-{label}-results.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    with DATASET.open(encoding="utf-8") as file:
        workload = [json.loads(line) for line in file if line.strip()]
    cache = Cache()
    keys = list(cache.client.scan_iter("embedding:*"))
    if keys:
        cache.client.delete(*keys)
    service = EmbeddingService(EMBEDDING_MODEL, cache=cache)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run("cold", service, workload)
    run("warm", service, workload)
    print("Saved independent embedding-cache results")


if __name__ == "__main__":
    main()
