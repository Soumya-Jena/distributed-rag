"""Measure one real local query-transformation miss followed by a hit."""

import csv
from pathlib import Path
from time import perf_counter

from src.cache import Cache
from src.query_transformer import QueryTransformer


OUTPUT = Path("experiments/day-13/transform-cache-results.csv")
QUERY = "How does PostgreSQL streaming replication work?"


def main():
    cache = Cache()
    keys = list(cache.client.scan_iter("query_transform:*"))
    if keys:
        cache.client.delete(*keys)
    transformer = QueryTransformer(cache=cache)
    rows = []
    for state in ("cold", "warm"):
        started = perf_counter()
        variants = transformer.transform(QUERY, max_new_tokens=32)
        elapsed = perf_counter() - started
        rows.append({
            "state": state,
            "cache_hit": int(transformer.last_cache_stats.hit),
            "cache_available": int(transformer.last_cache_stats.available),
            "cache_lookup_seconds": transformer.last_cache_stats.cache_lookup_seconds,
            "compute_seconds": transformer.last_cache_stats.compute_seconds,
            "end_to_end_seconds": elapsed,
            "semantic": variants.semantic,
            "technical": variants.technical,
            "alternate": variants.alternate,
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(*rows, sep="\n")


if __name__ == "__main__":
    main()
