"""Prove retrieval-cache isolation across a simulated corpus-version bump."""

import csv
from pathlib import Path

from src.cache import Cache, get_corpus_version
from src.db import get_connection
from src.retrieval_pipeline import RetrievalPipeline


OUTPUT = Path("experiments/day-13/cache-invalidation.csv")
QUERY = "How does PostgreSQL streaming replication work?"


def main():
    cache = Cache()
    keys = list(cache.client.scan_iter("retrieval:*"))
    if keys:
        cache.client.delete(*keys)
    pipeline = RetrievalPipeline(
        use_reranker=False, retrieval_mode="hybrid", cache=cache
    )
    before = get_corpus_version()
    first = pipeline.retrieve(QUERY, final_k=5)
    second = pipeline.retrieve(QUERY, final_k=5)
    old_keys = list(cache.client.scan_iter(f"retrieval:{before}:*"))

    # The ingestion path owns real version bumps. This controlled benchmark
    # changes only the namespace counter so no source document is modified.
    with get_connection() as conn:
        conn.execute(
            "UPDATE corpus_config SET corpus_version = corpus_version + 1, "
            "updated_at = NOW() WHERE id = 1"
        )
        conn.commit()
    after = get_corpus_version()
    third = pipeline.retrieve(QUERY, final_k=5)
    rows = [
        {"request": "first", "corpus_version": before,
         "retrieval_cache_hit": int(first.cache_stats.hit)},
        {"request": "repeat", "corpus_version": before,
         "retrieval_cache_hit": int(second.cache_stats.hit)},
        {"request": "after_version_bump", "corpus_version": after,
         "retrieval_cache_hit": int(third.cache_stats.hit)},
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if not old_keys or not all(cache.client.exists(key) for key in old_keys):
        raise RuntimeError("Expected the old versioned entry to remain until TTL expiry")
    if not (not first.cache_stats.hit and second.cache_stats.hit and not third.cache_stats.hit):
        raise RuntimeError("Unexpected invalidation sequence")
    print(f"Verified retrieval cache MISS after corpus {before} -> {after}")


if __name__ == "__main__":
    main()
