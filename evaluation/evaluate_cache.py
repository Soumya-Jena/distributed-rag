"""Benchmark cold and warm embedding/retrieval caches on 100 requests."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

from src.cache import Cache
from src.retrieval_pipeline import RetrievalPipeline


DATASET = Path("datasets/evaluation/cache_workload.jsonl")
OUTPUT = Path("experiments/day-13")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["cold", "warm"], required=True)
    parser.add_argument("--flush", action="store_true")
    args = parser.parse_args()

    cache = Cache()
    if not cache.ping():
        raise RuntimeError("Redis is unavailable; start it with docker compose up -d redis")
    if args.flush:
        cache.client.flushdb()

    with DATASET.open(encoding="utf-8") as file:
        workload = [json.loads(line) for line in file if line.strip()]
    pipeline = RetrievalPipeline(
        use_reranker=False, retrieval_mode="hybrid", cache=cache
    )
    rows = []
    for record in workload:
        started = perf_counter()
        result = pipeline.retrieve(record["query"], final_k=5)
        end_to_end = perf_counter() - started
        embedding = pipeline.retriever.embedding_service.last_cache_stats
        retrieval = result.cache_stats
        rows.append({
            **record,
            "corpus_version": result.corpus_version,
            "retrieval_cache_hit": int(retrieval.hit),
            "retrieval_cache_available": int(retrieval.available),
            "retrieval_cache_lookup_seconds": retrieval.cache_lookup_seconds,
            "embedding_cache_hit": int(embedding.hit),
            "embedding_cache_available": int(embedding.available),
            "embedding_cache_lookup_seconds": embedding.cache_lookup_seconds,
            "embedding_compute_seconds": embedding.compute_seconds,
            "vector_seconds": result.vector_seconds,
            "lexical_seconds": result.lexical_seconds,
            "fusion_seconds": result.fusion_seconds,
            "retrieval_seconds": (
                result.vector_seconds + result.lexical_seconds
                + result.fusion_seconds + result.rerank_seconds
            ),
            "end_to_end_seconds": end_to_end,
            "result_ids": "|".join(
                str(chunk.chunk_id) for chunk in result.final_chunks
            ),
        })
        print(
            f"{record['id']} retrieval={'HIT' if retrieval.hit else 'MISS'}",
            flush=True,
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{args.label}-cache-results.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
