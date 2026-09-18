"""Exercise response-cache provenance and hit behavior without costly Qwen output."""

import csv
from pathlib import Path
from time import perf_counter

from src.cache import Cache
from src.rag_service import RAGService


OUTPUT = Path("experiments/day-13/response-cache-results.csv")
QUERY = "How does PostgreSQL streaming replication work?"


class DeterministicGroundedGenerator:
    model_name = "cache-integration-fixture"
    last_input_token_count = 32
    last_output_token_count = 12

    @staticmethod
    def count_tokens(text):
        return len(text.split())

    def generate(self, messages, max_new_tokens=None):
        return (
            "EVIDENCE_STATUS: SUPPORTED\n\n"
            "Streaming replication sends WAL from a primary to standby "
            "servers. [S1]"
        )


def main():
    cache = Cache()
    keys = list(cache.client.scan_iter("response:*"))
    if keys:
        cache.client.delete(*keys)
    service = RAGService(
        use_reranker=False,
        generator=DeterministicGroundedGenerator(),
        cache=cache,
        response_cache_enabled=True,
    )
    rows = []
    for state in ("cold", "warm"):
        started = perf_counter()
        result = service.answer(QUERY)
        elapsed = perf_counter() - started
        rows.append({
            "state": state,
            "response_cache_hit": int(result.response_cache_hit),
            "end_to_end_seconds": elapsed,
            "answer": result.answer,
            "chunk_ids": "|".join(str(chunk.chunk_id) for chunk in result.chunks),
            "corpus_version": service.response_cache.key(QUERY, 5).split(":")[1],
            "synthetic_generator": 1,
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if rows[0]["response_cache_hit"] or not rows[1]["response_cache_hit"]:
        raise RuntimeError("Expected response cache MISS then HIT")
    print(*rows, sep="\n")


if __name__ == "__main__":
    main()
