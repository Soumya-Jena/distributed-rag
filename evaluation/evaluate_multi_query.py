"""Four-strategy retrieval benchmark with query-type breakdown."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.metrics import hit_at_k, reciprocal_rank
from src.multi_query_retriever import MultiQueryRetriever
from src.query_transformer import QueryVariants, missing_identifiers


DATASET = Path("datasets/evaluation/query_transform_questions.jsonl")
OUTPUT = Path("experiments/day-11")
STRATEGIES = ("original", "rewrite_only", "original_rewrite", "multi_query")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, strategy, query_type="ALL"):
    selected = [
        row for row in rows
        if row["strategy"] == strategy
        and (query_type == "ALL" or row["query_type"] == query_type)
    ]
    original = {row["id"]: row for row in rows if row["strategy"] == "original"}
    misses_at_1 = [row for row in selected if original[row["id"]]["hit@1"] == 0]
    original_hits_at_1 = [
        row for row in selected if original[row["id"]]["hit@1"] == 1
    ]
    misses_at_5 = [row for row in selected if original[row["id"]]["hit@5"] == 0]
    original_hits_at_5 = [
        row for row in selected if original[row["id"]]["hit@5"] == 1
    ]
    return {
        "strategy": strategy, "query_type": query_type, "questions": len(selected),
        "hit@1": mean(row["hit@1"] for row in selected),
        "hit@5": mean(row["hit@5"] for row in selected),
        "mrr": mean(row["reciprocal_rank"] for row in selected),
        "candidate_hit@20": mean(row["candidate_hit@20"] for row in selected),
        "rescue_rate@1": (
            mean(row["hit@1"] for row in misses_at_1) if misses_at_1 else ""
        ),
        "regression_rate@1": (
            mean(row["hit@1"] == 0 for row in original_hits_at_1)
            if original_hits_at_1 else ""
        ),
        "rescue_rate@5": (
            mean(row["hit@5"] for row in misses_at_5) if misses_at_5 else ""
        ),
        "regression_rate@5": (
            mean(row["hit@5"] == 0 for row in original_hits_at_5)
            if original_hits_at_5 else ""
        ),
        "mean_retrieval_seconds": mean(row["retrieval_seconds"] for row in selected),
        "mean_query_count": mean(row["query_count"] for row in selected),
    }


def main():
    with DATASET.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    types = {"DIRECT", "PARAPHRASE", "VAGUE", "EXACT_IDENTIFIER", "MULTI_PART"}
    counts = defaultdict(int)
    for record in records:
        counts[record["query_type"]] += 1
    if len(records) != 50 or set(counts) != types or set(counts.values()) != {10}:
        raise RuntimeError("Expected exactly ten questions in each of five query types")

    retriever = MultiQueryRetriever(use_reranker=False)
    rows = []
    for record in records:
        variants = QueryVariants(
            record["question"], record["semantic"],
            record["technical"], record["alternate"],
        )
        for strategy in STRATEGIES:
            result = retriever.retrieve(
                record["question"], variants, strategy=strategy, final_k=5
            )
            relevant = record["relevant_sources"]
            row = {
                "id": record["id"], "query_type": record["query_type"],
                "strategy": strategy, "question": record["question"],
                "query_count": len(result.query_variants),
                "hit@1": hit_at_k(result.final_chunks, relevant, 1),
                "hit@5": hit_at_k(result.final_chunks, relevant, 5),
                "reciprocal_rank": reciprocal_rank(result.final_chunks, relevant),
                "candidate_hit@20": hit_at_k(result.fused_candidates, relevant, 20),
                "top_source": Path(result.final_chunks[0].source_path).name if result.final_chunks else "",
                "relevant_rank": next((i for i, c in enumerate(result.final_chunks, 1) if Path(c.source_path).name in relevant), ""),
                "retrieval_seconds": result.retrieval_seconds,
                "found_by": "|".join(result.final_chunks[0].found_by) if result.final_chunks else "",
                "missing_identifiers": "|".join(sorted(set().union(
                    missing_identifiers(record["question"], record["semantic"]),
                    missing_identifiers(record["question"], record["technical"]),
                    missing_identifiers(record["question"], record["alternate"]),
                ))),
            }
            rows.append(row)
            print(record["id"], strategy, row["hit@5"], flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "strategy-results.csv", rows)
    summaries = [summarize(rows, strategy) for strategy in STRATEGIES]
    breakdown = [
        summarize(rows, strategy, query_type)
        for strategy in STRATEGIES for query_type in sorted(types)
    ]
    write_csv(OUTPUT / "strategy-comparison.csv", summaries)
    write_csv(OUTPUT / "query-type-comparison.csv", breakdown)
    print("Saved multi-query benchmark results")


if __name__ == "__main__":
    main()
