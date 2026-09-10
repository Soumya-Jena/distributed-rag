import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter

from src.fusion import reciprocal_rank_fusion
from src.lexical_retriever import LexicalRetriever
from src.metrics import hit_at_k, reciprocal_rank, source_name
from src.reranker import Reranker
from src.retriever import Retriever


DATASET = Path("datasets/evaluation/hybrid_questions.jsonl")
OUTPUT_DIR = Path("experiments/day-08")


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def source_list(chunks, limit=None):
    values = chunks if limit is None else chunks[:limit]
    return "|".join(source_name(chunk) for chunk in values)


def evaluate_architecture(chunks, relevant_sources):
    return {
        "hit_at_1": hit_at_k(chunks, relevant_sources, 1),
        "hit_at_5": hit_at_k(chunks, relevant_sources, 5),
        "mrr": reciprocal_rank(chunks, relevant_sources),
    }


def load_records():
    with DATASET.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def architecture_rows(answerable_rows):
    architectures = [
        ("vector", "vector"),
        ("vector-plus-reranker", "vector_reranked"),
        ("hybrid-rrf", "rrf"),
        ("hybrid-rrf-plus-reranker", "hybrid_reranked"),
    ]
    rows = []
    for name, prefix in architectures:
        rows.append(
            {
                "architecture": name,
                "questions": len(answerable_rows),
                "hit_at_1": mean(row[f"{prefix}_hit_at_1"] for row in answerable_rows),
                "hit_at_5": mean(row[f"{prefix}_hit_at_5"] for row in answerable_rows),
                "mrr": mean(row[f"{prefix}_mrr"] for row in answerable_rows),
                "mean_seconds": mean(row[f"{prefix}_seconds"] for row in answerable_rows),
                "p95_seconds": percentile(
                    [row[f"{prefix}_seconds"] for row in answerable_rows], 0.95
                ),
            }
        )
    return rows


def query_type_rows(answerable_rows):
    groups = defaultdict(list)
    for row in answerable_rows:
        groups[row["query_type"]].append(row)

    output = []
    for query_type, rows in sorted(groups.items()):
        output.append(
            {
                "query_type": query_type,
                "questions": len(rows),
                "vector_hit_at_1": mean(row["vector_hit_at_1"] for row in rows),
                "lexical_hit_at_20": mean(row["lexical_hit_at_20"] for row in rows),
                "union_candidate_hit": mean(row["union_candidate_hit"] for row in rows),
                "rrf_hit_at_1": mean(row["rrf_hit_at_1"] for row in rows),
                "rrf_hit_at_5": mean(row["rrf_hit_at_5"] for row in rows),
                "rrf_mrr": mean(row["rrf_mrr"] for row in rows),
                "hybrid_reranked_hit_at_1": mean(
                    row["hybrid_reranked_hit_at_1"] for row in rows
                ),
                "hybrid_reranked_mrr": mean(
                    row["hybrid_reranked_mrr"] for row in rows
                ),
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="hybrid")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--final-k", type=int, default=5)
    args = parser.parse_args()

    records = load_records()
    vector_retriever = Retriever()
    lexical_retriever = LexicalRetriever()
    reranker = Reranker()
    rows = []

    for record in records:
        vector_started = perf_counter()
        vector = vector_retriever.search(record["question"], top_k=args.candidate_k)
        vector_seconds = perf_counter() - vector_started

        lexical_started = perf_counter()
        lexical = lexical_retriever.search(record["question"], top_k=args.candidate_k)
        lexical_seconds = perf_counter() - lexical_started

        fusion_started = perf_counter()
        fused = reciprocal_rank_fusion(
            vector, lexical, rrf_k=args.rrf_k, top_k=args.candidate_k
        )
        fusion_seconds = perf_counter() - fusion_started

        rerank_started = perf_counter()
        vector_reranked = [
            item.chunk
            for item in reranker.rerank(
                record["question"], vector, top_k=args.final_k
            )
        ]
        vector_rerank_seconds = perf_counter() - rerank_started

        rerank_started = perf_counter()
        hybrid_reranked = [
            item.chunk
            for item in reranker.rerank(
                record["question"], fused, top_k=args.final_k
            )
        ]
        hybrid_rerank_seconds = perf_counter() - rerank_started

        relevant = record["relevant_sources"]
        answerable = record["answerable"]
        vector_metrics = evaluate_architecture(vector[: args.final_k], relevant)
        lexical_metrics = evaluate_architecture(lexical, relevant)
        rrf_metrics = evaluate_architecture(fused[: args.final_k], relevant)
        vector_reranked_metrics = evaluate_architecture(vector_reranked, relevant)
        hybrid_reranked_metrics = evaluate_architecture(hybrid_reranked, relevant)
        vector_hit_20 = hit_at_k(vector, relevant, args.candidate_k) if answerable else 0
        lexical_hit_20 = hit_at_k(lexical, relevant, args.candidate_k) if answerable else 0

        row = {
            "id": record["id"],
            "query_type": record["query_type"],
            "question": record["question"],
            "answerable": answerable,
            "vector_hit_at_20": vector_hit_20,
            "lexical_hit_at_20": lexical_hit_20,
            "union_candidate_hit": int(bool(vector_hit_20 or lexical_hit_20)),
            "lexical_rescue": int(not vector_hit_20 and lexical_hit_20),
            "semantic_rescue": int(vector_hit_20 and not lexical_hit_20),
            "vector_hit_at_1": vector_metrics["hit_at_1"],
            "vector_hit_at_5": vector_metrics["hit_at_5"],
            "vector_mrr": vector_metrics["mrr"],
            "lexical_hit_at_1": lexical_metrics["hit_at_1"],
            "lexical_hit_at_5": lexical_metrics["hit_at_5"],
            "lexical_mrr": lexical_metrics["mrr"],
            "rrf_hit_at_1": rrf_metrics["hit_at_1"],
            "rrf_hit_at_5": rrf_metrics["hit_at_5"],
            "rrf_mrr": rrf_metrics["mrr"],
            "vector_reranked_hit_at_1": vector_reranked_metrics["hit_at_1"],
            "vector_reranked_hit_at_5": vector_reranked_metrics["hit_at_5"],
            "vector_reranked_mrr": vector_reranked_metrics["mrr"],
            "hybrid_reranked_hit_at_1": hybrid_reranked_metrics["hit_at_1"],
            "hybrid_reranked_hit_at_5": hybrid_reranked_metrics["hit_at_5"],
            "hybrid_reranked_mrr": hybrid_reranked_metrics["mrr"],
            "vector_sources": source_list(vector, args.final_k),
            "lexical_sources": source_list(lexical, args.final_k),
            "rrf_sources": source_list(fused, args.final_k),
            "hybrid_reranked_sources": source_list(hybrid_reranked),
            "vector_seconds": vector_seconds,
            "lexical_seconds": lexical_seconds,
            "fusion_seconds": fusion_seconds,
            "vector_rerank_seconds": vector_rerank_seconds,
            "hybrid_rerank_seconds": hybrid_rerank_seconds,
            "vector_reranked_seconds": vector_seconds + vector_rerank_seconds,
            "rrf_seconds": vector_seconds + lexical_seconds + fusion_seconds,
            "hybrid_reranked_seconds": (
                vector_seconds + lexical_seconds + fusion_seconds + hybrid_rerank_seconds
            ),
        }
        rows.append(row)
        print(
            record["id"],
            record["query_type"],
            f"vector={vector_metrics['hit_at_1']}",
            f"lexical20={lexical_hit_20}",
            f"rrf={rrf_metrics['hit_at_1']}",
            f"full={hybrid_reranked_metrics['hit_at_1']}",
        )

    answerable_rows = [row for row in rows if row["answerable"]]
    prefix = OUTPUT_DIR / args.label
    write_csv(prefix.with_name(f"{args.label}-results.csv"), rows)
    write_csv(
        prefix.with_name(f"{args.label}-architecture-comparison.csv"),
        architecture_rows(answerable_rows),
    )
    write_csv(
        prefix.with_name(f"{args.label}-query-type-comparison.csv"),
        query_type_rows(answerable_rows),
    )

    lexical_rescues = sum(row["lexical_rescue"] for row in answerable_rows)
    semantic_rescues = sum(row["semantic_rescue"] for row in answerable_rows)
    rescue_rows = [
        {
            "answerable_questions": len(answerable_rows),
            "vector_hit_at_20": mean(row["vector_hit_at_20"] for row in answerable_rows),
            "lexical_hit_at_20": mean(row["lexical_hit_at_20"] for row in answerable_rows),
            "union_candidate_hit": mean(row["union_candidate_hit"] for row in answerable_rows),
            "lexical_rescues": lexical_rescues,
            "lexical_rescue_rate": lexical_rescues / len(answerable_rows),
            "semantic_rescues": semantic_rescues,
            "semantic_rescue_rate": semantic_rescues / len(answerable_rows),
        }
    ]
    write_csv(prefix.with_name(f"{args.label}-rescue-summary.csv"), rescue_rows)

    print(f"Saved Day 8 results under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
