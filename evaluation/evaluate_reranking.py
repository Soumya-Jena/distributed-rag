import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from evaluation.offline_retriever import OfflineRetriever
from src.metrics import hit_at_k, reciprocal_rank
from src.reranker import Reranker


OUTPUT = Path("experiments/day-07")
CANDIDATE_SIZES = (5, 10, 20, 40)


def load_questions():
    with Path("datasets/evaluation/questions.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip() and json.loads(line)["answerable"]]


def percentile(values, value):
    return float(np.percentile(values, value))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    retriever = OfflineRetriever()
    reranker = Reranker()
    questions = load_questions()
    rows = []
    score_rows = []

    for record in questions:
        question = record["question"]
        relevant = record["relevant_sources"]
        started = perf_counter()
        baseline = retriever.search(question, top_k=5)
        baseline_seconds = perf_counter() - started

        for candidate_k in CANDIDATE_SIZES:
            started = perf_counter()
            candidates = retriever.search(question, top_k=candidate_k)
            vector_seconds = perf_counter() - started
            started = perf_counter()
            all_reranked_items = reranker.rerank(
                question, candidates, top_k=len(candidates)
            )
            rerank_seconds = perf_counter() - started
            reranked_items = all_reranked_items[:5]
            reranked = [item.chunk for item in reranked_items]
            vector_rank = next(
                (rank for rank, chunk in enumerate(candidates, 1) if Path(chunk.source_path).name in relevant),
                0,
            )
            reranked_rank = next(
                (rank for rank, chunk in enumerate(reranked, 1) if Path(chunk.source_path).name in relevant),
                0,
            )
            rows.append(
                {
                    "id": record["id"],
                    "question": question,
                    "candidate_k": candidate_k,
                    "baseline_hit@1": hit_at_k(baseline, relevant, 1),
                    "baseline_hit@3": hit_at_k(baseline, relevant, 3),
                    "baseline_hit@5": hit_at_k(baseline, relevant, 5),
                    "baseline_mrr": reciprocal_rank(baseline, relevant),
                    "candidate_hit": hit_at_k(candidates, relevant, candidate_k),
                    "reranked_hit@1": hit_at_k(reranked, relevant, 1),
                    "reranked_hit@3": hit_at_k(reranked, relevant, 3),
                    "reranked_hit@5": hit_at_k(reranked, relevant, 5),
                    "reranked_mrr": reciprocal_rank(reranked, relevant),
                    "vector_rank": vector_rank,
                    "reranked_rank": reranked_rank,
                    "movement": (vector_rank - reranked_rank) if reranked_rank else -5,
                    "winner_original_rank": reranked_items[0].original_rank,
                    "winner_movement": reranked_items[0].original_rank - 1,
                    "baseline_vector_seconds": baseline_seconds,
                    "vector_seconds": vector_seconds,
                    "rerank_seconds": rerank_seconds,
                    "total_seconds": vector_seconds + rerank_seconds,
                    "pairs": len(candidates),
                    "pairs_per_second": len(candidates) / rerank_seconds,
                    "top_rerank_score": reranked_items[0].rerank_score if reranked_items else None,
                }
            )
            if candidate_k == 20:
                for rerank_rank, item in enumerate(all_reranked_items, start=1):
                    score_rows.append(
                        {
                            "id": record["id"],
                            "source": Path(item.chunk.source_path).name,
                            "chunk_index": item.chunk.chunk_index,
                            "relevant_source": int(Path(item.chunk.source_path).name in relevant),
                            "vector_rank": item.original_rank,
                            "rerank_rank": rerank_rank,
                            "rerank_score": item.rerank_score,
                        }
                    )
            print(record["id"], candidate_k, vector_rank, reranked_rank, flush=True)

    with (OUTPUT / "reranking-results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with (OUTPUT / "reranker-score-distribution.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=score_rows[0].keys())
        writer.writeheader()
        writer.writerows(score_rows)

    summaries = []
    for candidate_k in CANDIDATE_SIZES:
        selected = [row for row in rows if row["candidate_k"] == candidate_k]
        summaries.append(
            {
                "candidate_k": candidate_k,
                "candidate_hit": mean(row["candidate_hit"] for row in selected),
                "reranked_hit_at_1": mean(row["reranked_hit@1"] for row in selected),
                "reranked_hit_at_3": mean(row["reranked_hit@3"] for row in selected),
                "reranked_hit_at_5": mean(row["reranked_hit@5"] for row in selected),
                "reranked_mrr": mean(row["reranked_mrr"] for row in selected),
                "vector_p50_seconds": percentile([row["vector_seconds"] for row in selected], 50),
                "vector_p95_seconds": percentile([row["vector_seconds"] for row in selected], 95),
                "rerank_p50_seconds": percentile([row["rerank_seconds"] for row in selected], 50),
                "rerank_p95_seconds": percentile([row["rerank_seconds"] for row in selected], 95),
                "total_p95_seconds": percentile([row["total_seconds"] for row in selected], 95),
                "avg_pairs_per_second": mean(row["pairs_per_second"] for row in selected),
            }
        )
    with (OUTPUT / "candidate-k-comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
