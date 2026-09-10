import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from src.fusion import reciprocal_rank_fusion
from src.lexical_retriever import LexicalRetriever
from src.metrics import hit_at_k, reciprocal_rank
from src.retriever import Retriever


DATASET = Path("datasets/evaluation/hybrid_questions.jsonl")
OUTPUT = Path("experiments/day-08/rrf-k-comparison.csv")
RRF_VALUES = (10, 30, 60, 100)


def main():
    with DATASET.open(encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip() and json.loads(line)["answerable"]
        ]

    vector_retriever = Retriever()
    lexical_retriever = LexicalRetriever()
    rankings = []
    for record in records:
        vector = vector_retriever.search(record["question"], top_k=20)
        lexical = lexical_retriever.search(record["question"], top_k=20)
        rankings.append((record, vector, lexical))

    rows = []
    for rrf_k in RRF_VALUES:
        hits_at_1 = []
        hits_at_5 = []
        reciprocal_ranks = []
        fusion_times = []
        for record, vector, lexical in rankings:
            started = perf_counter()
            fused = reciprocal_rank_fusion(
                vector, lexical, rrf_k=rrf_k, top_k=20
            )
            fusion_times.append(perf_counter() - started)
            relevant = record["relevant_sources"]
            hits_at_1.append(hit_at_k(fused, relevant, 1))
            hits_at_5.append(hit_at_k(fused, relevant, 5))
            reciprocal_ranks.append(reciprocal_rank(fused, relevant))
        rows.append(
            {
                "rrf_k": rrf_k,
                "questions": len(records),
                "hit_at_1": mean(hits_at_1),
                "hit_at_5": mean(hits_at_5),
                "mrr": mean(reciprocal_ranks),
                "mean_fusion_seconds": mean(fusion_times),
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
