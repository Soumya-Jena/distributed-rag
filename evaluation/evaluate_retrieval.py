import argparse
import csv
import json
import re

from pathlib import Path
from statistics import mean
from time import perf_counter

from src.metrics import hit_at_k, recall_at_k, reciprocal_rank
from src.retriever import Retriever


DATASET = Path("datasets/evaluation/questions.jsonl")


def load_dataset():
    records = []
    with DATASET.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def valid_label(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError(
            "label may contain only letters, numbers, dots, underscores, and hyphens"
        )
    return value


def percentile(values, percentile_value):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, type=valid_label)
    args = parser.parse_args()

    output = Path("experiments/day-05") / f"{args.label}-retrieval.csv"
    dataset = load_dataset()
    retriever = Retriever()
    results = []

    for record in dataset:
        if not record["answerable"]:
            continue

        started = perf_counter()
        chunks = retriever.search(record["question"], top_k=5)
        retrieval_seconds = perf_counter() - started

        result = {
            "id": record["id"],
            "question": record["question"],
            "hit@1": hit_at_k(chunks, record["relevant_sources"], 1),
            "hit@3": hit_at_k(chunks, record["relevant_sources"], 3),
            "hit@5": hit_at_k(chunks, record["relevant_sources"], 5),
            "recall@5": recall_at_k(chunks, record["relevant_sources"], 5),
            "reciprocal_rank": reciprocal_rank(
                chunks, record["relevant_sources"]
            ),
            "top_1_source": Path(chunks[0].source_path).name if chunks else "",
            "top_1_similarity": chunks[0].similarity if chunks else None,
            "retrieval_seconds": retrieval_seconds,
        }
        results.append(result)
        print(record["id"], record["question"], result["hit@5"])

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    latencies = [row["retrieval_seconds"] for row in results]
    print("\nBASELINE RETRIEVAL")
    print("==================")
    print(f"Questions     : {len(results)}")
    print(f"Hit@1         : {mean(r['hit@1'] for r in results):.3f}")
    print(f"Hit@3         : {mean(r['hit@3'] for r in results):.3f}")
    print(f"Hit@5         : {mean(r['hit@5'] for r in results):.3f}")
    print(f"Recall@5      : {mean(r['recall@5'] for r in results):.3f}")
    print(f"MRR           : {mean(r['reciprocal_rank'] for r in results):.3f}")
    print(f"Retrieval avg : {mean(latencies):.6f}s")
    print(f"Retrieval P95 : {percentile(latencies, 0.95):.6f}s")
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
