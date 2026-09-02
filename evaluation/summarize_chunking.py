import csv

from pathlib import Path
from statistics import mean


CONFIGURATIONS = [
    (100, 20),
    (150, 30),
    (200, 40),
    (240, 40),
]


def read_rows(path):
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def percentile(values, percentile_value):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def main():
    directory = Path("experiments/day-05")
    output = directory / "retrieval-comparison.csv"
    summaries = []

    for chunk_size, overlap in CONFIGURATIONS:
        label = f"chunk-{chunk_size}-{overlap}"
        corpus = read_rows(directory / f"{label}-corpus.csv")[0]
        retrieval = read_rows(directory / f"{label}-retrieval.csv")
        latencies = [float(row["retrieval_seconds"]) for row in retrieval]

        summaries.append(
            {
                "label": label,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "chunks": int(corpus["chunks"]),
                "avg_tokens": float(corpus["avg_tokens"]),
                "relation_bytes": int(corpus["relation_bytes"]),
                "ingestion_seconds": float(corpus["ingestion_seconds"]),
                "hit_at_1": mean(float(row["hit@1"]) for row in retrieval),
                "hit_at_3": mean(float(row["hit@3"]) for row in retrieval),
                "hit_at_5": mean(float(row["hit@5"]) for row in retrieval),
                "recall_at_5": mean(float(row["recall@5"]) for row in retrieval),
                "mrr": mean(float(row["reciprocal_rank"]) for row in retrieval),
                "avg_top_1_similarity": mean(
                    float(row["top_1_similarity"]) for row in retrieval
                ),
                "retrieval_avg_seconds": mean(latencies),
                "retrieval_p95_seconds": percentile(latencies, 0.95),
            }
        )

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    for summary in summaries:
        print(summary)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
