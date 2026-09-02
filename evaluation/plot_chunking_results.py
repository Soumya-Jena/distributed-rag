import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


DIRECTORY = Path("experiments/day-05")


def rows(path):
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    retrieval = rows(DIRECTORY / "retrieval-comparison.csv")
    rag = rows(DIRECTORY / "rag-comparison.csv")
    overlap = []
    for amount in (0, 20, 40, 80):
        label = f"overlap-200-{amount}"
        result = rows(DIRECTORY / f"{label}-retrieval.csv")
        corpus = rows(DIRECTORY / f"{label}-corpus.csv")[0]
        overlap.append(
            {
                "label": label,
                "overlap": amount,
                "chunks": int(corpus["chunks"]),
                "avg_top_1_similarity": mean(
                    float(row["top_1_similarity"]) for row in result
                ),
                "mrr": mean(float(row["reciprocal_rank"]) for row in result),
            }
        )

    with (DIRECTORY / "overlap-comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=overlap[0].keys())
        writer.writeheader()
        writer.writerows(overlap)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    sizes = [int(row["chunk_size"]) for row in retrieval]
    axes[0].plot(
        sizes,
        [float(row["avg_top_1_similarity"]) for row in retrieval],
        marker="o",
        label="Average top-1 similarity",
    )
    axes[0].plot(sizes, [float(row["mrr"]) for row in retrieval], marker="s", label="MRR")
    axes[0].set(title="Retrieval by chunk size", xlabel="Chunk size (tokens)", ylim=(0, 1.05))
    axes[0].legend(fontsize=8)

    labels = [row["label"].replace("chunk-", "") for row in rag]
    axes[1].bar(labels, [float(row["answer_correctness"]) for row in rag], label="Correctness")
    axes[1].plot(labels, [float(row["groundedness"]) for row in rag], color="black", marker="o", label="Groundedness")
    axes[1].set(title="Manual RAG quality", xlabel="Size / overlap", ylim=(0, 1.05))
    axes[1].legend(fontsize=8)

    overlaps = [row["overlap"] for row in overlap]
    axes[2].plot(overlaps, [row["avg_top_1_similarity"] for row in overlap], marker="o")
    axes[2].set(title="Overlap-only comparison", xlabel="Overlap (200-token chunks)", ylim=(0.6, 0.75))

    figure.tight_layout()
    output = DIRECTORY / "chunking-comparison.png"
    figure.savefig(output, dpi=180)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
