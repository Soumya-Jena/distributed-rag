import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT = Path("experiments/day-06")


def read(path):
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    retrieval = read(OUTPUT / "retrieval-comparison.csv")
    rag = read(OUTPUT / "rag-comparison.csv")
    names = ["MiniLM", "BGE-small", "E5-small"]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(names, [float(row["mrr"]) for row in retrieval], color=["#2b7bba", "#f28e2b", "#59a14f"])
    axes[0].set(title="Embedding model vs MRR", ylabel="MRR", ylim=(0, 1.08))

    rag_names = ["MiniLM", "E5-small"]
    x = range(len(rag_names))
    axes[1].bar([value - 0.18 for value in x], [float(row["answer_correctness"]) for row in rag], width=0.36, label="Correctness")
    axes[1].bar([value + 0.18 for value in x], [float(row["unsupported_refusal"]) for row in rag], width=0.36, label="Unsupported refusal")
    axes[1].set_xticks(list(x), rag_names)
    axes[1].set(title="End-to-end RAG quality", ylim=(0, 1.08))
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(OUTPUT / "embedding-comparison.png", dpi=180)
    print(f"Saved {OUTPUT / 'embedding-comparison.png'}")


if __name__ == "__main__":
    main()
