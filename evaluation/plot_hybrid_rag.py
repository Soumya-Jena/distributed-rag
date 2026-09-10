import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("experiments/day-08")


def main():
    with (OUTPUT / "rag-comparison.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    labels = [row["architecture"].replace("-plus-", "+\n") for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    axes[0].bar(x - 0.2, [float(row["correctness"]) for row in rows], 0.2, label="Correctness")
    axes[0].bar(x, [float(row["groundedness"]) for row in rows], 0.2, label="Groundedness")
    axes[0].bar(x + 0.2, [float(row["citation_support"]) for row in rows], 0.2, label="Citation support")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_title("RAG quality")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, [1000 * float(row["retrieval_p95_seconds"]) for row in rows], color="#7c3aed")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("P95 milliseconds")
    axes[1].set_title("Retrieval latency inside RAG")

    fig.suptitle("Downstream RAG architecture comparison")
    fig.tight_layout()
    fig.savefig(OUTPUT / "hybrid-rag-comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
