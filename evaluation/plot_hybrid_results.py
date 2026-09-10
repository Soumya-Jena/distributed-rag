import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("experiments/day-08")


def read(name):
    with (OUTPUT / name).open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    architectures = read("hybrid-architecture-comparison.csv")
    query_types = read("hybrid-query-type-comparison.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    labels = [row["architecture"].replace("-plus-", "+\n") for row in architectures]
    x = np.arange(len(labels))

    axes[0].bar(x - 0.18, [float(row["hit_at_1"]) for row in architectures], 0.36, label="Hit@1")
    axes[0].bar(x + 0.18, [float(row["mrr"]) for row in architectures], 0.36, label="MRR")
    axes[0].set_ylim(0.75, 1.02)
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_title("Retrieval quality")
    axes[0].legend()

    axes[1].bar(x, [1000 * float(row["p95_seconds"]) for row in architectures], color="#d97706")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("P95 milliseconds")
    axes[1].set_title("Retrieval latency")

    type_labels = [row["query_type"] for row in query_types]
    type_x = np.arange(len(type_labels))
    axes[2].bar(type_x - 0.25, [float(row["vector_hit_at_1"]) for row in query_types], 0.25, label="Vector Hit@1")
    axes[2].bar(type_x, [float(row["lexical_hit_at_20"]) for row in query_types], 0.25, label="Lexical Hit@20")
    axes[2].bar(type_x + 0.25, [float(row["rrf_hit_at_1"]) for row in query_types], 0.25, label="RRF Hit@1")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xticks(type_x, type_labels)
    axes[2].set_title("Quality by query type")
    axes[2].legend(fontsize=8)

    fig.suptitle("Hybrid lexical + semantic retrieval")
    fig.tight_layout()
    fig.savefig(OUTPUT / "hybrid-retrieval-comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
