"""Plot retrieval quality and latency for the four query strategies."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


INPUT = Path("experiments/day-11/strategy-comparison.csv")
OUTPUT = Path("experiments/day-11/strategy-comparison.png")


def main():
    with INPUT.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    labels = [
        "Original", "Rewrite", "Original + rewrite", "Original + 3 variants"
    ]
    hit_at_1 = [float(row["hit@1"]) for row in rows]
    latency = [float(row["mean_retrieval_seconds"]) for row in rows]

    figure, quality_axis = plt.subplots(figsize=(10, 5.5))
    bars = quality_axis.bar(labels, hit_at_1, color="#3274A1", width=0.6)
    quality_axis.set_ylim(0.75, 1.03)
    quality_axis.set_ylabel("Hit@1")
    quality_axis.set_title("Query strategy: rank-one quality and retrieval latency")
    quality_axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, hit_at_1):
        quality_axis.text(
            bar.get_x() + bar.get_width() / 2, value + 0.006,
            f"{value:.2f}", ha="center", fontsize=9,
        )

    latency_axis = quality_axis.twinx()
    latency_axis.plot(labels, latency, color="#E1812C", marker="o", linewidth=2)
    latency_axis.set_ylabel("Mean retrieval seconds")
    latency_axis.set_ylim(0, max(latency) * 1.25)
    figure.tight_layout()
    figure.savefig(OUTPUT, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
