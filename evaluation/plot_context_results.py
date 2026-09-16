"""Plot the context strategy token/retention trade-off."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT = Path("experiments/day-12")


def read(path):
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    rows = read(OUTPUT / "context-strategy-comparison.csv")
    labels = [row["strategy"] for row in rows]
    tokens = [float(row["mean_context_tokens"]) for row in rows]
    retained = [float(row["relevant_source_retention"]) * 100 for row in rows]

    figure, left = plt.subplots(figsize=(9, 5))
    positions = range(len(rows))
    bars = left.bar(positions, tokens, color="#4C78A8")
    left.set_ylabel("Mean context tokens")
    left.set_xticks(list(positions), labels)
    left.bar_label(bars, fmt="%.0f")
    right = left.twinx()
    right.plot(positions, retained, color="#E45756", marker="o", linewidth=2)
    right.set_ylabel("Relevant-source retention (%)")
    right.set_ylim(0, 105)
    left.set_title("Context reduction versus evidence retention")
    figure.tight_layout()
    figure.savefig(OUTPUT / "context-strategy-comparison.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
