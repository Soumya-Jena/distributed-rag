import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("experiments/day-09")


def read(name):
    with (OUTPUT / name).open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    comparison = read("grounding-comparison.csv")
    traps = read("trap-type-comparison.csv")
    metrics = [
        "faithfulness",
        "correct_refusal_rate",
        "partial_handling_accuracy",
        "citation_coverage",
        "citation_support",
        "answer_correctness",
    ]
    labels = [
        "Faithfulness",
        "Correct refusal",
        "Partial handling",
        "Citation coverage",
        "Citation support",
        "Correctness",
    ]
    x = np.arange(len(metrics))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for offset, row, color in (
        (-width / 2, comparison[0], "#64748b"),
        (width / 2, comparison[1], "#2563eb"),
    ):
        axes[0].bar(
            x + offset,
            [float(row[metric]) for metric in metrics],
            width,
            label=row["prompt_mode"],
            color=color,
        )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_title("Grounding metrics")
    axes[0].legend()

    trap_x = np.arange(len(traps))
    axes[1].bar(
        trap_x - width / 2,
        [float(row["baseline_hallucination_rate"]) for row in traps],
        width,
        label="baseline",
        color="#dc2626",
    )
    axes[1].bar(
        trap_x + width / 2,
        [float(row["strict_hallucination_rate"]) for row in traps],
        width,
        label="strict",
        color="#16a34a",
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(
        trap_x,
        [row["trap_type"].replace("_", " ") for row in traps],
        rotation=25,
        ha="right",
    )
    axes[1].set_title("Unsupported-claim rate by trap type")
    axes[1].legend()
    fig.suptitle("Baseline versus strict grounding prompt")
    fig.tight_layout()
    fig.savefig(OUTPUT / "grounding-comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
