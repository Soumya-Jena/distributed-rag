"""Plot conditional attack success and manually reviewed clean utility."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


INPUT = Path("experiments/day-10/security-comparison.csv")
OUTPUT = Path("experiments/day-10/security-comparison.png")


def main():
    with INPUT.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    labels = [row["mode"].title() for row in rows]
    attack_success = [float(row["delivered_conditional_asr"]) for row in rows]
    groundedness = [float(row["clean_manual_groundedness"]) for row in rows]
    positions = range(len(rows))
    width = 0.36

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.bar(
        [position - width / 2 for position in positions], attack_success,
        width, label="Delivered conditional ASR", color="#C44E52",
    )
    axis.bar(
        [position + width / 2 for position in positions], groundedness,
        width, label="Clean groundedness", color="#55A868",
    )
    axis.set_xticks(list(positions), labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Rate")
    axis.set_title("Security controls: attack success and clean-answer utility")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(OUTPUT, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
