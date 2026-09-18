"""Plot cold/warm serving-path cache latency."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT = Path("experiments/day-13")


def main():
    with (OUTPUT / "cache-comparison.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    metrics = ("latency_p50_seconds", "latency_p95_seconds", "latency_p99_seconds")
    labels = ("P50", "P95", "P99")
    x = range(len(metrics))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5))
    cold = [float(rows[0][field]) * 1000 for field in metrics]
    warm = [float(rows[1][field]) * 1000 for field in metrics]
    axis.bar([value - width / 2 for value in x], cold, width, label="Cold")
    axis.bar([value + width / 2 for value in x], warm, width, label="Warm")
    axis.set_xticks(list(x), labels)
    axis.set_ylabel("Latency (ms)")
    axis.set_title("Cold versus warm retrieval-serving latency")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "cache-latency-comparison.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
