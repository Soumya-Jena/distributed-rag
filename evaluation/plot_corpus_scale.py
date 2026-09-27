import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path("experiments/day-16")


def read_csv(name):
    with (OUTPUT_DIR / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    latency = read_csv("latency-summary.csv")
    storage = read_csv("storage.csv")
    ingestion = read_csv("ingestion-throughput.csv")
    x = [int(row["chunks"]) for row in latency]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(x, [float(row["vector_p95_ms"]) for row in latency], marker="o", label="Exact vector P95")
    axes[0, 0].plot(x, [float(row["lexical_p95_ms"]) for row in latency], marker="o", label="GIN lexical P95")
    axes[0, 0].plot(x, [float(row["hybrid_p95_ms"]) for row in latency], marker="o", label="Hybrid P95")
    axes[0, 0].set_title("Retrieval latency")
    axes[0, 0].set_ylabel("milliseconds")
    axes[0, 0].legend()

    axes[0, 1].plot(
        [int(row["chunks"]) for row in storage],
        [int(row["total_relation_bytes"]) / 1024**2 for row in storage],
        marker="o",
    )
    axes[0, 1].set_title("Table + indexes")
    axes[0, 1].set_ylabel("MiB")

    completed = [row for row in ingestion if int(row["inserted_chunks"]) > 0]
    axes[1, 0].plot(
        [int(row["target_chunks"]) for row in completed],
        [float(row["rows_per_second"]) for row in completed],
        marker="o",
    )
    axes[1, 0].set_title("Database write throughput")
    axes[1, 0].set_ylabel("rows / second")

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.90,
        "Semantic quality vs corpus size: not plotted\n"
        "Track B uses synthetic vectors.\n"
        "The frozen real-corpus baseline is reported separately;\n"
        "no semantic quality claim is made from synthetic data.",
        va="top",
        fontsize=11,
    )
    for axis in axes.flat[:3]:
        axis.set_xlabel("chunks")
        axis.grid(alpha=0.25)
        axis.ticklabel_format(style="plain", axis="x")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "corpus-scale.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

