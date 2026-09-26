import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    source = Path("experiments/day-15/saturation-summary.csv")
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for mode in ("unique", "repeat"):
        selected = sorted(
            (row for row in rows if row["mode"] == mode),
            key=lambda row: int(row["concurrency"]),
        )
        x = [int(row["concurrency"]) for row in selected]
        axes[0].plot(x, [float(row["goodput_rps"]) for row in selected], marker="o", label=mode)
        axes[1].plot(x, [float(row["p95_ms"]) for row in selected], marker="o", label=f"{mode} P95")
        axes[1].plot(x, [float(row["p99_ms"]) for row in selected], marker="x", linestyle="--", label=f"{mode} P99")
    axes[0].set(title="Goodput vs concurrency", xlabel="Concurrent users", ylabel="Successful responses/s")
    axes[1].set(title="Tail latency vs concurrency", xlabel="Concurrent users", ylabel="Milliseconds")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend()
    figure.suptitle("Load harness validation (not Qwen capacity)")
    figure.tight_layout()
    output = source.parent / "load-curves.png"
    figure.savefig(output, dpi=160)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
