import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT = Path("experiments/day-07")


def read(name):
    with (OUTPUT / name).open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    rag = read("rag-comparison.csv")
    candidates = read("candidate-k-comparison.csv")

    quality, quality_axis = plt.subplots(figsize=(6, 4.5))
    labels = ["Vector only", "Vector + reranker"]
    x = range(2)
    quality_axis.bar([i - 0.18 for i in x], [float(row["correctness"]) for row in rag], 0.36, label="Correctness")
    quality_axis.bar([i + 0.18 for i in x], [float(row["unsupported_refusal"]) for row in rag], 0.36, label="Unsupported refusal")
    quality_axis.set_xticks(list(x), labels)
    quality_axis.set_ylim(0, 1.08)
    quality_axis.set_title("RAG quality: vector vs reranker")
    quality_axis.legend()
    quality.tight_layout()
    quality.savefig(OUTPUT / "reranking-quality.png", dpi=180)

    latency, latency_axis = plt.subplots(figsize=(6, 4.5))
    candidate_k = [int(row["candidate_k"]) for row in candidates]
    latency_axis.plot(candidate_k, [float(row["rerank_p95_seconds"]) * 1000 for row in candidates], marker="o", label="Rerank P95")
    latency_axis.plot(candidate_k, [float(row["total_p95_seconds"]) * 1000 for row in candidates], marker="s", label="Retrieval + rerank P95")
    latency_axis.set(title="Candidate size vs latency", xlabel="Candidate K", ylabel="Latency (ms)")
    latency_axis.legend()
    latency.tight_layout()
    latency.savefig(OUTPUT / "candidate-k-latency.png", dpi=180)
    print("Saved reranking-quality.png and candidate-k-latency.png")


if __name__ == "__main__":
    main()
