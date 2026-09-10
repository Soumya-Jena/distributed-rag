import csv
from pathlib import Path
from statistics import mean

import numpy as np


OUTPUT = Path("experiments/day-08")
FILES = (
    ("vector", "vector-rag.csv"),
    ("vector-plus-reranker", "vector-reranked-rag.csv"),
    ("hybrid-rrf", "hybrid-rag.csv"),
    ("hybrid-rrf-plus-reranker", "full-hybrid-rag.csv"),
)
SCORES = {
    "q001": (1, 1, "Correct and grounded; the model omitted citations."),
    "q002": (
        0,
        0,
        "Overstates MVCC as preventing conflicts/overwrites beyond the supplied evidence.",
    ),
    "q003": (1, 1, "Correct and grounded; the model omitted citations."),
    "q004": (
        1,
        1,
        "Primary answer is correct and grounded; output reaches the token limit.",
    ),
}


def number(row, name):
    value = row.get(name, "")
    return float(value) if value not in ("", None) else 0.0


def main():
    summaries = []
    for architecture, filename in FILES:
        path = OUTPUT / filename
        with path.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        for row in rows:
            if row["answerable"].lower() == "true":
                correctness, groundedness, notes = SCORES[row["id"]]
                row["manual_correctness"] = correctness
                row["manual_groundedness"] = groundedness
                row["manual_citation_support"] = 0
                row["notes"] = notes
            else:
                score = int("don't have enough information" in row["answer"].lower())
                row["manual_correctness"] = score
                row["manual_groundedness"] = score
                row["manual_citation_support"] = ""
                row["notes"] = "Unsupported-question refusal graded manually."

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        answerable = [row for row in rows if row["answerable"].lower() == "true"]
        unsupported = [row for row in rows if row["answerable"].lower() == "false"]
        retrieval = [number(row, "retrieval_seconds") for row in rows]
        generation = [number(row, "generation_seconds") for row in answerable]
        end_to_end = [
            number(row, "retrieval_seconds") + number(row, "generation_seconds")
            for row in answerable
        ]
        summaries.append(
            {
                "architecture": architecture,
                "correctness": mean(number(row, "manual_correctness") for row in answerable),
                "groundedness": mean(number(row, "manual_groundedness") for row in answerable),
                "citation_support": mean(number(row, "manual_citation_support") for row in answerable),
                "unsupported_refusal": mean(number(row, "manual_correctness") for row in unsupported),
                "vector_mean_seconds": mean(number(row, "vector_seconds") for row in rows),
                "lexical_mean_seconds": mean(number(row, "lexical_seconds") for row in rows),
                "fusion_mean_seconds": mean(number(row, "fusion_seconds") for row in rows),
                "rerank_mean_seconds": mean(number(row, "rerank_seconds") for row in rows),
                "retrieval_p95_seconds": float(np.percentile(retrieval, 95)),
                "generation_p95_seconds": float(np.percentile(generation, 95)),
                "end_to_end_p95_seconds": float(np.percentile(end_to_end, 95)),
            }
        )

    with (OUTPUT / "rag-comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
