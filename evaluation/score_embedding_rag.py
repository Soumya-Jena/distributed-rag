import csv
from pathlib import Path
from statistics import mean

import numpy as np


OUTPUT = Path("experiments/day-06")
LABELS = ["embedding-minilm", "embedding-e5-small-v2"]
SCORES = {
    "q001": (1, 1, "Correct and grounded; no citation."),
    "q002": (0, 0, "Material overstatement about conflict/overwrite prevention."),
    "q003": (1, 1, "Correct and grounded; no citation."),
    "q004": (1, 1, "Primary answer is correct and grounded; output is truncated."),
}


def main():
    summaries = []
    for label in LABELS:
        path = OUTPUT / f"{label}-rag.csv"
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
                manual_refusal = int(
                    "don't have enough information" in row["answer"].lower()
                    and "tokyo" not in row["answer"].lower()
                )
                row["manual_correctness"] = manual_refusal
                row["manual_groundedness"] = manual_refusal
                row["manual_citation_support"] = ""
                row["notes"] = "Unsupported question refusal graded manually."

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        answerable = [row for row in rows if row["answerable"].lower() == "true"]
        unsupported = [row for row in rows if row["answerable"].lower() == "false"]
        generation = [float(row["generation_seconds"]) for row in answerable]
        summaries.append(
            {
                "label": label,
                "answer_correctness": mean(float(row["manual_correctness"]) for row in answerable),
                "groundedness": mean(float(row["manual_groundedness"]) for row in answerable),
                "citation_support": mean(float(row["manual_citation_support"]) for row in answerable),
                "unsupported_refusal": mean(float(row["manual_correctness"]) for row in unsupported),
                "avg_prompt_tokens": mean(int(row["prompt_tokens"]) for row in answerable),
                "generation_p95_seconds": float(np.percentile(generation, 95)),
            }
        )

    with (OUTPUT / "rag-comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
