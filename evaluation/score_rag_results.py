import csv

from pathlib import Path
from statistics import mean


DIRECTORY = Path("experiments/day-05")

LABELS = [
    "chunk-100-20",
    "chunk-150-30",
    "chunk-240-40",
    "final",
]

# Binary manual rubric:
# 1 = no material factual/grounding error for the evaluation question.
# 0 = at least one material error or unsupported claim.
MANUAL_SCORES = {
    "chunk-100-20": {
        "q001": (1, 1, 0, "Correct and source-supported, but no citation."),
        "q002": (
            0,
            0,
            0,
            "Overstates conflict avoidance and adds unsupported availability/latency framing.",
        ),
        "q003": (1, 1, 0, "Correct and source-supported, but no citation."),
        "q004": (1, 1, 0, "Correct and source-supported, but no citation."),
    },
    "chunk-150-30": {
        "q001": (1, 1, 0, "Correct and source-supported, but no citation."),
        "q002": (1, 1, 0, "Correct and source-supported, but no citation."),
        "q003": (
            0,
            0,
            0,
            "Incorrectly suggests that resource limits drive scheduler placement.",
        ),
        "q004": (
            0,
            0,
            0,
            "Incorrectly says replicas handle reads/writes; the source assigns that to the leader.",
        ),
    },
    "chunk-240-40": {
        "q001": (1, 1, 0, "Correct and source-supported, but no citation."),
        "q002": (
            0,
            0,
            0,
            "Incorrectly attributes wraparound prevention and reliable replication support to MVCC.",
        ),
        "q003": (1, 1, 0, "Correct and source-supported, but no citation."),
        "q004": (1, 1, 0, "Correct and source-supported, but no citation."),
    },
}
MANUAL_SCORES["final"] = MANUAL_SCORES["chunk-100-20"]


def load(path):
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    summaries = []

    for label in LABELS:
        path = DIRECTORY / f"{label}-rag.csv"
        rows = load(path)

        for row in rows:
            if row["answerable"].lower() == "true":
                correctness, groundedness, citation_support, notes = MANUAL_SCORES[
                    label
                ][row["id"]]
                row["manual_correctness"] = correctness
                row["manual_groundedness"] = groundedness
                row["manual_citation_support"] = citation_support
                row["notes"] = notes
            else:
                row["manual_correctness"] = int(row["refused"])
                row["manual_groundedness"] = int(row["refused"])
                row["manual_citation_support"] = ""
                row["notes"] = "Correct code-enforced refusal; citation not applicable."

        save(path, rows)

        answerable = [row for row in rows if row["answerable"].lower() == "true"]
        unanswerable = [row for row in rows if row["answerable"].lower() == "false"]
        summaries.append(
            {
                "label": label,
                "answer_correctness": mean(
                    float(row["manual_correctness"]) for row in answerable
                ),
                "groundedness": mean(
                    float(row["manual_groundedness"]) for row in answerable
                ),
                "citation_support": mean(
                    float(row["manual_citation_support"]) for row in answerable
                ),
                "refusal_accuracy": mean(
                    float(row["refused"]) for row in unanswerable
                ),
                "avg_prompt_tokens": mean(
                    int(row["prompt_tokens"]) for row in answerable
                ),
                "avg_retrieval_seconds": mean(
                    float(row["retrieval_seconds"]) for row in rows
                ),
                "avg_generation_seconds": mean(
                    float(row["generation_seconds"]) for row in answerable
                ),
            }
        )

    output = DIRECTORY / "rag-comparison.csv"
    save(output, summaries)
    for summary in summaries:
        print(summary)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
