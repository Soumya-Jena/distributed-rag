"""Apply the documented manual review and summarize context RAG quality."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


OUTPUT = Path("experiments/day-12")
RESULTS = OUTPUT / "context-rag-results.csv"

# The 20-token run is intentionally graded as produced. A correct score requires
# a substantially correct answer to the whole question; faithfulness means the
# generated claims are supported even when the answer is incomplete.
CORRECTNESS = {
    "qt001": {"full": 1, "deduplicated": 1, "extractive": 1, "budgeted": 1},
    "qt011": {"full": 1, "deduplicated": 1, "extractive": 1, "budgeted": 1},
    "qt021": {"full": 1, "deduplicated": 1, "extractive": 1, "budgeted": 1},
    "qt031": {"full": 1, "deduplicated": 1, "extractive": 0, "budgeted": 0},
    "qt041": {"full": 0, "deduplicated": 0, "extractive": 0, "budgeted": 0},
}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    with RESULTS.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        correct = CORRECTNESS[row["id"]][row["strategy"]]
        row["manual_correctness"] = correct
        row["manual_faithfulness"] = 1
        row["manual_citation_support"] = int(row["has_citation"] == "1")
        if row["id"] == "qt031" and row["strategy"] in {
            "extractive", "budgeted"
        }:
            row["notes"] = (
                "Incomplete numbered-list opening; no role was actually stated."
            )
        elif row["id"] == "qt041":
            row["notes"] = (
                "Incomplete: omitted the asynchronous-standby data-loss risk."
            )
    write_csv(RESULTS, rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)
    summary = []
    for strategy, selected in grouped.items():
        generated = [
            float(row["generation_seconds"]) for row in selected
            if float(row["generation_seconds"]) > 0
        ]
        summary.append({
            "strategy": strategy,
            "questions": len(selected),
            "mean_context_tokens": mean(
                int(row["context_tokens"]) for row in selected
            ),
            "mean_prompt_tokens": mean(
                int(row["prompt_tokens"]) for row in selected
            ),
            "mean_output_tokens": mean(
                int(row["output_tokens"]) for row in selected
            ),
            "mean_generation_seconds": mean(generated) if generated else "reused",
            "correctness": mean(
                int(row["manual_correctness"]) for row in selected
            ),
            "faithfulness": mean(
                int(row["manual_faithfulness"]) for row in selected
            ),
            "citation_presence": mean(
                int(row["has_citation"]) for row in selected
            ),
            "citation_support": mean(
                int(row["manual_citation_support"]) for row in selected
            ),
        })
    write_csv(OUTPUT / "context-rag-comparison.csv", summary)
    print(*summary, sep="\n")


if __name__ == "__main__":
    main()
