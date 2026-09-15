"""Validate manual rewrite review and summarize it by query type."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


INPUT = Path("experiments/day-11/query-transform-review.csv")
MANUAL = Path("experiments/day-11/manual-query-transform-review.csv")
OUTPUT = Path("experiments/day-11/transformation-summary.csv")
TYPES = {"DIRECT", "PARAPHRASE", "VAGUE", "EXACT_IDENTIFIER", "MULTI_PART"}


def yes(value):
    return value.strip().lower() == "yes"


def meaning_score(value):
    return {"yes": 1.0, "partial": 0.5, "no": 0.0}[value.strip().lower()]


def summarize(label, rows):
    return {
        "query_type": label,
        "questions": len(rows),
        "meaning_preservation": mean(meaning_score(row["meaning_preserved"]) for row in rows),
        "identifier_preservation": mean(yes(row["identifiers_preserved"]) for row in rows),
        "hallucinated_terminology_rate": mean(yes(row["hallucinated_terminology"]) for row in rows),
        "fallback_rate": mean(int(row["fallback_used"]) for row in rows),
        "mean_transformation_seconds": mean(float(row["transformation_seconds"]) for row in rows),
    }


def main():
    with INPUT.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    with MANUAL.open(encoding="utf-8", newline="") as file:
        manual_rows = list(csv.DictReader(file))
    if len(rows) != 20 or len({row["id"] for row in rows}) != 20:
        raise RuntimeError("Expected exactly 20 unique transformation reviews")
    manual = {row["id"]: row for row in manual_rows}
    if len(manual_rows) != 20 or set(manual) != {row["id"] for row in rows}:
        raise RuntimeError("Manual review must cover the exact 20 generated transformations")
    for row in rows:
        row.update(manual[row["id"]])
    allowed_meaning = {"yes", "partial", "no"}
    allowed_binary = {"yes", "no"}
    for row in rows:
        if row["meaning_preserved"].strip().lower() not in allowed_meaning:
            raise RuntimeError(f"Missing meaning judgment for {row['id']}")
        for field in ("identifiers_preserved", "hallucinated_terminology"):
            if row[field].strip().lower() not in allowed_binary:
                raise RuntimeError(f"Missing {field} judgment for {row['id']}")

    groups = defaultdict(list)
    for row in rows:
        groups[row["query_type"]].append(row)
    if set(groups) != TYPES or set(map(len, groups.values())) != {4}:
        raise RuntimeError("Expected four reviewed questions per query type")
    summaries = [summarize("ALL", rows)] + [
        summarize(query_type, groups[query_type]) for query_type in sorted(TYPES)
    ]
    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0])
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
