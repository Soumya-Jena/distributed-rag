import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


OUTPUT = Path("experiments/day-09")
RAW_FILES = {
    "baseline": OUTPUT / "baseline-grounding.csv",
    "strict": OUTPUT / "strict-grounding.csv",
}
ANNOTATIONS = OUTPUT / "manual-annotations.csv"


def read_csv(path):
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def integer(row, name):
    value = row.get(name, "")
    return int(value) if value not in ("", None) else 0


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(mode, rows):
    full = [row for row in rows if row["answerability"] == "full"]
    partial = [row for row in rows if row["answerability"] == "partial"]
    unsupported = [row for row in rows if row["answerability"] == "none"]
    total_claims = sum(integer(row, "total_claims") for row in rows)
    supported = sum(integer(row, "supported_claims") for row in rows)
    partially_supported = sum(
        integer(row, "partially_supported_claims") for row in rows
    )
    unsupported_claims = sum(integer(row, "unsupported_claims") for row in rows)
    cited = sum(integer(row, "claims_with_citations") for row in rows)
    supported_citations = sum(
        integer(row, "citations_that_support_claim") for row in rows
    )
    return {
        "prompt_mode": mode,
        "questions": len(rows),
        "total_claims": total_claims,
        "supported_claims": supported,
        "partially_supported_claims": partially_supported,
        "unsupported_claims": unsupported_claims,
        "faithfulness": ratio(supported, total_claims),
        "unsupported_claim_rate": ratio(unsupported_claims, total_claims),
        "explicit_status_rate": mean(bool(row["evidence_status"]) for row in rows),
        "explicit_status_accuracy": mean(
            row["evidence_status"]
            == {"full": "SUPPORTED", "partial": "PARTIAL", "none": "INSUFFICIENT"}[
                row["answerability"]
            ]
            for row in rows
        ),
        "correct_refusal_rate": mean(
            row["observed_status"] == "INSUFFICIENT" for row in unsupported
        ),
        "over_refusal_rate": mean(
            row["observed_status"] == "INSUFFICIENT" for row in full
        ),
        "partial_handling_accuracy": mean(
            integer(row, "manual_partial_handling") for row in partial
        ),
        "citation_coverage": ratio(cited, total_claims),
        "citation_support": ratio(supported_citations, cited),
        "invalid_citations": sum(integer(row, "invalid_citation_count") for row in rows),
        "answer_correctness": mean(integer(row, "manual_correctness") for row in rows),
        "answer_groundedness": mean(integer(row, "manual_groundedness") for row in rows),
        "mean_generation_seconds": mean(float(row["generation_seconds"]) for row in rows),
    }


def trap_rows(review_rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in review_rows:
        if row["trap_type"] != "none":
            groups[row["trap_type"]][row["prompt_mode"]].append(row)

    output = []
    for trap_type in sorted(groups):
        row = {"trap_type": trap_type}
        for mode in ("baseline", "strict"):
            values = groups[trap_type].get(mode, [])
            total = sum(integer(item, "total_claims") for item in values)
            unsupported = sum(integer(item, "unsupported_claims") for item in values)
            row[f"{mode}_questions"] = len(values)
            row[f"{mode}_hallucination_rate"] = ratio(unsupported, total)
            row[f"{mode}_correctness"] = (
                mean(integer(item, "manual_correctness") for item in values)
                if values
                else 0.0
            )
        output.append(row)
    return output


def main():
    annotations = {
        (row["prompt_mode"], row["id"]): row for row in read_csv(ANNOTATIONS)
    }
    review = []
    for mode, path in RAW_FILES.items():
        for row in read_csv(path):
            annotation = annotations[(mode, row["id"])]
            for field in (
                "total_claims",
                "supported_claims",
                "partially_supported_claims",
                "unsupported_claims",
                "claims_with_citations",
                "citations_that_support_claim",
                "manual_correctness",
                "manual_groundedness",
                "manual_partial_handling",
                "notes",
            ):
                row[field] = annotation[field]
            review.append(row)

    if len(annotations) != 70 or len(review) != 70:
        raise ValueError("Expected 70 manual annotations and 70 reviewed rows")
    for row in review:
        classified = (
            integer(row, "supported_claims")
            + integer(row, "partially_supported_claims")
            + integer(row, "unsupported_claims")
        )
        if classified != integer(row, "total_claims"):
            raise ValueError(f"Claim counts do not balance for {row['prompt_mode']} {row['id']}")

    write_csv(OUTPUT / "grounding-review.csv", review)
    summaries = [
        summarize(mode, [row for row in review if row["prompt_mode"] == mode])
        for mode in ("baseline", "strict")
    ]
    write_csv(OUTPUT / "grounding-comparison.csv", summaries)
    write_csv(OUTPUT / "trap-type-comparison.csv", trap_rows(review))
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
