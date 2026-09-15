"""Generate and review real Qwen rewrites before retrieval integration."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

from src.query_transformer import (
    QueryTransformer, extract_identifiers, missing_identifiers,
)


DATASET = Path("datasets/evaluation/query_transform_questions.jsonl")
OUTPUT = Path("experiments/day-11/query-transform-review.csv")
FIELDS = (
    "id", "query_type", "original", "semantic", "technical", "alternate",
    "fallback_used", "original_identifiers", "missing_semantic_identifiers",
    "missing_technical_identifiers", "missing_alternate_identifiers",
    "transformation_seconds", "meaning_preserved", "identifiers_preserved",
    "hallucinated_terminology", "notes",
)


def write_rows(rows):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if min(args.limit, args.batch_size, args.max_new_tokens) < 1:
        parser.error("all numeric options must be positive")
    with DATASET.open(encoding="utf-8") as file:
        all_records = [json.loads(line) for line in file if line.strip()]
    type_order = ("DIRECT", "PARAPHRASE", "VAGUE", "EXACT_IDENTIFIER", "MULTI_PART")
    records = []
    while len(records) < min(args.limit, len(all_records)):
        added = False
        for query_type in type_order:
            candidate = next(
                (
                    record for record in all_records
                    if record["query_type"] == query_type and record not in records
                ),
                None,
            )
            if candidate and len(records) < args.limit:
                records.append(candidate)
                added = True
        if not added:
            break
    existing = {}
    if OUTPUT.exists():
        with OUTPUT.open(encoding="utf-8", newline="") as file:
            existing = {row["id"]: row for row in csv.DictReader(file)}
    transformer = QueryTransformer()
    for offset in range(0, len(records), args.batch_size):
        batch = [record for record in records[offset:offset + args.batch_size] if record["id"] not in existing]
        if not batch:
            continue
        print("Transforming " + ", ".join(record["id"] for record in batch), flush=True)
        started = perf_counter()
        variants = transformer.transform_batch(
            [record["question"] for record in batch], args.max_new_tokens
        )
        elapsed = (perf_counter() - started) / len(batch)
        for record, transformed in zip(batch, variants):
            missing = {
                name: sorted(missing_identifiers(transformed.original, value))
                for name, value in (
                    ("semantic", transformed.semantic),
                    ("technical", transformed.technical),
                    ("alternate", transformed.alternate),
                )
            }
            existing[record["id"]] = {
                "id": record["id"], "query_type": record["query_type"],
                "original": transformed.original, "semantic": transformed.semantic,
                "technical": transformed.technical, "alternate": transformed.alternate,
                "fallback_used": int(len(set(transformed.all())) == 1),
                "original_identifiers": "|".join(sorted(extract_identifiers(transformed.original))),
                "missing_semantic_identifiers": "|".join(missing["semantic"]),
                "missing_technical_identifiers": "|".join(missing["technical"]),
                "missing_alternate_identifiers": "|".join(missing["alternate"]),
                "transformation_seconds": elapsed,
                "meaning_preserved": "", "identifiers_preserved": "",
                "hallucinated_terminology": "", "notes": "",
            }
        write_rows([existing[r["id"]] for r in records if r["id"] in existing])
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
