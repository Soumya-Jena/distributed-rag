import csv
import json

from pathlib import Path

from src.metrics import (
    citations_valid,
    has_citation,
    keyword_coverage,
    refused,
)

from src.rag_service import RAGService


DATASET = Path(
    "datasets/evaluation/questions.jsonl"
)

OUTPUT = Path(
    "experiments/day-04/"
    "rag_results.csv"
)


def load_dataset():

    records = []

    with DATASET.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def main():

    dataset = load_dataset()

    rag = RAGService()

    results = []

    for record in dataset:

        print(
            f"\nRunning {record['id']}..."
        )

        result = rag.answer(
            record["question"],
            top_k=5,
        )

        row = {
            "id": record["id"],

            "question": (
                record["question"]
            ),

            "answerable": (
                record["answerable"]
            ),

            "answer": result.answer,

            "keyword_coverage": (
                keyword_coverage(
                    result.answer,
                    record[
                        "expected_keywords"
                    ],
                )
            ),

            "has_citation": (
                has_citation(
                    result.answer
                )
            ),

            "citations_valid": (
                citations_valid(
                    result.answer,
                    len(result.chunks),
                )
            ),

            "refused": refused(
                result.answer
            ),

            "retrieval_seconds": (
                result.retrieval_seconds
            ),

            "generation_seconds": (
                result.generation_seconds
            ),

            # Fill these manually later.
            "manual_correctness": "",
            "manual_groundedness": "",
            "manual_citation_support": "",
            "notes": "",
        }

        results.append(row)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()

        writer.writerows(results)

    print(
        f"\nResults saved to {OUTPUT}"
    )


if __name__ == "__main__":
    main()