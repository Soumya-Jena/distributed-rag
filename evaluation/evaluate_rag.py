import argparse
import csv
import json
import re

from pathlib import Path
from statistics import mean

from src.metrics import citations_valid, has_citation, keyword_coverage, refused
from src.rag_service import RAGService


DATASET = Path("datasets/evaluation/questions.jsonl")


def load_dataset():
    records = []
    with DATASET.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def valid_label(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError(
            "label may contain only letters, numbers, dots, underscores, and hyphens"
        )
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, type=valid_label)
    args = parser.parse_args()

    output = Path("experiments/day-05") / f"{args.label}-rag.csv"
    dataset = load_dataset()
    rag = RAGService()
    results = []

    for record in dataset:
        print(f"\nRunning {record['id']}...")
        result = rag.answer(record["question"], top_k=5)
        results.append(
            {
                "id": record["id"],
                "question": record["question"],
                "answerable": record["answerable"],
                "answer": result.answer,
                "keyword_coverage": keyword_coverage(
                    result.answer, record["expected_keywords"]
                ),
                "has_citation": has_citation(result.answer),
                "citations_valid": citations_valid(
                    result.answer, len(result.chunks)
                ),
                "refused": refused(result.answer),
                "retrieved_count": len(result.chunks),
                "prompt_tokens": result.prompt_tokens,
                "retrieval_seconds": result.retrieval_seconds,
                "generation_seconds": result.generation_seconds,
                "manual_correctness": "",
                "manual_groundedness": "",
                "manual_citation_support": "",
                "notes": "",
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    generated = [row for row in results if row["generation_seconds"] > 0]
    print(f"\nQuestions          : {len(results)}")
    print(f"Generated answers  : {len(generated)}")
    if generated:
        print(
            "Avg prompt tokens  : "
            f"{mean(row['prompt_tokens'] for row in generated):.1f}"
        )
        print(
            "Avg generation sec : "
            f"{mean(row['generation_seconds'] for row in generated):.3f}"
        )
    print(f"Results saved to   : {output}")


if __name__ == "__main__":
    main()
