import csv
import json

from pathlib import Path
from statistics import mean

from src.metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)

from src.retriever import Retriever


DATASET = Path(
    "datasets/evaluation/questions.jsonl"
)

OUTPUT = Path(
    "experiments/day-04/"
    "retrieval_results.csv"
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

    retriever = Retriever()

    results = []

    for record in dataset:

        # Only evaluate retrieval quality
        # on answerable questions.

        if not record["answerable"]:
            continue

        chunks = retriever.search(
            record["question"],
            top_k=5,
        )

        result = {
            "id": record["id"],
            "question": record["question"],

            "hit@1": hit_at_k(
                chunks,
                record["relevant_sources"],
                1,
            ),

            "hit@3": hit_at_k(
                chunks,
                record["relevant_sources"],
                3,
            ),

            "hit@5": hit_at_k(
                chunks,
                record["relevant_sources"],
                5,
            ),

            "recall@5": recall_at_k(
                chunks,
                record["relevant_sources"],
                5,
            ),

            "reciprocal_rank": (
                reciprocal_rank(
                    chunks,
                    record[
                        "relevant_sources"
                    ],
                )
            ),

            "top_1_source": (
                Path(
                    chunks[0].source_path
                ).name
                if chunks
                else ""
            ),

            "top_1_similarity": (
                chunks[0].similarity
                if chunks
                else None
            ),
        }

        results.append(result)

        print(
            record["id"],
            record["question"],
            result["hit@5"],
        )

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

    print("\nBASELINE RETRIEVAL")
    print("==================")

    print(
        f"Questions : {len(results)}"
    )

    print(
        f"Hit@1     : "
        f"{mean(r['hit@1'] for r in results):.3f}"
    )

    print(
        f"Hit@3     : "
        f"{mean(r['hit@3'] for r in results):.3f}"
    )

    print(
        f"Hit@5     : "
        f"{mean(r['hit@5'] for r in results):.3f}"
    )

    print(
        f"Recall@5  : "
        f"{mean(r['recall@5'] for r in results):.3f}"
    )

    print(
        f"MRR       : "
        f"{mean(r['reciprocal_rank'] for r in results):.3f}"
    )

    print(
        f"\nSaved to: {OUTPUT}"
    )


if __name__ == "__main__":
    main()