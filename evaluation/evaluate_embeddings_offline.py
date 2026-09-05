import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from src.chunking import chunk_text
from src.document_loader import discover_documents, load_document
from src.embedding_service import EmbeddingService
from src.metrics import hit_at_k, recall_at_k, reciprocal_rank


MODELS = [
    ("embedding-minilm", "sentence-transformers/all-MiniLM-L6-v2"),
    ("embedding-bge-small", "BAAI/bge-small-en-v1.5"),
    ("embedding-e5-small-v2", "intfloat/e5-small-v2"),
]
OUTPUT = Path("experiments/day-06")


@dataclass
class Chunk:
    source_path: str
    chunk_index: int
    content: str
    similarity: float = 0.0


def percentile(values, fraction):
    return float(np.percentile(values, fraction * 100))


def load_questions():
    with Path("datasets/evaluation/questions.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def evaluate(label, model_name, questions):
    service = EmbeddingService(model_name)
    chunks = []
    for path in discover_documents(Path("datasets/raw")):
        for item in chunk_text(load_document(path), service.model, 100, 20):
            chunks.append(Chunk(str(path), item["chunk_index"], item["content"]))

    started = perf_counter()
    document_embeddings = service.encode_documents([chunk.content for chunk in chunks])
    embedding_seconds = perf_counter() - started

    rows = []
    for record in questions:
        if not record["answerable"]:
            continue
        started = perf_counter()
        query_embedding = service.encode_query(record["question"])
        similarities = document_embeddings @ query_embedding
        order = np.argsort(-similarities)
        ranked = [
            Chunk(
                chunks[index].source_path,
                chunks[index].chunk_index,
                chunks[index].content,
                float(similarities[index]),
            )
            for index in order
        ]
        latency = perf_counter() - started
        relevant = record["relevant_sources"]
        rows.append(
            {
                "id": record["id"],
                "question": record["question"],
                "hit@1": hit_at_k(ranked, relevant, 1),
                "hit@3": hit_at_k(ranked, relevant, 3),
                "hit@5": hit_at_k(ranked, relevant, 5),
                "recall@5": recall_at_k(ranked, relevant, 5),
                "reciprocal_rank": reciprocal_rank(ranked, relevant),
                "correct_source_rank": next(
                    (i for i, chunk in enumerate(ranked, 1) if Path(chunk.source_path).name in relevant),
                    0,
                ),
                "top_1_source": Path(ranked[0].source_path).name,
                "top_1_chunk_index": ranked[0].chunk_index,
                "top_1_similarity": ranked[0].similarity,
                "retrieval_seconds": latency,
            }
        )

    with (OUTPUT / f"{label}-retrieval.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / f"{label}-embedding.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["embedding_model", "chunks", "embedding_seconds", "chunks_per_second"])
        writer.writeheader()
        writer.writerow(
            {
                "embedding_model": model_name,
                "chunks": len(chunks),
                "embedding_seconds": embedding_seconds,
                "chunks_per_second": len(chunks) / embedding_seconds,
            }
        )

    latencies = [row["retrieval_seconds"] for row in rows]
    summary = {
        "label": label,
        "embedding_model": model_name,
        "hit_at_1": mean(row["hit@1"] for row in rows),
        "hit_at_3": mean(row["hit@3"] for row in rows),
        "hit_at_5": mean(row["hit@5"] for row in rows),
        "recall_at_5": mean(row["recall@5"] for row in rows),
        "mrr": mean(row["reciprocal_rank"] for row in rows),
        "retrieval_p50_seconds": percentile(latencies, 0.50),
        "retrieval_p95_seconds": percentile(latencies, 0.95),
        "chunks_per_second": len(chunks) / embedding_seconds,
    }
    print(summary)
    return summary


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    summaries = [evaluate(label, model, questions) for label, model in MODELS]
    with (OUTPUT / "retrieval-comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
