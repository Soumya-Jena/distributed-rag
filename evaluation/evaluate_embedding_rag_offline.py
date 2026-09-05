import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from src.chunking import chunk_text
from src.config import MIN_RETRIEVAL_SIMILARITY
from src.document_loader import discover_documents, load_document
from src.embedding_service import EmbeddingService
from src.generator import LocalGenerator
from src.metrics import citations_valid, has_citation, keyword_coverage, refused
from src.prompt import SYSTEM_PROMPT, build_user_prompt
from src.retriever import RetrievedChunk


MODELS = [
    ("embedding-minilm", "sentence-transformers/all-MiniLM-L6-v2"),
    ("embedding-e5-small-v2", "intfloat/e5-small-v2"),
]
OUTPUT = Path("experiments/day-06")
REFUSAL = "I don't have enough information in the provided sources to answer that question."


@dataclass
class CorpusChunk:
    title: str
    source_path: str
    chunk_index: int
    content: str


def load_questions():
    with Path("datasets/evaluation/questions.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_corpus(service):
    chunks = []
    for path in discover_documents(Path("datasets/raw")):
        for item in chunk_text(load_document(path), service.model, 100, 20):
            chunks.append(CorpusChunk(path.stem, str(path), item["chunk_index"], item["content"]))
    embeddings = service.encode_documents([chunk.content for chunk in chunks])
    return chunks, embeddings


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Preserve the original six-question RAG baseline for historical comparison.
    questions = [
        record for record in load_questions()
        if record["id"] in {"q001", "q002", "q003", "q004", "q005", "q006"}
    ]
    generator = LocalGenerator()

    for label, model_name in MODELS:
        service = EmbeddingService(model_name)
        corpus, embeddings = build_corpus(service)
        rows = []
        for record in questions:
            print(f"\n{label}: {record['id']}", flush=True)
            started = perf_counter()
            query_embedding = service.encode_query(record["question"])
            similarities = embeddings @ query_embedding
            order = np.argsort(-similarities)[:5]
            chunks = [
                RetrievedChunk(
                    chunk_id=int(index),
                    title=corpus[index].title,
                    source_path=corpus[index].source_path,
                    chunk_index=corpus[index].chunk_index,
                    content=corpus[index].content,
                    similarity=float(similarities[index]),
                )
                for index in order
                if float(similarities[index]) >= MIN_RETRIEVAL_SIMILARITY
            ]
            retrieval_seconds = perf_counter() - started

            if chunks:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(record["question"], chunks)},
                ]
                started = perf_counter()
                answer = generator.generate(messages)
                generation_seconds = perf_counter() - started
                prompt_tokens = generator.last_input_token_count
            else:
                answer = REFUSAL
                generation_seconds = 0.0
                prompt_tokens = 0

            rows.append(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "answerable": record["answerable"],
                    "answer": answer,
                    "keyword_coverage": keyword_coverage(answer, record["expected_keywords"]),
                    "has_citation": has_citation(answer),
                    "citations_valid": citations_valid(answer, len(chunks)),
                    "refused": refused(answer),
                    "retrieved_count": len(chunks),
                    "prompt_tokens": prompt_tokens,
                    "retrieval_seconds": retrieval_seconds,
                    "generation_seconds": generation_seconds,
                    "manual_correctness": "",
                    "manual_groundedness": "",
                    "manual_citation_support": "",
                    "notes": "",
                }
            )

        with (OUTPUT / f"{label}-rag.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
