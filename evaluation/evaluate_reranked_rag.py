import csv
import json
import shutil
from pathlib import Path

from evaluation.offline_retriever import OfflineRetriever
from src.metrics import citations_valid, has_citation, keyword_coverage, refused
from src.rag_service import RAGService
from src.retrieval_pipeline import RetrievalPipeline


OUTPUT = Path("experiments/day-07")
BASELINE = Path("experiments/day-06/embedding-minilm-rag.csv")


def questions():
    selected = {"q001", "q002", "q003", "q004", "q005", "q006"}
    with Path("datasets/evaluation/questions.jsonl").open(encoding="utf-8") as file:
        return [
            record
            for line in file
            if line.strip()
            for record in [json.loads(line)]
            if record["id"] in selected
        ]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASELINE, OUTPUT / "vector-rag.csv")

    pipeline = RetrievalPipeline(
        use_reranker=True,
        retriever=OfflineRetriever(),
    )
    rag = RAGService(retrieval_pipeline=pipeline)
    rows = []
    for record in questions():
        print(f"Reranked RAG: {record['id']}", flush=True)
        result = rag.answer(record["question"], top_k=5)
        rows.append(
            {
                "id": record["id"],
                "question": record["question"],
                "answerable": record["answerable"],
                "answer": result.answer,
                "keyword_coverage": keyword_coverage(result.answer, record["expected_keywords"]),
                "has_citation": has_citation(result.answer),
                "citations_valid": citations_valid(result.answer, len(result.chunks)),
                "refused": refused(result.answer),
                "retrieved_count": len(result.chunks),
                "prompt_tokens": result.prompt_tokens,
                "retrieval_seconds": result.retrieval_seconds,
                "vector_seconds": result.vector_seconds,
                "rerank_seconds": result.rerank_seconds,
                "generation_seconds": result.generation_seconds,
                "manual_correctness": "",
                "manual_groundedness": "",
                "manual_citation_support": "",
                "notes": "",
            }
        )

    with (OUTPUT / "reranked-rag.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
