import csv
import json
import shutil
from pathlib import Path

from src.generator import LocalGenerator
from src.hybrid_retriever import HybridRetriever
from src.metrics import citations_valid, has_citation, keyword_coverage, refused
from src.rag_service import RAGService
from src.reranker import Reranker
from src.retrieval_pipeline import RetrievalPipeline


OUTPUT = Path("experiments/day-08")
DATASET = Path("datasets/evaluation/questions.jsonl")


def questions():
    selected = {"q001", "q002", "q003", "q004", "q005", "q006"}
    with DATASET.open(encoding="utf-8") as file:
        return [
            record
            for line in file
            if line.strip()
            for record in [json.loads(line)]
            if record["id"] in selected
        ]


def evaluate(name, service, records):
    rows = []
    for record in records:
        print(f"{name}: {record['id']}", flush=True)
        result = service.answer(record["question"], top_k=5)
        rows.append(
            {
                "id": record["id"],
                "question": record["question"],
                "answerable": record["answerable"],
                "answer": result.answer,
                "keyword_coverage": keyword_coverage(
                    result.answer, record["expected_keywords"]
                ),
                "has_citation": has_citation(result.answer),
                "citations_valid": citations_valid(result.answer, len(result.chunks)),
                "refused": refused(result.answer),
                "retrieved_count": len(result.chunks),
                "prompt_tokens": result.prompt_tokens,
                "retrieval_seconds": result.retrieval_seconds,
                "vector_seconds": result.vector_seconds,
                "lexical_seconds": result.lexical_seconds,
                "fusion_seconds": result.fusion_seconds,
                "rerank_seconds": result.rerank_seconds,
                "generation_seconds": result.generation_seconds,
                "manual_correctness": "",
                "manual_groundedness": "",
                "manual_citation_support": "",
                "notes": "",
            }
        )

    path = OUTPUT / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUTPUT.parent / "day-07/vector-rag.csv", OUTPUT / "vector-rag.csv")
    shutil.copyfile(
        OUTPUT.parent / "day-07/reranked-rag.csv",
        OUTPUT / "vector-reranked-rag.csv",
    )

    generator = LocalGenerator()
    hybrid_retriever = HybridRetriever()
    no_rerank_pipeline = RetrievalPipeline(
        use_reranker=False,
        retrieval_mode="hybrid",
        retriever=hybrid_retriever.vector_retriever,
        hybrid_retriever=hybrid_retriever,
    )
    full_pipeline = RetrievalPipeline(
        use_reranker=True,
        retrieval_mode="hybrid",
        retriever=hybrid_retriever.vector_retriever,
        hybrid_retriever=hybrid_retriever,
        reranker=Reranker(),
    )
    records = questions()
    evaluate(
        "hybrid-rag",
        RAGService(retrieval_pipeline=no_rerank_pipeline, generator=generator),
        records,
    )
    evaluate(
        "full-hybrid-rag",
        RAGService(retrieval_pipeline=full_pipeline, generator=generator),
        records,
    )


if __name__ == "__main__":
    main()
