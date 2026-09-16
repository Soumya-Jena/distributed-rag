"""Generate a balanced answer-quality sample for context strategies."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

from src.context_optimizer import ContextOptimizer
from src.generator import LocalGenerator
from src.grounding import extract_citations, extract_evidence_status
from src.metrics import citations_valid, has_citation, refused
from src.output_guard import guard_output
from src.retrieval_pipeline import RetrievalPipeline
from src.security_policy import build_security_messages


DATASET = Path("datasets/evaluation/query_transform_questions.jsonl")
OUTPUT = Path("experiments/day-12/context-rag-results.csv")
SELECTED_IDS = {
    "qt001", "qt011", "qt021", "qt031", "qt041",
}
STRATEGIES = (
    ("full", {}),
    ("deduplicated", {"dedup_threshold": 0.92}),
    ("extractive", {"dedup_threshold": 0.92, "keep_ratio": 0.50}),
    ("budgeted", {
        "dedup_threshold": 0.92, "keep_ratio": 0.50,
        "token_budget": 250,
    }),
)


def load_questions(limit=None):
    with DATASET.open(encoding="utf-8") as file:
        rows = [
            json.loads(line) for line in file if line.strip()
            and json.loads(line)["id"] in SELECTED_IDS
        ]
    rows.sort(key=lambda row: row["id"])
    return rows[:limit] if limit else rows


def write_rows(rows):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    generator = LocalGenerator()
    pipeline = RetrievalPipeline(use_reranker=False, retrieval_mode="hybrid")
    optimizer = ContextOptimizer(
        pipeline.retriever.embedding_service, generator.count_tokens,
        dedup_threshold=0.92, keep_ratio=0.50, token_budget=250,
    )
    records = load_questions(args.limit)
    completed = set()
    rows = []
    if OUTPUT.exists():
        with OUTPUT.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        completed = {(row["id"], row["strategy"]) for row in rows}
    answer_cache = {}
    for row in rows:
        fragments = json.loads(row["source_fragments"])
        signature = tuple(fragment["optimized_content"] for fragment in fragments)
        answer_cache[(row["id"], signature)] = row["answer"]

    for record in records:
        retrieval = pipeline.retrieve(record["question"], final_k=5)
        pending = []
        for strategy, options in STRATEGIES:
            if (record["id"], strategy) in completed:
                continue
            result = optimizer.optimize(
                record["question"], retrieval.final_chunks,
                strategy=strategy, **options,
            )
            messages, _ = build_security_messages(
                record["question"], result.chunks, "layered"
            )
            tokenized_prompt = generator.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True,
            )
            prompt_tokens = len(tokenized_prompt["input_ids"])
            pending.append((strategy, result, messages, prompt_tokens))
        if not pending:
            continue

        for strategy, result, messages, prompt_tokens in pending:
            signature = tuple(chunk.optimized_content for chunk in result.chunks)
            cached_answer = answer_cache.get((record["id"], signature))
            if cached_answer is None:
                started = perf_counter()
                answer = generator.generate(
                    messages, max_new_tokens=args.max_new_tokens
                )
                elapsed = perf_counter() - started
                answer_cache[(record["id"], signature)] = answer
                generation_note = ""
            else:
                answer = cached_answer
                elapsed = 0.0
                generation_note = "Generation reused for identical context."
            answer, validation = guard_output(answer)
            relevant_names = set(record["relevant_sources"])
            retained_names = {
                Path(chunk.source_path).name for chunk in result.chunks
            }
            rows.append({
                "id": record["id"],
                "query_type": record["query_type"],
                "question": record["question"],
                "strategy": strategy,
                "relevant_sources": "|".join(record["relevant_sources"]),
                "retrieved_sources": "|".join(
                    chunk.source_path for chunk in result.chunks
                ),
                "relevant_source_retained": int(bool(
                    relevant_names & retained_names
                )),
                "chunks": len(result.chunks),
                "original_context_tokens": result.original_tokens,
                "context_tokens": result.optimized_tokens,
                "compression_ratio": result.compression_ratio,
                "prompt_tokens": prompt_tokens,
                "output_tokens": generator.count_tokens(answer),
                "generation_seconds": elapsed,
                "evidence_status": extract_evidence_status(answer) or "",
                "has_citation": has_citation(answer),
                "citations_valid": int(citations_valid(
                    answer, len(result.chunks)
                )),
                "refused": refused(answer),
                "output_blocked": int(not validation.safe),
                "answer": answer,
                "source_fragments": json.dumps([{
                    "source_id": chunk.source_id,
                    "selected_sentence_indices": chunk.selected_sentence_indices,
                    "original_content": chunk.original_content,
                    "optimized_content": chunk.optimized_content,
                } for chunk in result.chunks], separators=(",", ":")),
                "manual_correctness": "",
                "manual_faithfulness": "",
                "manual_citation_support": "",
                "notes": generation_note,
            })
            write_rows(rows)
            print(
                f"Saved {record['id']} {strategy} ({len(rows)} rows)",
                flush=True,
            )


if __name__ == "__main__":
    main()
