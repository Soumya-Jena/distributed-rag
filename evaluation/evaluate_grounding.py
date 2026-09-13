import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

from src.generator import LocalGenerator
from src.config import MIN_RETRIEVAL_SIMILARITY
from src.grounding import (
    extract_citations,
    extract_evidence_status,
    invalid_citations,
)
from src.metrics import refused, source_name
from src.prompt import build_user_prompt, get_system_prompt
from src.retrieval_pipeline import RetrievalPipeline


DATASET = Path("datasets/evaluation/grounding_questions.jsonl")
OUTPUT = Path("experiments/day-09")
FIELDNAMES = [
    "id",
    "question",
    "answerability",
    "trap_type",
    "prompt_mode",
    "generated_answer",
    "evidence_status",
    "observed_status",
    "retrieved_sources",
    "source_count",
    "citation_count",
    "invalid_citation_count",
    "prompt_tokens",
    "retrieval_seconds",
    "vector_seconds",
    "lexical_seconds",
    "fusion_seconds",
    "generation_seconds",
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
]


def load_records():
    with DATASET.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def token_count(generator, messages):
    encoded = generator.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    return len(encoded)


def derived_status(answer):
    status = extract_evidence_status(answer)
    if status:
        return status
    if refused(answer):
        return "INSUFFICIENT"
    return "UNDECLARED"


def build_row(record, mode, chunks, retrieval, answer, prompt_tokens, generation_seconds):
    citations = extract_citations(answer)
    return {
        "id": record["id"],
        "question": record["question"],
        "answerability": record["answerability"],
        "trap_type": record["trap_type"],
        "prompt_mode": mode,
        "generated_answer": answer,
        "evidence_status": extract_evidence_status(answer) or "",
        "observed_status": derived_status(answer),
        "retrieved_sources": "|".join(source_name(chunk) for chunk in chunks),
        "source_count": len(chunks),
        "citation_count": len(citations),
        "invalid_citation_count": len(invalid_citations(answer, len(chunks))),
        "prompt_tokens": prompt_tokens,
        "retrieval_seconds": retrieval.vector_seconds + retrieval.lexical_seconds + retrieval.fusion_seconds + retrieval.rerank_seconds,
        "vector_seconds": retrieval.vector_seconds,
        "lexical_seconds": retrieval.lexical_seconds,
        "fusion_seconds": retrieval.fusion_seconds,
        "generation_seconds": generation_seconds,
        "total_claims": "",
        "supported_claims": "",
        "partially_supported_claims": "",
        "unsupported_claims": "",
        "claims_with_citations": "",
        "citations_that_support_claim": "",
        "manual_correctness": "",
        "manual_groundedness": "",
        "manual_partial_handling": "",
        "notes": "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "strict"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    records = load_records()
    pipeline = RetrievalPipeline(use_reranker=False, retrieval_mode="hybrid")
    prepared = []
    for record in records:
        retrieval = pipeline.retrieve(record["question"], final_k=5)
        chunks = [
            chunk
            for chunk in retrieval.final_chunks
            if getattr(chunk, "lexical_score", None) is not None
            or (
                getattr(chunk, "similarity", None) is not None
                and chunk.similarity >= MIN_RETRIEVAL_SIMILARITY
            )
        ]
        prepared.append((record, retrieval, chunks))

    generator = LocalGenerator()
    rows_by_id = {}
    pending = []
    refusal = "I don't have enough information in the provided sources to answer that question."
    for record, retrieval, chunks in prepared:
        if not chunks:
            answer = (
                f"EVIDENCE_STATUS: INSUFFICIENT\n\n{refusal}"
                if args.mode == "strict"
                else refusal
            )
            rows_by_id[record["id"]] = build_row(
                record, args.mode, chunks, retrieval, answer, 0, 0.0
            )
            continue
        messages = [
            {"role": "system", "content": get_system_prompt(args.mode)},
            {"role": "user", "content": build_user_prompt(record["question"], chunks)},
        ]
        pending.append((record, retrieval, chunks, messages))

    output = OUTPUT / f"{args.label}.csv"
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        print(
            f"{args.mode}: " + ", ".join(item[0]["id"] for item in batch),
            flush=True,
        )
        started = perf_counter()
        answers = generator.generate_batch(
            [item[3] for item in batch],
            max_new_tokens=args.max_new_tokens,
        )
        elapsed_per_answer = (perf_counter() - started) / len(batch)
        for (record, retrieval, chunks, messages), answer in zip(batch, answers):
            rows_by_id[record["id"]] = build_row(
                record,
                args.mode,
                chunks,
                retrieval,
                answer,
                token_count(generator, messages),
                elapsed_per_answer,
            )
        write_rows(
            output,
            [rows_by_id[record["id"]] for record in records if record["id"] in rows_by_id],
        )

    write_rows(output, [rows_by_id[record["id"]] for record in records])
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
