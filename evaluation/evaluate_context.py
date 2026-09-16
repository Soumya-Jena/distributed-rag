"""Evaluate context optimization over the frozen 50-question query dataset."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer

from src.config import EMBEDDING_MODEL, GENERATION_MODEL
from src.context_optimizer import ContextOptimizer
from src.retrieval_pipeline import RetrievalPipeline


DATASET = Path("datasets/evaluation/query_transform_questions.jsonl")
OUTPUT = Path("experiments/day-12")
CORE = (
    ("full", {}),
    ("deduplicated", {"dedup_threshold": 0.92}),
    ("extractive", {"dedup_threshold": 0.92, "keep_ratio": 0.50}),
    ("budgeted", {
        "dedup_threshold": 0.92, "keep_ratio": 0.50, "token_budget": 250,
    }),
)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def relevant_retained(chunks, relevant_sources):
    names = {Path(chunk.source_path).name for chunk in chunks}
    return int(bool(names & set(relevant_sources)))


def result_row(record, label, result):
    relevant_chunks = sum(
        Path(chunk.source_path).name in record["relevant_sources"]
        for chunk in result.chunks
    )
    return {
        "id": record["id"], "query_type": record["query_type"],
        "strategy": label, "question": record["question"],
        "chunks_before": result.candidates_before,
        "chunks_after": result.candidates_after,
        "unique_sources": result.unique_sources,
        "original_context_tokens": result.original_tokens,
        "context_tokens": result.optimized_tokens,
        "removed_tokens": result.removed_tokens,
        "compression_ratio": result.compression_ratio,
        "relevant_source_retained": relevant_retained(
            result.chunks, record["relevant_sources"]
        ),
        "relevant_chunks": relevant_chunks,
        "relevant_evidence_density_proxy": (
            relevant_chunks / (result.optimized_tokens / 100)
            if result.optimized_tokens else 0.0
        ),
        "optimization_seconds": result.optimization_seconds,
        "sources": "|".join(chunk.source_path for chunk in result.chunks),
        "source_fragments": json.dumps([
            {
                "source_id": chunk.source_id,
                "selected_sentence_indices": chunk.selected_sentence_indices,
                "original_tokens": chunk.original_tokens,
                "optimized_tokens": chunk.optimized_tokens,
            }
            for chunk in result.chunks
        ], separators=(",", ":")),
    }


def summarize(rows, group_fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    summary = []
    for key, selected in groups.items():
        row = dict(zip(group_fields, key))
        row.update({
            "questions": len(selected),
            "mean_context_tokens": mean(r["context_tokens"] for r in selected),
            "mean_removed_tokens": mean(r["removed_tokens"] for r in selected),
            "mean_compression_ratio": mean(r["compression_ratio"] for r in selected),
            "relevant_source_retention": mean(
                r["relevant_source_retained"] for r in selected
            ),
            "mean_unique_sources": mean(r["unique_sources"] for r in selected),
            "mean_relevant_evidence_density_proxy": mean(
                r["relevant_evidence_density_proxy"] for r in selected
            ),
            "mean_optimization_seconds": mean(
                r["optimization_seconds"] for r in selected
            ),
        })
        summary.append(row)
    return summary


def main():
    with DATASET.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    if len(records) != 50:
        raise RuntimeError("Expected the frozen 50-question dataset")

    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    pipeline = RetrievalPipeline(use_reranker=False, retrieval_mode="hybrid")
    if pipeline.retriever.embedding_service.model_name != EMBEDDING_MODEL:
        raise RuntimeError("Embedding configuration drifted")
    optimizer = ContextOptimizer(
        pipeline.retriever.embedding_service, counter,
        dedup_threshold=0.92, keep_ratio=0.50, token_budget=500,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)

    prepared = []
    for record in records:
        retrieval = pipeline.retrieve(record["question"], final_k=5)
        prepared.append((record, retrieval.final_chunks))

    core_rows = []
    for record, chunks in prepared:
        for label, options in CORE:
            result = optimizer.optimize(record["question"], chunks, label, **options)
            core_rows.append(result_row(record, label, result))
        print(f"core {record['id']}", flush=True)
    write_csv(OUTPUT / "context-strategy-results.csv", core_rows)
    write_csv(
        OUTPUT / "context-strategy-comparison.csv",
        summarize(core_rows, ["strategy"]),
    )
    write_csv(
        OUTPUT / "context-query-type-comparison.csv",
        summarize(core_rows, ["strategy", "query_type"]),
    )

    sweep_rows = []
    for threshold in (0.85, 0.90, 0.92, 0.95):
        for record, chunks in prepared:
            result = optimizer.optimize(
                record["question"], chunks, "deduplicated",
                dedup_threshold=threshold,
            )
            row = result_row(record, f"threshold-{threshold:.2f}", result)
            row["sweep"] = "threshold"
            row["value"] = threshold
            sweep_rows.append(row)
    for ratio in (1.00, 0.75, 0.50, 0.25):
        for record, chunks in prepared:
            result = optimizer.optimize(
                record["question"], chunks, "extractive", keep_ratio=ratio,
            )
            row = result_row(record, f"ratio-{ratio:.2f}", result)
            row["sweep"] = "ratio"
            row["value"] = ratio
            sweep_rows.append(row)
    # 250 is a corpus-scaled diagnostic; the requested 500-2000 range is
    # retained so the experiment also documents where the budget stops binding.
    for budget in (250, 500, 750, 1000, 1500, 2000):
        for record, chunks in prepared:
            result = optimizer.optimize(
                record["question"], chunks, "budgeted", token_budget=budget,
            )
            row = result_row(record, f"budget-{budget}", result)
            row["sweep"] = "budget"
            row["value"] = budget
            sweep_rows.append(row)
    for neighbor_window in (0, 1):
        for record, chunks in prepared:
            result = optimizer.optimize(
                record["question"], chunks, "extractive",
                keep_ratio=0.50, neighbor_window=neighbor_window,
            )
            row = result_row(
                record, f"neighbor-{neighbor_window}", result
            )
            row["sweep"] = "neighbor_window"
            row["value"] = neighbor_window
            sweep_rows.append(row)
    write_csv(OUTPUT / "context-sweep-results.csv", sweep_rows)
    write_csv(
        OUTPUT / "context-sweep-comparison.csv",
        summarize(sweep_rows, ["sweep", "value"]),
    )
    print("Saved context optimization results")


if __name__ == "__main__":
    main()
