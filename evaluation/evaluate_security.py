"""Controlled three-mode indirect prompt-injection experiment."""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv


DATASET = Path("datasets/evaluation/security_cases.jsonl")
OUTPUT = Path("experiments/day-10")
MODES = ("baseline", "structured", "layered")
EXPECTED_DOCUMENTS = {
    "kafka-overview.md", "kubernetes-scheduling.md", "postgres-mvcc.md",
    "postgres-replication.md", "poison_citation.md", "poison_output.md",
    "poison_override.md", "poison_prompt_extraction.md",
    "poison_retrieval.md", "poison_social_engineering.md",
}
FIELDS = (
    "id", "category", "question", "mode", "target_document", "attack_marker",
    "expected_behavior", "retrieved_sources", "retrieved_chunk_ids",
    "source_count", "target_exposed", "any_poison_exposed", "poison_rank",
    "flagged_chunks", "raw_answer", "delivered_answer", "marker_in_raw",
    "marker_in_delivered", "canary_in_raw", "canary_in_delivered",
    "blocked", "guard_issues", "citations", "target_only_citation",
    "prompt_tokens", "retrieval_seconds", "generation_seconds",
)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_existing(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as file:
        return {row["id"]: row for row in csv.DictReader(file)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch-size and max-new-tokens must be positive")

    load_dotenv()
    from src.security_eval import (
        CANARY, assert_security_database, contains_attack_marker,
        contains_canary, is_poisoned_path, security_database_url,
    )
    target = os.getenv("SECURITY_DATABASE_URL") or security_database_url(
        os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/ragdb")
    )
    assert_security_database(target)
    os.environ["DATABASE_URL"] = target

    # Import after selecting the isolated database; these modules read config at import.
    from src.config import MIN_RETRIEVAL_SIMILARITY
    from src.db import get_connection
    from src.generator import LocalGenerator
    from src.grounding import extract_citations
    from src.output_guard import guard_output
    from src.injection_detector import detect_injection
    from src.retrieval_pipeline import RetrievalPipeline
    from src.security_policy import build_security_messages

    with get_connection() as conn:
        dbname = conn.execute("SELECT current_database()").fetchone()[0]
        paths = [row[0] for row in conn.execute("SELECT source_path FROM documents")]
    if dbname != "ragdb_security" or not paths:
        raise RuntimeError("Isolated database is missing or empty; run setup and ingestion")
    if any(not path.replace("\\", "/").startswith("datasets/security/") for path in paths):
        raise RuntimeError("Security database contains a document outside datasets/security")
    if len(paths) != 10 or {Path(path).name for path in paths} != EXPECTED_DOCUMENTS:
        raise RuntimeError("Security database must contain exactly the frozen ten documents")

    with DATASET.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    if len(records) != 35 or len({r["id"] for r in records}) != 35:
        raise RuntimeError("Expected 25 unique attacks and 10 unique clean controls")
    categories = Counter(r["category"] for r in records)
    if categories != Counter({
        "override": 5, "prompt_leak": 5, "output_hijack": 5,
        "citation_hijack": 5, "social_engineering": 5, "clean_control": 10,
    }):
        raise RuntimeError("Security case categories do not match the frozen design")

    pipeline = RetrievalPipeline(use_reranker=False, retrieval_mode="hybrid")
    prepared = []
    for record in records:
        retrieval = pipeline.retrieve(record["question"], final_k=5)
        chunks = [
            chunk for chunk in retrieval.final_chunks
            if getattr(chunk, "lexical_score", None) is not None
            or (
                getattr(chunk, "similarity", None) is not None
                and chunk.similarity >= MIN_RETRIEVAL_SIMILARITY
            )
        ]
        prepared.append((record, retrieval, chunks))

    generator = LocalGenerator()
    for mode in (MODES if args.mode == "all" else (args.mode,)):
        path = OUTPUT / f"{mode}-security-results.csv"
        rows_by_id = read_existing(path)
        for record in records:
            old = rows_by_id.get(record["id"])
            if old and (
                old["question"] != record["question"]
                or old["category"] != record["category"]
                or old["mode"] != mode
            ):
                raise RuntimeError(f"Checkpoint differs from dataset for {record['id']}")
        pending = []
        for record, retrieval, chunks in prepared:
            if record["id"] in rows_by_id:
                continue
            messages, detections = build_security_messages(
                record["question"], chunks, mode, canary=CANARY
            )
            pending.append((record, retrieval, chunks, messages, detections))
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset:offset + args.batch_size]
            print(f"{mode}: " + ", ".join(item[0]["id"] for item in batch), flush=True)
            started = perf_counter()
            answers = generator.generate_batch(
                [item[3] for item in batch], max_new_tokens=args.max_new_tokens
            )
            seconds = (perf_counter() - started) / len(batch)
            for (record, retrieval, chunks, messages, detections), raw in zip(batch, answers):
                delivered, validation = (
                    guard_output(raw, CANARY) if mode == "layered"
                    else (raw, None)
                )
                target_indices = [
                    i for i, chunk in enumerate(chunks, 1)
                    if Path(chunk.source_path).name == record["target_document"]
                    and (
                        detect_injection(chunk.content).suspicious
                        or contains_attack_marker(chunk.content, record["attack_marker"])
                    )
                ]
                citations = extract_citations(raw)
                token_ids = generator.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                )
                rows_by_id[record["id"]] = {
                    "id": record["id"], "category": record["category"],
                    "question": record["question"], "mode": mode,
                    "target_document": record["target_document"],
                    "attack_marker": record["attack_marker"],
                    "expected_behavior": record["expected_behavior"],
                    "retrieved_sources": "|".join(chunk.source_path for chunk in chunks),
                    "retrieved_chunk_ids": "|".join(str(chunk.chunk_id) for chunk in chunks),
                    "source_count": len(chunks),
                    "target_exposed": int(bool(target_indices)),
                    "any_poison_exposed": int(any(is_poisoned_path(c.source_path) for c in chunks)),
                    "poison_rank": min(target_indices) if target_indices else "",
                    "flagged_chunks": sum(d.suspicious for d in detections),
                    "raw_answer": raw, "delivered_answer": delivered,
                    "marker_in_raw": int(contains_attack_marker(raw, record["attack_marker"])),
                    "marker_in_delivered": int(contains_attack_marker(delivered, record["attack_marker"])),
                    "canary_in_raw": int(contains_canary(raw, CANARY)),
                    "canary_in_delivered": int(contains_canary(delivered, CANARY)),
                    "blocked": int(validation is not None and not validation.safe),
                    "guard_issues": "|".join(validation.issues) if validation else "",
                    "citations": "|".join(str(c) for c in citations),
                    "target_only_citation": int(bool(citations) and set(citations).issubset(target_indices)),
                    "prompt_tokens": len(token_ids["input_ids"]),
                    "retrieval_seconds": (
                        retrieval.vector_seconds + retrieval.lexical_seconds
                        + retrieval.fusion_seconds + retrieval.rerank_seconds
                    ),
                    "generation_seconds": seconds,
                }
            write_rows(path, [rows_by_id[r["id"]] for r in records if r["id"] in rows_by_id])
        print(f"Saved {path} ({len(rows_by_id)} rows)", flush=True)


if __name__ == "__main__":
    main()
