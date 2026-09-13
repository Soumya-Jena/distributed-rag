"""Recount checkpointed prompt lengths using actual Qwen input_ids."""

import csv
import os
from pathlib import Path

from dotenv import load_dotenv

from evaluation.evaluate_security import FIELDS, MODES, OUTPUT
from src.security_eval import CANARY, assert_security_database, security_database_url


def main():
    load_dotenv()
    target = os.getenv("SECURITY_DATABASE_URL") or security_database_url(
        os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/ragdb")
    )
    assert_security_database(target)
    os.environ["DATABASE_URL"] = target
    from transformers import AutoTokenizer
    from src.config import GENERATION_MODEL, MIN_RETRIEVAL_SIMILARITY
    from src.retrieval_pipeline import RetrievalPipeline
    from src.security_policy import build_security_messages

    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    pipeline = RetrievalPipeline(use_reranker=False, retrieval_mode="hybrid")
    for mode in MODES:
        path = OUTPUT / f"{mode}-security-results.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            chunks = [
                chunk for chunk in pipeline.retrieve(row["question"], final_k=5).final_chunks
                if getattr(chunk, "lexical_score", None) is not None
                or (
                    getattr(chunk, "similarity", None) is not None
                    and chunk.similarity >= MIN_RETRIEVAL_SIMILARITY
                )
            ]
            ids = "|".join(str(chunk.chunk_id) for chunk in chunks)
            if ids != row["retrieved_chunk_ids"]:
                raise RuntimeError(f"Retrieval changed for {mode}/{row['id']}")
            messages, _ = build_security_messages(
                row["question"], chunks, mode, canary=CANARY
            )
            encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
            row["prompt_tokens"] = len(encoded["input_ids"])
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
        print(f"Recounted {len(rows)} prompt lengths in {path}")


if __name__ == "__main__":
    main()
