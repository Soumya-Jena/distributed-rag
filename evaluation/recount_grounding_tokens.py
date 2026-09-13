"""Repair Day-9 prompt-token counts after BatchEncoding length was misread."""

import csv
import os
from pathlib import Path

from transformers import AutoTokenizer

from src.config import GENERATION_MODEL, MIN_RETRIEVAL_SIMILARITY
from src.db import get_connection
from src.metrics import source_name
from src.prompt import build_user_prompt, get_system_prompt
from src.retrieval_pipeline import RetrievalPipeline


OUTPUT = Path("experiments/day-09")


def main():
    with get_connection() as conn:
        if conn.execute("SELECT current_database()").fetchone()[0] != "ragdb":
            raise RuntimeError("Refusing to recount grounding prompts outside ragdb")
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    pipeline = RetrievalPipeline(use_reranker=False, retrieval_mode="hybrid")
    for mode in ("baseline", "strict"):
        path = OUTPUT / f"{mode}-grounding.csv"
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            fields = reader.fieldnames
            rows = list(reader)
        for row in rows:
            retrieval = pipeline.retrieve(row["question"], final_k=5)
            chunks = [
                chunk for chunk in retrieval.final_chunks
                if getattr(chunk, "lexical_score", None) is not None
                or (
                    getattr(chunk, "similarity", None) is not None
                    and chunk.similarity >= MIN_RETRIEVAL_SIMILARITY
                )
            ]
            if "|".join(source_name(chunk) for chunk in chunks) != row["retrieved_sources"]:
                raise RuntimeError(f"Retrieval changed for {mode}/{row['id']}")
            if not chunks:
                row["prompt_tokens"] = 0
                continue
            messages = [
                {"role": "system", "content": get_system_prompt(mode)},
                {"role": "user", "content": build_user_prompt(row["question"], chunks)},
            ]
            encoding = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
            row["prompt_tokens"] = len(encoding["input_ids"])
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
        print(f"Corrected {len(rows)} prompt counts in {path}")


if __name__ == "__main__":
    main()
