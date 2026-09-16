"""Repair or refresh token counts with the actual generator tokenizer."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from src.config import GENERATION_MODEL
from src.security_policy import build_security_messages


PATH = Path("experiments/day-12/context-rag-results.csv")


def main():
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    with PATH.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        chunks = []
        for fragment in json.loads(row["source_fragments"]):
            source_path, chunk_index = fragment["source_id"].rsplit("#", 1)
            chunks.append(SimpleNamespace(
                content=fragment["optimized_content"],
                title=Path(source_path).stem,
                source_path=source_path,
                chunk_index=int(chunk_index),
            ))
        messages, _ = build_security_messages(row["question"], chunks, "layered")
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True,
        )
        row["prompt_tokens"] = len(encoded["input_ids"])
        row["output_tokens"] = len(tokenizer.encode(
            row["answer"], add_special_tokens=False
        ))
    with PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Recounted {len(rows)} rows with {GENERATION_MODEL}")


if __name__ == "__main__":
    main()
