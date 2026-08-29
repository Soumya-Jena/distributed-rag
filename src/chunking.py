from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)


def chunk_text(
    text: str,
    model: SentenceTransformer,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
):

    if overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    tokenizer = model.tokenizer

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    chunks = []

    start = 0
    chunk_index = 0

    while start < len(token_ids):

        end = min(
            start + chunk_size,
            len(token_ids),
        )

        chunk_token_ids = token_ids[start:end]

        chunk_text = tokenizer.decode(
            chunk_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        if chunk_text:

            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "content": chunk_text,
                    "token_count": len(
                        chunk_token_ids
                    ),
                }
            )

        if end == len(token_ids):
            break

        start += chunk_size - overlap
        chunk_index += 1

    return chunks