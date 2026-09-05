import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

from src.chunking import chunk_text
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL
from src.db import (
    get_connection,
    initialize_database,
)
from src.document_loader import (
    discover_documents,
    load_document,
)
from src.embedding_service import EmbeddingService


def ingest_document(
    path: Path,
    embedding_service: EmbeddingService,
):

    print(f"\nLoading: {path}")

    content = load_document(path)

    if not content.strip():
        print("Skipping empty document.")
        return

    chunks = chunk_text(
        content,
        embedding_service.model,
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    started = perf_counter()
    embeddings = embedding_service.encode_documents(texts)
    embedding_seconds = perf_counter() - started

    metadata = {
        "extension": path.suffix.lower(),
    }

    with get_connection() as conn:

        document = conn.execute(
            """
            INSERT INTO documents (
                title,
                source_path,
                content,
                metadata,
                updated_at
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                NOW()
            )
            ON CONFLICT (source_path)
            DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            RETURNING id;
            """,
            (
                path.stem,
                str(path),
                content,
                json.dumps(metadata),
            ),
        ).fetchone()

        document_id = document[0]

        # Makes ingestion idempotent.
        # Re-running ingestion replaces the chunks
        # belonging to this document.
        conn.execute(
            """
            DELETE FROM chunks
            WHERE document_id = %s;
            """,
            (document_id,),
        )

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):

            chunk_metadata = {
                "source": str(path),
            }

            conn.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_index,
                    content,
                    token_count,
                    embedding,
                    metadata
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                );
                """,
                (
                    document_id,
                    chunk["chunk_index"],
                    chunk["content"],
                    chunk["token_count"],
                    embedding,
                    json.dumps(chunk_metadata),
                ),
            )

        conn.commit()

    print(
        f"Stored {len(chunks)} embeddings."
    )
    print(f"Embedding time : {embedding_seconds:.3f}s")
    print(f"Chunks/sec     : {len(texts) / embedding_seconds:.2f}")
    return len(texts), embedding_seconds


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing documents",
    )
    parser.add_argument("--metrics-output", type=Path)

    args = parser.parse_args()

    initialize_database()

    embedding_service = EmbeddingService(EMBEDDING_MODEL)

    documents = discover_documents(
        args.directory
    )

    print(
        f"Found {len(documents)} documents."
    )

    total_chunks = 0
    total_embedding_seconds = 0.0
    for document in documents:

        chunks, seconds = ingest_document(
            document,
            embedding_service,
        )
        total_chunks += chunks
        total_embedding_seconds += seconds

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO corpus_config (id, embedding_model, chunk_size, chunk_overlap)
            VALUES (1, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                embedding_model = EXCLUDED.embedding_model,
                chunk_size = EXCLUDED.chunk_size,
                chunk_overlap = EXCLUDED.chunk_overlap,
                updated_at = NOW();
            """,
            (EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP),
        )
        conn.commit()

    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["embedding_model", "chunks", "embedding_seconds", "chunks_per_second"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "embedding_model": EMBEDDING_MODEL,
                    "chunks": total_chunks,
                    "embedding_seconds": total_embedding_seconds,
                    "chunks_per_second": total_chunks / total_embedding_seconds,
                }
            )


if __name__ == "__main__":
    main()
