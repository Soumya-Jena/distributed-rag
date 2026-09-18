import argparse
import csv
import hashlib
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


def file_sha256(path: Path):
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def ingest_document(
    path: Path,
    embedding_service: EmbeddingService,
):

    print(f"\nLoading: {path}")

    content_hash = file_sha256(path)
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT content_hash FROM documents WHERE source_path = %s",
            (str(path),),
        ).fetchone()
    if existing and existing[0] == content_hash:
        print("Unchanged; skipping chunking and embedding.")
        return 0, 0.0, False

    content = load_document(path)

    if not content.strip():
        print("Skipping empty document.")
        return 0, 0.0, False

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
                content_hash,
                metadata,
                updated_at
            )
            VALUES (
                %s,
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
                content_hash = EXCLUDED.content_hash,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            RETURNING id;
            """,
            (
                path.stem,
                str(path),
                content,
                content_hash,
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
    return len(texts), embedding_seconds, True


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

    with get_connection() as conn:
        existing_hashes = dict(conn.execute(
            "SELECT source_path, content_hash FROM documents"
        ).fetchall())
        existing_config = conn.execute(
            """
            SELECT embedding_model, chunk_size, chunk_overlap
            FROM corpus_config WHERE id = 1;
            """
        ).fetchone()
        content_changed = any(
            existing_hashes.get(str(document)) != file_sha256(document)
            for document in documents
        )
        config_changed = existing_config is not None and tuple(existing_config) != (
            EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
        )
        increment = int(content_changed or config_changed)
        # Advance the namespace before committing changed chunks. A crash can
        # cause a conservative extra invalidation, but never stale cache reuse.
        conn.execute(
            """
            INSERT INTO corpus_config (
                id, embedding_model, chunk_size, chunk_overlap, corpus_version
            )
            VALUES (1, %s, %s, %s, 1)
            ON CONFLICT (id) DO UPDATE SET
                embedding_model = EXCLUDED.embedding_model,
                chunk_size = EXCLUDED.chunk_size,
                chunk_overlap = EXCLUDED.chunk_overlap,
                corpus_version = corpus_config.corpus_version + %s,
                updated_at = NOW();
            """,
            (EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, increment),
        )
        conn.commit()
        version = conn.execute(
            "SELECT corpus_version FROM corpus_config WHERE id = 1"
        ).fetchone()[0]

    total_chunks = 0
    total_embedding_seconds = 0.0
    for document in documents:

        chunks, seconds, _changed = ingest_document(
            document,
            embedding_service,
        )
        total_chunks += chunks
        total_embedding_seconds += seconds
    print(f"Corpus version: {version} ({'changed' if increment else 'unchanged'})")

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
                    "chunks_per_second": (
                        total_chunks / total_embedding_seconds
                        if total_embedding_seconds else 0.0
                    ),
                }
            )


if __name__ == "__main__":
    main()
