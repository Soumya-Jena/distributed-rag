import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.chunking import chunk_text
from src.config import EMBEDDING_MODEL
from src.db import (
    get_connection,
    initialize_database,
)
from src.document_loader import (
    discover_documents,
    load_document,
)


def ingest_document(
    path: Path,
    model: SentenceTransformer,
):

    print(f"\nLoading: {path}")

    content = load_document(path)

    if not content.strip():
        print("Skipping empty document.")
        return

    chunks = chunk_text(
        content,
        model,
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing documents",
    )

    args = parser.parse_args()

    initialize_database()

    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    documents = discover_documents(
        args.directory
    )

    print(
        f"Found {len(documents)} documents."
    )

    for document in documents:

        ingest_document(
            document,
            model,
        )


if __name__ == "__main__":
    main()