import argparse

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL
from src.db import get_connection


def search(
    query: str,
    top_k: int = 5,
):

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    with get_connection() as conn:

        rows = conn.execute(
            """
            SELECT
                c.id,
                d.title,
                d.source_path,
                c.chunk_index,
                c.content,
                1 - (
                    c.embedding <=> %(embedding)s
                ) AS similarity
            FROM chunks c

            JOIN documents d
              ON d.id = c.document_id

            ORDER BY
                c.embedding <=> %(embedding)s

            LIMIT %(top_k)s;
            """,
            {
                "embedding": query_embedding,
                "top_k": top_k,
            },
        ).fetchall()

    return rows


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
        type=str,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    results = search(
        args.query,
        args.top_k,
    )

    print("\nQUERY")
    print("-----")
    print(args.query)

    print("\nRESULTS")
    print("-------")

    for rank, row in enumerate(
        results,
        start=1,
    ):

        (
            chunk_id,
            title,
            source,
            chunk_index,
            content,
            similarity,
        ) = row

        print(
            f"\n#{rank}"
            f"\nSimilarity : {similarity:.4f}"
            f"\nDocument   : {title}"
            f"\nChunk      : {chunk_index}"
            f"\nSource     : {source}"
            f"\n"
        )

        print(content[:700])

        print(
            "\n" + "-" * 70
        )


if __name__ == "__main__":
    main()