from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
)
from src.db import get_connection


@dataclass
class RetrievedChunk:
    chunk_id: int
    title: str
    source_path: str
    chunk_index: int
    content: str
    similarity: float


class Retriever:

    def __init__(self):

        print(
            f"Loading embedding model: "
            f"{EMBEDDING_MODEL}"
        )

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ):

        query_embedding = self.model.encode(
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

        return [
            RetrievedChunk(
                chunk_id=row[0],
                title=row[1],
                source_path=row[2],
                chunk_index=row[3],
                content=row[4],
                similarity=float(row[5]),
            )
            for row in rows
        ]