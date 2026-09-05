from dataclasses import dataclass

from src.config import (
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
)
from src.db import get_connection
from src.embedding_service import EmbeddingService


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

        self.embedding_service = EmbeddingService(EMBEDDING_MODEL)
        with get_connection() as conn:
            config = conn.execute(
                "SELECT embedding_model, chunk_size, chunk_overlap FROM corpus_config WHERE id = 1"
            ).fetchone()
        if not config:
            raise RuntimeError("Corpus configuration is missing. Re-ingest the corpus.")
        if config[0] != EMBEDDING_MODEL:
            raise RuntimeError(
                f"Corpus was embedded using {config[0]} but query model is "
                f"{EMBEDDING_MODEL}. Re-ingest the corpus before querying."
            )

    def search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ):

        query_embedding = self.embedding_service.encode_query(query)

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
