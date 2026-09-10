from dataclasses import dataclass

from src.db import get_connection


@dataclass
class LexicalChunk:
    chunk_id: int
    title: str
    source_path: str
    chunk_index: int
    content: str
    lexical_score: float


class LexicalRetriever:
    """PostgreSQL full-text retriever using the technical-token-friendly simple config."""

    def search(self, query: str, top_k: int = 20):
        with get_connection() as conn:
            rows = conn.execute(
                """
                WITH q AS (
                    SELECT websearch_to_tsquery('simple', %(query)s) AS query
                )
                SELECT
                    c.id,
                    d.title,
                    d.source_path,
                    c.chunk_index,
                    c.content,
                    ts_rank_cd(c.search_vector, q.query) AS lexical_score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                CROSS JOIN q
                WHERE c.search_vector @@ q.query
                ORDER BY lexical_score DESC, c.id
                LIMIT %(top_k)s;
                """,
                {"query": query, "top_k": top_k},
            ).fetchall()

        return [
            LexicalChunk(
                chunk_id=row[0],
                title=row[1],
                source_path=row[2],
                chunk_index=row[3],
                content=row[4],
                lexical_score=float(row[5]),
            )
            for row in rows
        ]
