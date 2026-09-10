import re
from collections import Counter

from src.lexical_retriever import LexicalChunk


TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def simple_tokens(text):
    """Approximate PostgreSQL's `simple` text-search normalization."""
    return TOKEN_PATTERN.findall(text.lower())


class OfflineLexicalRetriever:
    """Deterministic fallback for experiments when PostgreSQL is unavailable.

    PostgreSQL remains the production implementation. This fallback preserves
    the important `simple` + AND matching behavior, but its score is only a
    term-density proxy for `ts_rank_cd`.
    """

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self._tokens = [simple_tokens(chunk.content) for chunk in self.chunks]

    def search(self, query, top_k=20):
        query_terms = list(dict.fromkeys(simple_tokens(query)))
        if not query_terms:
            return []

        matches = []
        for chunk, tokens in zip(self.chunks, self._tokens):
            counts = Counter(tokens)
            if not all(counts[term] for term in query_terms):
                continue
            density = sum(counts[term] for term in query_terms) / max(len(tokens), 1)
            matches.append(
                LexicalChunk(
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    source_path=chunk.source_path,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    lexical_score=density,
                )
            )

        matches.sort(key=lambda item: (-item.lexical_score, item.chunk_id))
        return matches[:top_k]
