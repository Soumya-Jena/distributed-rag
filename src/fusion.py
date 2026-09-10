from dataclasses import dataclass


@dataclass
class HybridCandidate:
    chunk_id: int
    title: str
    source_path: str
    chunk_index: int
    content: str
    vector_rank: int | None
    lexical_rank: int | None
    vector_similarity: float | None
    lexical_score: float | None
    rrf_score: float

    @property
    def similarity(self):
        """Compatibility with existing result formatting and threshold code."""
        return self.vector_similarity


def reciprocal_rank_fusion(
    vector_results,
    lexical_results,
    rrf_k=60,
    top_k=20,
):
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    candidates = {}

    for rank, chunk in enumerate(vector_results, start=1):
        candidates[chunk.chunk_id] = HybridCandidate(
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            source_path=chunk.source_path,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            vector_rank=rank,
            lexical_rank=None,
            vector_similarity=chunk.similarity,
            lexical_score=None,
            rrf_score=1 / (rrf_k + rank),
        )

    for rank, chunk in enumerate(lexical_results, start=1):
        candidate = candidates.get(chunk.chunk_id)
        if candidate:
            candidate.lexical_rank = rank
            candidate.lexical_score = chunk.lexical_score
            candidate.rrf_score += 1 / (rrf_k + rank)
        else:
            candidates[chunk.chunk_id] = HybridCandidate(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                source_path=chunk.source_path,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                vector_rank=None,
                lexical_rank=rank,
                vector_similarity=None,
                lexical_score=chunk.lexical_score,
                rrf_score=1 / (rrf_k + rank),
            )

    return sorted(
        candidates.values(),
        key=lambda item: (-item.rrf_score, item.chunk_id),
    )[:top_k]
