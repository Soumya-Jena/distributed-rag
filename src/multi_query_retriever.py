"""Two-level RRF: vector/lexical per query, then across query variants."""

from dataclasses import dataclass, field
from time import perf_counter

from src.config import HYBRID_CANDIDATE_K, QUERY_VARIANTS, RRF_K
from src.hybrid_retriever import HybridRetriever
from src.query_transformer import QueryVariants
from src.reranker import Reranker


VARIANT_NAMES = ("original", "semantic", "technical", "alternate")


@dataclass
class MultiQueryCandidate:
    chunk_id: int
    title: str
    source_path: str
    chunk_index: int
    content: str
    rrf_score: float = 0.0
    found_by: list[str] = field(default_factory=list)
    query_ranks: dict[str, int] = field(default_factory=dict)
    vector_similarity: float | None = None
    lexical_score: float | None = None

    @property
    def similarity(self):
        return self.vector_similarity


@dataclass
class MultiQueryResult:
    query_variants: list[str]
    candidate_lists: list
    fused_candidates: list
    final_chunks: list
    transformation_seconds: float
    retrieval_seconds: float
    rerank_seconds: float


def cross_query_rrf(named_lists, rrf_k=RRF_K, top_k=HYBRID_CANDIDATE_K):
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    fused = {}
    for name, candidates in named_lists:
        for rank, chunk in enumerate(candidates, 1):
            candidate = fused.get(chunk.chunk_id)
            if candidate is None:
                candidate = MultiQueryCandidate(
                    chunk_id=chunk.chunk_id, title=chunk.title,
                    source_path=chunk.source_path, chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    vector_similarity=getattr(chunk, "similarity", None),
                    lexical_score=getattr(chunk, "lexical_score", None),
                )
                fused[chunk.chunk_id] = candidate
            candidate.rrf_score += 1 / (rrf_k + rank)
            candidate.found_by.append(name)
            candidate.query_ranks[name] = rank
    return sorted(fused.values(), key=lambda item: (-item.rrf_score, item.chunk_id))[:top_k]


class MultiQueryRetriever:
    def __init__(
        self, hybrid_retriever=None, reranker=None, use_reranker=False,
        variant_count=QUERY_VARIANTS,
    ):
        if variant_count not in {1, 2, 3}:
            raise ValueError("variant_count must be between one and three")
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.variant_count = variant_count
        self.use_reranker = use_reranker
        self.reranker = (reranker or Reranker()) if use_reranker else None

    def retrieve(self, original, variants: QueryVariants, strategy="multi_query", final_k=5):
        mapping = {
            "original": [("original", variants.original)],
            "rewrite_only": [("semantic", variants.semantic)],
            "original_rewrite": [
                ("original", variants.original), ("semantic", variants.semantic)
            ],
            "multi_query": list(zip(
                VARIANT_NAMES[:self.variant_count + 1],
                variants.all()[:self.variant_count + 1],
            )),
        }
        if strategy not in mapping:
            raise ValueError(f"Unknown query strategy: {strategy}")
        started = perf_counter()
        named_lists = []
        for name, query in mapping[strategy]:
            result = self.hybrid_retriever.search(
                query, vector_k=20, lexical_k=20, fused_k=20, rrf_k=RRF_K
            )
            named_lists.append((name, result.fused_results))
        retrieval_seconds = perf_counter() - started
        fused = cross_query_rrf(named_lists, top_k=20)
        rerank_seconds = 0.0
        if self.use_reranker:
            started = perf_counter()
            ranked = self.reranker.rerank(original, fused, top_k=final_k)
            final = [item.chunk for item in ranked]
            rerank_seconds = perf_counter() - started
        else:
            final = fused[:final_k]
        return MultiQueryResult(
            query_variants=[query for _, query in mapping[strategy]],
            candidate_lists=[items for _, items in named_lists],
            fused_candidates=fused, final_chunks=final,
            transformation_seconds=0.0, retrieval_seconds=retrieval_seconds,
            rerank_seconds=rerank_seconds,
        )
