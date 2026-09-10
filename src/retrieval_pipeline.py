from dataclasses import dataclass
from time import perf_counter

from src.config import (
    HYBRID_CANDIDATE_K,
    LEXICAL_CANDIDATE_K,
    RERANK_TOP_K,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_MODE,
    RRF_K,
)
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.retriever import RetrievedChunk, Retriever


@dataclass
class RetrievalResult:
    candidate_chunks: list
    final_chunks: list
    vector_seconds: float
    rerank_seconds: float
    lexical_seconds: float = 0.0
    fusion_seconds: float = 0.0
    reranked_chunks: list | None = None


class RetrievalPipeline:
    def __init__(
        self,
        use_reranker=True,
        retrieval_mode=RETRIEVAL_MODE,
        retriever=None,
        hybrid_retriever=None,
        reranker=None,
    ):
        self.retriever = retriever or Retriever()
        self.retrieval_mode = retrieval_mode
        if retrieval_mode not in {"vector", "hybrid"}:
            raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")
        self.hybrid_retriever = hybrid_retriever
        if retrieval_mode == "hybrid" and hybrid_retriever is None:
            self.hybrid_retriever = HybridRetriever(vector_retriever=self.retriever)
        self.use_reranker = use_reranker
        self.reranker = (reranker or Reranker()) if use_reranker else None

    def retrieve(
        self,
        query,
        candidate_k=RETRIEVAL_CANDIDATE_K,
        final_k=RERANK_TOP_K,
    ):
        lexical_seconds = 0.0
        fusion_seconds = 0.0
        if self.retrieval_mode == "hybrid":
            hybrid = self.hybrid_retriever.search(
                query,
                vector_k=candidate_k,
                lexical_k=LEXICAL_CANDIDATE_K,
                fused_k=HYBRID_CANDIDATE_K,
                rrf_k=RRF_K,
            )
            candidates = hybrid.fused_results
            vector_seconds = hybrid.vector_seconds
            lexical_seconds = hybrid.lexical_seconds
            fusion_seconds = hybrid.fusion_seconds
        else:
            started = perf_counter()
            candidates = self.retriever.search(query, top_k=candidate_k)
            vector_seconds = perf_counter() - started

        if not self.use_reranker:
            return RetrievalResult(
                candidate_chunks=candidates,
                final_chunks=candidates[:final_k],
                vector_seconds=vector_seconds,
                rerank_seconds=0.0,
                lexical_seconds=lexical_seconds,
                fusion_seconds=fusion_seconds,
            )

        started = perf_counter()
        reranked = self.reranker.rerank(query, candidates, top_k=final_k)
        rerank_seconds = perf_counter() - started
        return RetrievalResult(
            candidate_chunks=candidates,
            final_chunks=[item.chunk for item in reranked],
            vector_seconds=vector_seconds,
            rerank_seconds=rerank_seconds,
            lexical_seconds=lexical_seconds,
            fusion_seconds=fusion_seconds,
            reranked_chunks=reranked,
        )
