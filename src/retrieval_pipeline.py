from dataclasses import dataclass
from time import perf_counter

from src.config import RERANK_TOP_K, RETRIEVAL_CANDIDATE_K
from src.reranker import Reranker
from src.retriever import RetrievedChunk, Retriever


@dataclass
class RetrievalResult:
    candidate_chunks: list[RetrievedChunk]
    final_chunks: list[RetrievedChunk]
    vector_seconds: float
    rerank_seconds: float
    reranked_chunks: list | None = None


class RetrievalPipeline:
    def __init__(self, use_reranker=True, retriever=None, reranker=None):
        self.retriever = retriever or Retriever()
        self.use_reranker = use_reranker
        self.reranker = (reranker or Reranker()) if use_reranker else None

    def retrieve(
        self,
        query,
        candidate_k=RETRIEVAL_CANDIDATE_K,
        final_k=RERANK_TOP_K,
    ):
        started = perf_counter()
        candidates = self.retriever.search(query, top_k=candidate_k)
        vector_seconds = perf_counter() - started

        if not self.use_reranker:
            return RetrievalResult(
                candidate_chunks=candidates,
                final_chunks=candidates[:final_k],
                vector_seconds=vector_seconds,
                rerank_seconds=0.0,
            )

        started = perf_counter()
        reranked = self.reranker.rerank(query, candidates, top_k=final_k)
        rerank_seconds = perf_counter() - started
        return RetrievalResult(
            candidate_chunks=candidates,
            final_chunks=[item.chunk for item in reranked],
            vector_seconds=vector_seconds,
            rerank_seconds=rerank_seconds,
            reranked_chunks=reranked,
        )
