from dataclasses import dataclass
from time import perf_counter

from src.fusion import reciprocal_rank_fusion
from src.lexical_retriever import LexicalRetriever
from src.retriever import Retriever


@dataclass
class HybridRetrievalResult:
    vector_results: list
    lexical_results: list
    fused_results: list
    vector_seconds: float
    lexical_seconds: float
    fusion_seconds: float


class HybridRetriever:
    def __init__(self, vector_retriever=None, lexical_retriever=None):
        self.vector_retriever = vector_retriever or Retriever()
        self.lexical_retriever = lexical_retriever or LexicalRetriever()

    def search(
        self,
        query,
        vector_k=20,
        lexical_k=20,
        fused_k=20,
        rrf_k=60,
    ):
        started = perf_counter()
        vector_results = self.vector_retriever.search(query, top_k=vector_k)
        vector_seconds = perf_counter() - started

        started = perf_counter()
        lexical_results = self.lexical_retriever.search(query, top_k=lexical_k)
        lexical_seconds = perf_counter() - started

        started = perf_counter()
        fused_results = reciprocal_rank_fusion(
            vector_results,
            lexical_results,
            rrf_k=rrf_k,
            top_k=fused_k,
        )
        fusion_seconds = perf_counter() - started

        return HybridRetrievalResult(
            vector_results=vector_results,
            lexical_results=lexical_results,
            fused_results=fused_results,
            vector_seconds=vector_seconds,
            lexical_seconds=lexical_seconds,
            fusion_seconds=fusion_seconds,
        )
