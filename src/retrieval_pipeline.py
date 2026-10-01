from dataclasses import dataclass
from time import perf_counter

from src.config import (
    BULKHEAD_ACQUIRE_TIMEOUT_SECONDS,
    CACHE_ENABLED,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_RECOVERY_SECONDS,
    DEPENDENCY_RETRY_ATTEMPTS,
    HYBRID_CANDIDATE_K,
    LEXICAL_CANDIDATE_K,
    RERANK_TOP_K,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_MODE,
    RRF_K,
    RERANK_BULKHEAD_SLOTS,
    RETRY_BASE_DELAY_SECONDS,
)
from src.fault_injection import FaultInjector
from src.fusion import reciprocal_rank_fusion
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.retriever import RetrievedChunk, Retriever
from src.cache import Cache, CacheStats
from src.retrieval_cache import RetrievalCache
from src.tracing import observed_span
from src.resilience import (
    Bulkhead,
    CircuitBreaker,
    ResilienceState,
    RetrievalUnavailableError,
    ServiceMode,
    TransientDependencyError,
    retry_transient,
)


def _database_call(operation):
    """Retry only PostgreSQL connection-class failures, never arbitrary errors."""
    try:
        return operation()
    except Exception as error:
        sqlstate = getattr(error, "sqlstate", None)
        is_connection_error = (
            isinstance(sqlstate, str) and sqlstate.startswith("08")
        ) or (
            sqlstate is None
            and type(error).__name__ in {"OperationalError", "ConnectionTimeout"}
        )
        if is_connection_error:
            raise TransientDependencyError(str(error)) from error
        raise


def _retry_database(operation):
    return retry_transient(
        lambda: _database_call(operation),
        attempts=DEPENDENCY_RETRY_ATTEMPTS,
        base_delay=RETRY_BASE_DELAY_SECONDS,
    )


@dataclass
class RetrievalResult:
    candidate_chunks: list
    final_chunks: list
    vector_seconds: float
    rerank_seconds: float
    lexical_seconds: float = 0.0
    fusion_seconds: float = 0.0
    reranked_chunks: list | None = None
    cache_stats: CacheStats = CacheStats()
    corpus_version: int | None = None
    service_modes: list = None
    degraded_components: list = None


class RetrievalPipeline:
    def __init__(
        self,
        use_reranker=True,
        retrieval_mode=RETRIEVAL_MODE,
        retriever=None,
        hybrid_retriever=None,
        reranker=None,
        cache=None,
        fault_injector=None,
        vector_breaker=None,
        lexical_breaker=None,
        rerank_bulkhead=None,
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
        self.cache = cache or Cache(
            enabled=(
                CACHE_ENABLED
                and retriever is None
                and hybrid_retriever is None
            )
        )
        self.result_cache = RetrievalCache(
            self.cache, retrieval_mode, use_reranker
        )
        self.faults = fault_injector or FaultInjector()
        breaker_options = {
            "failure_threshold": CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            "recovery_seconds": CIRCUIT_BREAKER_RECOVERY_SECONDS,
        }
        self.vector_breaker = vector_breaker or CircuitBreaker(
            **breaker_options, name="vector_retrieval"
        )
        self.lexical_breaker = lexical_breaker or CircuitBreaker(
            **breaker_options, name="lexical_retrieval"
        )
        self.rerank_bulkhead = rerank_bulkhead or Bulkhead(
            RERANK_BULKHEAD_SLOTS,
            BULKHEAD_ACQUIRE_TIMEOUT_SECONDS,
            "reranker",
        )

    def retrieve(
        self,
        query,
        candidate_k=RETRIEVAL_CANDIDATE_K,
        final_k=RERANK_TOP_K,
    ):
        state = ResilienceState()
        with observed_span("cache.retrieval_lookup", {
            "cache.name": "retrieval",
        }) as cache_span:
            cached, cache_stats, cache_key, corpus_version = self.result_cache.get(
                query, candidate_k, final_k
            )
            cache_span.set_attribute("cache.hit", cached is not None)
        if cached is not None:
            candidates, final = cached
            return RetrievalResult(
                candidate_chunks=candidates,
                final_chunks=final,
                vector_seconds=0.0,
                rerank_seconds=0.0,
                lexical_seconds=0.0,
                fusion_seconds=0.0,
                cache_stats=cache_stats,
                corpus_version=corpus_version,
                service_modes=[],
                degraded_components=[],
            )

        lexical_seconds = 0.0
        fusion_seconds = 0.0
        if self.retrieval_mode == "hybrid":
            vector_candidates = []
            lexical_candidates = []
            vector_available = False
            lexical_available = False
            started = perf_counter()
            try:
                self.faults.apply("embedding")
                self.faults.apply("vector_retrieval")
                vector_candidates = self.vector_breaker.call(
                    lambda: _retry_database(
                        lambda: self.hybrid_retriever.vector_retriever.search(
                            query, top_k=candidate_k
                        )
                    )
                )
                vector_available = True
            except Exception:
                state.degrade(ServiceMode.DEGRADED_LEXICAL_ONLY, "vector_retrieval")
            vector_seconds = perf_counter() - started
            started = perf_counter()
            try:
                self.faults.apply("lexical_retrieval")
                lexical_candidates = self.lexical_breaker.call(
                    lambda: _retry_database(
                        lambda: self.hybrid_retriever.lexical_retriever.search(
                            query, top_k=LEXICAL_CANDIDATE_K
                        )
                    )
                )
                lexical_available = True
            except Exception:
                state.degrade(ServiceMode.DEGRADED_VECTOR_ONLY, "lexical_retrieval")
            lexical_seconds = perf_counter() - started
            if not vector_available and not lexical_available:
                raise RetrievalUnavailableError("Both retrieval branches are unavailable")
            started = perf_counter()
            if vector_candidates and lexical_candidates:
                candidates = reciprocal_rank_fusion(
                    vector_candidates,
                    lexical_candidates,
                    rrf_k=RRF_K,
                    top_k=HYBRID_CANDIDATE_K,
                )
            else:
                candidates = (vector_candidates or lexical_candidates)[:HYBRID_CANDIDATE_K]
            fusion_seconds = perf_counter() - started
        else:
            started = perf_counter()
            try:
                self.faults.apply("embedding")
                self.faults.apply("vector_retrieval")
                candidates = self.vector_breaker.call(
                    lambda: _retry_database(
                        lambda: self.retriever.search(query, top_k=candidate_k)
                    )
                )
            except Exception as exc:
                raise RetrievalUnavailableError("Vector retrieval is unavailable") from exc
            vector_seconds = perf_counter() - started

        if not self.use_reranker:
            result = RetrievalResult(
                candidate_chunks=candidates,
                final_chunks=candidates[:final_k],
                vector_seconds=vector_seconds,
                rerank_seconds=0.0,
                lexical_seconds=lexical_seconds,
                fusion_seconds=fusion_seconds,
                cache_stats=cache_stats,
                corpus_version=corpus_version,
                service_modes=state.modes,
                degraded_components=state.degraded_components,
            )
            self.result_cache.set(cache_key, result.candidate_chunks, result.final_chunks)
            return result

        started = perf_counter()
        with observed_span("rerank", {"rag.candidates": len(candidates)}):
            try:
                self.faults.apply("reranker")
                reranked = self.rerank_bulkhead.call(
                    lambda: self.reranker.rerank(query, candidates, top_k=final_k)
                )
            except Exception:
                state.degrade(ServiceMode.DEGRADED_NO_RERANK, "reranker")
                reranked = None
        rerank_seconds = perf_counter() - started
        final_chunks = (
            [item.chunk for item in reranked]
            if reranked is not None else candidates[:final_k]
        )
        result = RetrievalResult(
            candidate_chunks=candidates,
            final_chunks=final_chunks,
            vector_seconds=vector_seconds,
            rerank_seconds=rerank_seconds,
            lexical_seconds=lexical_seconds,
            fusion_seconds=fusion_seconds,
            reranked_chunks=reranked,
            cache_stats=cache_stats,
            corpus_version=corpus_version,
            service_modes=state.modes,
            degraded_components=state.degraded_components,
        )
        self.result_cache.set(cache_key, result.candidate_chunks, result.final_chunks)
        return result
