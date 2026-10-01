from dataclasses import dataclass, field
from time import perf_counter

from src.generator import LocalGenerator

from src.config import (
    GROUNDING_MODE,
    CONTEXT_DEDUP_THRESHOLD,
    CONTEXT_KEEP_RATIO,
    CONTEXT_NEIGHBOR_WINDOW,
    CONTEXT_STRATEGY,
    CONTEXT_TOKEN_BUDGET,
    GENERATION_MODEL,
    RESPONSE_CACHE_ENABLED,
    MIN_RETRIEVAL_SIMILARITY,
    RETRIEVAL_MODE,
    SECURITY_MODE,
    QUERY_TRANSFORM_ENABLED,
    QUERY_TRANSFORM_MODEL,
    USE_RERANKER,
    OBSERVABILITY_ENABLED,
    BULKHEAD_ACQUIRE_TIMEOUT_SECONDS,
    GENERATION_BULKHEAD_SLOTS,
    REQUEST_DEADLINE_SECONDS,
)

from src.prompt import (
    build_user_prompt,
    get_system_prompt,
)

from src.retriever import RetrievedChunk
from src.retrieval_pipeline import RetrievalPipeline
from src.multi_query_retriever import MultiQueryRetriever
from src.query_transformer import QueryTransformer
from src.context_optimizer import ContextOptimizer, STRATEGIES
from src.security_policy import build_security_messages
from src.output_guard import guard_output
from src.cache import Cache
from src.response_cache import ResponseCache
from src.grounding import extract_evidence_status
from src.runtime_metrics import (
    COMPRESSION_RATIO, EMPTY_RETRIEVAL, ERRORS, EVIDENCE_STATUS,
    FINAL_CONTEXT_CHUNKS, OBSERVABILITY_OVERHEAD, OUTPUT_TOKENS_TOTAL, REFUSALS,
    REQUEST_DURATION, REQUESTS, RETRIEVAL_CANDIDATES, STAGE_DURATION,
    SECURITY_EVENTS, TOKENS, bounded_error_type, record_cache_stats,
)
from src.structured_logging import log_event, query_metadata
from src.tracing import observed_span, telemetry_enabled, trace_id
from src.fault_injection import FaultInjector
from src.resilience import (
    Bulkhead,
    Deadline,
    GenerationUnavailableError,
    ResilienceState,
    SecurityUnavailableError,
    ServiceMode,
)
from src.context_optimizer import ContextOptimizationResult


@dataclass
class RAGResult:

    question: str

    answer: str

    chunks: list

    retrieval_seconds: float

    transformation_seconds: float

    vector_seconds: float

    rerank_seconds: float

    lexical_seconds: float

    fusion_seconds: float

    generation_seconds: float

    optimization_seconds: float

    original_context_tokens: int

    context_tokens: int

    output_tokens: int

    compression_ratio: float

    prompt_tokens: int

    retrieval_candidate_count: int

    cache_stats: dict = field(default_factory=dict)

    response_cache_hit: bool = False

    trace_id: str = ""

    status: str = "normal"

    service_modes: list = field(default_factory=list)

    degraded_components: list = field(default_factory=list)


class RAGService:

    def __init__(
        self,
        use_reranker=None,
        retrieval_mode=RETRIEVAL_MODE,
        retrieval_pipeline=None,
        generator=None,
        grounding_mode=GROUNDING_MODE,
        security_mode=SECURITY_MODE,
        query_strategy="original",
        query_transformer=None,
        multi_query_retriever=None,
        context_strategy=CONTEXT_STRATEGY,
        context_optimizer=None,
        cache=None,
        response_cache_enabled=RESPONSE_CACHE_ENABLED,
        observability_enabled=OBSERVABILITY_ENABLED,
        fault_injector=None,
        generation_bulkhead=None,
        request_deadline_seconds=REQUEST_DEADLINE_SECONDS,
        fallback_retrieval_pipeline=None,
    ):

        print(
            "Initializing RAG service..."
        )

        self.grounding_mode = grounding_mode
        self.system_prompt = get_system_prompt(grounding_mode)
        if security_mode not in {"baseline", "structured", "layered"}:
            raise ValueError(f"Unknown security mode: {security_mode}")
        if security_mode != "baseline" and grounding_mode != "strict":
            raise ValueError("Structured security modes require strict grounding")
        self.security_mode = security_mode
        self.generator = generator or LocalGenerator()
        if query_strategy not in {
            "original", "rewrite_only", "original_rewrite", "multi_query"
        }:
            raise ValueError(f"Unknown query strategy: {query_strategy}")
        if query_strategy != "original" and not QUERY_TRANSFORM_ENABLED:
            raise ValueError("Query transformation is disabled by configuration")
        self.query_strategy = query_strategy
        if context_strategy not in STRATEGIES:
            raise ValueError(f"Unknown context strategy: {context_strategy}")
        self.context_strategy = context_strategy
        self.cache = cache or Cache()
        self.response_cache_enabled = response_cache_enabled
        self.observability_enabled = observability_enabled
        self.faults = fault_injector or FaultInjector()
        self.request_deadline_seconds = request_deadline_seconds
        self.generation_bulkhead = generation_bulkhead or Bulkhead(
            GENERATION_BULKHEAD_SLOTS,
            BULKHEAD_ACQUIRE_TIMEOUT_SECONDS,
            "generation",
        )
        effective_reranker = USE_RERANKER if use_reranker is None else use_reranker
        self._fallback_retrieval_options = {
            "use_reranker": effective_reranker,
            "retrieval_mode": retrieval_mode,
        }
        self.retrieval_pipeline = None
        self.query_transformer = query_transformer
        self.multi_query_retriever = multi_query_retriever
        if query_strategy == "original":
            self.retrieval_pipeline = retrieval_pipeline or RetrievalPipeline(
                use_reranker=effective_reranker,
                retrieval_mode=retrieval_mode,
            )
        else:
            if query_transformer is None:
                shared_generator = (
                    self.generator
                    if getattr(self.generator, "model_name", None)
                    == QUERY_TRANSFORM_MODEL else None
                )
                self.query_transformer = QueryTransformer(
                    generator=shared_generator,
                    cache=self.cache,
                )
            self.multi_query_retriever = (
                multi_query_retriever
                or MultiQueryRetriever(use_reranker=effective_reranker)
            )
        self.fallback_retrieval_pipeline = fallback_retrieval_pipeline

        if context_optimizer is None:
            if context_strategy == "full":
                embedding_service = None
            elif query_strategy == "original":
                embedding_service = (
                    self.retrieval_pipeline.retriever.embedding_service
                )
            else:
                embedding_service = (
                    self.multi_query_retriever.hybrid_retriever
                    .vector_retriever.embedding_service
                )
            token_counter = getattr(
                self.generator, "count_tokens",
                lambda text: len(text.split()),
            )
            self.token_counter = token_counter
            context_optimizer = ContextOptimizer(
                embedding_service=embedding_service,
                token_counter=token_counter,
                dedup_threshold=CONTEXT_DEDUP_THRESHOLD,
                keep_ratio=CONTEXT_KEEP_RATIO,
                token_budget=CONTEXT_TOKEN_BUDGET,
                neighbor_window=CONTEXT_NEIGHBOR_WINDOW,
            )
        self.context_optimizer = context_optimizer
        if not hasattr(self, "token_counter"):
            self.token_counter = getattr(
                self.generator, "count_tokens", lambda text: len(text.split())
            )
        retrieval_fingerprint = (
            self.retrieval_pipeline.result_cache.config_fingerprint
            if self.retrieval_pipeline is not None
            and hasattr(self.retrieval_pipeline, "result_cache")
            else "multi-query"
        )
        self.response_cache = ResponseCache(
            self.cache, retrieval_fingerprint, context_strategy,
            query_strategy, grounding_mode, security_mode,
            generator_model=getattr(
                self.generator, "model_name", GENERATION_MODEL
            ),
        )


    def answer(self, question: str, top_k: int = 5):
        if not self.observability_enabled:
            with telemetry_enabled(False):
                return self._answer_impl(question, top_k)

        started = perf_counter()
        metadata = query_metadata(question)
        with observed_span("rag.request", {
            "rag.top_k": top_k,
            "rag.query_length": metadata["query_length"],
        }) as span:
            request_trace_id = trace_id()
            span.set_attribute("rag.query_hash", metadata["query_hash"])
            log_event("request_started", top_k=top_k, **metadata)
            try:
                result = self._answer_impl(question, top_k)
                result.trace_id = request_trace_id
                telemetry_started = perf_counter()
                try:
                    self.faults.apply("observability")
                    self._record_result(result)
                    OBSERVABILITY_OVERHEAD.observe(perf_counter() - telemetry_started)
                except Exception:
                    # Telemetry is intentionally fail-open: an exporter or metric
                    # failure must never convert a valid grounded answer into a 500.
                    pass
                span.set_attribute("rag.context_chunks", len(result.chunks))
                span.set_attribute("rag.response_cache_hit", result.response_cache_hit)
                span.add_event("request.completed", {
                    "evidence.status": extract_evidence_status(result.answer) or "missing"
                })
                REQUESTS.labels(status="success").inc()
                log_event(
                    "request_completed", context_chunks=len(result.chunks),
                    response_cache_hit=result.response_cache_hit,
                    duration_seconds=round(perf_counter() - started, 6),
                )
                return result
            except Exception as error:
                REQUESTS.labels(status="error").inc()
                ERRORS.labels(stage="request", error_type=bounded_error_type(error)).inc()
                log_event("request_failed", level=40, error_type=bounded_error_type(error))
                raise
            finally:
                REQUEST_DURATION.observe(perf_counter() - started)

    def _record_result(self, result):
        RETRIEVAL_CANDIDATES.observe(result.retrieval_candidate_count)
        FINAL_CONTEXT_CHUNKS.observe(len(result.chunks))
        if not result.chunks:
            EMPTY_RETRIEVAL.inc()
        for stage, seconds in {
            "query_transform": result.transformation_seconds,
            "vector_search": result.vector_seconds,
            "lexical_search": result.lexical_seconds,
            "rrf_fusion": result.fusion_seconds,
            "rerank": result.rerank_seconds,
            "context_optimize": result.optimization_seconds,
            "generation": result.generation_seconds,
        }.items():
            STAGE_DURATION.labels(stage=stage).observe(max(0.0, seconds))
        TOKENS.labels(kind="prompt").observe(result.prompt_tokens)
        TOKENS.labels(kind="context").observe(result.context_tokens)
        TOKENS.labels(kind="output").observe(result.output_tokens)
        if not result.response_cache_hit:
            OUTPUT_TOKENS_TOTAL.inc(max(0, result.output_tokens))
        COMPRESSION_RATIO.observe(max(0.0, min(1.0, result.compression_ratio)))
        record_cache_stats(result.cache_stats)
        status = extract_evidence_status(result.answer) or "missing"
        EVIDENCE_STATUS.labels(status=status.lower()).inc()
        if status == "INSUFFICIENT":
            REFUSALS.inc()

    def _answer_impl(
        self,
        question: str,
        top_k: int = 5,
    ):

        if not self.response_cache_enabled:
            return self._answer_uncached(question, top_k)

        key = self.response_cache.key(question, top_k)
        with observed_span("cache.response_lookup", {
            "cache.name": "response",
        }) as cache_span:
            value, chunks, stats = self.response_cache.get(key)
            cache_span.set_attribute("cache.hit", value is not None)
            cache_span.add_event(
                "cache.decision", {"cache.result": "hit" if value is not None else "miss"}
            )
        if value is not None:
            value["question"] = question
            value["chunks"] = chunks
            value["response_cache_hit"] = True
            value.setdefault("retrieval_candidate_count", len(chunks))
            value.setdefault("cache_stats", {})["response"] = stats.to_dict()
            return RAGResult(**value)
        if not stats.available:
            result = self._answer_uncached(question, top_k)
            result.cache_stats["response"] = stats.to_dict()
            return result

        with self.cache.lock(f"lock:{key}") as acquired:
            if not acquired:
                value, chunks, waited_stats = self.response_cache.wait_for_fill(key)
                if value is not None:
                    value["question"] = question
                    value["chunks"] = chunks
                    value["response_cache_hit"] = True
                    value.setdefault("retrieval_candidate_count", len(chunks))
                    value.setdefault("cache_stats", {})["response"] = (
                        waited_stats.to_dict()
                    )
                    return RAGResult(**value)
                return self._answer_uncached(question, top_k)

            # Double-check after acquiring the lock: another request may have
            # populated the entry between our initial miss and SET-NX.
            value, chunks, locked_stats = self.response_cache.get(key)
            if value is not None:
                value["question"] = question
                value["chunks"] = chunks
                value["response_cache_hit"] = True
                value.setdefault("retrieval_candidate_count", len(chunks))
                value.setdefault("cache_stats", {})["response"] = (
                    locked_stats.to_dict()
                )
                return RAGResult(**value)

            result = self._answer_uncached(question, top_k)
            result.cache_stats["response"] = stats.to_dict()
            if self.response_cache.eligible(result):
                self.response_cache.set(key, result)
            return result


    def _answer_uncached(
        self,
        question: str,
        top_k: int = 5,
    ):

        deadline = Deadline(self.request_deadline_seconds)
        state = ResilienceState()

        # -------------------------
        # Retrieval
        # -------------------------

        transformation_seconds = 0.0
        if self.query_strategy == "original":
            deadline.require("retrieval")
            with observed_span("retrieval", {"rag.final_k": top_k}):
                retrieval_result = self.retrieval_pipeline.retrieve(
                    question,
                    final_k=top_k,
                )
            vector_seconds = retrieval_result.vector_seconds
            lexical_seconds = retrieval_result.lexical_seconds
            fusion_seconds = retrieval_result.fusion_seconds
            cache_stat = getattr(retrieval_result, "cache_stats", None)
            retrieval_cache_stats = (
                cache_stat.to_dict() if cache_stat is not None else {}
            )
            embedding_service = getattr(
                getattr(self.retrieval_pipeline, "retriever", None),
                "embedding_service", None,
            )
            embedding_cache_stats = getattr(
                embedding_service, "last_cache_stats", None
            )
        else:
            started = perf_counter()
            deadline.require("query transformation")
            try:
                self.faults.apply("query_transform")
                with observed_span("query.transform"):
                    variants = self.query_transformer.transform(question)
            except Exception:
                state.degrade(ServiceMode.DEGRADED_QUERY, "query_transform")
                deadline.require("fallback retrieval")
                if self.fallback_retrieval_pipeline is None:
                    self.fallback_retrieval_pipeline = RetrievalPipeline(
                        **self._fallback_retrieval_options
                    )
                retrieval_result = self.fallback_retrieval_pipeline.retrieve(
                    question, final_k=top_k
                )
                transformation_seconds = perf_counter() - started
                vector_seconds = retrieval_result.vector_seconds
                lexical_seconds = retrieval_result.lexical_seconds
                fusion_seconds = retrieval_result.fusion_seconds
                retrieval_cache_stats = {}
                embedding_cache_stats = None
            else:
                transformation_seconds = perf_counter() - started
                deadline.require("retrieval")
                with observed_span("retrieval", {"rag.final_k": top_k}):
                    retrieval_result = self.multi_query_retriever.retrieve(
                        question, variants, strategy=self.query_strategy, final_k=top_k
                    )
                vector_seconds = 0.0
                lexical_seconds = 0.0
                fusion_seconds = retrieval_result.retrieval_seconds
                retrieval_cache_stats = getattr(retrieval_result, "cache_stats", {})
                embedding_cache_stats = None

        retrieval_modes = getattr(retrieval_result, "service_modes", []) or []
        retrieval_components = (
            getattr(retrieval_result, "degraded_components", []) or []
        )
        for index, mode in enumerate(retrieval_modes):
            component = (
                retrieval_components[index]
                if index < len(retrieval_components) else mode.value
            )
            state.degrade(mode, component, record=False)
        for component in retrieval_components:
            if component not in state.degraded_components:
                state.degraded_components.append(component)

        candidate_chunks = getattr(
            retrieval_result, "candidate_chunks",
            getattr(retrieval_result, "fused_candidates", []),
        )
        chunks = retrieval_result.final_chunks

        chunks = [chunk for chunk in chunks if self._passes_threshold(chunk)]

        with observed_span("context.optimize", {
            "rag.input_chunks": len(chunks),
            "rag.strategy": self.context_strategy,
        }):
            try:
                deadline.require("context optimization")
                self.faults.apply("context_optimizer")
                context_result = self.context_optimizer.optimize(
                    question, chunks, strategy=self.context_strategy
                )
            except Exception:
                state.degrade(ServiceMode.DEGRADED_RAW_CONTEXT, "context_optimizer")
                tokens = sum(self.token_counter(chunk.content) for chunk in chunks)
                context_result = ContextOptimizationResult(
                    chunks=chunks,
                    original_tokens=tokens,
                    optimized_tokens=tokens,
                    removed_tokens=0,
                    compression_ratio=1.0,
                    optimization_seconds=0.0,
                    candidates_before=len(chunks),
                    candidates_after=len(chunks),
                    unique_sources=len({chunk.source_path for chunk in chunks}),
                )
        chunks = context_result.chunks

        retrieval_seconds = (
            vector_seconds + retrieval_result.rerank_seconds
            + lexical_seconds + fusion_seconds
        )

        if not chunks:

            return RAGResult(
                question=question,
                answer=self._insufficient_answer(),
                chunks=[],
                retrieval_seconds=(
                    retrieval_seconds
                ),
                transformation_seconds=transformation_seconds,
                vector_seconds=vector_seconds,
                rerank_seconds=retrieval_result.rerank_seconds,
                lexical_seconds=lexical_seconds,
                fusion_seconds=fusion_seconds,
                generation_seconds=0.0,
                optimization_seconds=context_result.optimization_seconds,
                original_context_tokens=context_result.original_tokens,
                context_tokens=context_result.optimized_tokens,
                output_tokens=0,
                compression_ratio=context_result.compression_ratio,
                prompt_tokens=0,
                retrieval_candidate_count=len(candidate_chunks),
                cache_stats={
                    "retrieval": retrieval_cache_stats,
                    "embedding": (
                        embedding_cache_stats.to_dict()
                        if embedding_cache_stats else {}
                    ),
                },
                status=state.status,
                service_modes=[mode.value for mode in state.modes],
                degraded_components=state.degraded_components,
            )

        # -------------------------
        # Prompt
        # -------------------------

        if self.security_mode == "baseline":
            prompt = build_user_prompt(question, chunks)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            with observed_span("security.pre_generation", {
                "security.mode": self.security_mode,
            }) as security_span:
                try:
                    self.faults.apply("security_validator")
                    messages, security_metadata = build_security_messages(
                        question, chunks, self.security_mode
                    )
                except Exception as exc:
                    raise SecurityUnavailableError(
                        "Security validation is unavailable; failing closed"
                    ) from exc
                security_span.add_event("security.checked", {
                    "security.mode": self.security_mode,
                    "security.flagged_chunks": sum(
                        1 for item in security_metadata if item.suspicious
                    ),
                })
                flagged_count = sum(
                    1 for item in security_metadata if item.suspicious
                )
                SECURITY_EVENTS.labels(
                    decision="flagged" if flagged_count else "clean"
                ).inc()

        # -------------------------
        # Generation
        # -------------------------

        start = perf_counter()

        deadline.require("generation")
        with observed_span("generation"):
            try:
                self.faults.apply("generator")
                answer = self.generation_bulkhead.call(
                    lambda: self.generator.generate(messages)
                )
            except Exception as exc:
                raise GenerationUnavailableError("Generation is unavailable") from exc
        if self.security_mode == "layered":
            with observed_span("output.validate"):
                try:
                    self.faults.apply("output_validator")
                    answer, validation = guard_output(answer)
                except Exception as exc:
                    raise SecurityUnavailableError(
                        "Output validation is unavailable; failing closed"
                    ) from exc
                SECURITY_EVENTS.labels(
                    decision="allowed" if validation.safe else "blocked"
                ).inc()

        generation_seconds = (
            perf_counter()
            - start
        )

        return RAGResult(
            question=question,
            answer=answer,
            chunks=chunks,
            retrieval_seconds=(
                retrieval_seconds
            ),
            transformation_seconds=transformation_seconds,
            vector_seconds=vector_seconds,
            rerank_seconds=retrieval_result.rerank_seconds,
            lexical_seconds=lexical_seconds,
            fusion_seconds=fusion_seconds,
            generation_seconds=(
                generation_seconds
            ),
            optimization_seconds=context_result.optimization_seconds,
            original_context_tokens=context_result.original_tokens,
            context_tokens=context_result.optimized_tokens,
            output_tokens=getattr(self.generator, "last_output_token_count", 0),
            compression_ratio=context_result.compression_ratio,
            prompt_tokens=(
                self.generator.last_input_token_count
            ),
            retrieval_candidate_count=len(candidate_chunks),
            cache_stats={
                "retrieval": retrieval_cache_stats,
                "embedding": (
                    embedding_cache_stats.to_dict()
                    if embedding_cache_stats else {}
                ),
                "query_transform": (
                    self.query_transformer.last_cache_stats.to_dict()
                    if self.query_transformer is not None
                    and hasattr(self.query_transformer, "last_cache_stats")
                    else {}
                ),
            },
            status=state.status,
            service_modes=[mode.value for mode in state.modes],
            degraded_components=state.degraded_components,
        )

    @staticmethod
    def _passes_threshold(chunk):
        lexical_score = getattr(chunk, "lexical_score", None)
        similarity = getattr(chunk, "similarity", None)
        return lexical_score is not None or (
            similarity is not None and similarity >= MIN_RETRIEVAL_SIMILARITY
        )

    def _insufficient_answer(self):
        refusal = (
            "I don't have enough information in the provided sources to "
            "answer that question."
        )
        if self.grounding_mode == "strict":
            return f"EVIDENCE_STATUS: INSUFFICIENT\n\n{refusal}"
        return refusal
