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

    cache_stats: dict = field(default_factory=dict)

    response_cache_hit: bool = False


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
        effective_reranker = USE_RERANKER if use_reranker is None else use_reranker
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
            context_optimizer = ContextOptimizer(
                embedding_service=embedding_service,
                token_counter=token_counter,
                dedup_threshold=CONTEXT_DEDUP_THRESHOLD,
                keep_ratio=CONTEXT_KEEP_RATIO,
                token_budget=CONTEXT_TOKEN_BUDGET,
                neighbor_window=CONTEXT_NEIGHBOR_WINDOW,
            )
        self.context_optimizer = context_optimizer
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


    def answer(
        self,
        question: str,
        top_k: int = 5,
    ):

        if not self.response_cache_enabled:
            return self._answer_uncached(question, top_k)

        key = self.response_cache.key(question, top_k)
        value, chunks, stats = self.response_cache.get(key)
        if value is not None:
            value["question"] = question
            value["chunks"] = chunks
            value["response_cache_hit"] = True
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

        # -------------------------
        # Retrieval
        # -------------------------

        transformation_seconds = 0.0
        if self.query_strategy == "original":
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
            variants = self.query_transformer.transform(question)
            transformation_seconds = perf_counter() - started
            retrieval_result = self.multi_query_retriever.retrieve(
                question, variants, strategy=self.query_strategy, final_k=top_k
            )
            # Multi-query retrieval measures the two RRF levels together.
            vector_seconds = 0.0
            lexical_seconds = 0.0
            fusion_seconds = retrieval_result.retrieval_seconds
            retrieval_cache_stats = getattr(
                retrieval_result, "cache_stats", {}
            )
            embedding_cache_stats = None

        chunks = retrieval_result.final_chunks

        chunks = [chunk for chunk in chunks if self._passes_threshold(chunk)]

        context_result = self.context_optimizer.optimize(
            question, chunks, strategy=self.context_strategy
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
                cache_stats={
                    "retrieval": retrieval_cache_stats,
                    "embedding": (
                        embedding_cache_stats.to_dict()
                        if embedding_cache_stats else {}
                    ),
                },
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
            messages, _ = build_security_messages(
                question, chunks, self.security_mode
            )

        # -------------------------
        # Generation
        # -------------------------

        start = perf_counter()

        answer = self.generator.generate(
            messages
        )
        if self.security_mode == "layered":
            answer, _ = guard_output(answer)

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
