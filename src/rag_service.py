from dataclasses import dataclass
from time import perf_counter

from src.generator import LocalGenerator

from src.config import (
    GROUNDING_MODE,
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
from src.security_policy import build_security_messages
from src.output_guard import guard_output


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

    prompt_tokens: int


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
                    generator=shared_generator
                )
            self.multi_query_retriever = (
                multi_query_retriever
                or MultiQueryRetriever(use_reranker=effective_reranker)
            )


    def answer(
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

        chunks = retrieval_result.final_chunks

        chunks = [chunk for chunk in chunks if self._passes_threshold(chunk)]

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
                prompt_tokens=0,
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
            prompt_tokens=(
                self.generator.last_input_token_count
            ),
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
