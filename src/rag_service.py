from dataclasses import dataclass
from time import perf_counter

from src.generator import LocalGenerator

from src.config import (
    GROUNDING_MODE,
    MIN_RETRIEVAL_SIMILARITY,
    RETRIEVAL_MODE,
    USE_RERANKER,
)

from src.prompt import (
    build_user_prompt,
    get_system_prompt,
)

from src.retriever import RetrievedChunk
from src.retrieval_pipeline import RetrievalPipeline


@dataclass
class RAGResult:

    question: str

    answer: str

    chunks: list

    retrieval_seconds: float

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
    ):

        print(
            "Initializing RAG service..."
        )

        self.grounding_mode = grounding_mode
        self.system_prompt = get_system_prompt(grounding_mode)

        self.retrieval_pipeline = retrieval_pipeline or RetrievalPipeline(
            use_reranker=USE_RERANKER if use_reranker is None else use_reranker,
            retrieval_mode=retrieval_mode,
        )

        self.generator = generator or LocalGenerator()


    def answer(
        self,
        question: str,
        top_k: int = 5,
    ):

        # -------------------------
        # Retrieval
        # -------------------------

        retrieval_result = self.retrieval_pipeline.retrieve(
            question,
            final_k=top_k,
        )

        chunks = retrieval_result.final_chunks

        chunks = [chunk for chunk in chunks if self._passes_threshold(chunk)]

        retrieval_seconds = (
            retrieval_result.vector_seconds + retrieval_result.rerank_seconds
            + retrieval_result.lexical_seconds + retrieval_result.fusion_seconds
        )

        if not chunks:

            return RAGResult(
                question=question,
                answer=self._insufficient_answer(),
                chunks=[],
                retrieval_seconds=(
                    retrieval_seconds
                ),
                vector_seconds=retrieval_result.vector_seconds,
                rerank_seconds=retrieval_result.rerank_seconds,
                lexical_seconds=retrieval_result.lexical_seconds,
                fusion_seconds=retrieval_result.fusion_seconds,
                generation_seconds=0.0,
                prompt_tokens=0,
            )

        # -------------------------
        # Prompt
        # -------------------------

        prompt = build_user_prompt(
            question,
            chunks,
        )

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # -------------------------
        # Generation
        # -------------------------

        start = perf_counter()

        answer = self.generator.generate(
            messages
        )

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
            vector_seconds=retrieval_result.vector_seconds,
            rerank_seconds=retrieval_result.rerank_seconds,
            lexical_seconds=retrieval_result.lexical_seconds,
            fusion_seconds=retrieval_result.fusion_seconds,
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
