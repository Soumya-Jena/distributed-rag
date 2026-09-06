from dataclasses import dataclass
from time import perf_counter

from src.generator import LocalGenerator

from src.config import (
    MIN_RETRIEVAL_SIMILARITY,
    USE_RERANKER,
)

from src.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
)

from src.retriever import RetrievedChunk
from src.retrieval_pipeline import RetrievalPipeline


@dataclass
class RAGResult:

    question: str

    answer: str

    chunks: list[RetrievedChunk]

    retrieval_seconds: float

    vector_seconds: float

    rerank_seconds: float

    generation_seconds: float

    prompt_tokens: int


class RAGService:

    def __init__(self, use_reranker=None, retrieval_pipeline=None):

        print(
            "Initializing RAG service..."
        )

        self.retrieval_pipeline = retrieval_pipeline or RetrievalPipeline(
            use_reranker=USE_RERANKER if use_reranker is None else use_reranker
        )

        self.generator = LocalGenerator()


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

        chunks = [
            chunk
            for chunk in chunks
            if chunk.similarity
            >= MIN_RETRIEVAL_SIMILARITY
        ]

        retrieval_seconds = (
            retrieval_result.vector_seconds + retrieval_result.rerank_seconds
        )

        if not chunks:

            return RAGResult(
                question=question,
                answer=(
                    "I don't have enough "
                    "information in the "
                    "provided sources to "
                    "answer that question."
                ),
                chunks=[],
                retrieval_seconds=(
                    retrieval_seconds
                ),
                vector_seconds=retrieval_result.vector_seconds,
                rerank_seconds=retrieval_result.rerank_seconds,
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
                "content": SYSTEM_PROMPT,
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
            generation_seconds=(
                generation_seconds
            ),
            prompt_tokens=(
                self.generator.last_input_token_count
            ),
        )
