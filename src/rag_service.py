from dataclasses import dataclass
from time import perf_counter

from src.generator import LocalGenerator

from src.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
)

from src.retriever import (
    RetrievedChunk,
    Retriever,
)


@dataclass
class RAGResult:

    question: str

    answer: str

    chunks: list[RetrievedChunk]

    retrieval_seconds: float

    generation_seconds: float


class RAGService:

    def __init__(self):

        print(
            "Initializing RAG service..."
        )

        self.retriever = Retriever()

        self.generator = LocalGenerator()


    def answer(
        self,
        question: str,
        top_k: int = 5,
    ):

        # -------------------------
        # Retrieval
        # -------------------------

        start = perf_counter()

        chunks = self.retriever.search(
            question,
            top_k,
        )

        retrieval_seconds = (
            perf_counter()
            - start
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
                generation_seconds=0.0,
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
            generation_seconds=(
                generation_seconds
            ),
        )