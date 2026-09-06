import argparse

from src.rag_service import RAGService


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "question"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    reranker_group = parser.add_mutually_exclusive_group()
    reranker_group.add_argument(
        "--reranker",
        dest="use_reranker",
        action="store_true",
        help="Enable cross-encoder reranking",
    )
    reranker_group.add_argument(
        "--no-reranker",
        dest="use_reranker",
        action="store_false",
        help="Use vector retrieval without cross-encoder reranking",
    )
    parser.set_defaults(use_reranker=None)

    args = parser.parse_args()

    rag = RAGService(use_reranker=args.use_reranker)

    result = rag.answer(
        args.question,
        args.top_k,
    )

    print("\nANSWER")
    print("======")

    print(result.answer)

    print("\nSOURCES")
    print("=======")

    for index, chunk in enumerate(
        result.chunks,
        start=1,
    ):

        print(
            f"[S{index}] "
            f"{chunk.title} | "
            f"chunk={chunk.chunk_index} | "
            f"similarity="
            f"{chunk.similarity:.4f}"
        )

    print("\nPERFORMANCE")
    print("===========")

    print(
        f"Vector     : "
        f"{result.vector_seconds:.3f}s"
    )

    print(
        f"Reranking  : "
        f"{result.rerank_seconds:.3f}s"
    )

    print(
        f"Generation : "
        f"{result.generation_seconds:.3f}s"
    )

    print(
        f"Prompt tokens: "
        f"{result.prompt_tokens}"
    )


if __name__ == "__main__":
    main()
