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
        help="Disable cross-encoder reranking",
    )
    parser.set_defaults(use_reranker=None)

    parser.add_argument(
        "--retrieval-mode",
        choices=["vector", "hybrid"],
        default=None,
        help="Choose dense vector retrieval or hybrid vector + lexical retrieval",
    )

    args = parser.parse_args()

    rag_options = {"use_reranker": args.use_reranker}
    if args.retrieval_mode is not None:
        rag_options["retrieval_mode"] = args.retrieval_mode
    rag = RAGService(**rag_options)

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

        score_parts = []
        similarity = getattr(chunk, "similarity", None)
        if similarity is not None:
            score_parts.append(f"vector={similarity:.4f}")
        lexical_score = getattr(chunk, "lexical_score", None)
        if lexical_score is not None:
            score_parts.append(f"lexical={lexical_score:.4f}")
        rrf_score = getattr(chunk, "rrf_score", None)
        if rrf_score is not None:
            score_parts.append(f"rrf={rrf_score:.6f}")

        print(
            f"[S{index}] "
            f"{chunk.title} | "
            f"chunk={chunk.chunk_index} | "
            f"{' | '.join(score_parts)}"
        )

    print("\nPERFORMANCE")
    print("===========")

    print(
        f"Vector     : "
        f"{result.vector_seconds:.3f}s"
    )

    print(
        f"Lexical    : "
        f"{result.lexical_seconds:.3f}s"
    )

    print(
        f"Fusion     : "
        f"{result.fusion_seconds:.3f}s"
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
