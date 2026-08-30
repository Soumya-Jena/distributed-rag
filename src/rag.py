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

    args = parser.parse_args()

    rag = RAGService()

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
        f"Retrieval  : "
        f"{result.retrieval_seconds:.3f}s"
    )

    print(
        f"Generation : "
        f"{result.generation_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()