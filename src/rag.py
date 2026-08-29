import argparse

from time import perf_counter

from src.generator import LocalGenerator
from src.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from src.retriever import Retriever


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "question",
        type=str,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
    )

    args = parser.parse_args()

    # ----------------------------------
    # Step 1: Retrieval
    # ----------------------------------

    retriever = Retriever()

    retrieval_start = perf_counter()

    chunks = retriever.search(
        args.question,
        args.top_k,
    )

    retrieval_seconds = (
        perf_counter()
        - retrieval_start
    )

    if not chunks:

        print(
            "No relevant chunks were retrieved."
        )

        return

    # ----------------------------------
    # Step 2: Prompt construction
    # ----------------------------------

    user_prompt = build_user_prompt(
        args.question,
        chunks,
    )

    if args.show_context:

        print(
            "\nRETRIEVED CONTEXT"
        )

        print(
            "================="
        )

        print(
            user_prompt
        )

    # ----------------------------------
    # Step 3: Generation
    # ----------------------------------

    generator = LocalGenerator()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    print(
        "\n\nANSWER"
    )

    print(
        "======"
    )

    generation_start = perf_counter()

    answer_parts = []

    for text_fragment in generator.generate_stream(
        messages
    ):

        print(
            text_fragment,
            end="",
            flush=True,
        )

        answer_parts.append(
            text_fragment
        )

    answer = "".join(
        answer_parts
    ).strip()

    generation_seconds = (
        perf_counter()
        - generation_start
    )

    print()

    # ----------------------------------
    # Sources and performance
    # ----------------------------------

    print(
        "\n\nSOURCES"
    )

    print(
        "======="
    )

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        print(
            f"[S{index}] "
            f"{chunk.title} | "
            f"chunk={chunk.chunk_index} | "
            f"similarity={chunk.similarity:.4f}"
        )

        print(
            f"     {chunk.source_path}"
        )

    print(
        "\n\nPERFORMANCE"
    )

    print(
        "==========="
    )

    print(
        f"Retrieval  : "
        f"{retrieval_seconds:.3f} sec"
    )

    print(
        f"Generation : "
        f"{generation_seconds:.3f} sec"
    )


if __name__ == "__main__":
    main()
