import argparse

from src.retriever import Retriever


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
        type=str,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    retriever = Retriever()

    results = retriever.search(
        args.query,
        args.top_k,
    )

    print("\nQUERY")
    print("-----")
    print(args.query)

    print("\nRESULTS")
    print("-------")

    for rank, chunk in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n#{rank}"
            f"\nSimilarity : {chunk.similarity:.4f}"
            f"\nDocument   : {chunk.title}"
            f"\nChunk      : {chunk.chunk_index}"
            f"\nSource     : {chunk.source_path}"
        )

        print()

        print(
            chunk.content[:700]
        )

        print(
            "\n" + "-" * 70
        )


if __name__ == "__main__":
    main()