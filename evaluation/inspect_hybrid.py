import argparse

from src.hybrid_retriever import HybridRetriever


DEFAULT_QUERIES = [
    "MVCC",
    "How does PostgreSQL prevent readers from blocking writers?",
    "Kafka consumer group",
]


def inspect(query, retriever):
    result = retriever.search(query)
    print(f"\nQUERY: {query}")

    print("\nVECTOR")
    print("======")
    for rank, item in enumerate(result.vector_results[:5], start=1):
        print(rank, item.title, item.chunk_index, f"{item.similarity:.4f}")

    print("\nLEXICAL")
    print("=======")
    for rank, item in enumerate(result.lexical_results[:5], start=1):
        print(rank, item.title, item.chunk_index, f"{item.lexical_score:.4f}")

    print("\nFUSED")
    print("=====")
    for rank, item in enumerate(result.fused_results[:5], start=1):
        print(
            rank,
            item.title,
            item.chunk_index,
            f"vector={item.vector_rank}",
            f"lexical={item.lexical_rank}",
            f"rrf={item.rrf_score:.6f}",
        )

    print(
        "\nLATENCY",
        f"vector={result.vector_seconds:.4f}s",
        f"lexical={result.lexical_seconds:.4f}s",
        f"fusion={result.fusion_seconds:.6f}s",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="*")
    args = parser.parse_args()

    retriever = HybridRetriever()
    queries = [" ".join(args.query)] if args.query else DEFAULT_QUERIES
    for query in queries:
        inspect(query, retriever)


if __name__ == "__main__":
    main()
