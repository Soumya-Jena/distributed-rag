"""Inspect transformed-query retrieval with provenance."""

import argparse
from time import perf_counter

from src.multi_query_retriever import MultiQueryRetriever
from src.query_transformer import QueryTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument(
        "--strategy",
        choices=["original", "rewrite_only", "original_rewrite", "multi_query"],
        default="multi_query",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--reranker", action="store_true")
    args = parser.parse_args()
    transformer = QueryTransformer()
    started = perf_counter()
    variants = transformer.transform(args.query)
    transformation_seconds = perf_counter() - started
    result = MultiQueryRetriever(use_reranker=args.reranker).retrieve(
        args.query, variants, strategy=args.strategy, final_k=args.top_k
    )
    print("\nQUERY VARIANTS")
    for name, query in zip(
        ("original", "semantic", "technical", "alternate"), variants.all()
    ):
        print(f"{name:10}: {query}")
    print("\nRESULTS")
    for rank, chunk in enumerate(result.final_chunks, 1):
        found = ", ".join(chunk.found_by)
        ranks = ", ".join(f"{name}={value}" for name, value in chunk.query_ranks.items())
        print(
            f"#{rank} {chunk.title} chunk={chunk.chunk_index} "
            f"cross_query_rrf={chunk.rrf_score:.6f} found_by={found} ranks={ranks}"
        )
    print(f"\nTransformation: {transformation_seconds:.3f}s")
    print(f"Retrieval: {result.retrieval_seconds:.3f}s")
    print(f"Reranking: {result.rerank_seconds:.3f}s")


if __name__ == "__main__":
    main()
