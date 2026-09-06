import argparse

from evaluation.offline_retriever import OfflineRetriever
from src.retrieval_pipeline import RetrievalPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query",
        nargs="?",
        default="How does PostgreSQL streaming replication work?",
    )
    args = parser.parse_args()
    pipeline = RetrievalPipeline(retriever=OfflineRetriever(), use_reranker=True)
    result = pipeline.retrieve(args.query)

    print("\nVECTOR CANDIDATES\n=================")
    for rank, chunk in enumerate(result.candidate_chunks, start=1):
        print(rank, chunk.title, chunk.chunk_index, f"{chunk.similarity:.4f}")

    print("\nRERANKED\n========")
    for rank, item in enumerate(result.reranked_chunks or [], start=1):
        print(
            rank,
            item.chunk.title,
            item.chunk.chunk_index,
            f"original={item.original_rank}",
            f"score={item.rerank_score:.4f}",
        )


if __name__ == "__main__":
    main()
