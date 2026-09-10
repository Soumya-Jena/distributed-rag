import unittest

from src.reranker import RerankedChunk
from src.retrieval_pipeline import RetrievalPipeline
from src.retriever import RetrievedChunk


def chunks():
    return [
        RetrievedChunk(index, f"doc-{index}", f"doc-{index}.md", 0, str(index), 1 / index)
        for index in (1, 2, 3)
    ]


class StubRetriever:
    def search(self, query, top_k=5):
        return chunks()[:top_k]


class ReverseReranker:
    def rerank(self, query, candidates, top_k=5):
        return [
            RerankedChunk(chunk, original_rank, float(original_rank))
            for original_rank, chunk in reversed(list(enumerate(candidates, start=1)))
        ][:top_k]


class RetrievalPipelineTests(unittest.TestCase):
    def test_vector_only_preserves_candidate_order(self):
        result = RetrievalPipeline(
            use_reranker=False,
            retrieval_mode="vector",
            retriever=StubRetriever(),
        ).retrieve("query", candidate_k=3, final_k=2)

        self.assertEqual([chunk.chunk_id for chunk in result.final_chunks], [1, 2])
        self.assertEqual(result.rerank_seconds, 0.0)

    def test_reranker_can_change_final_order(self):
        result = RetrievalPipeline(
            use_reranker=True,
            retrieval_mode="vector",
            retriever=StubRetriever(),
            reranker=ReverseReranker(),
        ).retrieve("query", candidate_k=3, final_k=2)

        self.assertEqual([chunk.chunk_id for chunk in result.final_chunks], [3, 2])
        self.assertEqual(result.reranked_chunks[0].original_rank, 3)


if __name__ == "__main__":
    unittest.main()
