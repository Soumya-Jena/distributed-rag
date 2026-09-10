import unittest

from src.fusion import reciprocal_rank_fusion
from src.lexical_retriever import LexicalChunk
from src.retriever import RetrievedChunk


def vector_chunk(chunk_id, similarity=0.8):
    return RetrievedChunk(
        chunk_id,
        f"doc-{chunk_id}",
        f"doc-{chunk_id}.md",
        0,
        f"content-{chunk_id}",
        similarity,
    )


def lexical_chunk(chunk_id, lexical_score=0.5):
    return LexicalChunk(
        chunk_id,
        f"doc-{chunk_id}",
        f"doc-{chunk_id}.md",
        0,
        f"content-{chunk_id}",
        lexical_score,
    )


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_candidate_in_both_rankings_receives_both_contributions(self):
        results = reciprocal_rank_fusion(
            [vector_chunk(1), vector_chunk(2)],
            [lexical_chunk(2), lexical_chunk(3)],
            rrf_k=60,
            top_k=3,
        )

        self.assertEqual(results[0].chunk_id, 2)
        self.assertEqual(results[0].vector_rank, 2)
        self.assertEqual(results[0].lexical_rank, 1)
        self.assertAlmostEqual(results[0].rrf_score, 1 / 62 + 1 / 61)

    def test_raw_vector_and_lexical_scores_do_not_affect_fusion_order(self):
        results = reciprocal_rank_fusion(
            [vector_chunk(1, 0.01), vector_chunk(2, 0.99)],
            [lexical_chunk(3, 999.0)],
            rrf_k=60,
            top_k=3,
        )

        self.assertEqual([item.chunk_id for item in results], [1, 3, 2])

    def test_negative_rrf_constant_is_rejected(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], [], rrf_k=-1)


if __name__ == "__main__":
    unittest.main()
