import unittest
from types import SimpleNamespace

from src.multi_query_retriever import MultiQueryRetriever, cross_query_rrf
from src.query_transformer import QueryVariants
from src.rag_service import RAGService


def chunk(identifier, title=None):
    return SimpleNamespace(
        chunk_id=identifier, title=title or str(identifier), source_path=f"{identifier}.md",
        chunk_index=0, content=str(identifier), similarity=0.5, lexical_score=None,
    )


class MultiQueryTests(unittest.TestCase):
    def test_cross_query_rrf_records_provenance(self):
        result = cross_query_rrf([
            ("original", [chunk(1), chunk(2)]),
            ("semantic", [chunk(2), chunk(3)]),
        ], rrf_k=60, top_k=3)
        self.assertEqual(result[0].chunk_id, 2)
        self.assertEqual(result[0].found_by, ["original", "semantic"])
        self.assertEqual(result[0].query_ranks, {"original": 2, "semantic": 1})

    def test_original_strategy_makes_one_search(self):
        calls = []
        hybrid = SimpleNamespace(search=lambda query, **kwargs: (
            calls.append(query) or SimpleNamespace(fused_results=[chunk(1)])
        ))
        variants = QueryVariants("o", "s", "t", "a")
        result = MultiQueryRetriever(hybrid_retriever=hybrid).retrieve(
            "o", variants, strategy="original"
        )
        self.assertEqual(calls, ["o"])
        self.assertEqual(result.final_chunks[0].found_by, ["original"])

    def test_configurable_variant_count_keeps_original(self):
        calls = []
        hybrid = SimpleNamespace(search=lambda query, **kwargs: (
            calls.append(query) or SimpleNamespace(fused_results=[chunk(1)])
        ))
        variants = QueryVariants("o", "s", "t", "a")
        MultiQueryRetriever(
            hybrid_retriever=hybrid, variant_count=2
        ).retrieve("o", variants, strategy="multi_query")
        self.assertEqual(calls, ["o", "s", "t"])

    def test_rejects_invalid_variant_count(self):
        with self.assertRaises(ValueError):
            MultiQueryRetriever(hybrid_retriever=object(), variant_count=4)

    def test_reranker_receives_original_question(self):
        calls = []
        hybrid = SimpleNamespace(search=lambda query, **kwargs: SimpleNamespace(fused_results=[chunk(1)]))
        reranker = SimpleNamespace(rerank=lambda query, chunks, top_k: (
            calls.append(query) or [SimpleNamespace(chunk=chunks[0])]
        ))
        variants = QueryVariants("user question", "rewrite", "technical", "alternate")
        MultiQueryRetriever(hybrid, reranker, use_reranker=True).retrieve(
            "user question", variants, strategy="multi_query"
        )
        self.assertEqual(calls, ["user question"])

    def test_rag_service_keeps_security_after_multi_query_retrieval(self):
        variants = QueryVariants("question", "semantic", "technical", "alternate")
        transformer = SimpleNamespace(transform=lambda question: variants)
        retrieval = SimpleNamespace(
            final_chunks=[chunk(1)], retrieval_seconds=0.2, rerank_seconds=0.0,
        )
        multi = SimpleNamespace(retrieve=lambda *args, **kwargs: retrieval)
        generator = SimpleNamespace(
            generate=lambda messages: "EVIDENCE_STATUS: SUPPORTED\n\nAnswer [S1]",
            last_input_token_count=20,
        )
        service = RAGService(
            generator=generator, query_strategy="multi_query",
            query_transformer=transformer, multi_query_retriever=multi,
            grounding_mode="strict", security_mode="layered",
        )
        result = service.answer("question")
        self.assertEqual(result.chunks[0].chunk_id, 1)
        self.assertIn("[S1]", result.answer)
        self.assertGreaterEqual(result.transformation_seconds, 0)


if __name__ == "__main__":
    unittest.main()
