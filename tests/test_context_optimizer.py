import unittest
from types import SimpleNamespace

import numpy as np

from src.context_optimizer import (
    ContextOptimizer, NearDuplicateFilter, SentenceCompressor,
    deduplicate_exact, split_sentences,
)


def chunk(identifier, text, source="doc.md", index=None):
    return SimpleNamespace(
        chunk_id=identifier, title="Doc", source_path=source,
        chunk_index=identifier if index is None else index,
        content=text, similarity=0.5, lexical_score=None, rrf_score=0.1,
        found_by=["original"], query_ranks={"original": identifier},
    )


class FakeEmbeddings:
    def __init__(self, vectors=None, query=None):
        self.vectors = vectors or {}
        self.query = np.array(query or [1.0, 0.0])

    def encode_query(self, _query):
        return self.query

    def encode_documents(self, texts):
        return np.array([self.vectors[text] for text in texts])


class ContextOptimizerTests(unittest.TestCase):
    def test_sentence_splitter_preserves_markdown_lines(self):
        self.assertEqual(
            split_sentences("First fact. Second fact!\n- A list item"),
            ["First fact.", "Second fact!", "- A list item"],
        )

    def test_exact_deduplication_uses_source_and_chunk_index(self):
        first = chunk(1, "one", index=3)
        duplicate = chunk(9, "changed id", index=3)
        self.assertEqual(deduplicate_exact([first, duplicate]), [first])

    def test_near_duplicate_filter_keeps_higher_ranked_chunk(self):
        first = chunk(1, "first")
        duplicate = chunk(2, "duplicate", source="other.md")
        distinct = chunk(3, "distinct")
        embeddings = FakeEmbeddings({
            "first": [1.0, 0.0], "duplicate": [0.95, 0.1],
            "distinct": [0.0, 1.0],
        })
        result = NearDuplicateFilter(embeddings, 0.92).filter(
            [first, duplicate, distinct]
        )
        self.assertEqual([item.chunk_id for item in result], [1, 3])

    def test_sentence_selection_restores_original_order(self):
        text = "Low fact. High fact. Medium fact."
        embeddings = FakeEmbeddings({
            "Low fact.": [0.1, 0.9], "High fact.": [1.0, 0.0],
            "Medium fact.": [0.8, 0.2],
        })
        result, indices = SentenceCompressor(embeddings).select(
            "query", text, keep_ratio=0.5
        )
        self.assertEqual(indices, [1, 2])
        self.assertEqual(result, "High fact. Medium fact.")

    def test_neighbor_expansion_keeps_surrounding_context(self):
        text = "Before. Match. After. Last."
        embeddings = FakeEmbeddings({
            "Before.": [0.0, 1.0], "Match.": [1.0, 0.0],
            "After.": [0.0, 1.0], "Last.": [0.0, 1.0],
        })
        result, indices = SentenceCompressor(embeddings).select(
            "query", text, keep_ratio=0.25, neighbor_window=1
        )
        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(result, "Before. Match. After.")

    def test_budget_keeps_whole_sentences_and_provenance(self):
        text = "One two. Three four. Five six."
        embeddings = FakeEmbeddings({
            "One two. Three four. Five six.": [1.0, 0.0],
            "One two.": [1.0, 0.0], "Three four.": [0.9, 0.1],
            "Five six.": [0.8, 0.2],
        })
        optimizer = ContextOptimizer(
            embeddings, lambda value: len(value.split()),
            keep_ratio=1, token_budget=4,
        )
        result = optimizer.optimize("query", [chunk(7, text)], "budgeted")
        self.assertEqual(result.optimized_tokens, 4)
        self.assertEqual(result.chunks[0].content, "One two. Three four.")
        self.assertEqual(result.chunks[0].original_content, text)
        self.assertEqual(result.chunks[0].source_id, "doc.md#7")

    def test_unknown_strategy_is_rejected(self):
        optimizer = ContextOptimizer(
            FakeEmbeddings(), lambda value: len(value.split())
        )
        with self.assertRaises(ValueError):
            optimizer.optimize("query", [], "unknown")


if __name__ == "__main__":
    unittest.main()
