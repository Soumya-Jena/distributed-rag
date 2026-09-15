import unittest
from types import SimpleNamespace

from src.query_transformer import (
    QueryTransformer, extract_identifiers, missing_identifiers,
)


class QueryTransformerTests(unittest.TestCase):
    def test_extracts_exact_identifiers(self):
        self.assertEqual(
            extract_identifiers("Why does SQLSTATE 23505 mention synchronous_commit and WAL?"),
            {"SQLSTATE", "23505", "synchronous_commit", "WAL"},
        )

    def test_reports_identifier_loss(self):
        self.assertEqual(missing_identifiers("Explain WAL", "Explain logging"), {"WAL"})

    def test_parses_three_fields(self):
        output = "SEMANTIC: meaning\nTECHNICAL: terms\nALTERNATE: phrasing"
        variants = QueryTransformer._parse("original", output)
        self.assertEqual(variants.all(), ["original", "meaning", "terms", "phrasing"])

    def test_bad_output_falls_back_to_original(self):
        variants = QueryTransformer._parse("keep me", "a prose answer")
        self.assertEqual(variants.all(), ["keep me"] * 4)

    def test_extra_output_falls_back_to_original(self):
        output = (
            "Here are the rewrites:\nSEMANTIC: meaning\n"
            "TECHNICAL: terms\nALTERNATE: phrasing"
        )
        variants = QueryTransformer._parse("keep me", output)
        self.assertEqual(variants.all(), ["keep me"] * 4)

    def test_generation_failure_falls_back(self):
        generator = SimpleNamespace(generate=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError()))
        variants = QueryTransformer(generator=generator).transform("safe original")
        self.assertEqual(variants.all(), ["safe original"] * 4)


if __name__ == "__main__":
    unittest.main()
