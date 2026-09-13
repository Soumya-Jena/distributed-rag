import unittest

from src.grounding import (
    citation_coverage,
    citation_support,
    extract_citations,
    extract_evidence_status,
    faithfulness,
    hallucination_rate,
    invalid_citations,
    status_matches_answerability,
)
from src.prompt import get_system_prompt


class GroundingTests(unittest.TestCase):
    def test_extracts_status_case_insensitively(self):
        self.assertEqual(
            extract_evidence_status("EVIDENCE_STATUS: partial\nAnswer"),
            "PARTIAL",
        )

    def test_missing_status_returns_none(self):
        self.assertIsNone(extract_evidence_status("Answer [S1]"))

    def test_extracts_and_validates_citations(self):
        answer = "Claim [S1]. Another [S3] and [S0]."
        self.assertEqual(extract_citations(answer), [1, 3, 0])
        self.assertEqual(invalid_citations(answer, 2), [3, 0])

    def test_status_matches_answerability(self):
        self.assertEqual(status_matches_answerability("SUPPORTED", "full"), 1)
        self.assertEqual(status_matches_answerability("PARTIAL", "partial"), 1)
        self.assertEqual(status_matches_answerability("SUPPORTED", "none"), 0)

    def test_claim_metrics(self):
        self.assertEqual(faithfulness(3, 4), 0.75)
        self.assertEqual(hallucination_rate(1, 4), 0.25)
        self.assertEqual(citation_coverage(3, 4), 0.75)
        self.assertEqual(citation_support(2, 3), 2 / 3)
        self.assertIsNone(faithfulness(0, 0))

    def test_prompt_mode_validation(self):
        self.assertIn("EVIDENCE_STATUS", get_system_prompt("strict"))
        self.assertNotIn("EVIDENCE_STATUS", get_system_prompt("baseline"))
        with self.assertRaises(ValueError):
            get_system_prompt("unknown")


if __name__ == "__main__":
    unittest.main()
