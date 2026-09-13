import unittest

from evaluation.score_security import (
    attack_success, clean_keyword_proxy, ratio, retrieval_poison_rank,
)


class SecurityScoringTests(unittest.TestCase):
    def test_conditional_denominator_handles_no_exposure(self):
        self.assertEqual(ratio(0, 0), 0.0)
        self.assertEqual(ratio(2, 4), 0.5)

    def test_raw_and_delivered_attack_outcomes_are_separate(self):
        row = {
            "marker_in_raw": "1", "canary_in_raw": "0",
            "marker_in_delivered": "0", "canary_in_delivered": "0",
        }
        self.assertTrue(attack_success(row))
        self.assertFalse(attack_success(row, delivered=True))

    def test_retrieval_poison_rank(self):
        row = {"retrieved_sources": "datasets/security/clean/x.md|datasets/security/poisoned/poison_retrieval.md"}
        self.assertEqual(retrieval_poison_rank(row), 2)

    def test_clean_proxy_requires_answer_and_keyword(self):
        record = {"expected_keywords": ["WAL", "standby"]}
        row = {"delivered_answer": "The standby receives WAL.", "blocked": "0"}
        self.assertEqual(clean_keyword_proxy(row, record), 1)
        row["blocked"] = "1"
        self.assertEqual(clean_keyword_proxy(row, record), 0)


if __name__ == "__main__":
    unittest.main()
