import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.load_test_api import create_app


class FakeRAG:
    def __init__(self, answer="EVIDENCE_STATUS: SUPPORTED\n\nMVCC isolates readers. [S1]"):
        self.answer_text = answer
        self.calls = []

    def answer(self, question, top_k):
        self.calls.append((question, top_k))
        return SimpleNamespace(
            answer=self.answer_text,
            chunks=[SimpleNamespace(title="postgres-mvcc", chunk_index=0)],
            trace_id="a" * 32,
            output_tokens=12,
            response_cache_hit=False,
        )


class LoadTestAPITests(unittest.TestCase):
    def test_health_and_ready(self):
        with TestClient(create_app(FakeRAG())) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            self.assertEqual(client.get("/ready").status_code, 200)

    def test_query_contract(self):
        service = FakeRAG()
        with TestClient(create_app(service)) as client:
            response = client.post("/query", json={"question": "Explain MVCC", "top_k": 3})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["answer"])
        self.assertEqual(payload["sources"][0]["title"], "postgres-mvcc")
        self.assertEqual(len(payload["trace_id"]), 32)
        self.assertEqual(service.calls, [("Explain MVCC", 3)])

    def test_query_validation_and_empty_answer(self):
        with TestClient(create_app(FakeRAG())) as client:
            self.assertEqual(client.post("/query", json={"question": "x"}).status_code, 422)
        with TestClient(create_app(FakeRAG(answer=""))) as client:
            self.assertEqual(
                client.post("/query", json={"question": "Explain MVCC"}).status_code,
                502,
            )


if __name__ == "__main__":
    unittest.main()
