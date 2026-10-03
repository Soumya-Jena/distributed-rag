import unittest
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from src.api.app import create_app as create_api_app
from src.generation.client import HTTPGenerationClient, LocalGenerationClient
from src.model_server.app import create_app as create_model_app
from src.model_server.generator import FakeGenerator


class GenerationClientTests(unittest.TestCase):
    def test_http_generation_client_obeys_internal_contract(self):
        def handler(request):
            self.assertEqual(request.url.path, "/internal/generate")
            return httpx.Response(200, json={
                "text": "grounded answer",
                "input_tokens": 12,
                "output_tokens": 3,
            })

        http_client = httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://model-server:8010",
        )
        client = HTTPGenerationClient(client=http_client)
        answer = client.generate(
            [{"role": "user", "content": "question"}], max_new_tokens=10
        )
        self.assertEqual(answer, "grounded answer")
        self.assertEqual(client.last_input_token_count, 12)
        self.assertEqual(client.last_output_token_count, 3)

    def test_local_adapter_preserves_offline_generator(self):
        generator = SimpleNamespace(
            model_name="local",
            last_input_token_count=2,
            last_output_token_count=1,
            count_tokens=lambda text: len(text.split()),
            generate=lambda messages, max_new_tokens: "local answer",
        )
        client = LocalGenerationClient(generator)
        self.assertEqual(
            client.generate([{"role": "user", "content": "q"}], max_new_tokens=8),
            "local answer",
        )
        self.assertEqual(client.count_tokens("two tokens"), 2)


class ModelServerTests(unittest.TestCase):
    def test_liveness_readiness_and_generation(self):
        with TestClient(create_model_app(FakeGenerator())) as client:
            self.assertEqual(client.get("/live").json(), {"status": "alive"})
            self.assertEqual(client.get("/ready").status_code, 200)
            response = client.post("/internal/generate", json={
                "messages": [{"role": "user", "content": "question"}],
                "max_new_tokens": 16,
            })
        self.assertEqual(response.status_code, 200)
        self.assertIn("[S1]", response.json()["text"])
        self.assertGreater(response.json()["output_tokens"], 0)

    def test_model_bulkhead_rejects_excess_work(self):
        app = create_model_app(FakeGenerator())
        with TestClient(app) as client:
            app.state.slots.acquire()
            try:
                response = client.post("/internal/generate", json={
                    "messages": [{"role": "user", "content": "question"}],
                    "max_new_tokens": 16,
                })
            finally:
                app.state.slots.release()
        self.assertEqual(response.status_code, 429)


class DeploymentAPITests(unittest.TestCase):
    def fake_service(self):
        result = SimpleNamespace(
            answer="EVIDENCE_STATUS: SUPPORTED\n\nAnswer [S1]",
            chunks=[SimpleNamespace(title="postgres-mvcc", chunk_index=0)],
            trace_id="a" * 32,
            output_tokens=8,
            response_cache_hit=False,
            status="normal",
            service_modes=[],
            degraded_components=[],
        )
        return SimpleNamespace(
            answer=lambda _question, _top_k: result,
            generator=SimpleNamespace(close=lambda: None),
        )

    def test_liveness_does_not_call_dependencies(self):
        checks = []
        app = create_api_app(
            self.fake_service(),
            readiness_check=lambda _service: checks.append(True) or {},
        )
        with TestClient(app) as client:
            self.assertEqual(client.get("/live").json(), {"status": "alive"})
        self.assertEqual(checks, [])

    def test_readiness_and_query_contract(self):
        checks = {
            "postgres": True,
            "embedding": True,
            "generation": True,
            "security": True,
        }
        app = create_api_app(
            self.fake_service(), readiness_check=lambda _service: checks
        )
        with TestClient(app) as client:
            self.assertEqual(client.get("/ready").status_code, 200)
            response = client.post(
                "/api/v1/query",
                headers={"x-request-id": "request-123"},
                json={"question": "Explain MVCC", "top_k": 3},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-request-id"], "request-123")
        self.assertEqual(response.json()["request_id"], "request-123")

    def test_failed_hard_readiness_returns_503(self):
        checks = {
            "postgres": False,
            "embedding": True,
            "generation": True,
            "security": True,
        }
        app = create_api_app(
            self.fake_service(), readiness_check=lambda _service: checks
        )
        with TestClient(app) as client:
            response = client.get("/ready")
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
