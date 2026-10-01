import random
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.fault_injection import FaultInjector, InjectedFault
from src.load_test_api import create_app
from src.resilience import (
    Bulkhead,
    BulkheadFullError,
    CircuitBreaker,
    CircuitOpenError,
    Deadline,
    DeadlineExceededError,
    GenerationUnavailableError,
    RetrievalUnavailableError,
    SecurityUnavailableError,
    TransientDependencyError,
    retry_transient,
)
from src.retrieval_pipeline import RetrievalPipeline
from src.rag_service import RAGService


def chunk(identifier, *, vector=True):
    return SimpleNamespace(
        chunk_id=identifier,
        title=str(identifier),
        source_path=f"{identifier}.md",
        chunk_index=0,
        content=str(identifier),
        similarity=0.8 if vector else None,
        lexical_score=None if vector else 1.0,
    )


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class ResiliencePrimitiveTests(unittest.TestCase):
    def test_fault_injection_is_impossible_in_production(self):
        with self.assertRaises(RuntimeError):
            FaultInjector(enabled=True, environment="production")

    def test_fault_injector_supports_deterministic_failure_and_delay(self):
        sleeps = []
        injector = FaultInjector(
            enabled=True,
            environment="test",
            actions={"database": "delay:0.25", "generator": "fail"},
            sleeper=sleeps.append,
        )
        injector.apply("database")
        self.assertEqual(sleeps, [0.25])
        with self.assertRaises(InjectedFault):
            injector.apply("generator")

    def test_deadline_rejects_new_work_after_budget_expires(self):
        clock = FakeClock()
        deadline = Deadline(1.0, clock=clock)
        clock.now = 1.1
        with self.assertRaises(DeadlineExceededError):
            deadline.require("generation")

    def test_retry_is_bounded_and_only_retries_transient_failures(self):
        attempts = []
        sleeps = []

        def eventually_succeeds():
            attempts.append(1)
            if len(attempts) < 3:
                raise TransientDependencyError("temporary")
            return "ok"

        result = retry_transient(
            eventually_succeeds,
            attempts=3,
            base_delay=0.1,
            sleeper=sleeps.append,
            rng=random.Random(1),
        )
        self.assertEqual(result, "ok")
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(sleeps), 2)

        permanent_attempts = []
        with self.assertRaises(ValueError):
            retry_transient(
                lambda: permanent_attempts.append(1) or (_ for _ in ()).throw(
                    ValueError("permanent")
                )
            )
        self.assertEqual(len(permanent_attempts), 1)

    def test_circuit_breaker_opens_then_recovers_through_half_open(self):
        clock = FakeClock()
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_seconds=10,
            clock=clock,
        )
        for _ in range(2):
            with self.assertRaises(OSError):
                breaker.call(lambda: (_ for _ in ()).throw(OSError("down")))
        self.assertEqual(breaker.state, "open")
        with self.assertRaises(CircuitOpenError):
            breaker.call(lambda: "should not run")
        clock.now = 11
        self.assertEqual(breaker.call(lambda: "recovered"), "recovered")
        self.assertEqual(breaker.state, "closed")

    def test_bulkhead_rejects_excess_work(self):
        bulkhead = Bulkhead(1, acquire_timeout=0, name="generation")
        bulkhead.semaphore.acquire()
        try:
            with self.assertRaises(BulkheadFullError):
                bulkhead.call(lambda: None)
        finally:
            bulkhead.semaphore.release()


class RetrievalDegradationTests(unittest.TestCase):
    def pipeline(self, actions):
        vector = SimpleNamespace(search=lambda *_args, **_kwargs: [chunk("vector")])
        lexical = SimpleNamespace(
            search=lambda *_args, **_kwargs: [chunk("lexical", vector=False)]
        )
        hybrid = SimpleNamespace(vector_retriever=vector, lexical_retriever=lexical)
        return RetrievalPipeline(
            use_reranker=False,
            retrieval_mode="hybrid",
            retriever=vector,
            hybrid_retriever=hybrid,
            fault_injector=FaultInjector(
                enabled=True,
                environment="test",
                actions=actions,
            ),
        )

    def test_vector_failure_falls_back_to_lexical(self):
        result = self.pipeline({"vector_retrieval": "fail"}).retrieve("question")
        self.assertEqual(result.final_chunks[0].chunk_id, "lexical")
        self.assertEqual(result.service_modes[0].value, "degraded_lexical_only")

    def test_embedding_failure_falls_back_to_lexical(self):
        result = self.pipeline({"embedding": "fail"}).retrieve("question")
        self.assertEqual(result.final_chunks[0].chunk_id, "lexical")
        self.assertEqual(result.service_modes[0].value, "degraded_lexical_only")

    def test_lexical_failure_falls_back_to_vector(self):
        result = self.pipeline({"lexical_retrieval": "fail"}).retrieve("question")
        self.assertEqual(result.final_chunks[0].chunk_id, "vector")
        self.assertEqual(result.service_modes[0].value, "degraded_vector_only")

    def test_both_retrieval_branches_down_fail_closed(self):
        pipeline = self.pipeline({
            "vector_retrieval": "fail",
            "lexical_retrieval": "fail",
        })
        with self.assertRaises(RetrievalUnavailableError):
            pipeline.retrieve("question")

    def test_healthy_empty_retrieval_is_not_misclassified_as_outage(self):
        empty = SimpleNamespace(search=lambda *_args, **_kwargs: [])
        hybrid = SimpleNamespace(vector_retriever=empty, lexical_retriever=empty)
        result = RetrievalPipeline(
            use_reranker=False,
            retrieval_mode="hybrid",
            retriever=empty,
            hybrid_retriever=hybrid,
        ).retrieve("no matching evidence")
        self.assertEqual(result.final_chunks, [])
        self.assertEqual(result.service_modes, [])


class FailureContractTests(unittest.TestCase):
    def test_hard_dependency_failure_maps_to_http_503(self):
        service = SimpleNamespace(
            answer=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RetrievalUnavailableError("retrieval unavailable")
            )
        )
        with TestClient(create_app(service)) as client:
            response = client.post("/query", json={"question": "Explain MVCC"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["mode"], "unavailable_retrieval")

    def service(self, actions, security_mode="layered", observability=False):
        source = chunk("source")
        retrieval = SimpleNamespace(
            candidate_chunks=[source],
            final_chunks=[source],
            vector_seconds=0.01,
            rerank_seconds=0.0,
            lexical_seconds=0.01,
            fusion_seconds=0.0,
            service_modes=[],
            degraded_components=[],
            cache_stats=None,
        )
        retrieval_pipeline = SimpleNamespace(retrieve=lambda *_args, **_kwargs: retrieval)
        generator = SimpleNamespace(
            generate=lambda _messages: "EVIDENCE_STATUS: SUPPORTED\n\nAnswer [S1]",
            count_tokens=lambda text: len(text.split()),
            last_input_token_count=20,
            last_output_token_count=8,
        )
        return RAGService(
            retrieval_pipeline=retrieval_pipeline,
            generator=generator,
            response_cache_enabled=False,
            observability_enabled=observability,
            grounding_mode="strict",
            security_mode=security_mode,
            fault_injector=FaultInjector(
                enabled=True,
                environment="test",
                actions=actions,
            ),
        )

    def test_context_optimizer_failure_uses_raw_context(self):
        result = self.service({"context_optimizer": "fail"}).answer("Explain source")
        self.assertEqual(result.status, "degraded")
        self.assertIn("degraded_raw_context", result.service_modes)
        self.assertTrue(result.answer)

    def test_generator_failure_is_hard_failure(self):
        with self.assertRaises(GenerationUnavailableError):
            self.service({"generator": "fail"}).answer("Explain source")

    def test_security_component_failure_fails_closed(self):
        with self.assertRaises(SecurityUnavailableError):
            self.service({"security_validator": "fail"}).answer("Explain source")

    def test_observability_failure_fails_open(self):
        result = self.service(
            {"observability": "fail"}, observability=True
        ).answer("Explain source")
        self.assertTrue(result.answer)


if __name__ == "__main__":
    unittest.main()
