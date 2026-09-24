import io
import json
import logging
import unittest

from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import generate_latest

from src.runtime_metrics import REQUESTS, STAGE_DURATION, bounded_error_type
from src.structured_logging import LOGGER, configure_logging, log_event, query_metadata
from src.tracing import observed_span, telemetry_enabled, trace_id


class ObservabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exporter = InMemorySpanExporter()
        trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(cls.exporter))

    def setUp(self):
        self.exporter.clear()

    def test_metrics_use_bounded_labels(self):
        REQUESTS.labels(status="success").inc()
        STAGE_DURATION.labels(stage="generation").observe(0.02)
        payload = generate_latest().decode("utf-8")
        self.assertIn('rag_requests_total{status="success"}', payload)
        self.assertIn('rag_stage_duration_seconds_count{stage="generation"}', payload)
        self.assertNotIn("question=", payload)
        self.assertNotIn("trace_id=", payload)

    def test_nested_spans_share_trace_without_content_attributes(self):
        with observed_span("rag.request"):
            active_id = trace_id()
            with observed_span("retrieval", {"rag.final_k": 5}):
                pass
        spans = self.exporter.get_finished_spans()
        self.assertEqual({span.context.trace_id for span in spans}, {int(active_id, 16)})
        self.assertEqual({span.name for span in spans}, {"rag.request", "retrieval"})
        self.assertNotIn("query", str([span.attributes for span in spans]).lower())

    def test_log_contains_hash_and_trace_but_not_raw_query(self):
        stream = io.StringIO()
        old_handlers = LOGGER.handlers[:]
        LOGGER.handlers.clear()
        handler = logging.StreamHandler(stream)
        from pythonjsonlogger.json import JsonFormatter
        handler.setFormatter(JsonFormatter("%(event)s %(trace_id)s %(query_hash)s %(query_length)s"))
        LOGGER.addHandler(handler)
        try:
            secret = "What is my private account number?"
            with observed_span("rag.request"):
                log_event("request_started", **query_metadata(secret))
            entry = json.loads(stream.getvalue())
            self.assertEqual(entry["event"], "request_started")
            self.assertEqual(len(entry["trace_id"]), 32)
            self.assertNotIn(secret, stream.getvalue())
        finally:
            LOGGER.handlers[:] = old_handlers

    def test_error_taxonomy_is_bounded(self):
        self.assertEqual(bounded_error_type(ValueError("secret")), "validation")
        self.assertEqual(bounded_error_type(RuntimeError("secret")), "internal")

    def test_tracing_can_be_disabled_for_ab_measurement(self):
        with telemetry_enabled(False):
            with observed_span("rag.request"):
                pass
        self.assertEqual(self.exporter.get_finished_spans(), ())


if __name__ == "__main__":
    unittest.main()
