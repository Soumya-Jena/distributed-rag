"""Generate reviewable metric and trace samples from deterministic work."""

import json
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import generate_latest

from src.runtime_metrics import REQUESTS, REQUEST_DURATION, STAGE_DURATION
from src.tracing import observed_span


def main():
    output = Path("experiments/day-14")
    output.mkdir(parents=True, exist_ok=True)
    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    with observed_span("rag.request", {"rag.top_k": 5}):
        with observed_span("retrieval"):
            with observed_span("embedding"):
                pass
            with observed_span("vector_search", {"rag.candidates": 20}):
                pass
        with observed_span("context.optimize", {"rag.output_chunks": 5}):
            pass
        with observed_span("generation"):
            pass
    REQUESTS.labels(status="success").inc()
    REQUEST_DURATION.observe(.125)
    STAGE_DURATION.labels(stage="generation").observe(.05)
    spans = [{
        "name": span.name,
        "trace_id": f"{span.context.trace_id:032x}",
        "span_id": f"{span.context.span_id:016x}",
        "parent_span_id": f"{span.parent.span_id:016x}" if span.parent else None,
        "attributes": dict(span.attributes),
        "status": span.status.status_code.name,
    } for span in exporter.get_finished_spans()]
    (output / "trace-sample.json").write_text(json.dumps(spans, indent=2), encoding="utf-8")
    (output / "metrics-sample.txt").write_bytes(generate_latest())
    print(f"Wrote {len(spans)} spans and a Prometheus snapshot")


if __name__ == "__main__":
    main()
