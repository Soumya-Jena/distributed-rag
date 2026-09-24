"""Send a content-free trace to the configured exporter for smoke testing."""

from opentelemetry import trace

from src.structured_logging import log_event
from src.tracing import observed_span


def main():
    with observed_span("rag.request", {"rag.top_k": 5}):
        with observed_span("retrieval"):
            log_event("verification_trace", top_k=5)
    trace.get_tracer_provider().shutdown()


if __name__ == "__main__":
    main()
