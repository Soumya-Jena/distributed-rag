"""OpenTelemetry setup and safe span helpers."""

from contextlib import contextmanager
from contextvars import ContextVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode

from src.config import OTEL_ENDPOINT, OTEL_EXPORTER


def configure_tracing(exporter=None):
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        return provider
    provider = TracerProvider(resource=Resource.create({"service.name": "distributed-rag"}))
    selected = exporter
    if selected is None and OTEL_EXPORTER == "console":
        selected = ConsoleSpanExporter()
    elif selected is None and OTEL_EXPORTER == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        selected = OTLPSpanExporter(endpoint=OTEL_ENDPOINT)
    if selected is not None:
        provider.add_span_processor(BatchSpanProcessor(selected))
    trace.set_tracer_provider(provider)
    return provider


configure_tracing()
tracer = trace.get_tracer("distributed-rag")
_tracing_enabled = ContextVar("rag_tracing_enabled", default=True)


def trace_id():
    value = trace.get_current_span().get_span_context().trace_id
    return f"{value:032x}" if value else "0" * 32


@contextmanager
def observed_span(name, attributes=None):
    """Record a span without attaching queries, chunks, or generated text."""
    if not _tracing_enabled.get():
        yield trace.get_current_span()
        return
    with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
        try:
            yield span
        except Exception as error:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, type(error).__name__))
            raise


@contextmanager
def telemetry_enabled(enabled):
    token = _tracing_enabled.set(enabled)
    try:
        yield
    finally:
        _tracing_enabled.reset(token)
