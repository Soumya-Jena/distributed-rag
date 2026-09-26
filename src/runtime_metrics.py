"""Low-cardinality Prometheus metrics for the online RAG pipeline."""

from threading import Lock

from prometheus_client import Counter, Gauge, Histogram, start_http_server


STAGES = (
    "query_transform", "embedding", "vector_search", "lexical_search",
    "rrf_fusion", "rerank", "context_optimize", "security_check",
    "generation", "output_validation",
)

REQUESTS = Counter(
    "rag_requests_total", "Completed RAG requests", ("status",)
)
REQUEST_DURATION = Histogram(
    "rag_request_duration_seconds", "End-to-end request latency",
    buckets=(.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120),
)
STAGE_DURATION = Histogram(
    "rag_stage_duration_seconds", "Pipeline stage latency", ("stage",),
    buckets=(.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60),
)
RETRIEVAL_CANDIDATES = Histogram(
    "rag_retrieval_candidates", "Candidates returned before context selection",
    buckets=(0, 1, 3, 5, 10, 20, 40, 80),
)
FINAL_CONTEXT_CHUNKS = Histogram(
    "rag_final_context_chunks", "Chunks passed to generation",
    buckets=(0, 1, 2, 3, 5, 8, 10, 20),
)
EMPTY_RETRIEVAL = Counter(
    "rag_empty_retrieval_total", "Requests with no usable evidence"
)
TOKENS = Histogram(
    "rag_tokens", "Prompt, context, and output token counts", ("kind",),
    buckets=(0, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000),
)
COMPRESSION_RATIO = Histogram(
    "rag_context_compression_ratio", "Optimized/original context token ratio",
    buckets=(0, .1, .25, .5, .75, .9, 1),
)
CACHE_OPERATIONS = Counter(
    "rag_cache_operations_total", "Cache lookup outcomes", ("cache", "result")
)
CACHE_LOOKUP_DURATION = Histogram(
    "rag_cache_lookup_duration_seconds", "Cache lookup latency", ("cache",),
    buckets=(.0001, .0005, .001, .005, .01, .025, .05, .1, .25, .5),
)
EVIDENCE_STATUS = Counter(
    "rag_evidence_status_total", "Generated evidence status", ("status",)
)
REFUSALS = Counter("rag_refusals_total", "Insufficient-evidence refusals")
SECURITY_EVENTS = Counter(
    "rag_security_events_total", "Security decisions", ("decision",)
)
ERRORS = Counter(
    "rag_errors_total", "Bounded pipeline errors", ("stage", "error_type")
)
OBSERVABILITY_OVERHEAD = Histogram(
    "rag_observability_overhead_seconds", "Time spent recording telemetry",
    buckets=(.00001, .00005, .0001, .0005, .001, .005, .01, .05),
)
IN_FLIGHT = Gauge(
    "rag_in_flight_requests", "Current HTTP RAG requests being processed"
)
HTTP_REQUESTS = Counter(
    "rag_http_requests_total", "Load-boundary outcomes", ("status",)
)
GOODPUT = Counter(
    "rag_goodput_total", "Successful non-empty RAG responses"
)
OUTPUT_TOKENS_TOTAL = Counter(
    "rag_output_tokens_total", "Aggregate generated output tokens"
)

_server_lock = Lock()
_server_started = False


def start_metrics_server(port=9108):
    """Start the process-wide endpoint once; never once per request."""
    global _server_started
    with _server_lock:
        if not _server_started:
            start_http_server(port)
            _server_started = True
    return _server_started


def record_cache_stats(stats_by_layer):
    for cache_name, values in stats_by_layer.items():
        if not values:
            continue
        result = "unavailable" if not values.get("available", True) else (
            "hit" if values.get("hit") else "miss"
        )
        CACHE_OPERATIONS.labels(cache=cache_name, result=result).inc()
        CACHE_LOOKUP_DURATION.labels(cache=cache_name).observe(
            max(0.0, float(values.get("cache_lookup_seconds", 0.0)))
        )


def bounded_error_type(error):
    if isinstance(error, (TimeoutError, ConnectionError)):
        return "dependency"
    if isinstance(error, (ValueError, TypeError)):
        return "validation"
    return "internal"
