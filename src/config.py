import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()


def read_secret(name, default=None):
    configured = os.getenv(f"{name.upper()}_FILE")
    path = Path(configured or f"/run/secrets/{name}")
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return default


DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
DATABASE_PORT = int(os.getenv("DATABASE_PORT", "5432"))
DATABASE_NAME = os.getenv("DATABASE_NAME", "ragdb")
DATABASE_USER = os.getenv("DATABASE_USER", "rag")
DATABASE_PASSWORD = read_secret(
    "postgres_password", os.getenv("DATABASE_PASSWORD", "rag")
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    (
        f"postgresql://{quote_plus(DATABASE_USER)}:"
        f"{quote_plus(DATABASE_PASSWORD)}@{DATABASE_HOST}:"
        f"{DATABASE_PORT}/{DATABASE_NAME}"
    ),
)


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


GENERATION_MODEL = os.getenv(
    "GENERATION_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct",
)

GENERATION_BACKEND = os.getenv("GENERATION_BACKEND", "local").lower()
GENERATION_URL = os.getenv("GENERATION_URL", "http://localhost:8010")
GENERATION_TIMEOUT_SECONDS = float(
    os.getenv("GENERATION_TIMEOUT_SECONDS", "120")
)


CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "100",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "20",
    )
)


RETRIEVAL_TOP_K = int(
    os.getenv(
        "RETRIEVAL_TOP_K",
        "5",
    )
)


MIN_RETRIEVAL_SIMILARITY = float(
    os.getenv(
        "MIN_RETRIEVAL_SIMILARITY",
        "0.30",
    )
)


MAX_NEW_TOKENS = int(
    os.getenv(
        "MAX_NEW_TOKENS",
        "350",
    )
)

RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
)

RETRIEVAL_CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() in {
    "1",
    "true",
    "yes",
}

RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
LEXICAL_CANDIDATE_K = int(os.getenv("LEXICAL_CANDIDATE_K", "20"))
HYBRID_CANDIDATE_K = int(os.getenv("HYBRID_CANDIDATE_K", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))

GROUNDING_MODE = os.getenv("GROUNDING_MODE", "strict").lower()
SECURITY_MODE = os.getenv("SECURITY_MODE", "layered").lower()
QUERY_TRANSFORM_MODEL = os.getenv(
    "QUERY_TRANSFORM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
)
QUERY_VARIANTS = int(os.getenv("QUERY_VARIANTS", "3"))
QUERY_TRANSFORM_ENABLED = os.getenv("QUERY_TRANSFORM_ENABLED", "true").lower() in {
    "1", "true", "yes",
}
CONTEXT_STRATEGY = os.getenv("CONTEXT_STRATEGY", "full").lower()
CONTEXT_DEDUP_THRESHOLD = float(os.getenv("CONTEXT_DEDUP_THRESHOLD", "0.92"))
CONTEXT_KEEP_RATIO = float(os.getenv("CONTEXT_KEEP_RATIO", "0.50"))
CONTEXT_TOKEN_BUDGET = int(os.getenv("CONTEXT_TOKEN_BUDGET", "1000"))
CONTEXT_NEIGHBOR_WINDOW = int(os.getenv("CONTEXT_NEIGHBOR_WINDOW", "0"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in {
    "1", "true", "yes",
}
EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", "86400"))
TRANSFORM_CACHE_TTL = int(os.getenv("TRANSFORM_CACHE_TTL", "86400"))
RETRIEVAL_CACHE_TTL = int(os.getenv("RETRIEVAL_CACHE_TTL", "3600"))
RESPONSE_CACHE_TTL = int(os.getenv("RESPONSE_CACHE_TTL", "600"))
RESPONSE_CACHE_ENABLED = os.getenv("RESPONSE_CACHE_ENABLED", "false").lower() in {
    "1", "true", "yes",
}

OBSERVABILITY_ENABLED = os.getenv("OBSERVABILITY_ENABLED", "true").lower() in {
    "1", "true", "yes",
}
METRICS_SERVER_ENABLED = os.getenv("METRICS_SERVER_ENABLED", "false").lower() in {
    "1", "true", "yes",
}
METRICS_PORT = int(os.getenv("METRICS_PORT", "9108"))
OTEL_EXPORTER = os.getenv("OTEL_EXPORTER", "none").lower()
OTEL_ENDPOINT = os.getenv("OTEL_ENDPOINT", "http://localhost:4318/v1/traces")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
FAULT_INJECTION_ENABLED = os.getenv("FAULT_INJECTION_ENABLED", "false").lower() in {
    "1", "true", "yes",
}
DATABASE_CONNECT_TIMEOUT = int(os.getenv("DATABASE_CONNECT_TIMEOUT", "2"))
RETRIEVAL_STATEMENT_TIMEOUT_MS = int(
    os.getenv("RETRIEVAL_STATEMENT_TIMEOUT_MS", "500")
)
REQUEST_DEADLINE_SECONDS = float(os.getenv("REQUEST_DEADLINE_SECONDS", "120"))
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(
    os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")
)
CIRCUIT_BREAKER_RECOVERY_SECONDS = float(
    os.getenv("CIRCUIT_BREAKER_RECOVERY_SECONDS", "10")
)
GENERATION_BULKHEAD_SLOTS = int(os.getenv("GENERATION_BULKHEAD_SLOTS", "1"))
RERANK_BULKHEAD_SLOTS = int(os.getenv("RERANK_BULKHEAD_SLOTS", "1"))
BULKHEAD_ACQUIRE_TIMEOUT_SECONDS = float(
    os.getenv("BULKHEAD_ACQUIRE_TIMEOUT_SECONDS", "1")
)
DEPENDENCY_RETRY_ATTEMPTS = int(os.getenv("DEPENDENCY_RETRY_ATTEMPTS", "2"))
RETRY_BASE_DELAY_SECONDS = float(os.getenv("RETRY_BASE_DELAY_SECONDS", "0.05"))

QUERY_TRANSFORM_PROMPT_VERSION = "v1"
GROUNDING_PROMPT_VERSION = "v3"
SECURITY_PROMPT_VERSION = "v2"
RETRIEVAL_PIPELINE_VERSION = "v1"
