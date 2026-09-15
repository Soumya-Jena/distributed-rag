import os

from dotenv import load_dotenv

load_dotenv()


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://rag:rag@localhost:5432/ragdb",
)


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


GENERATION_MODEL = os.getenv(
    "GENERATION_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct",
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
