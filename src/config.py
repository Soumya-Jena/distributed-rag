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
        "200",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "40",
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
