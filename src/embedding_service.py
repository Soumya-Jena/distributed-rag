from time import perf_counter

import numpy as np
from sentence_transformers import SentenceTransformer

from src.cache import Cache, CacheStats, fingerprint, hash_text, normalize_query
from src.config import EMBEDDING_CACHE_TTL


MODEL_CONFIGS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "query_prefix": "",
        "document_prefix": "",
        "dimension": 384,
    },
    "BAAI/bge-small-en-v1.5": {
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "document_prefix": "",
        "dimension": 384,
    },
    "intfloat/e5-small-v2": {
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
        "dimension": 384,
    },
}


class EmbeddingService:
    def __init__(self, model_name, cache=None):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported embedding model: {model_name}")

        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.cache = cache or Cache()
        self.cache_fingerprint = fingerprint({
            "model": model_name,
            "normalize": True,
            "query_prefix": self.config["query_prefix"],
        })
        self.last_cache_stats = CacheStats()
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        actual_dimension = self.model.get_embedding_dimension()
        expected_dimension = self.config["dimension"]
        if actual_dimension != expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: {actual_dimension} != {expected_dimension}"
            )
        self.tokenizer = self.model.tokenizer
        print(f"Dimension : {actual_dimension}")
        print(f"Max seq   : {self.model.max_seq_length}")

    def encode_documents(self, texts, batch_size=32):
        prefix = self.config["document_prefix"]
        return self.model.encode(
            [prefix + text for text in texts],
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def _encode_query_uncached(self, query):
        return self.model.encode(
            self.config["query_prefix"] + query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_query(self, query):
        normalized = normalize_query(query)
        key = (
            f"embedding:{self.cache_fingerprint}:"
            f"{hash_text(normalized)}"
        )
        cached, lookup = self.cache.lookup_json(key)
        if cached is not None:
            self.last_cache_stats = lookup
            return np.asarray(cached, dtype=np.float32)

        started = perf_counter()
        embedding = self._encode_query_uncached(normalized)
        compute_seconds = perf_counter() - started
        self.cache.set_json(
            key, embedding.tolist(), EMBEDDING_CACHE_TTL
        )
        self.last_cache_stats = CacheStats(
            hit=False,
            cache_lookup_seconds=lookup.cache_lookup_seconds,
            compute_seconds=compute_seconds,
            available=lookup.available,
        )
        return embedding
