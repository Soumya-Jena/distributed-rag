"""Identifier-only retrieval-result cache with live PostgreSQL hydration."""

from types import SimpleNamespace

from src.cache import fingerprint, get_corpus_version, hash_text, normalize_query
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    HYBRID_CANDIDATE_K,
    LEXICAL_CANDIDATE_K,
    RERANK_MODEL,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_CACHE_TTL,
    RETRIEVAL_PIPELINE_VERSION,
    RRF_K,
)
from src.db import get_connection


SCORE_FIELDS = (
    "similarity", "vector_similarity", "lexical_score", "rrf_score",
    "vector_rank", "lexical_rank",
)


def describe_chunk(chunk):
    value = {"chunk_id": int(chunk.chunk_id)}
    for field in SCORE_FIELDS:
        score = getattr(chunk, field, None)
        if score is not None:
            value[field] = score
    return value


def hydrate_chunks(descriptors):
    if not descriptors:
        return []
    ids = [item["chunk_id"] for item in descriptors]
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT c.id, d.title, d.source_path, c.chunk_index, c.content
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id = ANY(%s);
            """,
            (ids,),
        ).fetchall()
    content = {
        row[0]: {
            "title": row[1], "source_path": row[2],
            "chunk_index": row[3], "content": row[4],
        }
        for row in rows
    }
    chunks = []
    for descriptor in descriptors:
        current = content.get(descriptor["chunk_id"])
        if current is None:
            continue
        fields = {**descriptor, **current}
        if "similarity" not in fields:
            fields["similarity"] = fields.get("vector_similarity")
        chunks.append(SimpleNamespace(**fields))
    return chunks


class RetrievalCache:
    def __init__(self, cache, retrieval_mode, use_reranker):
        self.cache = cache
        self.config_fingerprint = fingerprint({
            "pipeline_version": RETRIEVAL_PIPELINE_VERSION,
            "embedding_model": EMBEDDING_MODEL,
            "candidate_k": RETRIEVAL_CANDIDATE_K,
            "lexical_k": LEXICAL_CANDIDATE_K,
            "hybrid_k": HYBRID_CANDIDATE_K,
            "rrf_k": RRF_K,
            "reranker": RERANK_MODEL if use_reranker else None,
            "retrieval_mode": retrieval_mode,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        })

    def key(self, query, candidate_k, final_k, corpus_version=None):
        version = get_corpus_version() if corpus_version is None else corpus_version
        normalized = normalize_query(query)
        request_fingerprint = fingerprint({
            "candidate_k": candidate_k,
            "final_k": final_k,
            "query_strategy": "original",
        })
        return (
            f"retrieval:{version}:{self.config_fingerprint}:"
            f"{request_fingerprint}:{hash_text(normalized)}"
        ), version

    def get(self, query, candidate_k, final_k):
        if not self.cache.enabled:
            value, stats = self.cache.lookup_json("disabled")
            return value, stats, None, None
        key, version = self.key(query, candidate_k, final_k)
        value, stats = self.cache.lookup_json(key)
        if value is None:
            return None, stats, key, version
        candidates = hydrate_chunks(value["candidate_chunks"])
        final = hydrate_chunks(value["final_chunks"])
        # A missing ID means the database and corpus version disagree. Fail closed
        # on the cache entry and recompute instead of serving partial rankings.
        if len(candidates) != len(value["candidate_chunks"]) or len(final) != len(
            value["final_chunks"]
        ):
            self.cache.delete(key)
            return None, stats.__class__(
                hit=False,
                cache_lookup_seconds=stats.cache_lookup_seconds,
                available=stats.available,
            ), key, version
        return (candidates, final), stats, key, version

    def set(self, key, candidates, final):
        if key is None:
            return False
        return self.cache.set_json(
            key,
            {
                "candidate_chunks": [describe_chunk(chunk) for chunk in candidates],
                "final_chunks": [describe_chunk(chunk) for chunk in final],
            },
            RETRIEVAL_CACHE_TTL,
        )
