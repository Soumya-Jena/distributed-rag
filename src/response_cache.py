"""Strict, experimental final-response caching with evidence provenance."""

import time
from dataclasses import fields

from src.cache import fingerprint, get_corpus_version, hash_text, normalize_query
from src.config import (
    CONTEXT_DEDUP_THRESHOLD,
    CONTEXT_KEEP_RATIO,
    CONTEXT_NEIGHBOR_WINDOW,
    CONTEXT_TOKEN_BUDGET,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    GROUNDING_PROMPT_VERSION,
    HYBRID_CANDIDATE_K,
    LEXICAL_CANDIDATE_K,
    QUERY_TRANSFORM_MODEL,
    QUERY_TRANSFORM_PROMPT_VERSION,
    QUERY_VARIANTS,
    RERANK_MODEL,
    RETRIEVAL_CANDIDATE_K,
    RRF_K,
    RESPONSE_CACHE_TTL,
    SECURITY_PROMPT_VERSION,
)
from src.grounding import extract_evidence_status
from src.metrics import citations_valid
from src.retrieval_cache import describe_chunk, hydrate_chunks


class ResponseCache:
    def __init__(
        self, cache, retrieval_fingerprint, context_strategy,
        query_strategy, grounding_mode, security_mode,
        generator_model=GENERATION_MODEL,
    ):
        self.cache = cache
        self.pipeline_fingerprint = fingerprint({
            "retrieval": retrieval_fingerprint,
            "embedding_model": EMBEDDING_MODEL,
            "candidate_k": RETRIEVAL_CANDIDATE_K,
            "lexical_k": LEXICAL_CANDIDATE_K,
            "hybrid_k": HYBRID_CANDIDATE_K,
            "rrf_k": RRF_K,
            "reranker": RERANK_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "context_strategy": context_strategy,
            "context_dedup_threshold": CONTEXT_DEDUP_THRESHOLD,
            "context_keep_ratio": CONTEXT_KEEP_RATIO,
            "context_token_budget": CONTEXT_TOKEN_BUDGET,
            "context_neighbor_window": CONTEXT_NEIGHBOR_WINDOW,
            "query_strategy": query_strategy,
            "query_transform_model": QUERY_TRANSFORM_MODEL,
            "query_transform_prompt": QUERY_TRANSFORM_PROMPT_VERSION,
            "query_variants": QUERY_VARIANTS,
            "generator": generator_model,
            "grounding_mode": grounding_mode,
            "grounding_prompt": GROUNDING_PROMPT_VERSION,
            "security_mode": security_mode,
            "security_prompt": SECURITY_PROMPT_VERSION,
        })

    def key(self, question, top_k):
        version = get_corpus_version()
        request = fingerprint({"top_k": top_k})
        return (
            f"response:{version}:{self.pipeline_fingerprint}:{request}:"
            f"{hash_text(normalize_query(question))}"
        )

    def get(self, key):
        value, stats = self.cache.lookup_json(key)
        if value is None:
            return None, None, stats
        chunks = hydrate_chunks(value.pop("chunk_descriptors"))
        if len(chunks) != value.pop("chunk_count"):
            self.cache.delete(key)
            return None, None, stats.__class__(
                hit=False,
                cache_lookup_seconds=stats.cache_lookup_seconds,
                available=stats.available,
            )
        return value, chunks, stats

    def set(self, key, result):
        value = {
            field.name: getattr(result, field.name)
            for field in fields(result)
            if field.name not in {"question", "chunks", "response_cache_hit"}
        }
        value["chunk_descriptors"] = [
            describe_chunk(chunk) for chunk in result.chunks
        ]
        value["chunk_count"] = len(result.chunks)
        return self.cache.set_json(key, value, RESPONSE_CACHE_TTL)

    @staticmethod
    def eligible(result):
        status = extract_evidence_status(result.answer)
        return bool(
            result.chunks
            and status in {"SUPPORTED", "PARTIAL"}
            and citations_valid(result.answer, len(result.chunks))
        )

    def wait_for_fill(self, key, timeout=2.0, interval=0.05):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            time.sleep(interval)
            value, chunks, stats = self.get(key)
            if value is not None:
                return value, chunks, stats
        return None, None, None
