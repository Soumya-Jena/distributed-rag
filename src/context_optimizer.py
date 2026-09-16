"""Deterministic post-retrieval context selection and compression."""

import math
import re
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np


STRATEGIES = ("full", "deduplicated", "extractive", "budgeted")


@dataclass
class OptimizedChunk:
    source_id: str
    chunk_id: int
    title: str
    source_path: str
    chunk_index: int
    original_content: str
    optimized_content: str
    original_tokens: int
    optimized_tokens: int
    selected_sentence_indices: list[int] = field(default_factory=list)
    similarity: float | None = None
    vector_similarity: float | None = None
    lexical_score: float | None = None
    rrf_score: float | None = None
    found_by: list[str] = field(default_factory=list)
    query_ranks: dict[str, int] = field(default_factory=dict)

    @property
    def content(self):
        return self.optimized_content


@dataclass
class ContextOptimizationResult:
    chunks: list[OptimizedChunk]
    original_tokens: int
    optimized_tokens: int
    removed_tokens: int
    compression_ratio: float
    optimization_seconds: float
    candidates_before: int
    candidates_after: int
    unique_sources: int


def split_sentences(text):
    """Split prose and Markdown lines without downloading tokenizer data."""
    sentences = []
    for block in re.split(r"\n+", text.strip()):
        block = block.strip()
        if not block:
            continue
        parts = re.split(r"(?<=[.!?])\s+", block)
        sentences.extend(part.strip() for part in parts if part.strip())
    return sentences


def deduplicate_exact(chunks):
    seen = set()
    selected = []
    for chunk in chunks:
        key = (chunk.source_path, chunk.chunk_index)
        if key not in seen:
            seen.add(key)
            selected.append(chunk)
    return selected


class NearDuplicateFilter:
    def __init__(self, embedding_service, threshold=0.92):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        self.embedding_service = embedding_service
        self.threshold = threshold

    def filter(self, chunks):
        if len(chunks) <= 1:
            return list(chunks)
        embeddings = self.embedding_service.encode_documents(
            [chunk.content for chunk in chunks]
        )
        selected = []
        selected_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            if any(
                float(np.dot(embedding, existing)) >= self.threshold
                for existing in selected_embeddings
            ):
                continue
            selected.append(chunk)
            selected_embeddings.append(embedding)
        return selected


class SentenceCompressor:
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service

    def select(self, query, text, keep_ratio=0.5, neighbor_window=0):
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be greater than zero and at most one")
        if neighbor_window < 0:
            raise ValueError("neighbor_window must be non-negative")
        sentences = split_sentences(text)
        if len(sentences) <= 1 or keep_ratio == 1:
            return text, list(range(len(sentences)))
        query_embedding = self.embedding_service.encode_query(query)
        embeddings = self.embedding_service.encode_documents(sentences)
        scored = sorted(
            (
                (float(np.dot(query_embedding, embedding)), index)
                for index, embedding in enumerate(embeddings)
            ),
            reverse=True,
        )
        keep = max(1, math.ceil(len(sentences) * keep_ratio))
        chosen = {index for _, index in scored[:keep]}
        expanded = set(chosen)
        for index in chosen:
            expanded.update(range(
                max(0, index - neighbor_window),
                min(len(sentences), index + neighbor_window + 1),
            ))
        indices = sorted(expanded)
        return " ".join(sentences[index] for index in indices), indices


class ContextOptimizer:
    def __init__(
        self, embedding_service, token_counter, dedup_threshold=0.92,
        keep_ratio=0.5, token_budget=1000, neighbor_window=0,
    ):
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        self.token_counter = token_counter
        self.near_duplicates = NearDuplicateFilter(
            embedding_service, dedup_threshold
        )
        self.compressor = SentenceCompressor(embedding_service)
        self.keep_ratio = keep_ratio
        self.token_budget = token_budget
        self.neighbor_window = neighbor_window

    def _wrap(self, chunk, content=None, indices=None):
        original = getattr(chunk, "original_content", chunk.content)
        optimized = content if content is not None else chunk.content
        source_path = chunk.source_path
        chunk_index = chunk.chunk_index
        return OptimizedChunk(
            source_id=f"{source_path}#{chunk_index}",
            chunk_id=getattr(chunk, "chunk_id", f"{source_path}#{chunk_index}"),
            title=chunk.title, source_path=source_path, chunk_index=chunk_index,
            original_content=original, optimized_content=optimized,
            original_tokens=self.token_counter(original),
            optimized_tokens=self.token_counter(optimized),
            selected_sentence_indices=(indices if indices is not None else list(
                range(len(split_sentences(optimized)))
            )),
            similarity=getattr(chunk, "similarity", None),
            vector_similarity=getattr(chunk, "vector_similarity", None),
            lexical_score=getattr(chunk, "lexical_score", None),
            rrf_score=getattr(chunk, "rrf_score", None),
            found_by=list(getattr(chunk, "found_by", [])),
            query_ranks=dict(getattr(chunk, "query_ranks", {})),
        )

    def _compress(self, query, chunks, keep_ratio, neighbor_window):
        result = []
        for chunk in chunks:
            text, indices = self.compressor.select(
                query, chunk.content, keep_ratio, neighbor_window
            )
            result.append(self._wrap(chunk, text, indices))
        return result

    def _fit_budget(self, chunks, budget):
        selected = []
        remaining = budget
        for chunk in chunks:
            if chunk.optimized_tokens <= remaining:
                selected.append(chunk)
                remaining -= chunk.optimized_tokens
                continue
            sentences = split_sentences(chunk.optimized_content)
            kept = []
            for sentence in sentences:
                candidate = " ".join(kept + [sentence])
                if self.token_counter(candidate) <= remaining:
                    kept.append(sentence)
            if kept:
                text = " ".join(kept)
                local_indices = [
                    i for i, sentence in enumerate(sentences) if sentence in kept
                ]
                original_indices = [
                    chunk.selected_sentence_indices[i] for i in local_indices
                ]
                selected.append(self._wrap(
                    chunk, text, original_indices,
                ))
                remaining -= self.token_counter(text)
            if remaining <= 0:
                break
        return selected

    def optimize(
        self, query, chunks, strategy="full", keep_ratio=None,
        token_budget=None, dedup_threshold=None, neighbor_window=None,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown context strategy: {strategy}")
        started = perf_counter()
        original = list(chunks)
        original_tokens = sum(self.token_counter(chunk.content) for chunk in original)
        exact = deduplicate_exact(original)
        if dedup_threshold is not None:
            near_filter = NearDuplicateFilter(
                self.near_duplicates.embedding_service, dedup_threshold
            )
        else:
            near_filter = self.near_duplicates

        if strategy == "full":
            optimized = [self._wrap(chunk) for chunk in original]
        else:
            deduplicated = near_filter.filter(exact)
            if strategy == "deduplicated":
                optimized = [self._wrap(chunk) for chunk in deduplicated]
            else:
                optimized = self._compress(
                    query, deduplicated,
                    self.keep_ratio if keep_ratio is None else keep_ratio,
                    self.neighbor_window if neighbor_window is None else neighbor_window,
                )
                if strategy == "budgeted":
                    optimized = self._fit_budget(
                        optimized,
                        self.token_budget if token_budget is None else token_budget,
                    )

        optimized_tokens = sum(chunk.optimized_tokens for chunk in optimized)
        removed = max(0, original_tokens - optimized_tokens)
        return ContextOptimizationResult(
            chunks=optimized, original_tokens=original_tokens,
            optimized_tokens=optimized_tokens, removed_tokens=removed,
            compression_ratio=(removed / original_tokens if original_tokens else 0.0),
            optimization_seconds=perf_counter() - started,
            candidates_before=len(original), candidates_after=len(optimized),
            unique_sources=len({chunk.source_path for chunk in optimized}),
        )
