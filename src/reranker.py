from dataclasses import dataclass

import torch
from sentence_transformers import CrossEncoder

from src.config import RERANK_MODEL
from src.retriever import RetrievedChunk


@dataclass
class RerankedChunk:
    chunk: RetrievedChunk
    original_rank: int
    rerank_score: float


class Reranker:
    def __init__(self, model_name=RERANK_MODEL):
        self.model_name = model_name
        print(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name, activation_fn=torch.nn.Sigmoid())

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 5):
        if not chunks:
            return []
        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self.model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False,
        )
        ranked = [
            RerankedChunk(chunk=chunk, original_rank=index, rerank_score=float(score))
            for index, (chunk, score) in enumerate(zip(chunks, scores), start=1)
        ]
        ranked.sort(key=lambda item: item.rerank_score, reverse=True)
        return ranked[:top_k]
