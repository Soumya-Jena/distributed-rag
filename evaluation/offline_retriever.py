from pathlib import Path

import numpy as np

from src.chunking import chunk_text
from src.document_loader import discover_documents, load_document
from src.embedding_service import EmbeddingService
from src.retriever import RetrievedChunk


class OfflineRetriever:
    """Equivalent normalized-cosine retriever for experiments when ragdb is offline."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_service = EmbeddingService(model_name)
        self.chunks = []
        for path in discover_documents(Path("datasets/raw")):
            for item in chunk_text(
                load_document(path), self.embedding_service.model, 100, 20
            ):
                self.chunks.append(
                    RetrievedChunk(
                        chunk_id=len(self.chunks),
                        title=path.stem,
                        source_path=str(path),
                        chunk_index=item["chunk_index"],
                        content=item["content"],
                        similarity=0.0,
                    )
                )
        self.embeddings = self.embedding_service.encode_documents(
            [chunk.content for chunk in self.chunks]
        )

    def search(self, query, top_k=5):
        query_embedding = self.embedding_service.encode_query(query)
        similarities = self.embeddings @ query_embedding
        order = np.argsort(-similarities)[:top_k]
        return [
            RetrievedChunk(
                chunk_id=self.chunks[index].chunk_id,
                title=self.chunks[index].title,
                source_path=self.chunks[index].source_path,
                chunk_index=self.chunks[index].chunk_index,
                content=self.chunks[index].content,
                similarity=float(similarities[index]),
            )
            for index in order
        ]
