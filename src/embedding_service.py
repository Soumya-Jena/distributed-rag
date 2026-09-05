from sentence_transformers import SentenceTransformer


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
    def __init__(self, model_name):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported embedding model: {model_name}")

        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
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

    def encode_query(self, query):
        return self.model.encode(
            self.config["query_prefix"] + query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
