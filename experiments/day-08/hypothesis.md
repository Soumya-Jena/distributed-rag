# Day 8 — Hybrid Lexical + Semantic Retrieval

## Research Question

Does combining lexical full-text search with dense vector retrieval improve retrieval and downstream RAG quality, particularly for technical identifiers and exact terminology?

## Hypothesis

Dense retrieval will perform strongly for conceptual and paraphrased questions.

Lexical retrieval will perform better for exact technical terms such as parameter names, acronyms, error codes, and identifiers.

Combining both retrieval methods with Reciprocal Rank Fusion should improve candidate recall and ranking robustness.

A cross-encoder reranker should further improve the ordering of the fused candidate set.

## Controlled Variables

- Four-document corpus
- 100-token chunks with 20-token overlap
- `sentence-transformers/all-MiniLM-L6-v2` embeddings
- `cross-encoder/ms-marco-MiniLM-L6-v2` reranker
- Qwen2.5-1.5B-Instruct generation model
- Prompt and generation settings
- Fifty answerable evaluation questions plus four unsupported controls
- Twenty candidates per retrieval branch and five final context chunks
- Local machine and PostgreSQL instance
