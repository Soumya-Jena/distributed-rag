# Day 6 — Embedding Model Evaluation

## Research Question

How does embedding-model choice affect semantic retrieval and downstream RAG quality while chunking remains fixed?

## Hypothesis

Retrieval-specific embedding models such as BGE and E5 will outperform the general MiniLM baseline on technical question-to-passage retrieval, at the cost of additional embedding latency or resource use.

## Controlled Variables

- Corpus and evaluation questions
- Chunk size: 100 tokens
- Chunk overlap: 20 tokens
- Top-K: 5
- Vector dimension: 384
- Cosine distance
- PostgreSQL/pgvector database
- Qwen2.5-1.5B-Instruct generator
- Prompt, hardware, and manual scoring rubric

## Independent Variable

The embedding model: MiniLM, BGE-small-en-v1.5, or E5-small-v2.
