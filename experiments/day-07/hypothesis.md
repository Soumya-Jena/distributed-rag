# Day 7 — Cross-Encoder Reranking

## Research Question

Does second-stage cross-encoder reranking improve retrieval ranking and downstream RAG quality enough to justify its additional latency?

## Hypothesis

Dense vector retrieval provides good candidate recall, but cosine similarity alone may not rank the most useful passages first. A cross-encoder that jointly evaluates each query and candidate should improve Hit@1 and MRR while preserving Hit@5, at the cost of additional latency.

## Controlled Variables

- Corpus and evaluation dataset
- MiniLM embedding model
- 100-token chunks with 20-token overlap
- Qwen generator and prompt
- Normalized cosine first-stage retrieval
- Hardware

## Independent Variables

- Reranking enabled versus disabled
- Candidate set size: 5, 10, 20, or 40

The reranker is `cross-encoder/ms-marco-MiniLM-L6-v2`, and the final context always contains at most five chunks.
