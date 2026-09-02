# Day 5 — Chunk Size Experiment

## Research question

How do chunk size and overlap affect retrieval quality and downstream RAG answer quality?

## Hypothesis

Smaller chunks may improve retrieval precision because each vector represents a narrower semantic concept. However, chunks that are too small may lose surrounding context needed by the language model.

Larger chunks may preserve more context but dilute the semantic representation and introduce irrelevant text. Moderate overlap should reduce information loss at chunk boundaries.

I expect `150/30` or `200/40` to provide the best overall balance. The `100/20` configuration may produce focused retrieval but fragmented context, while `240/40` may reduce chunk count at the cost of less focused embeddings and larger prompts.

This hypothesis is recorded before completing the controlled four-configuration suite. One preliminary `100/20` retrieval file existed beforehand, but it will be regenerated using the finalized measurement workflow.

## Controlled variables

- Corpus: the same four source documents in `datasets/raw`
- Evaluation questions: `datasets/evaluation/questions.jsonl`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Generation model: `Qwen/Qwen2.5-1.5B-Instruct`
- Retrieval depth: Top 5
- Vector metric: cosine distance/similarity
- Database: PostgreSQL 17 with pgvector
- Hardware: the same local CPU-only machine

## Independent variables

| Label | Chunk size | Overlap |
|---|---:|---:|
| `chunk-100-20` | 100 | 20 |
| `chunk-150-30` | 150 | 30 |
| `chunk-200-40` | 200 | 40 |
| `chunk-240-40` | 240 | 40 |
