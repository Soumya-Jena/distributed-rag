# Distributed RAG

An experimental lab for building, evaluating, and benchmarking distributed retrieval-augmented generation systems.

## Project structure

- `src/ingestion` — document loading and preprocessing
- `src/embeddings` — embedding generation and indexing
- `src/retrieval` — distributed retrieval components
- `src/generation` — generation pipelines
- `src/evaluation` — quality and performance evaluation
- `notebooks` — exploratory analysis
- `datasets` — local dataset placeholders and documentation
- `experiments` — experiment configurations and results
- `benchmarks` — repeatable performance benchmarks
- `tests` — automated tests

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```
