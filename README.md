# Distributed RAG

An experimental lab for building, evaluating, and benchmarking retrieval-augmented generation systems with PostgreSQL, pgvector, full-text search, and local language models.

## Project structure

- `src/ingest.py` — chunk, embed, and store documents
- `src/retriever.py` — dense pgvector retrieval
- `src/lexical_retriever.py` — PostgreSQL full-text retrieval
- `src/fusion.py` — Reciprocal Rank Fusion (RRF)
- `src/hybrid_retriever.py` — vector + lexical candidate generation
- `src/reranker.py` — optional cross-encoder second stage
- `src/rag_service.py` — retrieval, prompting, and local generation
- `evaluation` — repeatable retrieval and RAG experiments
- `datasets` — source documents and evaluation questions
- `experiments` — versioned measurements, reports, and charts
- `tests` — unit tests for retrieval orchestration and fusion

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
docker compose up -d
python -m src.ingest datasets/raw
```

## Hybrid retrieval

The default retrieval path combines dense MiniLM embeddings with PostgreSQL full-text search over a generated `tsvector` column. RRF combines the two rankings without mixing incomparable cosine and `ts_rank_cd` scores. Cross-encoder reranking remains available but is disabled by default because it did not improve the current benchmark.

```powershell
python -m src.rag "How does PostgreSQL streaming replication work?"
python -m src.rag "What is MVCC?" --retrieval-mode hybrid --reranker
python -m src.rag "What is MVCC?" --grounding-mode strict
python -m evaluation.inspect_hybrid
python -m evaluation.evaluate_hybrid --label hybrid
```

Configuration lives in `.env`; copy `.env.example` when setting up a new environment.

## Grounding evaluation

The selected strict evidence contract requires `SUPPORTED`, `PARTIAL`, or `INSUFFICIENT` status, forbids filling gaps from model memory, and requests claim-level citations. The baseline prompt remains available for reproducible A/B evaluation.

```powershell
python -m evaluation.evaluate_grounding --mode baseline --label baseline-grounding --batch-size 4 --max-new-tokens 32
python -m evaluation.evaluate_grounding --mode strict --label strict-grounding --batch-size 4 --max-new-tokens 32
python -m evaluation.score_grounding
python -m evaluation.plot_grounding_results
```

See `experiments/day-09/report.md` for the claim-level review and limitations.
