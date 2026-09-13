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

## Security experiment

The security test uses a separate `ragdb_security` database and `datasets/security`; it does not ingest synthetic attacks into the normal corpus. Six synthetic poisoned documents include instruction override, prompt extraction, citation hijack, output hijack, social-engineering, and retrieval-poisoning cases. No real credentials are used.

```powershell
python -m evaluation.setup_security_db
$previousDatabaseUrl = $env:DATABASE_URL
$env:DATABASE_URL = "postgresql://rag:rag@localhost:5432/ragdb_security"
python -m src.ingest datasets/security
python -m evaluation.annotate_security_provenance
$env:DATABASE_URL = $previousDatabaseUrl
python -m evaluation.evaluate_security --mode all --batch-size 4 --max-new-tokens 32
python -m evaluation.recount_security_tokens
python -m evaluation.score_security
```

The evaluator always checks that it is connected to `ragdb_security`; it refuses to run on `ragdb`. It saves one raw-result CSV per mode and checkpoints after each model batch. `SECURITY_MODE=layered` enables source delimiting, injection flags, and an output guard for synthetic test markers. Neither regex detection nor marker blocking proves the answer is factually grounded. See `experiments/day-10/report.md` for results and limitations.
