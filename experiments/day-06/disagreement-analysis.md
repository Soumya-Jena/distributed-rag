# Embedding Disagreement Analysis

All models ranked the correct document first for every answerable question. The disagreements below concern which 100-token chunk within that document was ranked first; this is still important because the generator receives chunk text, not merely a document name.

| Question | MiniLM chunk | BGE chunk | E5 chunk | Observation |
|---|---:|---:|---:|---|
| q002 — MVCC and visibility | 3 | 0 | 0 | MiniLM favored the explicit concluding contrast between MVCC and replication; BGE/E5 favored the opening definition and row-version explanation. |
| q003 — selecting a node | 0 | 3 | 3 | MiniLM favored the scheduler overview/filtering passage; BGE/E5 favored the later placement and kubelet context. |
| q009 — logical vs physical replication | 4 | 1 | 4 | MiniLM/E5 selected the direct physical/logical comparison; BGE selected the asynchronous/synchronous replication passage, which is less directly responsive. |
| q010 — snapshots and visible versions | 0 | 0 | 1 | MiniLM/BGE selected the opening MVCC passage; E5 selected the chunk that explicitly describes per-isolation-level snapshots. E5's choice is more precise. |
| q015 — after Pod binding | 0 | 3 | 3 | BGE/E5 selected the exact kubelet/container-runtime passage; MiniLM selected the scheduler overview. BGE/E5 provide better generation context. |

## Interpretation

Retrieval metrics saturated because the corpus contains only four strongly separated topics. Chunk-level inspection reveals useful differences hidden by document-level Hit@K and MRR. E5 did particularly well on paraphrased procedural questions (`q010`, `q015`), consistent with its query/passage training format. BGE's `q009` selection is a regression despite its correct-document rank.

These are hypotheses from a small corpus, not general model claims. A future evaluation should add chunk-level relevance labels so these differences can be scored directly rather than inspected manually.
