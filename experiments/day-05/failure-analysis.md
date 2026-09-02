# Day 5 — Retrieval and Boundary Analysis

## Retrieval failures

There were no source-ranking failures in the four-question answerable dataset. Every configuration achieved Hit@1, Hit@3, Hit@5, Recall@5, and MRR of `1.000`. This is a limitation of the small, topically distinct evaluation set: the aggregate metrics are saturated and cannot distinguish configurations.

The top-1 similarity scores still changed:

| Question | 100/20 | 150/30 | 200/40 | 240/40 |
|---|---:|---:|---:|---:|
| PostgreSQL replication | 0.8039 | 0.8128 | 0.8084 | 0.7767 |
| PostgreSQL MVCC | 0.6871 | 0.6463 | 0.5873 | 0.6970 |
| Kubernetes scheduling | 0.6597 | 0.6874 | 0.6662 | 0.7074 |
| Kafka overview | 0.6000 | 0.5419 | 0.5350 | 0.5112 |

### Smaller chunks helped Kafka retrieval

The Kafka question's top-1 similarity fell from `0.6000` at `100/20` to `0.5112` at `240/40`. The smaller chunk concentrated on Kafka's primary event-streaming purpose. The larger chunk mixed that definition with partitioning, replication, brokers, and consumer behavior, diluting the query-specific semantic signal.

### Larger chunks helped MVCC and scheduling retrieval

MVCC similarity was highest at `240/40` (`0.6970`) and lowest at `200/40` (`0.5873`). The 240-token MVCC chunk retained the definition, snapshot visibility rules, isolation levels, and concurrency behavior together. Similarly, Kubernetes scheduling improved to `0.7074` at `240/40`, where filtering and scoring context remained in one chunk.

### Rank metrics hid meaningful changes

Although MRR was tied, average top-1 similarity ranged from `0.6492` (`200/40`) to `0.6877` (`100/20`). This demonstrates why a perfect Hit@K or MRR on a tiny dataset must not be treated as proof that configurations are equivalent.

## Boundary effects

Exact exported chunks are preserved in `boundaries/`.

### Case 1 — PostgreSQL replication

At `100/20`, the explanation spans multiple chunks: the initial WAL transfer and replay are in chunk 0, asynchronous behavior continues into chunk 1, and synchronous acknowledgements cross into chunk 2. The 20-token overlap retains part of the primary/standby transaction discussion at each boundary, but a single retrieved chunk may omit failover or durability trade-offs.

At `200/40`, most transfer and asynchronous behavior stays in chunk 0, while synchronous replication, failover, slots, and physical/logical distinctions occupy chunk 1. This is more complete context for generation without sending the whole document.

### Case 2 — Kubernetes scheduling

At `100/20`, filtering, scoring, resource requests, and kubelet execution are distributed across four chunks. Overlap preserves transitions, but the final 40-token chunk explains that the scheduler selects a node while the kubelet starts containers; this distinction can be missed if only the first result is used.

At `240/40`, filtering, scoring, and resource-request behavior mostly remain together, explaining the higher top-1 similarity. The final node-affinity and kubelet material still falls into a second chunk.

### Case 3 — PostgreSQL MVCC

At `100/20`, the final 30-token chunk contains the important contrast that MVCC controls visibility while WAL handles replication. A question comparing MVCC and replication may need adjacent chunks. At `240/40`, nearly the entire MVCC explanation is preserved in the first chunk, improving context completeness but increasing prompt size.

## Candidates selected for full RAG evaluation

- `100/20`: highest average top-1 similarity and most focused chunks.
- `150/30`: near-best similarity with 35% fewer chunks than `100/20`; expected balance candidate.
- `240/40`: strongest MVCC and Kubernetes similarities and a useful large-context contrast.

The `200/40` baseline remains in the retrieval comparison but is not sent through the expensive full-RAG run because it had the lowest average top-1 similarity among configurations with otherwise identical rank metrics.
