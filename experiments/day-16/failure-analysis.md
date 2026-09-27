# Failure and limitation analysis

## 500K tail-latency outlier

At 500K, exact-vector warm P50/P95 were 142.0/164.1 ms, but P99 reached 1,481.8 ms; hybrid P99 reached 1,508.0 ms. The raw query-level samples are preserved. The outlier coincided with a working set larger than comfortable host memory and should not be hidden by reporting only P50. A repeated isolated run would be required to distinguish cache eviction, Docker scheduling, storage I/O, and unrelated host activity.

## 1M safe stop

The 1M tier was not attempted. After 500K, host memory was 83.8% with 2.49 GiB available. The measured table plus indexes occupied 1.10 GB, so a roughly linear projection to 1M would consume most of the remaining memory margin and likely violate the fixed 90% safety gate. The largest healthy measured tier is therefore 500K.

## Semantic rank degradation

The frozen real corpus contains only 4 documents, 17 chunks, and 1,457 recorded tokens. Its fixed 50 labelled questions scored Hit@1/3/5, Recall@5, and MRR of 1.0. No genuine, diverse 10K–500K semantic corpus was available locally. Synthetic random vectors are unsuitable distractors for a quality claim, so `rank-degradation.csv` explicitly records scaled rank as N/A instead of manufacturing a result. Consequently, a “worst ten degraded questions” analysis is not applicable.

## End-to-end generation

Generation was excluded from the primary experiment to isolate retrieval. A 20-question end-to-end sample was also skipped: the preceding load experiment showed the local Qwen path at zero goodput with host memory reaching 93%, which already violates this experiment's 90% guard. Repeating that unsafe condition would not improve the retrieval-scale conclusion.

## Scope

The templated text and seeded vectors test PostgreSQL, pgvector, GIN, batching, storage, and query mechanics. They do not represent natural document diversity, production embedding distributions, multi-user concurrency, or semantic relevance.
