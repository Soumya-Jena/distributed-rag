# RAG caching and invalidation

## Research question

How much serving latency and repeated computation can be eliminated through caching while preserving freshness, grounding, and citation correctness?

## Hypothesis

Embedding and query-transformation caching should provide low-risk latency savings because their outputs depend mainly on stable model and prompt inputs. Retrieval caching should save more work for repeated queries but must be isolated whenever the corpus changes. Full-response caching should provide the largest saving but carries the greatest freshness and correctness risk.

Including corpus and configuration fingerprints in every relevant key should prevent stale cross-version reuse. Redis failure should degrade performance, not correctness or availability.

## Controlled variables

- Four-document corpus and current chunk configuration
- MiniLM embedding model
- Hybrid vector and lexical retrieval with RRF
- No reranker and original-query serving path
- Full context, strict grounding, and layered security
- Local hardware and frozen 100-request workload

## Independent variables

- Cache layer
- Hit or miss state
- TTL
- Corpus version
