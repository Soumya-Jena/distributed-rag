# Query Transformation and Multi-Query Retrieval

## Research question

Can query rewriting and multi-query expansion improve retrieval for vague, paraphrased, and terminology-mismatched questions without materially hurting exact-term retrieval or increasing latency excessively?

## Hypothesis

The original query should remain strongest for exact technical identifiers. A semantic rewrite may help when user vocabulary differs from the documentation. Multiple meaning-preserving queries may increase difficult-question candidate recall, but poor rewrites can invent terminology, dilute exact identifiers, drift from intent, and add latency.

## Controlled variables

Corpus, 100/20 chunking, MiniLM embeddings, PostgreSQL lexical retrieval, first-level RRF, no reranker (the measured retrieval winner), Qwen generator, strict grounding, layered security controls, evaluation questions, and hardware remain fixed.

## Independent variable

1. Original query only
2. Semantic rewrite only
3. Original plus semantic rewrite
4. Original plus semantic, technical, and alternate variants

All ensemble strategies preserve the user's original query. When a cross-encoder is enabled for later experiments, final candidates are reranked against the original user question, never a generated rewrite.
