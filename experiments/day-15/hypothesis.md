# Day 15 — Load and Concurrency Testing

## Research Question

How does the RAG platform behave as concurrency increases, and which component becomes the first scalability bottleneck?

## Hypothesis

End-to-end throughput will initially increase with concurrency. Beyond a saturation point, additional concurrency will produce little throughput improvement while P95/P99 latency and queueing increase sharply.

Local LLM generation is expected to become the dominant bottleneck before PostgreSQL vector/lexical retrieval. Caching should materially improve repeated-query workloads but have less impact on unique-query workloads.

## Controlled Variables

Corpus, models, retrieval architecture, prompt, security policy, cache configuration, hardware, and observability configuration are frozen from the preceding experiments.
