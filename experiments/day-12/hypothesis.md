# Context selection and compression

## Research question

Can retrieved context be substantially reduced while preserving or improving answer correctness, faithfulness, citation quality, and relevant-source coverage?

## Hypothesis

Overlapping chunks and retrieval rankings introduce redundant or weakly relevant text. Removing near duplicates and retaining query-relevant sentences should reduce context and prompt tokens without materially lowering answer quality. Excessive compression will remove qualifiers or one side of multi-part evidence, increasing partial or unsupported answers.

## Controlled variables

- Four-document, 17-chunk corpus and 100/20 chunking
- MiniLM embeddings
- Hybrid vector plus PostgreSQL lexical retrieval and RRF
- Original-query strategy, the selected low-latency default from the previous experiment
- No cross-encoder reranker, the measured selected setting
- Strict grounding and layered security controls
- Qwen2.5-1.5B-Instruct with deterministic decoding
- Frozen evaluation questions and local hardware

The only independent variable is the post-retrieval context strategy: full chunks, near-deduplicated chunks, 50% extractive sentence selection, or extractive selection under a global token budget.
