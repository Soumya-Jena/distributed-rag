# Context selection and compression checklist

- [x] Freeze the selected corpus, retrieval, grounding, security, query, and generator settings.
- [x] Capture the Top-5 full-context baseline.
- [x] Count context, prompt, and output tokens with the Qwen tokenizer.
- [x] Preserve original chunks and source/chunk/sentence provenance.
- [x] Implement exact and embedding-based near-duplicate removal.
- [x] Sweep near-duplicate thresholds 0.85, 0.90, 0.92, and 0.95.
- [x] Implement deterministic sentence splitting and original-query sentence scoring.
- [x] Restore original sentence order and test a one-sentence neighbor window.
- [x] Sweep 100%, 75%, 50%, and 25% retained sentences.
- [x] Test global budgets of 500, 750, 1000, 1500, and 2000 tokens.
- [x] Add a corpus-scaled 250-token diagnostic because the requested budgets do not bind after extraction.
- [x] Avoid mid-sentence budget truncation.
- [x] Record source diversity and inspect multi-part questions.
- [x] Compare full, de-duplicated, extractive, and budgeted strategies.
- [x] Calculate compression ratio and a project-specific evidence-density proxy.
- [x] Run the 50-question retrieval-side evaluation by query type.
- [x] Run and manually review a balanced local-Qwen answer-quality sample.
- [x] Record compression-caused failures.
- [x] Produce a comparison chart and final decision report.
- [x] Integrate the strategy into the RAG CLI and document reproduction commands.
- [x] Add focused optimizer tests and run the full test suite.

Abstractive compression was intentionally not selected: extractive compression already exposed a quality risk, while a second generative stage would add latency and a new hallucination surface.
