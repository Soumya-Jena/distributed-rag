# Context-optimization metric rubric

- **Context tokens**: actual Qwen tokenizer count for evidence text only.
- **Prompt tokens**: actual tokenized chat-template input supplied to Qwen.
- **Output tokens**: actual Qwen tokenizer count for generated text.
- **Compression ratio**: `1 - optimized_context_tokens / original_context_tokens`.
- **Relevant-source retention**: whether at least one labeled source remains after optimization.
- **Unique sources**: number of distinct source paths represented after optimization.
- **Correctness**: manual score 1, 0.5, or 0 for correct, partial, or wrong.
- **Faithfulness**: manual score 1, 0.5, or 0 according to support from the exact optimized fragments.
- **Citation support**: manual score 1, 0.5, or 0 according to whether cited fragments support the associated answer claims.
- **Evidence-density proxy**: labeled relevant chunks per 100 context tokens. This is explicitly a project-specific retrieval diagnostic, not claim-level factual density.
- **Compression-caused regression**: baseline correctness exceeds compressed correctness for the same question.
- **Quality-preserving compression**: largest observed token reduction without an obvious correctness, faithfulness, citation-support, or source-retention decline on this small dataset.

Candidate and source metrics are automatic. Answer-quality scores require review of the generated answer against the exact retained fragments; keyword presence is never labeled correctness.
