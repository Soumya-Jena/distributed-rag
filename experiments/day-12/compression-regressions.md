# Compression regression review

## Method

The five-question generation sample contains one representative question from each frozen query category. A compression-caused regression requires the full-context answer to be correct and the compressed answer to be incorrect for the same question. Reviews use the exact retained fragments saved in `context-rag-results.csv`.

## qt031 — WAL role (`EXACT_IDENTIFIER`)

The full-context answer states that WAL records database changes and begins explaining the primary's role. Both compressed answers instead open a numbered list—“serve two main purposes: 1.”—but hit the 20-token experiment cap before stating either purpose.

- Full correctness: 1
- Extractive correctness: 0
- Budgeted correctness: 0
- Relevant source retained: yes
- Root cause: a generation-format/output-budget interaction, not loss of the labeled document

This still counts as an observed answer regression because the delivered compressed answer is incomplete. It does not prove that the compressor removed the necessary fact.

## qt041 — multi-part question

Every strategy explained the basic replication path but omitted the second requested fact: an asynchronous standby can lose recent commits when it has not received the latest WAL. Because the full baseline also failed, this is a generator/output-cap failure rather than a compression-caused regression.

## Conclusion

One of five sampled questions regressed under compression, and the only multi-part sample failed for every strategy. The sample is too small and the output cap too restrictive to declare 50% extraction quality-preserving. Full context remains the default.
