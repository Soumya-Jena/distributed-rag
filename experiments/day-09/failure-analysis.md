# Grounding Failure Analysis

## Unsupported additions

The baseline invented a database-size value, PostgreSQL 9.0 as the introduction version, an incorrect SQLSTATE meaning, and an xCluster conflict-resolution mechanism. The strict prompt removed the database-size and version inventions but retained the SQLSTATE and xCluster failures.

## Overclaims and false premises

Both prompts accepted a false zero-RPO guarantee and the false premise that synchronous replication is required for MVCC. The model treated related replication excerpts as permission to answer rather than checking whether they established the premise.

## Unsupported causal inference

The baseline correctly rejected the claim that obsolete row versions prove the cause of a production CPU spike. The strict arm regressed and said they may contribute to the spike despite no CPU evidence. A stronger prompt improved aggregate faithfulness but did not monotonically improve every trap category.

## Partial answers

Only strict g025 successfully answered the supported consumer-group behavior and stated that the default rebalance timeout was absent. The other 19 partial-arm responses across both modes described the supported mechanism but did not address the missing number, version, or configuration value before the output cap.

## Evidence-status compliance

The strict arm emitted an explicit status for 10 of 35 questions (28.6%), and only 5 statuses matched expected answerability (14.3%). It sometimes labeled an answer `SUPPORTED` while its own text said the fact was not supplied, showing that format presence is not semantic correctness.

## Citations

Neither prompt produced a citation. Citation coverage and citation support are therefore both zero. Invalid-citation count is also zero, but that is not a success: no citation labels existed to validate.

## Truncation

The controlled 32-token generation cap made the complete 70-response local experiment feasible but truncated many answers. This reduced correctness and partial handling, and it gave the strict prompt less room after its evidence-status line. Faithfulness scores include only factual claims actually emitted; incomplete fragments are not counted as claims.

## Recommended follow-up

- Enforce structured status with constrained decoding or a separate classifier.
- Add few-shot examples for partial answers and false-premise rejection.
- Generate citations structurally instead of relying only on prompt compliance.
- Repeat on faster hardware with a larger output cap.
- Add an independent citation-entailment evaluator after manual labels establish ground truth.
