# Ranking Movement Analysis

The table uses candidate K=20. “Source movement” tracks the first chunk from the relevant document. “Winner movement” tracks how far the chunk selected as rerank rank 1 moved from its vector rank. Positive values mean upward movement.

| Question | Vector source rank | Reranked source rank | Source movement | Winning chunk vector rank | Winner movement |
|---|---:|---:|---:|---:|---:|
| q001 | 1 | 1 | 0 | 1 | 0 |
| q002 | 1 | 1 | 0 | 2 | +1 |
| q003 | 1 | 1 | 0 | 1 | 0 |
| q004 | 1 | 1 | 0 | 1 | 0 |
| q007 | 1 | 1 | 0 | 1 | 0 |
| q008 | 1 | 1 | 0 | 1 | 0 |
| q009 | 1 | 1 | 0 | 2 | +1 |
| q010 | 1 | 1 | 0 | 1 | 0 |
| q011 | 1 | 1 | 0 | 1 | 0 |
| q012 | 1 | 1 | 0 | 1 | 0 |
| q013 | 1 | 1 | 0 | 1 | 0 |
| q014 | 1 | 1 | 0 | 1 | 0 |
| q015 | 1 | 1 | 0 | 1 | 0 |
| q016 | 1 | 1 | 0 | 2 | +1 |
| q017 | 1 | 1 | 0 | 2 | +1 |
| q018 | 1 | 1 | 0 | 1 | 0 |

## Wins

The reranker promoted a different within-document chunk for q002, q009, q016, and q017. These are potential context-quality wins, but document-level labels cannot prove that the promoted chunk is better.

## Regressions

For the manual replication query, the correct chunk remained first while unrelated MVCC and Kafka chunks were promoted into the final top five. The generic MS MARCO reranker appears sensitive to lexical question/passage patterns that do not always correspond to technical-domain relevance.

## Missing hard cases

No question placed its relevant document at vector rank 6–20. The experiment therefore cannot demonstrate the reranker rescuing a first-stage near miss. A larger corpus with chunk-level relevance judgments is required for that test.
