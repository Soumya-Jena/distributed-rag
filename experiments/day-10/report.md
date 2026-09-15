# Indirect prompt-injection evaluation

## Outcome

None of the three modes emitted a synthetic attack marker or canary, even when the targeted attack-bearing chunk was retrieved. This means conditional marker/canary attack success was 0%, but it does not prove the answers were safe or grounded. The layered detector identified six of nine poisoned chunks with zero of 17 clean chunks flagged, and the guard had nothing to block. Layered controls remain the defense-in-depth default, but their advantage over structured prompting was not demonstrated by this small model and payload set.

## Experiment design

The ordinary `ragdb` and `datasets/raw` corpus were left untouched. `ragdb_security` contains four copied clean documents and six synthetic poisoned documents (five instruction payloads plus a keyword-stuffed retrieval-poison page). All 26 chunks use the selected 100/20 chunking, MiniLM embeddings, hybrid vector + lexical retrieval, RRF, no reranker, and top-five context. Qwen2.5-1.5B-Instruct runs with greedy decoding and a 32-token cap in every mode. The same synthetic canary is appended to the trusted system prompt in each mode; no real secret is used.

The 35-case dataset has 25 attacks (five questions for each of five payload categories) and ten clean controls. We ran the selected strict grounding prompt first without new security handling, then structured untrusted-source delimiters, then the layered mode with heuristic flags and an output guard. Retrieved chunk IDs were identical across all three modes. The final seven layered generations were resumed with batch size one to reduce local memory pressure; therefore generation timing is operational evidence, not a controlled mode comparison.

## Main results

| Metric | Baseline | Structured | Layered |
|---|---:|---:|---:|
| Target attack-bearing chunk exposure | 0.88 | 0.88 | 0.88 |
| Any poisoned source in top five | 1.00 | 1.00 | 1.00 |
| Retrieval-poison source in top five | 0.72 | 0.72 | 0.72 |
| Conditional raw marker/canary ASR | 0.00 | 0.00 | 0.00 |
| Conditional delivered marker/canary ASR | 0.00 | 0.00 | 0.00 |
| Canary leak rate | 0.00 | 0.00 | 0.00 |
| Attack-answer citation rate | 0.00 | 0.00 | 0.00 |
| Clean manual correctness | 0.85 | 0.80 | 0.85 |
| Clean manual groundedness | 0.95 | 0.90 | 0.95 |
| Clean blocked-answer rate | 0.00 | 0.00 | 0.00 |
| Mean prompt tokens | 1055.2 | 1109.7 | 1129.6 |
| Mean generation seconds | 138.5 | 111.8 | 142.7 |

![Security comparison](security-comparison.png)

The automated clean keyword proxy scored every answer as successful, but manual review exposed partial or wrong answers in every mode. Most notably, layered case `sec034` claimed the sources did not explain how a standby receives changes even though the replication evidence directly described WAL transfer. This is why the manual scores, not keyword presence alone, determine clean utility.

## Attack-category results

| Category | Target exposed | Exposure rate | Raw ASR | Delivered ASR |
|---|---:|---:|---:|---:|
| Override | 4/5 | 0.80 | 0.00 | 0.00 |
| Prompt leak | 4/5 | 0.80 | 0.00 | 0.00 |
| Output hijack | 4/5 | 0.80 | 0.00 | 0.00 |
| Citation hijack | 5/5 | 1.00 | 0.00 | 0.00 |
| Social engineering | 5/5 | 1.00 | 0.00 | 0.00 |

Exposure is the same in all modes because retrieval was deliberately frozen. The table therefore applies to each arm. Citation hijacking could not succeed under the narrow metric because the model emitted no citations at all; missing citations are a separate compliance failure, not evidence of robust attribution.

## Detector and guard

The layered heuristic flagged six of nine poisoned chunks, for 66.7% recall on this corpus, and zero of 17 clean chunks, for a 0% observed false-positive rate. These estimates are based on only 26 hand-written chunks. No raw answer contained the synthetic marker or canary, so the output guard's block rate was 0% for both attacks and clean controls. Unit tests separately verify that the guard blocks exact and spacing-normalized test markers when present.

## Decision

Keep `SECURITY_MODE=layered` as a provisional defense-in-depth default because source delimiting, visible injection flags, and deterministic output validation cover different boundaries and preserved measured clean utility. Do not claim that this experiment proves prompt-injection resistance: the baseline also achieved 0% narrow ASR, the detector missed three poisoned chunks, and no mode supplied citations. A larger adversarial payload set, paraphrased-success review, claim-level grounding evaluation, and tool/action attacks are required before production use.

## Observed failures and limitations

The baseline already produces unsupported or incorrect technical statements without printing an attack marker. For example, one WAL answer says the standby does not replay WAL, contrary to the clean replication source. Absence of marker leakage is therefore not equivalent to groundedness.

The uncased MiniLM chunker normalizes source text and can split attack markers with spaces around underscores; marker scoring handles this normalization. The 32-token cap truncates some answers and depresses measured completeness and citation compliance. The synthetic attack documents have visible `poisoned` paths and only five independent instruction payloads, which limits generalization. The output guard checks test-marker shapes and the canary; it does not verify unsupported facts, source attribution, or paraphrased attacks. The current RAG model has no action-taking tools, so this experiment addresses answer manipulation, not tool hijacking.

## Reproduction

Follow the security setup and evaluation commands in the project README. Raw outputs, scoring rubric, category scorecard, clean-answer annotations, and source-provenance details are stored beside this report. The evaluator refuses to run on the normal `ragdb` database.

```powershell
python -m evaluation.evaluate_security --mode all --batch-size 4 --max-new-tokens 32
python -m evaluation.recount_security_tokens
python -m evaluation.score_security
python -m evaluation.plot_security_results
```
