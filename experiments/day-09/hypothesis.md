# Day 9 — Hallucination and Groundedness Evaluation

## Research Question

When relevant evidence is supplied by the RAG pipeline, does the generation model restrict its factual claims to information supported by that evidence?

## Hypothesis

The existing grounding prompt will reduce hallucinations, but the model will occasionally:

1. use prior knowledge not contained in retrieved context,
2. strengthen qualified statements into absolute claims,
3. answer questions for which evidence is incomplete,
4. attach citations that exist but do not support the claim.

A stricter evidence contract should improve faithfulness and refusal behavior, potentially at the cost of increased over-refusal.

## Controlled Variables

- Corpus
- Chunking
- Embedding model
- Vector retrieval
- Lexical retrieval
- RRF configuration
- Cross-encoder setting
- Top-K
- Generation model
- Generation parameters
- Hardware

## Independent Variable

Grounding instruction: baseline versus strict evidence contract.

## Evidence states

- `SUPPORTED`: enough evidence exists to answer substantially all of the question.
- `PARTIAL`: the evidence supports only part of the requested answer.
- `INSUFFICIENT`: the evidence does not support a useful answer.

## Failure taxonomy

- Unsupported addition
- Overclaim
- Citation mismatch
- Correct but ungrounded statement
