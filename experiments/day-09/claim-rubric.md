# Claim-Level Grounding Rubric

Each factual claim in a generated answer is classified independently.

## `SUPPORTED`

The retrieved excerpt directly establishes the claim without materially changing its scope or certainty.

## `PARTIALLY_SUPPORTED`

Some of the claim follows from the excerpt, but it adds an unsupported detail, strengthens qualified language, or combines supported and unsupported propositions.

## `UNSUPPORTED`

The retrieved excerpts do not establish the claim. A statement can be factually true in the outside world and still be unsupported in this RAG evaluation.

## Citation fields

- `claims_with_citations`: factual claims carrying at least one citation label.
- `citations_that_support_claim`: cited claims whose cited excerpt directly supports the associated statement.
- `invalid_citation_count`: labels outside the supplied `[S1]` through `[Sn]` range; computed automatically.

Citation-label validity and citation support are different measurements.

## Answer-level fields

- `manual_correctness`: `1` if the response correctly handles the requested answer under the supplied evidence, otherwise `0`.
- `manual_groundedness`: `1` if it contains no unsupported factual claim, otherwise `0`.
- `manual_partial_handling`: for partial questions, `1` only when the supported part is answered and the missing part is explicitly left unresolved.

## Formulas

```text
Faithfulness = supported claims / total factual claims
Hallucination rate = unsupported claims / total factual claims
Citation coverage = cited factual claims / total factual claims
Citation support = citations supporting their claims / cited factual claims
```

Partially supported claims remain visible as a separate count rather than being silently merged into either numerator.
