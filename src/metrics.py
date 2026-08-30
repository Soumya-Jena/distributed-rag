import re

from pathlib import Path


REFUSAL_TEXT = (
    "i don't have enough information "
    "in the provided sources"
)


def source_name(chunk):
    return Path(
        chunk.source_path
    ).name


def hit_at_k(
    chunks,
    relevant_sources,
    k,
):

    relevant = set(
        relevant_sources
    )

    retrieved = {
        source_name(chunk)
        for chunk in chunks[:k]
    }

    return int(
        bool(relevant & retrieved)
    )


def recall_at_k(
    chunks,
    relevant_sources,
    k,
):

    if not relevant_sources:
        return None

    relevant = set(
        relevant_sources
    )

    retrieved = {
        source_name(chunk)
        for chunk in chunks[:k]
    }

    found = relevant & retrieved

    return (
        len(found)
        / len(relevant)
    )


def reciprocal_rank(
    chunks,
    relevant_sources,
):

    relevant = set(
        relevant_sources
    )

    for rank, chunk in enumerate(
        chunks,
        start=1,
    ):

        if source_name(chunk) in relevant:
            return 1 / rank

    return 0.0


def keyword_coverage(
    answer,
    expected_keywords,
):

    if not expected_keywords:
        return None

    answer_lower = answer.lower()

    found = sum(
        1
        for keyword in expected_keywords
        if keyword.lower() in answer_lower
    )

    return (
        found
        / len(expected_keywords)
    )


def extract_citations(answer):

    matches = re.findall(
        r"\[S(\d+)\]",
        answer,
    )

    return [
        int(match)
        for match in matches
    ]


def has_citation(answer):

    return int(
        len(
            extract_citations(answer)
        ) > 0
    )


def citations_valid(
    answer,
    retrieved_count,
):

    citations = extract_citations(
        answer
    )

    if not citations:
        return False

    return all(
        1 <= citation <= retrieved_count
        for citation in citations
    )


def refused(answer):

    return int(
        REFUSAL_TEXT
        in answer.lower()
    )