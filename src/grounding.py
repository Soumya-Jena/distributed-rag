import re


VALID_STATUSES = {
    "SUPPORTED",
    "PARTIAL",
    "INSUFFICIENT",
}


def extract_evidence_status(answer):
    match = re.search(
        r"EVIDENCE_STATUS:\s*(SUPPORTED|PARTIAL|INSUFFICIENT)",
        answer,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).upper()


def extract_citations(answer):
    return [int(value) for value in re.findall(r"\[S(\d+)\]", answer)]


def invalid_citations(answer, source_count):
    return [
        citation
        for citation in extract_citations(answer)
        if citation < 1 or citation > source_count
    ]


def status_matches_answerability(status, answerability):
    expected = {
        "full": "SUPPORTED",
        "partial": "PARTIAL",
        "none": "INSUFFICIENT",
    }
    return int(status == expected[answerability])


def faithfulness(supported_claims, total_claims):
    if total_claims == 0:
        return None
    return supported_claims / total_claims


def hallucination_rate(unsupported_claims, total_claims):
    if total_claims == 0:
        return None
    return unsupported_claims / total_claims


def citation_coverage(claims_with_citations, total_claims):
    if total_claims == 0:
        return None
    return claims_with_citations / total_claims


def citation_support(citations_that_support_claim, claims_with_citations):
    if claims_with_citations == 0:
        return None
    return citations_that_support_claim / claims_with_citations
