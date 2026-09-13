"""Test-marker output block. Not a factual-grounding verifier."""

import re
from dataclasses import dataclass



BLOCKED_RESPONSE = (
    "The response was blocked because retrieved content attempted to "
    "influence the model's instructions."
)


@dataclass(frozen=True)
class OutputValidation:
    safe: bool
    issues: tuple[str, ...]


def validate_output(answer: str, canary: str | None = None) -> OutputValidation:
    issues = []
    if canary and canary.casefold() in answer.casefold():
        issues.append("security_canary_leak")
    if re.search(r"attack\s*_?\s*marker\s*_?\s*[a-z0-9_]+", answer, re.IGNORECASE):
        issues.append("attack_marker")
    return OutputValidation(not issues, tuple(issues))


def guard_output(answer: str, canary: str | None = None) -> tuple[str, OutputValidation]:
    validation = validate_output(answer, canary)
    return (answer if validation.safe else BLOCKED_RESPONSE), validation
