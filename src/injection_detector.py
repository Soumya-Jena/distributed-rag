"""Heuristic annotation of untrusted retrieved text, not a security boundary."""

import re
from dataclasses import dataclass


PATTERNS = (
    r"ignore\s+(?:all\s+)?previous\s+instructions",
    r"reveal\s+(?:the\s+)?(?:hidden\s+)?system\s+(?:prompt|instructions)",
    r"output\s+only",
    r"higher\s+priority\s+than",
    r"start\s+(?:the\s+)?(?:answer|response)\s+with",
    r"stop\s+answering",
    r"regardless\s+of\s+the\s+question",
    r"security\s+team\s+has\s+approved",
    r"hidden\s+(?:marker|instructions)",
)


@dataclass(frozen=True)
class InjectionDetection:
    suspicious: bool
    matches: tuple[str, ...]


def detect_injection(text: str) -> InjectionDetection:
    matches = tuple(
        pattern for pattern in PATTERNS
        if re.search(pattern, text, re.IGNORECASE)
    )
    return InjectionDetection(bool(matches), matches)
