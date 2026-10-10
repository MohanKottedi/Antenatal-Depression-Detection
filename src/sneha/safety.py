
import re


RISK_PATTERNS = [
    r"\bsuicid(?:e|al)\b",
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\bwant to die\b",
    r"\bwish i were dead\b",
    r"\bbetter off dead\b",
    r"\bhurt myself\b",
    r"\bharm myself\b",
    r"\bself[- ]harm\b",
    r"\bcan't keep myself safe\b",
    r"\bcannot keep myself safe\b",
]

CLEAR_DENIALS = [
    r"\bno suicidal thoughts\b",
    r"\bnot suicidal\b",
    r"\bdon't have thoughts of hurting myself\b",
    r"\bdo not have thoughts of hurting myself\b",
    r"\bno thoughts of hurting myself\b",
    r"\bnot having thoughts of harming myself\b",
]


def may_indicate_self_harm(message: str) -> bool:
    """Conservative keyword trigger for a safety follow-up."""

    text = message.lower().strip()

    if any(re.search(pattern, text) for pattern in CLEAR_DENIALS):
        return False

    return any(
        re.search(pattern, text)
        for pattern in RISK_PATTERNS
    )
