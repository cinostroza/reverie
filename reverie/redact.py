"""Secret redaction, applied before first write (HLD 13.2).

Redaction is a *pre-write* control, not cleanup. Once a token reaches the
episode buffer it is on disk and, after consolidation, possibly paraphrased
into prose where no regex will ever find it again.

This catches known formats plus high-entropy strings. It will not catch a
proprietary internal identifier that happens to be sensitive -- see HLD 13.3.
`reverie forget` is the backstop that actually works.
"""

from __future__ import annotations

import math
import re
from collections import Counter

__all__ = ["redact", "shannon_entropy", "RedactionStats"]

# Known-format credentials. Ordered most to least specific.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("private_key", re.compile(r"-----BEGIN[ A-Z]*PRIVATE KEY-----[\s\S]*?-----END[ A-Z]*PRIVATE KEY-----")),
    ("jwt", re.compile(r"\bey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("conn_string", re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s:@/]+:[^\s:@/]+@[^\s/]+")),
    ("bearer", re.compile(r"(?i)\b(?:bearer|token|api[-_]?key)\s*[:=]\s*[\"']?([A-Za-z0-9_\-.]{16,})[\"']?")),
    ("env_assign", re.compile(
        r"(?i)\b([A-Z_]*(?:SECRET|PASSWORD|PASSWD|TOKEN|APIKEY|API_KEY|CREDENTIAL)[A-Z_]*)\s*=\s*[\"']?([^\s\"']{6,})[\"']?"
    )),
]

# Strings that look random enough to be a credential even without a known
# prefix. Tuned deliberately conservative: a false positive destroys a memory,
# a false negative is caught by the pattern list or by `forget`.
_CANDIDATE = re.compile(r"\b[A-Za-z0-9+/=_-]{28,}\b")
_ENTROPY_THRESHOLD = 3.9

# Things that trip the entropy heuristic but are not secrets.
_ENTROPY_ALLOW = re.compile(
    r"(?i)^(?:[0-9a-f]{7,40}|sha256:[0-9a-f]+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$"
)


class RedactionStats:
    __slots__ = ("counts",)

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def __repr__(self) -> str:
        return f"RedactionStats(total={self.total}, {dict(self.counts)})"


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def redact(text: str, stats: RedactionStats | None = None) -> str:
    """Replace credentials with typed placeholders.

    Placeholders are typed (``[REDACTED:aws_key]``) so a memory retains the
    *shape* of what happened -- "the deploy failed because the AWS key was
    wrong" survives, the key itself does not.
    """
    if not text:
        return text
    stats = stats if stats is not None else RedactionStats()

    for name, pattern in _PATTERNS:
        def _sub(m: re.Match[str], _name: str = name) -> str:
            stats.counts[_name] += 1
            if _name == "env_assign":
                return f"{m.group(1)}=[REDACTED:{_name}]"
            if _name == "bearer":
                return m.group(0).replace(m.group(1), f"[REDACTED:{_name}]")
            return f"[REDACTED:{_name}]"

        text = pattern.sub(_sub, text)

    def _entropy_sub(m: re.Match[str]) -> str:
        token = m.group(0)
        if _ENTROPY_ALLOW.match(token):
            return token
        if shannon_entropy(token) >= _ENTROPY_THRESHOLD:
            stats.counts["high_entropy"] += 1
            return "[REDACTED:high_entropy]"
        return token

    return _CANDIDATE.sub(_entropy_sub, text)
