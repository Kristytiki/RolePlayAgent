"""Parse Speech / Action / Thought output. Robust fallback when the model
doesn't follow the format (CHAI is role-play tuned, not instruction tuned —
roughly 50% of replies follow the format cleanly in our smoke tests).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Match `Speech: foo` / `Action: bar` / `Thought: baz` at line starts.
# Keep capturing until the next labeled line or end-of-string.
_SAT_LINE_RE = re.compile(
    r"^\s*(Speech|Action|Thought)\s*[:：]\s*(.*?)(?=^\s*(?:Speech|Action|Thought)\s*[:：]|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class ParsedReply:
    speech: str
    action: str | None = None
    thought: str | None = None
    raw: str = ""

    @property
    def has_structure(self) -> bool:
        return self.action is not None or self.thought is not None


def parse_sat(text: str) -> ParsedReply:
    """Best-effort extraction of S/A/T sections from a CHAI reply.

    Falls back to treating the whole text as `speech` if no labels match.
    Strips surrounding markdown like '*action*' or '[thought]' that the model
    sometimes uses.
    """
    if not text:
        return ParsedReply(speech="", raw=text)

    speech_parts: list[str] = []
    action_parts: list[str] = []
    thought_parts: list[str] = []

    found_any = False
    for m in _SAT_LINE_RE.finditer(text):
        found_any = True
        kind = m.group(1).strip().lower()
        body = m.group(2).strip().rstrip(".·•— -").strip()
        if not body:
            continue
        if kind == "speech":
            speech_parts.append(body)
        elif kind == "action":
            action_parts.append(_strip_wrappers(body))
        elif kind == "thought":
            thought_parts.append(_strip_wrappers(body))

    if not found_any:
        return ParsedReply(speech=text.strip(), raw=text)

    return ParsedReply(
        speech=" ".join(speech_parts).strip(),
        action=" ".join(action_parts).strip() or None,
        thought=" ".join(thought_parts).strip() or None,
        raw=text,
    )


_WRAPPER_RE = re.compile(r"^[\(\[\{*_`\"'\s]+|[\)\]\}*_`\"'\s]+$")


def _strip_wrappers(s: str) -> str:
    return _WRAPPER_RE.sub("", s)
