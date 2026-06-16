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

    CHAI is role-play tuned, not instruction tuned, so it often labels only
    `Speech:` and leaves the action / thought as un-labelled narrative
    *before* the Speech line. We salvage that:

      - regex pulls every `Speech:` / `Action:` / `Thought:` block
      - if only Speech is labelled but there is un-labelled prose before it,
        we attribute that prose to action (parenthesised) or thought
        (bracketed) when wrapper characters give it away
      - last-ditch fallback: whole reply becomes `speech`
    """
    if not text:
        return ParsedReply(speech="", raw=text)

    speech_parts: list[str] = []
    action_parts: list[str] = []
    thought_parts: list[str] = []

    matches = list(_SAT_LINE_RE.finditer(text))
    found_any = bool(matches)

    for m in matches:
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

    # Salvage un-labelled narrative that appears before the first Speech:.
    # Common CHAI shape:
    #   "<narrative paragraphs about what they observe / feel>\n\nSpeech: ..."
    # We split such lines into action vs thought based on bracket/paren markers.
    if found_any and not action_parts and not thought_parts:
        first_label_at = matches[0].start()
        prefix = text[:first_label_at].strip()
        if prefix:
            for raw_line in (l.strip() for l in prefix.split("\n") if l.strip()):
                stripped = _strip_wrappers(raw_line)
                if not stripped:
                    continue
                if raw_line.startswith("(") or raw_line.startswith("*"):
                    action_parts.append(stripped)
                elif raw_line.startswith("[") or raw_line.startswith("_"):
                    thought_parts.append(stripped)
                else:
                    # Default un-labelled prose → thought (inner narrative voice).
                    thought_parts.append(stripped)

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
