"""Render retrieved persona chunks as ChatHaruhi-style few-shot turns."""
from __future__ import annotations

from roleplaychatbotservice.chai import ChaiMessage
from roleplaychatbotservice.personas.retrieval.utterance import RetrievedChunk

_KIND_LABEL = {
    "utterance":  "as you have said before",
    "thought":    "your inner thought in a similar moment",
    "experience": "a past experience of yours",
}


def render_retrieved_chunks(
    retrieved: list[RetrievedChunk],
    persona_name: str,  # currently unused; kept for future per-character templates
) -> list[ChaiMessage]:
    """Translate retrieved chunks into chat_history turns.

    CHAI treats every line in chat_history as user-side input regardless of
    `sender`, so the framing has to live in the message text. We use a
    third-person annotation that nudges the model toward this voice/situation
    without pretending it was an actual prior turn in *this* conversation.
    """
    out: list[ChaiMessage] = []
    for rc in retrieved:
        label = _KIND_LABEL.get(rc.chunk.kind, "context")
        out.append(ChaiMessage(
            sender="Bot",
            message=f"[{label} (sim={rc.score:.2f})] {rc.chunk.text}",
        ))
    return out
