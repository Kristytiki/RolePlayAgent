"""Top-level chat_history assembler. The only function that knows the full
ordering of header → GCA → few-shot → greeting → live turns.
"""
from __future__ import annotations

from roleplaychatbotservice.chai import ChaiMessage
from roleplaychatbotservice.personas.retrieval import RetrievedChunk, RetrievedScene
from roleplaychatbotservice.personas.schema import BotPersona
from roleplaychatbotservice.prompts.few_shot import render_retrieved_chunks
from roleplaychatbotservice.prompts.gca import render_gca_block
from roleplaychatbotservice.prompts.header import build_persona_greeting, build_persona_header


def build_chat_history(
    persona: BotPersona,
    live_turns: list[ChaiMessage],
    retrieved: list[RetrievedChunk] | None = None,
    retrieved_scene: RetrievedScene | None = None,
    enable_sat: bool = False,
) -> list[ChaiMessage]:
    history = build_persona_header(persona)
    history.extend(render_gca_block(retrieved_scene, persona, enable_sat=enable_sat))
    if retrieved:
        history.extend(render_retrieved_chunks(retrieved, persona.name))
    history.append(build_persona_greeting(persona))
    history.extend(live_turns)
    return history
