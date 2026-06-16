"""Build the flattened system_prompt for a Strands Agent turn.

Strands hands `system_prompt: str` to model.stream(); ChaiModel injects it
as the leading Bot turn in CHAI's chat_history. So everything we want CHAI
to see *before* the live conversation history goes here:

  1. Safety prompt
  2. Persona profile (CoSER profile_markdown)
  3. GCA scene block (scenario + my motivation + others)
  4. ChatHaruhi-style retrieved few-shot
  5. S·A·T output format instructions
  6. Greeting (so CHAI sees "you are about to greet the user with X")
"""
from __future__ import annotations

from roleplaychatbotservice.personas.retrieval import RetrievedChunk, RetrievedScene
from roleplaychatbotservice.personas.schema import BotPersona
from roleplaychatbotservice.prompts.gca import (
    SAT_INSTRUCTION,
    _is_same_character,
)


def build_system_prompt(
    persona: BotPersona,
    retrieved_chunks: list[RetrievedChunk] | None = None,
    retrieved_scene: RetrievedScene | None = None,
    enable_sat: bool = True,
) -> str:
    parts: list[str] = []
    parts.append(persona.safety_prompt)
    parts.append("")
    parts.append(f"You are {persona.name}.")
    if persona.profile_markdown:
        parts.append(persona.profile_markdown.strip())
    parts.append("")

    # GCA: situation match
    if retrieved_scene is not None:
        scn = retrieved_scene.scene
        if scn.topic or scn.scenario:
            label = scn.topic.strip() or scn.scenario.strip()[:300]
            if label:
                parts.append(f"[Setting] The current moment resembles a past scene: {label}")

        own = next(
            (kc for kc in scn.key_characters
             if _is_same_character(kc.name, persona.name)),
            None,
        )
        if own and own.thought:
            parts.append(f"[Your motivation in such moments] {own.thought}")

        others = [
            kc for kc in scn.key_characters
            if not _is_same_character(kc.name, persona.name) and kc.thought
        ]
        if others:
            line = "; ".join(f"{kc.name}: {kc.thought}" for kc in others[:3])
            parts.append(f"[Others present in such moments] {line}")
        parts.append("")

    # ChatHaruhi-style few-shot (utterance retrieval)
    if retrieved_chunks:
        parts.append("[For tone reference — your past lines / thoughts / experiences in similar moments]")
        for rc in retrieved_chunks:
            kind = rc.chunk.kind
            text = rc.chunk.text
            tag = {
                "utterance": "said",
                "thought":   "thought",
                "experience":"experienced",
            }.get(kind, kind)
            parts.append(f"  - ({tag}) {text}")
        parts.append("")

    if enable_sat:
        parts.append(SAT_INSTRUCTION)
        parts.append("")

    parts.append(f'Open by greeting the user with: "{persona.greeting}"')

    return "\n".join(parts).strip()
