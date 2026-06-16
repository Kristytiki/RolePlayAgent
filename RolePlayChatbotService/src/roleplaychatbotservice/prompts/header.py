"""Persona header — the static, per-character preamble of CHAI's chat_history.

Layered structure (CHAI-style):
  1. safety prompt
  2. user "Alright" acknowledgment
  3. character profile (CoSER's profile_markdown verbatim, since it already
     covers GCA's five elements: background / personality / motivation /
     relationships / growth-arc in markdown)
  4. (optional) retrieved few-shot — added by chat_history.build_chat_history
  5. greeting
"""
from __future__ import annotations

from roleplaychatbotservice.chai import ChaiMessage
from roleplaychatbotservice.personas.schema import BotPersona


def build_persona_header(persona: BotPersona) -> list[ChaiMessage]:
    intro = (
        f"I am {persona.name}.\n{persona.profile_markdown}".strip()
        if persona.profile_markdown
        else f"I am {persona.name}."
    )
    return [
        ChaiMessage(sender="Bot", message=persona.safety_prompt),
        ChaiMessage(sender="User", message="Alright"),
        ChaiMessage(sender="Bot", message=intro),
    ]


def build_persona_greeting(persona: BotPersona) -> ChaiMessage:
    return ChaiMessage(sender=persona.name, message=persona.greeting)
