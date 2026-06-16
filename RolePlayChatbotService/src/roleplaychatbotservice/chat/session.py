"""ChatSession — wraps a Strands Agent.

Each session owns one Strands `Agent` with its own conversation manager and
session manager. We intentionally do NOT keep our own `live_turns` list any
more — that's `agent.messages`, managed by Strands (sliding window, rolling
summary, file persistence all happen for free inside Strands).

The Agent's `system_prompt` is rebuilt by the pipeline before each turn so
GCA / RAG content is fresh per user message.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

from roleplaychatbotservice.personas.schema import BotPersona

if TYPE_CHECKING:
    from strands import Agent


@dataclass
class ChatSession:
    session_id: str
    persona: BotPersona
    user_name: str
    agent: "Agent"
    lock: Lock = field(default_factory=Lock, repr=False, compare=False)
