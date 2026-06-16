"""Build a Strands Agent that uses CHAI as a third-person summarizer.

CHAI is a role-play model, but a smoke test (see EXAMPLES.md) showed it
follows a 'Summarizer' bot_name + system_prompt cleanly — outputting tight
third-person bullets that preserve names, plot beats, and emotional state.

We get a free Tier-2 memory backend (rolling summary) without needing a
separate model API key.
"""
from __future__ import annotations

from strands import Agent

from roleplaychatbotservice.chai import ChaiClient
from roleplaychatbotservice.strands_integration.chai_model import ChaiModel


SUMMARIZER_SYSTEM_PROMPT = (
    "You are Summarizer — a neutral analyst, not a role-play character. "
    "Compress the given role-play transcript into terse third-person bullets. "
    "Preserve character names, established facts, plot beats, relationships, "
    "and emotional state. Do not add commentary or narrate. Output only the "
    "bullets, beginning each line with '- '."
)


def build_summarizer_agent(chai_client: ChaiClient) -> Agent:
    model = ChaiModel(chai_client=chai_client, bot_name="Summarizer", user_name="System")
    return Agent(
        model=model,
        system_prompt=SUMMARIZER_SYSTEM_PROMPT,
        callback_handler=None,
    )
