"""Build a per-session Strands Agent wired with our persona, memory, and
session-persistence stack.

Memory architecture (Strands official):
  * `SummarizingConversationManager` (Tier 1 sliding + Tier 2 summary)
    rolls oldest 30% of history into bullet form via a separate Agent
    using CHAI in Summarizer-persona mode (see strands_integration/summarizer.py).
  * `FileSessionManager` persists message history to SESSION_DIR/<sid>/...
    survives uvicorn restarts.

The persona's identity / GCA elements are NOT baked into the Agent at
construction time. We rebuild the Agent's `system_prompt` per turn in
`pipeline.chat_turn` so retrieved-scene and few-shot chunks track the
current user message.
"""
from __future__ import annotations

import logging
from pathlib import Path

from strands import Agent
from strands.agent.conversation_manager import SummarizingConversationManager
from strands.session.file_session_manager import FileSessionManager

from roleplaychatbotservice.chai import ChaiClient
from roleplaychatbotservice.personas.schema import BotPersona
from roleplaychatbotservice.strands_integration import ChaiModel

logger = logging.getLogger(__name__)


def build_chat_agent(
    persona: BotPersona,
    user_name: str,
    session_id: str,
    chai_client: ChaiClient,
    summarizer_agent: Agent,
    session_dir: str,
    summary_ratio: float = 0.3,
    preserve_recent_messages: int = 10,
) -> Agent:
    """Construct one Strands Agent for a (persona, session) pair.

    The Agent will auto-restore its message history from `session_dir/<sid>/`
    if a session by that id already exists on disk.
    """
    Path(session_dir).mkdir(parents=True, exist_ok=True)

    model = ChaiModel(
        chai_client=chai_client,
        bot_name=persona.name,
        user_name=user_name,
    )

    conv_manager = SummarizingConversationManager(
        summary_ratio=summary_ratio,
        preserve_recent_messages=preserve_recent_messages,
        summarization_agent=summarizer_agent,
    )

    session_manager = FileSessionManager(
        session_id=session_id,
        storage_dir=session_dir,
    )

    # system_prompt is a placeholder — overwritten per turn in pipeline.
    return Agent(
        model=model,
        system_prompt=f"You are {persona.name}.",
        conversation_manager=conv_manager,
        session_manager=session_manager,
        callback_handler=None,
    )
