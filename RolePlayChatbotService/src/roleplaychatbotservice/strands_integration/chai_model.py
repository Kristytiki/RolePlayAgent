"""Strands-compatible Model adapter wrapping our ChaiClient.

CHAI is not in Strands' built-in providers (Bedrock / Anthropic / OpenAI /
Ollama). This adapter implements `strands.models.Model.stream()` so a
Strands `Agent` can be configured to talk to CHAI.

Honest scope note: CHAI is a role-play model that returns one full answer
per HTTP call (no native streaming). We emit Strands' streaming event
protocol — messageStart → contentBlockDelta → messageStop — from the single
response to satisfy the interface. Tool-use is not supported (CHAI is not
instruction-tuned for tool calls), so any `tool_specs` argument is logged
and ignored.

This adapter is exposed for two reasons:
  1. Lets future code use Strands' SummarizingConversationManager + Agent
     loop on top of CHAI. (The chat/pipeline.py module currently uses our
     own pipeline and does NOT route through Strands; that's intentional —
     for the simple per-turn shape, our pipeline is more direct.)
  2. Demonstrates we *can* drop into Strands cleanly when the project grows
     to need agent loops, multi-step reasoning, or model swapping.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterable
from typing import Any

from roleplaychatbotservice.chai import ChaiClient, ChaiMessage, ChaiRequest

logger = logging.getLogger(__name__)


def _strands_messages_to_chai(messages: list[Any]) -> list[ChaiMessage]:
    """Translate Strands `list[Message]` → CHAI `chat_history` shape.

    Strands `Message` is a TypedDict with `{role: 'user'|'assistant', content: list[ContentBlock]}`.
    Each ContentBlock has shape `{text: str}` for plain text. We flatten content into
    a single CHAI message per turn; non-text blocks are ignored.
    """
    out: list[ChaiMessage] = []
    for m in messages:
        role = m.get("role", "user")
        sender = "Bot" if role == "assistant" else "User"
        text_parts: list[str] = []
        for block in m.get("content", []) or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        text = "\n".join(p for p in text_parts if p)
        if text:
            out.append(ChaiMessage(sender=sender, message=text))
    return out


class ChaiModel:
    """Strands `Model` subclass for CHAI Guanaco endpoint.

    We don't inherit `strands.models.Model` directly to avoid coupling our
    test surface to Strands' abstract internals. The methods below match the
    abstract signatures so duck-typing through Strands' Agent works.
    Mark the class via the registry trick if needed.
    """

    # CHAI is stateless from Strands' perspective — every call sends the full
    # chat_history, the upstream server doesn't carry session state.
    stateful: bool = False

    def __init__(
        self,
        chai_client: ChaiClient,
        bot_name: str = "Bot",
        user_name: str = "User",
        system_prefix: str | None = None,
    ) -> None:
        """`bot_name` doubles as the persona signal to CHAI's prompt template
        (it expands to `{bot_name} [safe for work edition]:`). Use a neutral
        name like 'Summarizer' for non-roleplay tasks; CHAI then drops out of
        character and behaves like a generic assistant. Verified live."""
        self._client = chai_client
        self._config: dict[str, Any] = {
            "bot_name": bot_name,
            "user_name": user_name,
            "system_prefix": system_prefix,
        }

    # -- Strands Model interface ---------------------------------------------

    def update_config(self, **model_config: Any) -> None:
        self._config.update(model_config)

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)

    async def stream(
        self,
        messages: list[Any],
        tool_specs: list[Any] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[dict[str, Any]]:
        if tool_specs:
            logger.warning("ChaiModel does not support tools; %d spec(s) ignored",
                           len(tool_specs))

        chat_history = _strands_messages_to_chai(messages)

        # Inject system_prompt as a leading Bot turn — CHAI has no system role.
        if system_prompt:
            chat_history = [
                ChaiMessage(sender="Bot", message=system_prompt),
                ChaiMessage(sender="User", message="Alright"),
                *chat_history,
            ]

        request = ChaiRequest(
            bot_name=self._config["bot_name"],
            user_name=self._config["user_name"],
            chat_history=chat_history,
        )
        response = await self._client.achat(request)

        # Emit Strands' streaming protocol. CHAI is non-streaming, so we
        # synthesise the standard event sequence in one shot:
        #   messageStart → contentBlockStart → contentBlockDelta → contentBlockStop → messageStop
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": response.text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}

    async def structured_output(self, *args: Any, **kwargs: Any):
        # CHAI is not instruction-tuned; we don't support structured output.
        raise NotImplementedError("ChaiModel does not support structured_output")

    def count_tokens(self, *args: Any, **kwargs: Any) -> int:
        # Rough: 1 token ≈ 4 chars (server-side cap is 2048 input, 80 output).
        # We don't have a tokenizer here; this is best-effort for Strands' budget hooks.
        try:
            messages = args[0] if args else kwargs.get("messages", [])
            total_chars = 0
            for m in messages:
                for block in m.get("content", []) or []:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        total_chars += len(block["text"])
            return max(1, total_chars // 4)
        except Exception:
            return 1
