"""Pipeline: orchestrates one user-message → bot-reply turn via Strands Agent.

  user msg →  retrieve top-k utterances (RAG layer 2)
           →  retrieve top-1 analogous scene (GCA situation matching)
           →  rebuild system_prompt with GCA + RAG content baked in
           →  agent.invoke_async(user_message) — Strands handles:
                * appending user msg to agent.messages
                * SummarizingConversationManager (sliding + Tier-2 summary)
                * FileSessionManager (per-session disk persistence)
                * calling ChaiModel → CHAI POST
                * appending assistant msg to agent.messages
           →  parse Speech/Action/Thought from result.message
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from roleplaychatbotservice.chat.session import ChatSession
from roleplaychatbotservice.personas.embedder import QwenEmbedder
from roleplaychatbotservice.personas.retrieval import (
    RetrievedChunk,
    RetrievedScene,
    SceneIndex,
    UtteranceIndex,
)
from roleplaychatbotservice.prompts.parse import ParsedReply, parse_sat
from roleplaychatbotservice.prompts.system import build_system_prompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatTurnContext:
    """Runtime dependencies handed to chat_turn(). Built once at app startup."""
    embedder: QwenEmbedder | None = None
    utterance_indices: dict[str, UtteranceIndex] | None = None
    scene_indices: dict[str, SceneIndex] | None = None
    rag_k: int = 4
    enable_gca: bool = True
    enable_sat: bool = True
    scene_min_score: float = 0.30


@dataclass(frozen=True)
class ChatTurnResult:
    raw_reply: str
    parsed: ParsedReply
    retrieved_scene_score: float | None = None
    retrieved_chunk_count: int = 0


async def chat_turn(session: ChatSession, user_message: str, ctx: ChatTurnContext) -> ChatTurnResult:
    """Run one user → bot turn. Mutates the underlying Strands Agent's history."""
    with session.lock:
        retrieved_chunks = _retrieve_chunks(session, user_message, ctx)
        retrieved_scene = _retrieve_scene(session, user_message, ctx) if ctx.enable_gca else None

        # Rebuild system_prompt fresh per turn so RAG/GCA tracks current input.
        session.agent.system_prompt = build_system_prompt(
            persona=session.persona,
            retrieved_chunks=retrieved_chunks,
            retrieved_scene=retrieved_scene,
            enable_sat=ctx.enable_sat,
        )

        try:
            result = await session.agent.invoke_async(user_message)
        except Exception:
            raise

    raw = _extract_text(result)
    parsed = parse_sat(raw)
    return ChatTurnResult(
        raw_reply=raw,
        parsed=parsed,
        retrieved_scene_score=retrieved_scene.score if retrieved_scene else None,
        retrieved_chunk_count=len(retrieved_chunks),
    )


def _extract_text(agent_result: object) -> str:
    """Flatten Strands AgentResult.message into plain text."""
    msg = getattr(agent_result, "message", None) or {}
    if isinstance(msg, dict):
        parts: list[str] = []
        for block in msg.get("content", []) or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(p for p in parts if p).strip()
    return str(msg)


def _retrieve_chunks(session: ChatSession, user_message: str, ctx: ChatTurnContext) -> list[RetrievedChunk]:
    if ctx.embedder is None or ctx.utterance_indices is None or ctx.rag_k <= 0:
        return []
    index = ctx.utterance_indices.get(session.persona.id)
    if index is None:
        return []
    try:
        retrieved = index.search(user_message, ctx.embedder, k=ctx.rag_k)
    except Exception as exc:
        logger.warning("rag retrieval failed for persona=%s: %s", session.persona.id, exc)
        return []
    if retrieved:
        logger.info("rag persona=%s k=%d top=%.3f",
                    session.persona.id, len(retrieved), retrieved[0].score)
    return retrieved


def _retrieve_scene(session: ChatSession, user_message: str, ctx: ChatTurnContext) -> RetrievedScene | None:
    if ctx.embedder is None or ctx.scene_indices is None:
        return None
    index = ctx.scene_indices.get(session.persona.id)
    if index is None:
        return None
    # Pull the last 1-2 turns from agent.messages so situation matching uses
    # ongoing context, not just literal user words.
    recent = []
    for m in (session.agent.messages or [])[-2:]:
        for block in m.get("content", []) or []:
            text = block.get("text") if isinstance(block, dict) else None
            if text:
                recent.append(text)
    query = " ".join([*recent, user_message]).strip()

    try:
        rs = index.search(query, ctx.embedder, k=1)
    except Exception as exc:
        logger.warning("scene retrieval failed for persona=%s: %s", session.persona.id, exc)
        return None
    if not rs:
        return None
    top = rs[0]
    if top.score < ctx.scene_min_score:
        logger.debug("scene below threshold: persona=%s score=%.3f", session.persona.id, top.score)
        return None
    logger.info("scene persona=%s topic=%r score=%.3f",
                session.persona.id, top.scene.topic[:60], top.score)
    return top
