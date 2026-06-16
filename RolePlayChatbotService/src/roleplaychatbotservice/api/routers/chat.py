"""Chat session endpoints — backed by per-session Strands Agents."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from roleplaychatbotservice.api.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    HistoryResponse,
    HistoryTurn,
    PersonaSummary,
    SendMessageRequest,
    SendMessageResponse,
)
from roleplaychatbotservice.chat import (
    ChatSession,
    ChatTurnContext,
    build_chat_agent,
    chat_turn,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def _persona_summary(p) -> PersonaSummary:
    return PersonaSummary(
        id=p.id, name=p.name, source=p.source,
        avatar_emoji=p.avatar_emoji, tagline=p.tagline, tags=list(p.tags),
    )


def _turn_context(request: Request) -> ChatTurnContext:
    state = request.app.state
    s = state.settings
    return ChatTurnContext(
        embedder=state.embedder,
        utterance_indices=state.utterance_indices,
        scene_indices=state.scene_indices,
        rag_k=s.rag_k,
        enable_gca=s.enable_gca,
        enable_sat=s.enable_sat_format,
        scene_min_score=s.scene_min_score,
    )


@router.post("/sessions", response_model=CreateSessionResponse, status_code=201)
def create_session(body: CreateSessionRequest, request: Request) -> CreateSessionResponse:
    state = request.app.state
    persona = state.personas.get(body.persona_id)
    if persona is None:
        raise HTTPException(404, f"persona {body.persona_id!r} not found")

    sid = state.sessions.new_session_id()
    agent = build_chat_agent(
        persona=persona,
        user_name=body.user_name,
        session_id=sid,
        chai_client=state.chai_client,
        summarizer_agent=state.summarizer_agent,
        session_dir=state.settings.session_dir,
    )
    session = ChatSession(
        session_id=sid,
        persona=persona,
        user_name=body.user_name,
        agent=agent,
    )
    state.sessions.add(session)

    return CreateSessionResponse(
        session_id=sid,
        persona=_persona_summary(persona),
        greeting=persona.greeting,
    )


@router.post("/sessions/{session_id}/messages", response_model=SendMessageResponse)
async def send_message(session_id: str, body: SendMessageRequest, request: Request) -> SendMessageResponse:
    session = request.app.state.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"session {session_id!r} not found")
    try:
        result = await chat_turn(session, body.message, _turn_context(request))
    except Exception as exc:
        logger.exception("CHAI call failed for session %s", session_id)
        raise HTTPException(502, f"upstream model error: {exc!s}") from exc
    return SendMessageResponse(
        reply=result.parsed.speech or result.raw_reply,
        speech=result.parsed.speech or result.raw_reply,
        action=result.parsed.action,
        thought=result.parsed.thought,
        raw=result.raw_reply,
        retrieved_scene_score=result.retrieved_scene_score,
        retrieved_chunk_count=result.retrieved_chunk_count,
    )


@router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
def get_history(session_id: str, request: Request) -> HistoryResponse:
    session = request.app.state.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"session {session_id!r} not found")

    turns: list[HistoryTurn] = []
    for m in session.agent.messages or []:
        role = m.get("role", "")
        sender = session.persona.name if role == "assistant" else session.user_name
        for block in m.get("content", []) or []:
            text = block.get("text") if isinstance(block, dict) else None
            if text:
                turns.append(HistoryTurn(sender=sender, message=text))
    return HistoryResponse(
        session_id=session.session_id,
        persona_id=session.persona.id,
        turns=turns,
    )


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, request: Request) -> None:
    if not request.app.state.sessions.delete(session_id):
        raise HTTPException(404, f"session {session_id!r} not found")
