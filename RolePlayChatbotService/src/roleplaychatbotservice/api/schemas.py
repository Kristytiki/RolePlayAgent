"""Wire-format Pydantic models for the HTTP API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PersonaSummary(BaseModel):
    id: str
    name: str
    source: str
    avatar_emoji: str
    tagline: str
    tags: list[str]


class PersonaDetail(PersonaSummary):
    profile_markdown: str
    greeting: str


class CreateSessionRequest(BaseModel):
    persona_id: str
    user_name: str = Field(default="Reader", min_length=1, max_length=64)


class CreateSessionResponse(BaseModel):
    session_id: str
    persona: PersonaSummary
    greeting: str


class SendMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class SendMessageResponse(BaseModel):
    reply: str                     # speech (visible part); falls back to full text
    speech: str                    # parsed speech (== reply when format is followed)
    action: str | None = None
    thought: str | None = None
    raw: str | None = None         # the unparsed model output, for debugging
    retrieved_scene_score: float | None = None
    retrieved_chunk_count: int = 0


class HistoryTurn(BaseModel):
    sender: str
    message: str


class HistoryResponse(BaseModel):
    session_id: str
    persona_id: str
    turns: list[HistoryTurn]


class PersonaSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=20)


class PersonaSearchResult(BaseModel):
    persona: PersonaSummary
    score: float


class PersonaSearchResponse(BaseModel):
    query: str
    results: list[PersonaSearchResult]
