"""Persona browsing + selection-RAG endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from roleplaychatbotservice.api.schemas import (
    PersonaDetail,
    PersonaSearchRequest,
    PersonaSearchResponse,
    PersonaSearchResult,
    PersonaSummary,
)

router = APIRouter(prefix="/personas", tags=["personas"])


def _summary(p) -> PersonaSummary:
    return PersonaSummary(
        id=p.id, name=p.name, source=p.source,
        avatar_emoji=p.avatar_emoji, tagline=p.tagline, tags=list(p.tags),
    )


@router.get("", response_model=list[PersonaSummary])
def list_personas(request: Request) -> list[PersonaSummary]:
    return [_summary(p) for p in request.app.state.personas.values()]


@router.post("/search", response_model=PersonaSearchResponse)
def search_personas(body: PersonaSearchRequest, request: Request) -> PersonaSearchResponse:
    index = request.app.state.selection_index
    if index is None:
        raise HTTPException(503, "selection index not available (RAG disabled)")
    matches = index.search(body.query, k=body.k)
    return PersonaSearchResponse(
        query=body.query,
        results=[
            PersonaSearchResult(persona=_summary(m.persona), score=m.score)
            for m in matches
        ],
    )


@router.get("/{persona_id}", response_model=PersonaDetail)
def get_persona(persona_id: str, request: Request) -> PersonaDetail:
    p = request.app.state.personas.get(persona_id)
    if p is None:
        raise HTTPException(404, f"persona {persona_id!r} not found")
    return PersonaDetail(
        id=p.id, name=p.name, source=p.source,
        avatar_emoji=p.avatar_emoji, tagline=p.tagline, tags=list(p.tags),
        profile_markdown=p.profile_markdown, greeting=p.greeting,
    )
