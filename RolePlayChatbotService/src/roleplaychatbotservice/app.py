"""FastAPI app factory + uvicorn launcher."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from roleplaychatbotservice.api.routers import chat as chat_router
from roleplaychatbotservice.api.routers import personas as personas_router
from roleplaychatbotservice.chai import ChaiClient
from roleplaychatbotservice.chat import SessionRegistry
from roleplaychatbotservice.config import get_settings
from roleplaychatbotservice.personas import load_personas
from roleplaychatbotservice.personas.embedder import QwenEmbedder
from roleplaychatbotservice.personas.retrieval import (
    PersonaSelectionIndex,
    build_indices,
    build_scene_indices,
)
from roleplaychatbotservice.strands_integration import build_summarizer_agent

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    personas_list = load_personas()
    app.state.settings = settings
    app.state.personas = {p.id: p for p in personas_list}
    app.state.chai_client = ChaiClient(api_key=settings.chai_api_key)
    app.state.sessions = SessionRegistry()
    app.state.summarizer_agent = build_summarizer_agent(app.state.chai_client)

    # RAG + GCA: embedder + utterance/scene/selection indices.
    if settings.enable_rag and personas_list:
        embedder = QwenEmbedder(model_id=settings.embedding_model)
        app.state.embedder = embedder
        app.state.utterance_indices = build_indices(personas_list, embedder)
        app.state.selection_index = PersonaSelectionIndex(personas_list, embedder)
        app.state.scene_indices = build_scene_indices(personas_list, embedder) if settings.enable_gca else {}
        logger.info(
            "RAG ready: utt=%d scene=%d selection=%d personas",
            len(app.state.utterance_indices), len(app.state.scene_indices), len(personas_list),
        )
    else:
        app.state.embedder = None
        app.state.utterance_indices = {}
        app.state.scene_indices = {}
        app.state.selection_index = None
        logger.info("RAG disabled")

    logger.info("startup: %d personas %s", len(personas_list), [p.id for p in personas_list])
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="RolePlay Chatbot Service", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(personas_router.router)
    app.include_router(chat_router.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()


def run() -> None:
    """Entry point for `roleplaychatbotservice` script."""
    import uvicorn
    uvicorn.run("roleplaychatbotservice.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
