"""Active ChatSessions, keyed by session_id.

Strands' FileSessionManager handles disk persistence per-Agent — we just
hold the in-memory map of session_id → ChatSession. Restoration on startup
is up to the caller (it needs a live ChaiClient to rebuild the Agent).
"""
from __future__ import annotations

import logging
import uuid
from threading import Lock

from roleplaychatbotservice.chat.session import ChatSession

logger = logging.getLogger(__name__)


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._lock = Lock()

    def new_session_id(self) -> str:
        return uuid.uuid4().hex

    def add(self, session: ChatSession) -> None:
        with self._lock:
            self._sessions[session.session_id] = session
        logger.info(
            "registered session %s persona=%s user=%s",
            session.session_id, session.persona.id, session.user_name,
        )

    def get(self, session_id: str) -> ChatSession | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
