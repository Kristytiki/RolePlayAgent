"""Minimal client for the CHAI Guanaco onsite chat endpoint.

Smoke-tested against the live endpoint. Notable behaviours observed:
- response field is `model_output`, not `response`
- server enforces `max_output_tokens=80`, replies often get truncated mid-sentence
- `chat_history[*].sender` is largely ignored: every message is rendered as a
  `<|im_start|>user` block in the model_input. Persona signalling must therefore
  be embedded in the message text, not relied upon via the sender field.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field

import httpx

logger = logging.getLogger(__name__)

CHAI_ENDPOINT = "http://guanaco-submitter.guanaco-backend.k2.chaiverse.com/endpoints/onsite/chat"
DEFAULT_TIMEOUT = 30.0


@dataclass(frozen=True)
class ChaiMessage:
    sender: str
    message: str


@dataclass(frozen=True)
class ChaiRequest:
    bot_name: str
    user_name: str
    chat_history: list[ChaiMessage]
    memory: str = ""
    prompt: str = ""

    def to_payload(self) -> dict:
        return {
            "memory": self.memory,
            "prompt": self.prompt,
            "bot_name": self.bot_name,
            "user_name": self.user_name,
            "chat_history": [asdict(m) for m in self.chat_history],
        }


@dataclass(frozen=True)
class ChaiResponse:
    text: str
    raw: dict = field(repr=False)


class ChaiError(RuntimeError):
    pass


class ChaiClient:
    """Thin httpx wrapper around the CHAI onsite chat endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = CHAI_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        key = api_key or os.environ.get("CHAI_API_KEY")
        if not key:
            raise ChaiError("CHAI_API_KEY is required (env or constructor arg)")
        self._endpoint = endpoint
        self._headers = {"Authorization": f"Bearer {key}"}
        self._timeout = timeout

    def chat(self, request: ChaiRequest) -> ChaiResponse:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(self._endpoint, headers=self._headers, json=request.to_payload())
        return self._parse(resp)

    async def achat(self, request: ChaiRequest) -> ChaiResponse:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._endpoint, headers=self._headers, json=request.to_payload())
        return self._parse(resp)

    @staticmethod
    def _parse(resp: httpx.Response) -> ChaiResponse:
        if resp.status_code != 200:
            raise ChaiError(f"CHAI {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        text = data.get("model_output", "").strip()
        if not text:
            raise ChaiError(f"empty model_output, raw keys={list(data)}")
        return ChaiResponse(text=text, raw=data)
