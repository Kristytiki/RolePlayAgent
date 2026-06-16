# RolePlayChatbotService

Backend for the role-play chatbot. Wraps CHAI's stateless model API behind a FastAPI service with persona retrieval (RAG) and CoSER/ChatHaruhi-sourced character data.

See `.claude/design.md` for the full design.

## Project layout

```
src/roleplaychatbotservice/
├── __init__.py            ← console-script entry, 4 lines
├── app.py                 ← FastAPI app + lifespan
├── config.py              ← Settings (CHAI key, RAG flags)
├── chai/                  ← upstream model client
│   └── client.py
├── personas/              ← persona data + RAG
│   ├── schema.py          ← BotPersona, RagChunk, Scene
│   ├── loader.py          ← CoSER → BotPersona
│   ├── embedder.py        ← QwenEmbedder
│   └── retrieval/
│       ├── selection.py   ← RAG layer 1: which persona
│       └── utterance.py   ← RAG layer 2: per-character chunks
├── prompts/               ← prompt assembly lives here
│   ├── header.py          ← persona profile injection
│   ├── few_shot.py        ← retrieved chunks → CHAI lines
│   └── chat_history.py    ← top-level assembler
├── chat/                  ← session state + turn orchestration
│   ├── session.py         ← pure state (live_turns)
│   ├── registry.py        ← session_id → ChatSession
│   └── pipeline.py        ← chat_turn() — wires RAG into CHAI calls
└── api/
    ├── schemas.py
    └── routers/
        ├── personas.py    ← list / get / search
        └── chat.py        ← session CRUD + send message
```

## Quick start

```bash
uv sync
export CHAI_API_KEY=...

# One-time: download datasets and extract our 9 (currently) CoSER personas.
uv run python scripts/inspect_datasets.py    # downloads CoSER + ChatHaruhi
uv run python scripts/extract_personas.py    # writes personas/data/raw/

# Run the service. First run downloads Qwen3-Embedding-0.6B (~1.2GB) and embeds
# all persona corpora (~3 minutes); cached afterwards.
uv run uvicorn roleplaychatbotservice.app:app --port 8000
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/health` | Liveness check |
| `GET`  | `/personas` | List all personas |
| `POST` | `/personas/search` | RAG layer 1 — `{query, k}` → top-k personas |
| `GET`  | `/personas/{id}` | Persona detail |
| `POST` | `/chat/sessions` | Create chat session |
| `POST` | `/chat/sessions/{id}/messages` | Send user message → reply (RAG layer 2 fires inside) |
| `GET`  | `/chat/sessions/{id}/history` | Visible history |
| `DELETE` | `/chat/sessions/{id}` | End session |

## Datasets

- **CoSER** (`Neph0s/CoSER`, MIT) — literary personas with structured profile / experiences / thoughts / utterances. Currently using 9 characters: Mr. Darcy, Heathcliff, Sherlock Holmes, Jay Gatsby, Hermione Granger, Elizabeth Bennet, Anna Karenina, Atticus Finch, Scarlett O'Hara.
- **ChatHaruhi** (`silk-road/ChatHaruhi-54K-Role-Playing-Dialogue`, CC-BY-NC) — extracted but not yet wired (no CoSER-style profile).

Raw extracted persona data lives under `src/roleplaychatbotservice/personas/data/raw/{coser,haruhi}/*.json` so it's reviewable, editable, and decoupled from re-downloading.
