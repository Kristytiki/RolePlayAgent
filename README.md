# RolePlay Chatbot

CHAI take-home: a role-play chatbot built on top of CHAI's stateless model API,
with two-layer RAG, CoSER given-circumstance prompting, Strands-managed memory,
a React UI, and a CoSER-style evaluation harness.

## Layout

```
.
├── RolePlayChatbotService/   FastAPI backend (Strands Agent + ChaiModel + RAG + GCA)
├── RolePlayChatbotUI/        React + Vite, two screens: persona grid → chat
└── RolePlayChatbotEval/      CoSER-style multi-agent simulation + penalty-based judge
```

Each subdirectory has its own README.

## Architecture in one paragraph

A user picks a literary persona (one of 9 sourced from CoSER) — either by
browsing the grid or via a semantic search box (RAG layer 1). Each chat turn
hits a Strands `Agent` whose `ChaiModel` adapter wraps CHAI's HTTP endpoint;
before each call we (a) retrieve top-k utterances/thoughts/experiences from
the chosen character's corpus (RAG layer 2, ChatHaruhi-style few-shot) and
(b) match the conversational moment to the most analogous past scene from
CoSER's `character_datasets`, then assemble a system prompt that bakes in
GCA's five elements (scenario, character profile, in-scene motivation, other
characters' profiles, S/A/T output format). Strands' `SummarizingConversationManager`
rolls the oldest 30% of history into bullet summaries via a separate `Agent`
that reuses CHAI in a neutral "Summarizer" persona. `FileSessionManager`
persists everything to disk so sessions survive restarts.

See `RolePlayChatbotService/EXAMPLES.md` for live A/B comparisons (RAG+GCA off
vs on) on Mr. Darcy, Atticus Finch, Sherlock Holmes, and Scarlett O'Hara.

## Quick start

```bash
# 1. Backend
cd RolePlayChatbotService
uv sync
export CHAI_API_KEY=...
uv run python scripts/inspect_datasets.py    # one-time; downloads CoSER + ChatHaruhi
uv run python scripts/extract_personas.py    # one-time; writes data/raw/{coser,haruhi}/
uv run uvicorn roleplaychatbotservice.app:app --host 0.0.0.0 --port 8000
# First run downloads Qwen3-Embedding-0.6B (~1.2GB) and embeds persona corpora (~3 min)

# 2. UI (separate shell)
cd RolePlayChatbotUI
npm install
npm run dev -- --host 0.0.0.0 --port 5173

# 3. Eval (optional, separate shell, service must be running)
cd RolePlayChatbotEval
uv sync
uv run roleplaychatboteval --no-judge --max-cases 3
# Set ANTHROPIC_API_KEY to enable the penalty-based judge.
```

## Notes for reviewers

- **Tools used**: Claude Code (Sonnet 4.5/Opus). Most of this code was
  paired — design discussions, architecture decisions, and implementation
  drafts via the CLI agent. Verification via curl + a small custom A/B
  capture script (`scripts/ab_compare.py`).
- **Datasets**: only public, MIT/CC-licensed sources (`Neph0s/CoSER`,
  `silk-road/ChatHaruhi-54K-Role-Playing-Dialogue`). No persona profile is
  LLM-synthesized — every character profile comes verbatim from CoSER.
- **Model is hosted black-box**: no SFT, no fine-tuning. Everything in
  this codebase happens at inference time.
