# RolePlayChatbotEval

CoSER-style evaluation harness for the role-play chatbot service.

```
src/roleplaychatboteval/
├── __init__.py     ← exports
├── dataset.py      ← load CoSER test set + filter to our personas
├── simulate.py     ← multi-CHAI-agent simulation against the live service
├── judge.py        ← penalty-based 4-dim judge (Anthropic Claude)
├── report.py       ← aggregate + format summary table
└── cli.py          ← `roleplaychatboteval` console entry point
```

## What it does

1. Loads `Neph0s/CoSER`'s 200 held-out test conversations from your local HF cache
2. Keeps only cases where one of our 9 service personas is a `major_character`
3. For each kept case, opens one chat session per speaking character against
   the running RolePlayChatbotService and runs a fixed-alternation
   multi-agent simulation (≤12 turns by default) — this exercises our full
   stack: Strands `Agent` + `ChaiModel` + GCA + RAG + memory
4. Optionally invokes a judge LLM (Anthropic Claude Haiku) per case × 4
   dimensions × penalty-based deductions (per CoSER paper §4)
5. Aggregates per-dimension mean/std into `summary.txt`

## Running

```bash
# Service must already be live on :8000
cd ../RolePlayChatbotService
uv run uvicorn roleplaychatbotservice.app:app --port 8000 &
cd ../RolePlayChatbotEval
uv sync

# Skeleton run — emits transcripts only
uv run roleplaychatboteval --no-judge --max-cases 3

# Full run with scoring (set the key once)
export ANTHROPIC_API_KEY=...
uv run roleplaychatboteval --max-cases 8

# Filter cases by persona name as it appears in CoSER
uv run roleplaychatboteval --persona "Hermione Granger" --persona "Sherlock Holmes"
```

Outputs land in `.eval_runs/case_NNN/{case.json, simulation.json, score.json}`
plus a top-level `summary.txt`.

## Honest scope notes

- We use **fixed-alternation** speaker order, not CoSER's NSP model — the
  paper's NSP needs SFT, which we don't do.
- The judge is **Anthropic Haiku** by default (cheaper than the paper's
  GPT-4o); swap via `--model` if you want.
- Filtering to our 9 personas leaves ~8 cases out of 200, so this is a
  proof-of-pipeline more than a full benchmark. To compare against CoSER's
  reported numbers you would need the full CoSER persona library wired up.
