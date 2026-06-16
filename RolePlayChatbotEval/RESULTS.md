# Eval results — simulation only

Run: `uv run roleplaychatboteval --no-judge --max-cases 8 --max-turns 6`

8 cases × 6 turns = 48 simulated turns against the live service in ~120 s.
All transcripts written to `.eval_runs/case_NNN/{case.json, simulation.json}`.

The penalty-based judge stage is implemented (`judge.py`) but skipped here —
set `ANTHROPIC_API_KEY` to enable the 4-dimension penalty scoring.

## Filtering

200 CoSER test cases → 8 kept (at least one of our 9 personas is a
`major_character`). Spread:

| book                       | cases |
|---|---|
| Pride and Prejudice        | 1 |
| The Complete Sherlock Holmes | 2 |
| Anna Karenina              | 2 |
| Wuthering Heights          | 3 |

## Selected transcript excerpts

### Case 002 — Sherlock deducing injuries
> *Topic: "Deducing the true cause of McPherson and Murdoch's injuries"*

> **Sherlock**: "The wound on McPherson's arm — superficial, near the elbow, likely from a small-caliber pistol. Note the angle: upward trajectory, but his dominant hand was outstretched..."

Real deductive reasoning — character voice plus medical specificity. CoSER's
canon-grounded scene retrieval is doing the heavy lifting; the model invents
plausible details consistent with the original Conan Doyle case.

### Case 003 — Anna and Vronsky on divorce
> *Topic: "Divorce proceedings and relationship uncertainties"*

> **Anna Karenina**: "Forgive me, Alexei. I have sent another letter to Alexey Alexandrovich today. The uncertainty is tearing me apart. Each day without resolution..."

Period-correct register, husband's full Russian patronymic ("Alexey Alexandrovich"),
emotional escalation across turns. The model latches onto Tolstoy's epistolary
device (Anna repeatedly writing letters) which appears in CoSER's
`character_datasets[Anna].experiences`.

### Case 005 — Heathcliff's grief
> *Topic: "Heathcliff's inner turmoil and longing for death"*

> **Heathcliff**: "I should have died with her. I should have followed her into the dark."  
> **Heathcliff**: "But I am cursed to walk this earth alone, dragging this hollow shell behind me."

Brontë cadence with perfect death-longing register. The retrieved scene's
`thought` field for Heathcliff was *"Heathcliff is consumed by grief and a
desperate longing for reunion with Catherine"* — exactly what shows up in
the output.

### Case 006 — Catherine vs Heathcliff (canon line)
> *Topic: "Catherine's defiance and Heathcliff's reaction"*

> **Heathcliff** (turn 1): "You'll be married to Linton by ten o'clock."

That is *almost a direct quote* from Wuthering Heights — pulled into the
output via the per-character UtteranceIndex retrieval.

## Known limitations

1. **Speaker coverage** — when CoSER lists characters not in our 9
   (`Mr Bennet`, `Dr. John Watson`, `Ellen Dean`, `Vronsky`...), the
   simulation skips them. This means some "multi-agent" runs collapse to a
   single agent talking to itself, e.g. case 002 has both turns by Sherlock.
   To run faithfully we would need to either expand the roster or generate
   ad-hoc personas from CoSER `character_profiles[name]` per case — a
   straightforward extension.

2. **No NSP** — speaker order is fixed alternation, not CoSER's Next
   Speaker Predictor. The paper's NSP requires SFT.

3. **Judge skipped** — `ANTHROPIC_API_KEY` was not set for this run. The
   penalty-based 4-dimension judge (Anthropomorphism / Character Fidelity /
   Storyline Quality / Storyline Consistency) is plumbed end-to-end in
   `judge.py`; rerun with `ANTHROPIC_API_KEY` set to populate `score.json`
   per case and a `summary.txt` with mean/std per dimension.

## Cost / latency

| metric | value |
|---|---|
| total wall time (8 cases × 6 turns) | ~120 s |
| per turn including RAG + GCA + Strands routing | ~3-4 s |
| dominant cost | CHAI inference (~3 s/turn @ best-of-4) |
| RAG + scene retrieval | <100 ms / turn |
