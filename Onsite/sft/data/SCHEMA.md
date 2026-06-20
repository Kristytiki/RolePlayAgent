# SFT Data Schema for Chaiverse

## Why this matters

CoSER 30k SFT → 27% win-rate (below 30% baseline). Step count made it worse.
Root cause: the data **distribution** doesn't match what Chai users want, not
the data quantity. Before fine-tuning more, design the schema explicitly to
match Chai's inference setup and user preferences.

## Inference-time constraints (from `onsite_submit.py`)

These are HARD constraints — anything you train on must fit them:

| Constraint | Value | Implication for training |
|---|---|---|
| `max_input_tokens` | 2048 | system + history must fit in 2k |
| `max_output_tokens` | ≤80 (Chai validation) | each assistant target ≤ 60-70 tokens |
| `stopping_words` | `['\n']` | targets are **single-line** |
| `bot_template` | `<\|im_start\|>assistant\n{bot_name}: {message}<\|im_end\|>\n` | Chai prepends `{bot_name}:` at inference |
| `response_template` | `<\|im_start\|>assistant\n{bot_name}:` | model only generates AFTER the colon |

→ Training assistant target value MUST:
1. Be a single line (no `\n`)
2. NOT start with `{bot_name}:` prefix
3. Be ≤ 60 tokens

## Soft constraints (Chai user preferences)

These come from observed top performers (Qwen3-30B-A3B + best_of=16 = 35.7%)
and PIPPA's c.ai-style data. Not enforced at format level but encoded by
prompting the synthesizer:

| Decision | Choice | Reason |
|---|---|---|
| Reply length | 15-40 tokens | sweet spot for chat (CoSER's 50-80 was too long) |
| Action grammar | `*action*` | Pygmalion / c.ai convention; PIPPA uses it |
| Inner thought | **NONE** (no `[thought]`) | CoSER's `[thought]` correlated with -3% win-rate |
| Name prefix | NONE | inference template already supplies it |
| Emoji density | ~30% messages have 1-2 emoji | matches mobile chat reality |
| NSFW level | soft flirt allowed; no explicit | Chai filter blocks hard NSFW anyway |
| System length | 150-250 tokens | enough for character voice, leaves headroom |
| Conversation depth | 4 turns user / 4 turns asst | short, dense — not narrative |
| Tone | conversational, modern texting | avoid "Indeed!" / "Ah, my friend" lecture voice |

## ShareGPT format (one record)

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "You are <name>. <2-3 sentence persona>.\n\nSetting: a casual chat where someone has just messaged you. You speak in short conversational messages like in a texting app — no long monologues, no formal narration."
    },
    {
      "from": "human",
      "value": "===Conversation Start===\n\n<opener from user>"
    },
    {"from": "assistant", "value": "<reply, 15-40 tok, single line, no name prefix>"},
    {"from": "human",     "value": "<follow-up>"},
    {"from": "assistant", "value": "<reply>"},
    {"from": "human",     "value": "<follow-up>"},
    {"from": "assistant", "value": "<reply>"},
    {"from": "human",     "value": "<follow-up>"},
    {"from": "assistant", "value": "<reply>"}
  ]
}
```

**Why `===Conversation Start===` first user turn?** Mirrors the CoSER ShareGPT
format the existing preprocessing pipeline expects, so the same `train.py`
loader works without changes.

## Character pool design

24 characters split across the categories Chai users actually pick:

- **anime** (10): Hatsune Miku, Yor Forger, Levi Ackerman, Marin Kitagawa, Makima, Megumin, Rei Ayanami, Asuka Langley, Power, Mai Sakurajima
- **OC** (5): Aria, Kael, Nox, Junie, Vera (mix of dragon-blooded barmaid, half-elf bounty hunter, succubus cafe owner, etc — generic but specific personas)
- **fantasy** (4): Lyra Stormveil, Caius the Black, Selene, Roan
- **modern** (5): Kira (art student), Daniel (skater barista), Hana (K-pop trainee), Theo (hockey captain), Izzy (librarian / punk bassist)

Each character has a 2-line persona used to seed the system message. 1k samples /
24 characters = ~42 conversations per character → enough variation but each
character voice gets reinforced.

## User opener pool

20 hand-written openers covering the actual Chai user "first message" shapes:

```
"hey 👋 what are you up to right now?"
"you free tonight?"
"oh hi, didn't expect to run into you here"
"so... we need to talk"
"*walks up to you* hey stranger 😏"
"long day?"
"missed you"
... (20 total)
```

Mix of: greetings / questions / *action* openers / mood openers / playful flirt.
~30% include emoji to seed the same density in the model output.

## Generation pipeline

```
for each of N=1000 samples:
    archetype, name, persona = random.choice(CHAR_POOL)
    opener                   = random.choice(USER_OPENERS)
    
    system_prompt = OPUS_SYSTEM_INSTRUCTIONS    # the rules above
    user_msg      = f"CHARACTER: {name}\nPROFILE: {persona}\nOPENER: {opener}"
    
    response = claude-opus-4-8.generate(
        system = system_prompt,
        messages = [{role: user, content: user_msg}],
        max_tokens = 1200,
    )
    
    # Schema: {"turns": [{role, content}, {role, content}, ...]}
    # 8 turns total (4 user + 4 assistant after opener)
    # Validate JSON, drop newlines from each content, build ShareGPT record
```

## Validation rules (post-generation)

Drop a record if:
- Any assistant turn contains `\n`
- Any assistant turn starts with `{name}:` prefix
- Any assistant turn is empty
- Any assistant turn is > 70 tokens
- Fewer than 2 assistant turns total
- Opus output fails JSON parse / Pydantic validation

Expected drop rate: ~5-10% (Opus is mostly compliant).

## Cost estimate

- N=1000 records, ~1200 output tokens each = ~1.2M output tokens
- Opus 4.8 pricing: ~$15/M output, ~$3/M input
- Total: **~$20-25** for 1000 records
- Wall time: 8-12 minutes at concurrency=16

## Comparison vs existing datasets

| Source | Schema match | Quantity | Cost | Win-rate (so far) |
|---|---|---|---|---|
| CoSER (literary novels) | ❌ wrong tone, has [thought], long | 30k | free | **27%** ↓↓ |
| PIPPA (c.ai dump) | ⚠️ right tone, mixed quality, NSFW heavy | 10k | free | TBD (eval running) |
| Hieunguyenminh | ⚠️ QA-style, lecture voice | 5k | free | not trained |
| **Synthetic (this)** | ✅ enforced match | **1k** | **$25** | **TBD** |

Hypothesis: 1k schema-matched data > 30k unmatched data.

## Hyperparam recommendation for the synthetic data

Given each sample is 8 turns × ~30 tokens = ~250 tokens of supervision:
- `max_steps = 200-300` (1k samples × 1 epoch ≈ 250 steps at eff bs 8)
- `lora_r = 16` (lighter, like the PIPPA-r16 run)
- `lora_alpha = 32`
- `lr = 1e-4` (not the 2e-4 that over-shot CoSER)
- `max_seq_length = 2048`
- `train_on_responses_only` masking via Qwen ChatML markers

## Next steps after this corpus exists

1. Train Qwen2.5-3B with the lighter config above. ~10-15 min on L4.
2. Submit + 90 min eval.
3. If win-rate ≥ 32% (matches best base+gen variant), this is the **first SFT
   that actually wins**. Then mix synthetic + PIPPA for a 5k second pass.
4. If still < 30% → SFT is genuinely the wrong tool for Chai's preference
   dimension; commit to best base × best gen final answer.
