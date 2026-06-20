# Onsite Write-Up — Chaiverse Submission

**Author:** Zheqi Wu
**Date:** 2026-06-20
**Repo:** https://github.com/Kristytiki/RolePlayAgent
**Goal:** Maximize Chaiverse win-rate within onsite time budget by exploring a 4-dimensional design space (base model × formatter × generation params × supervised fine-tune × LoRA ensemble).

---

## 1. Best submission

| Field | Value |
|---|---|
| **Submission ID** | `zheqiwu-qwen3-4b-anime-s_1701_v3` |
| **Console URL** | https://console.chaiverse.com/models/zheqiwu-qwen3-4b-anime-s_1701_v3 |
| **HF model** | https://huggingface.co/ZheqiWu/Qwen3-4B-Anime-step100 |
| **Win-rate** | **39.25%** (9,720 battles) |
| **Baseline** | 30.92% (`qwen-qwen2-5-3b-instruct_v18`, 10,486 battles) |
| **Δ vs baseline** | **+8.33 pp** (≈ +27% relative) |

### Each parameter, what it means

**Training:**

| Parameter | Value | Meaning |
|---|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` | Pretrained backbone we fine-tune on top of. 4B dense Qwen3 (Sept 2025), already instruction-tuned. |
| Fine-tune data | 100 records of `sft_chai_anime.json` | 100 Sonnet-4.6-generated anime/fandom/OC ShareGPT conversations, schema-matched to Chai inference (single-line, `*action*`, no `[thought]`, no name prefix). |
| LoRA `r` | 32 | Rank of the low-rank adapter matrices (`ΔW = A·B`). r=32 means each module learns ~30M trainable params (≈1.9% of base). |
| LoRA `alpha` | 64 | Scale factor — effective LoRA contribution at inference is `(alpha/r) · ΔW = 2 · ΔW`. |
| Learning rate | `1e-4` | Lower than the default `2e-4` to avoid CoSER's over-shoot pattern (loss kept dropping but win-rate fell). |
| Steps | 100 | Cosine schedule with `warmup_steps=10`. step100 narrowly beat step60 at ≥9k battles (39.25% vs 38.52%). |
| Effective batch | 8 | per-device 2 × grad-accum 4. |
| `max_seq_length` | 2048 | Matches Chai's `max_input_tokens=2048`; longer context wasted at inference. |
| Loss masking | assistant turns only | `train_on_responses_only` via ChatML markers (`<\|im_start\|>user\n` ↔ `<\|im_start\|>assistant\n`). The model only learns to predict character replies, not the user/system text. |

**Inference (generation_params at submit time):**

| Parameter | Value | Meaning |
|---|---|---|
| `best_of` | **64** | vLLM samples 64 candidate replies per turn; Chai's reranker picks the best one. The single highest-leverage knob — `bo=8 → bo=64` adds ~+5 pp on a fixed SFT model. |
| `max_output_tokens` | **80** | Maximum tokens emitted per reply. Chai validation hard-caps at 80; default was 64. |
| `stopping_words` | **`[]`** (empty) | Default was `["\n"]` (truncate at first newline → forces single-line replies). Removing it lets the model write 1-2 sentence replies, which Chai users prefer. |
| `temperature` | 1.0 | Sampling temperature. |
| `top_p` | 1.0 | Nucleus sampling threshold (effectively off). |
| `top_k` | 40 | Restricts sampling to top-40 tokens at each step. |
| `min_p`, `presence_penalty`, `frequency_penalty` | 0.0 | All defaults. |
| `max_input_tokens` | 2048 | Same value the model was fine-tuned at. |

The `best_of=64 + max_output=80 + stopping_words=[]` combination contributes
**+7.44 pp** on top of the base SFT model (`qwen3_4b_anime_step100` default
gen = 31.81% vs `bo64_long` = 39.25%) — the largest single lift in the
portfolio.

**Formatter (Chai bot template, ChatML):**

```python
{
  "memory_template":   "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template":   "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template":      "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template":     "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
  "truncate_by_message": True,
}
```

`response_template` ends with `{bot_name}:` so Chai prepends the character
name itself at inference. Training data must therefore strip the `Name: `
prefix from assistant turns — otherwise the model emits the name a second
time and you get `"Aria: Aria: ..."`.

---

## 2. Strategy: 4 design dimensions × portfolio search

| Dim | Tunable | GPU? | # variants explored |
|---|---|---|---|
| **A — Base model** | Which pretrained LLM Chaiverse serves | No | 11 |
| **B — Formatter** | Per-family chat template + optional system-prompt injection | No | 5 |
| **C — Generation params** | Sampling, `best_of`, `max_output_tokens`, stopping words | No | 5+ |
| **D — SFT** | LoRA fine-tune on roleplay data, push to HF, submit | Yes | 30+ |
| **E — LoRA ensemble** | Weight-space averaging of multiple same-base LoRA adapters | No (post-train) | several |

Single signal: `win_ratio` from Chaiverse, ~90 min eval per submission.
Submissions ran in parallel server-side, so we fired in batches.

---

## 3. Dim A — Base model scan (≤30B)

11 base models tried; per-family formatter applied (wrong template tanks win-rate).

| Family | Models | Template |
|---|---|---|
| **Qwen2.5** | 0.5B / 1.5B / 3B Instruct | ChatML (`<\|im_start\|>...<\|im_end\|>`) |
| **Qwen3** | 1.7B / 4B-Instruct-2507 / 30B-A3B-Instruct-2507 (MoE) | ChatML |
| **Llama-3.2** | 1B / 3B Instruct | Header-id (`<\|start_header_id\|>...<\|eot_id\|>`) |
| **Gemma** | gemma-3-1b-it, gemma-4-E2B-it | Turns (`<start_of_turn>...<end_of_turn>`) |
| **SmolLM2** | 1.7B Instruct | ChatML-like |

**Key finding:** Qwen3-30B-A3B (MoE, ~3B active) and Qwen3-4B beat all smaller
bases. Gemma-4-E2B failed to deploy (FP8 multimodal preprocessor missing
preprocessor_config.json on Chai's vLLM stack — all 5 Gemma submissions returned
0 battles).

---

## 4. Dim B — Formatter

The interesting variant injects a CoSER-style action/thought guide into the system message:

```
{memory}

Use [your thought] for thoughts which others can't see.
Use (your action) for actions which others can see.
```

Applied to Qwen2.5-3B, Qwen3-4B, Qwen3-30B-A3B, and Gemma-4-E2B. Did not move
the needle vs the equivalent no-injection submission — small effect, sometimes
slightly negative for SFT models that already encode the format.

### Per-family formatter (verbatim)

**ChatML (Qwen, SmolLM2):**
```python
{
  "memory_template":   "<|im_start|>system\n{memory}<|im_end|>\n",
  "prompt_template":   "<|im_start|>user\n{prompt}<|im_end|>\n",
  "bot_template":      "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
  "user_template":     "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
  "response_template": "<|im_start|>assistant\n{bot_name}:",
}
```

**Llama-3.x headered:**
```python
{
  "memory_template":   "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{memory}<|eot_id|>",
  "prompt_template":   "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>",
  "bot_template":      "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}: {message}<|eot_id|>",
  "user_template":     "<|start_header_id|>user<|end_header_id|>\n\n{user_name}: {message}<|eot_id|>",
  "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}:",
}
```

**Gemma turns:**
```python
{
  "memory_template":   "<start_of_turn>user\n{memory}<end_of_turn>\n",
  "prompt_template":   "<start_of_turn>user\n{prompt}<end_of_turn>\n",
  "bot_template":      "<start_of_turn>model\n{bot_name}: {message}<end_of_turn>\n",
  "user_template":     "<start_of_turn>user\n{user_name}: {message}<end_of_turn>\n",
  "response_template": "<start_of_turn>model\n{bot_name}:",
}
```

---

## 5. Dim C — Generation params

Default (Chai's onsite):
```python
temperature=1.0, top_p=1.0, top_k=40, min_p=0.0,
presence_penalty=0.0, frequency_penalty=0.0,
stopping_words=["\n"],
max_input_tokens=2048, max_output_tokens=64, best_of=8
```

Variants and observed effects:

| Override | Effect |
|---|---|
| `best_of=16 / 32 / 64` | **Largest single lever.** Reranker picks better candidate as N grows. anime-step60 went from 33.6% (bo=8) → 36.4% (bo=64). |
| `max_output_tokens=80, stopping_words=[]` | Allows longer free-form replies. Helped Qwen3-30B-A3B (38.9% vs 36.2% baseline). Note: Chai validation rejects `>80`; we initially tried 128 and got rejected. |
| `frequency_penalty=0.3` | Reduce repetition. Mild positive on small bases. |
| `temperature=0.8, top_p=0.9` | Conservative sampling. Helped Qwen2.5-3B (33.3%). |

---

## 6. Dim D — Supervised fine-tuning

**Stack:** Unsloth 2026.6.8 + TRL 0.18.2 + Transformers 5.5 + PyTorch 2.6.0+cu124, single NVIDIA L4 (24 GB), bf16, 4-bit quantized base.

**LoRA config:** `r=16, lora_alpha=32, dropout=0.0`, target Q/K/V/O + gate/up/down.
Lessons learned: the initial CoSER run with `r=32, lr=2e-4` over-shot (loss
dropped but win-rate fell as steps increased); we switched to `r=16, lr=1e-4`
for all subsequent runs.

**Training args:** per-device batch 2, grad-accum 4 (effective 8),
`max_seq_length=2048` to match Chai's `max_input_tokens`, cosine LR with
`warmup_steps=10`, `optim="adamw_8bit"`, `train_on_responses_only` masking via
ChatML markers.

### 6.1 Datasets (the central design decision)

All 7 datasets we trained against, in one place. Every dataset is preprocessed
to the same Chai-aligned ShareGPT format (single-line assistant targets, no
`{name}:` prefix, ≤80 tokens per turn).

| ID | Dataset | Origin | Generator | Format / style | Volume (raw → kept) | File | Preprocess script | Best win-rate observed |
|---|---|---|---|---|---|---|---|---|
| (a) | **CoSER** | `Neph0s/CoSER` (distilled from 771 books) | — (real text) | Literary `[thought] (action) "speech"`, multi-paragraph | 305,134 → 30,000 (random subset, seed 42) | `sft_chai_aligned.json` | `data/preprocess_coser.py` | 27.2% (Qwen2.5-3B step30) — **below baseline** |
| (b) | **PIPPA** | `PygmalionAI/PIPPA` (character.ai user dump) | — (real users) | Real chat RP, `*action*` convention | 16,832 → 1,944 (category whitelist + casual filter) | `sft_chai_pippa.json` | `data/preprocess_pippa.py` | 30.6% (Qwen2.5-3B r=16 + bo16) |
| (c) | **Hieunguyenminh** | `Hieunguyenminh/roleplay` HF dataset | — | QA-style "lecture" voice (Indeed!, Ah my friend...) | 5,755 → 5,000 | `sft_chai_hieu.json` | not trained in final round |
| (d) | **anime v1** ⭐ | self-generated | Claude Sonnet 4.6 | 8-line prompt, single-line ≤50 tok, anime tropes + fandom canon + dark OC | 100 (100 character specs × 1 conversation each, 100/100 valid) | `sft_chai_anime.json` | n/a (workflow-generated) | **39.25%** (Qwen3-4B step100 + bo64 + long) |
| (e) | **anime v2** | self-generated | Claude Sonnet 4.6 | Onsite-style few-shot (Hogwarts/Sylus), 30-50 turn cinematic, OpenAI `messages` w/ `weight=0/1` | 1,000 (100 base × 10 scenarios; 4 batches × 250-way concurrency) | `sft_chai_anime_v2.json` | n/a | 33.6% (Qwen3-4B step100) |
| (f) | **anime v3_gpt** | self-generated | GPT-OSS-120B (Bedrock) | Same prompt as v2 | 1,000 → 953 (47 dropped to schema drift) | `sft_chai_anime_v3_gpt.json` | n/a | 31.0% (Qwen3-4B step100) |
| (g) | **synth** | self-generated | Bedrock `us.anthropic.claude-opus-4-8` | Schema-controlled (`SCHEMA.md`): 24 personas × 20 openers, 4 user + 4 assistant turns, 15-40 tok, *action* only, ~30% emoji | 1,000 (async concurrency 16; ~$25 / 8 min) | `sft_chai_synth.json` | `data/gen_synthetic.py` | 31.3% (Qwen3-4B step60 + bo32) |

**Universal preprocessing** (applied to every dataset before SFT):

1. Strip leading `"<character>: "` from each assistant turn (Chai's `bot_template` prepends it at inference).
2. Collapse `\n\n` and internal newlines to single space (Chai uses `stopping_words=['\n']`).
3. Drop empty / multi-line / template-leftover turns.
4. Cap each turn at ~600 chars; cap conversations at 16-20 turns.

**Generation pipelines for the 4 self-generated datasets:**
- **anime v1 / v2**: Sonnet 4.6 fanned out across N character specs via a workflow (one subagent per spec, JSON-schema-validated). v1 = 100 specs simple prompt; v2 = 1000 specs with onsite-style few-shot (`assets/onsite.example`).
- **anime v3_gpt**: same v2 prompt sent to GPT-OSS-120B via Bedrock `boto3 invoke_model`, async concurrency 16.
- **synth**: `data/gen_synthetic.py` — Bedrock Opus 4.8, schema-controlled per `data/SCHEMA.md` (24 personas × 20 openers, strict per-turn rules: ≤40 tok, single-line, *action* only, no `[thought]`, no name prefix, ~30% emoji).

### 6.2 SFT result summary

| Base × Data | Best win-rate | Best step | Notes |
|---|---|---|---|
| Qwen2.5-3B + CoSER | 27.2% | 30 | **Worse than 30% no-SFT baseline.** More steps → worse (25.6% at 500). Distribution mismatch. |
| Qwen2.5-3B + PIPPA | 29.0% | r=16 | Better than CoSER but still below baseline. |
| Qwen2.5-3B + synth | 31.6% | 60 | First SFT to beat Qwen2.5-3B no-SFT. |
| **Qwen3-4B + anime v1** ⭐ | **36.4%** | **60 (+best_of=64)** | **Best SFT.** 100 records of distribution-matched data > 30k mismatched. |
| Qwen3-4B + anime v2 (1k Sonnet onsite-style) | 33.6% | 100 | More data didn't help; the format drift hurt. |
| Qwen3-4B + anime v3_gpt (1k GPT) | 31.0% | 100 | GPT-OSS generations less on-spec than Sonnet. |
| Llama-3.2-3B + anime | 30.1% | 100 (v2) | |

### 6.3 Why anime v1 outperformed everything despite 1/300 the data of CoSER

1. **Distribution match.** Sonnet 4.6 was prompted to write Chai-style RP (anime/tsundere/yandere/OC, single-line, `*action*` only). CoSER's literary distribution is what Chai users actively dislike.
2. **Format match.** Anime v1 outputs were already ≤50 tokens, single-line, `*action*` — no preprocessing battles.
3. **Less is more.** Smaller corpus + lighter LoRA (r=16) + early stopping (60 steps) avoids the over-fitting drift visible in CoSER's loss-vs-win-rate divergence.

**Loss is not a reliable training signal.** CoSER's loss continued to drop from
2.44 (step 30) → 1.70 (step 500) while win-rate fell from 27.2% → 25.6%.
For Chai's preference signal, only `win_ratio` matters.

### 6.4 anime v1 generation — exact prompt and pipeline (best-model recipe)

This is the dataset that produced the strongest fine-tune we shipped:
`Qwen3-4B-Anime-step100` + `best_of=64` + `max_output_tokens=80, stopping_words=[]`
= **39.25% win-rate** (`zheqiwu-qwen3-4b-anime-s_1701_v3`, 9,720 battles —
+8.33 pp over the 30.92% baseline; +0.35 pp over the un-fine-tuned
Qwen3-30B-A3B MoE submission). Reproducing here in full because it's the
load-bearing piece of the result.

**Code:** `Onsite/dpo/assets/anime_gen_workflow.js` — a workflow script that
fans out 100 subagents (one per character spec). Each subagent is asked to
emit a single-record ShareGPT JSON. Each call is independently schema-validated,
and 100/100 succeeded.

**Character pool (100 specs, 3 categories):**

- ~50 **fandom canon characters** with one-line canon descriptions:
  Asuka Langley, Shinobu Kocho, Sakura Matou, Makima, Ochaco Uraraka, Kafka,
  Mitsuri Kanroji, Power, Marin Kitagawa, Megumin, Saber, Ram, Senko-san,
  Mai Sakurajima, Rem, Mikasa Ackerman, Kurapika, Astolfo, Rei Ayanami,
  Yuno Gasai, March 7th, Raiden Shogun, Komi Shouko, Rin Tohsaka, Bakugo,
  Hu Tao, Silver Wolf, Yae Miko, Hisoka, Mei Hatsume, Killua, Historia,
  Power, Mio Akiyama, Nezuko, Bronya, etc.
- ~30 **anime tropes** parameterized by archetype, with two `(variant 2)`
  variations each:
  - tsundere demon lord summoned by accident
  - oneesan landlady with a hidden possessive streak
  - yandere childhood friend planning your "wedding" since age 6
  - kuudere robot maid programmed to serve you
  - tsundere classmate who keeps insulting you
  - kuudere swordswoman ghost haunting your inherited katana
  - dandere library girl who only opens up about books
  - genki energetic neko-girl streamer with dark backstory
  - mahou shoujo magical girl in disguise
  - tomboy childhood rival pro gamer
  - imouto adopted little sister who is too clingy
  - shy kitsune shrine maiden with nine hidden tails
  - ojou-sama heiress slumming at a coffee shop
  - yakuza heiress hiding at a maid cafe
  - energetic vampire stuck looking 14 but actually 400yo
  - …
- ~20 **dark OC** specs:
  - vampire mafia capo who kidnapped you to be her blood-source-fiancée
  - yandere knight liege who will slay all your enemies
  - cyberpunk netrunner streamer who spliced into your neural ports
  - corporate yandere CEO who bought your debt
  - kemonomimi assassin sent to kill you who fell asleep in your bed
  - eldritch elder god in the form of a tiny pouty cosplayer
  - …

**Output schema (enforced via JSON-schema):**
```json
{
  "system": "<200-700 char character profile + scenario hook>",
  "turns": [
    { "from": "human" | "assistant", "value": "<≤320 chars>" }
  ]   // 6-10 turns, starting with human
}
```

**Per-character prompt (verbatim, sent to Claude Sonnet 4.6 worker):**

```
You are generating a high-quality role-play SFT conversation for a Chaiverse
leaderboard model.

Chaiverse users want CUTE, ANIME, FANDOM, OC chatbot vibes — tsundere/yandere/
kuudere, fandom canon characters, monster-girl/vampire OC. Replies are SAMPLED
with stop=["\n"] and max_output=64 tokens, so single-line short outputs only.

## YOUR TARGET
{one of:}
  FANDOM CHARACTER: <name>
    Canon: <one-line canon description>
{or:}
  ANIME TROPE: <trope spec>
    Make up an original name fitting this archetype.
{or:}
  DARK OC: <spec>
    Make up an original name and an opening situation.

## OUTPUT SHAPE
A ShareGPT-style multi-turn conversation:
- system: 200-700 chars character profile + scenario hook
  (e.g. "You are X. <traits, voice, scenario>. The scene begins as user...")
- turns: 6-10 alternating turns, ALWAYS starting with from="human" first turn
  = "===Conversation Start===" empty bait, OR a scene-setting human action.
- assistant turns must be SINGLE LINE (no newline character), ≤ ~50 tokens,
  with *action* asterisks or (action) parens, and the character speaking
  IN-CHARACTER.
- assistant turns should NOT prefix with the character name.
- conversation should escalate naturally — the user pokes/teases/responds,
  the character replies in voice.
- VARY turn shapes: some pure dialogue, some pure action, some mix. Use
  *blushes* (clenches fist) [thinks: ...] sparingly when fitting the persona.
- AVOID generic openers like "Hello!" — pick something character-specific.
- AVOID AI-assistant disclaimers, NO breaking the fourth wall.
- Tone: cute, possessive, teasing, dark/dramatic — match the persona.
  Light NSFW innuendo is FINE for adult RP characters but no explicit sex.

Return ONLY the JSON object with system + turns. No prose, no markdown.
```

**Pipeline:**
1. `SPECS` — 100 records hand-curated (50 fandom + 30 trope + 20 dark OC, with two `variant 2` rotations of each trope/OC for diversity).
2. `pipeline(SPECS, gen_one_with_schema)` — fans out 100 concurrent Sonnet 4.6 calls, each constrained by the JSON schema above.
3. Output validated → drop failures → save as `Onsite/sft/assets/sft_chai_anime.json` (100 records, 173 KB).

**Why this prompt works:**

- Tells the generator the **exact downstream constraint** (`stop=["\n"]`,
  `max_output=64`) so it self-imposes single-line short outputs at generation
  time, instead of needing aggressive preprocessing later.
- Restricts the **distribution to Chai's actual user taste** (anime / fandom /
  OC, tsundere/yandere/kuudere) — no philosophy, no advice, no discussion.
- Forces the assistant turn to **omit the character-name prefix** (Chai's
  bot_template already supplies it), avoiding the double-prefix bug.
- Asks for **persona-specific scenario hooks** instead of generic "Hello!"
  openers, so each conversation looks like a different roleplay setting.

**Training recipe** (best at `step100` and `step60`):

```bash
cd Onsite/sft && source .venv/bin/activate
DATASET=anime BASE_MODEL=qwen3-4b \
    LORA_R=32 LORA_ALPHA=64 LR=1e-4 \
    MAX_STEPS=100 SNAPSHOT_AT=30,60,100 SAVE_LORA_EVERY=30 \
    python core/train.py
```

Other defaults from `core/train.py`:
- `per_device_batch=2`, `grad_accum=4` (effective batch 8)
- `max_seq_length=2048` (matches Chai's `max_input_tokens`)
- `lr_scheduler_type="cosine"`, `warmup_steps=10`
- `optim="adamw_8bit"`, `weight_decay=0.0`, `bf16=True`
- 4-bit quantized base via Unsloth
- `train_on_responses_only` masking via ChatML markers (`<|im_start|>user\n` ↔ `<|im_start|>assistant\n`)

**Submission recipe** (the +4 percentage-point lift on top of the SFT model):

```python
generation_params = {
    "temperature": 1.0, "top_p": 1.0, "top_k": 40, "min_p": 0.0,
    "presence_penalty": 0.0, "frequency_penalty": 0.0,
    "stopping_words": [],            # ← removed; allow multi-sentence replies
    "max_input_tokens": 2048,
    "max_output_tokens": 80,         # ← max Chai allows
    "best_of": 64,                   # ← reranker picks from 64 candidates
}
formatter = QWEN_CHATML  # see §4
```

Final sequence to ship the best fine-tune:

```bash
# 1. train (above)
# 2. push merged 16-bit
hf upload ZheqiWu/Qwen3-4B-Anime-step100 ../train_model/qwen3-4b-anime-step100 .

# 3. submit (variant qwen3_4b_anime_step100_bo64_long is in submit_batch.py)
cd .. && uv run --with requests python submit_batch.py \
    --only qwen3_4b_anime_step100_bo64_long
```

---

## 7. Dim E — LoRA ensemble (weight-space averaging)

**Why:** each LoRA over-fits to its data's quirks. Averaging weight deltas
across same-base adapters cancels idiosyncratic noise while preserving the
shared roleplay signal. Free at inference (one merged checkpoint).

**Hard requirements:** identical base model, identical `target_modules`,
identical LoRA `r`. Cannot ensemble Qwen3-4B with Qwen2.5-3B in weight space
(architectures differ); would require logits ensembling, which Chai's serving
stack does not expose.

**Tools:** `peft.PeftModel.add_weighted_adapter` with three combination
strategies tried:
- `linear` — weighted sum of deltas (default, baseline)
- `ties` — drops low-magnitude params per adapter, sign-vote, then merge
- `dare_linear` — random-drop LoRA params then rescale, then linear merge

**Available Qwen3-4B LoRAs (all r=16):**
- `qwen3-4b-anime-lora/{step30,60,90}` — Sonnet 4.6 anime v1 (peak 36.4%)
- `qwen3-4b-anime-v2-lora/*` — Sonnet 4.6 anime v2 (1k onsite-style)
- `qwen3-4b-anime-v3-gpt-lora/*` — GPT-OSS anime v3 (1k)

**Observed:**
- Same-data different-step ensembles (step30 + step60 + step90) collapse to
  ~step50 — low ROI.
- Same-data different-generator ensembles (Sonnet v1 + Sonnet v2 + GPT v3) are
  the more interesting axis: they average out generator-specific stylistic
  quirks while preserving the anime distribution.

---

## 8. Final win-rate ranking (≥5k battles, sorted by lift over baseline)

Baseline: `qwen-qwen2-5-3b-instruct_v18` = **30.92%** (10,486 battles), the
default Qwen2.5-3B-Instruct submission Chai shipped in the onsite kit.
**Δ** column = win-rate − baseline.

| Rank | Submission | Win | Δ vs baseline | Battles | Approach |
|---|---|---|---|---|---|
| 🥇 | `qwen3_4b_anime_step100_bo64_long` ⭐ **SFT** | **39.25%** | **+8.33 pp** | 9,720 | A+C+D: Qwen3-4B + anime-v1 + bo=64 + long |
| 🥈 | `qwen3_30b_a3b_long` | 38.90% | +7.98 pp | 10,144 | A+C: MoE base + long output (no SFT) |
| 🥉 | `qwen3_4b_anime_step60_bo64_long` **SFT** | 38.52% | +7.60 pp | 9,969 | A+C+D: anime SFT step60 + bo=64 + long |
| 4 | `qwen3_30b_a3b_bo16` | 36.24% | +5.32 pp | 10,149 | A+C: MoE + best_of=16 |
| 5 | `qwen3_4b_anime_step60_bo64_v2` SFT | 35.52% | +4.60 pp | 9,818 | A+C+D |
| 6 | `qwen3_4b_anime_step60_bo128` SFT | 35.31% | +4.39 pp | 9,994 | A+C+D: bo=128 |
| 7 | `qwen3_4b_long` | 35.31% | +4.39 pp | 10,509 | A+C: Qwen3-4B base + long |
| 8 | `qwen3_4b_anime_step60_bo64_tight` SFT | 35.28% | +4.36 pp | 9,836 | A+C+D |
| 9 | `qwen3_4b_anime_step100_long` SFT | 35.27% | +4.35 pp | 10,147 | A+C+D: bo=8 + long |
| 10 | `qwen3_4b_anime_step60_bo64` SFT | 34.91% | +3.99 pp | 10,207 | A+C+D |

**Takeaways:**
- The strongest single model is a **fine-tuned 4B**, not the 30B MoE base. `qwen3_4b_anime_step100_bo64_long` beats `qwen3_30b_a3b_long` by +0.35 pp using an active model 7× smaller (4B dense vs 30B MoE).
- 8 of the top-10 are Qwen3-4B with anime-v1 SFT — the **base × dataset combo** that consistently performed.
- `best_of=64 + max_output=80 + stopping_words=[]` is the single most lucrative gen-param config; it adds ≈ +7 pp on top of the SFT step100 default-gen submission.
- **Distribution match dominates volume** — 100 schema-matched anime records beat 30,000 literary records (CoSER) by ~12 pp on the same Qwen3-4B base.
- **Loss does not predict win-rate.** Always evaluate on Chaiverse, not training metrics.

---

## 9. Reproducibility

### Files in this repo
```
Onsite/
├── WRITEUP.md                        # this file
├── PLAN.md                            # original 4-dim portfolio plan
├── submit_batch.py                   # portfolio submitter (reads variants, fires to Chaiverse)
├── onsite_submit.py                   # legacy single-shot submitter
├── submissions.json / .xlsx          # full ledger with win-rate, gen-params, formatter
├── SUBMISSIONS.md                    # auto-appended submission log
├── sft/
│   ├── README.md                     # detailed SFT pipeline doc
│   ├── core/
│   │   ├── setup_env.sh              # uv venv + torch cu124 + Unsloth (pinned)
│   │   ├── download.sh               # pull base models + datasets from HF
│   │   ├── train.py                  # Unsloth LoRA SFT — env-driven
│   │   └── ensemble_loras.py         # peft.add_weighted_adapter wrapper
│   ├── data/
│   │   ├── SCHEMA.md                 # data schema design doc
│   │   ├── preprocess_coser.py       # CoSER → ShareGPT
│   │   ├── preprocess_pippa.py       # PIPPA → ShareGPT (category + casual filter)
│   │   ├── preprocess_hieu.py        # Hieunguyenminh → ShareGPT
│   │   └── gen_synthetic.py          # Bedrock Opus 4.8 schema-controlled synthesis
│   └── runs/
│       └── ensemble/run.sh           # build 3 ensembles + push HF
├── dpo/                               # DPO scaffolding (gen_candidates / build_pairs / train_dpo)
└── train_model/                       # merged 16-bit checkpoints (safetensors gitignored)
```

### Reproduction
```bash
cd Onsite/sft
export HF_TOKEN=<read+write>
export CHAI_DEVELOPER_KEY=<your CR_...>

# 1. env
bash core/setup_env.sh && source .venv/bin/activate

# 2. data + base
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir assets/Qwen3-4B-Instruct-2507
# anime v1 generation lives in another workflow that fans out 100 Sonnet workers;
# the resulting json is committed under assets/sft_chai_anime.json.

# 3. train (winning recipe)
DATASET=anime BASE_MODEL=qwen3-4b LORA_R=32 LORA_ALPHA=64 LR=1e-4 \
  MAX_STEPS=60 SNAPSHOT_AT=30,60 SAVE_LORA_EVERY=30 \
  python core/train.py

# 4. push merged
hf upload <user>/Qwen3-4B-Anime-step60 ../train_model/qwen3-4b-anime-step60

# 5. submit
cd .. && python submit_batch.py --only qwen3_4b_anime_step60_bo64
```

### Software environment
| Package | Version |
|---|---|
| Python | 3.11 |
| torch | 2.6.0+cu124 |
| transformers | 5.5.0 |
| trl | 0.18.2 |
| unsloth | 2026.6.8 (commit `7ce8dc7`) |
| peft | resolved via Unsloth |
| bitsandbytes | ≥0.43 |
| xformers | 0.0.29.post3 |
| GPU driver | NVIDIA 595.71.05 / CUDA 13.2 |
| GPU | NVIDIA L4, 24 GB |

### Datasets (HuggingFace)
- `Neph0s/CoSER` — license per dataset card
- `PygmalionAI/PIPPA` — license per dataset card
- `Hieunguyenminh/roleplay` — license per dataset card
- anime v1/v2/v3_gpt and synth — generated locally (Sonnet 4.6 / GPT-OSS-120B / Opus 4.8). Generation scripts in repo.

### HuggingFace model artifacts (public)
- https://huggingface.co/ZheqiWu/Qwen3-4B-Anime-step60 (best SFT)
- https://huggingface.co/ZheqiWu/Qwen3-4B-Anime-step100
- https://huggingface.co/ZheqiWu/Qwen3-4B-Anime-step30
- https://huggingface.co/ZheqiWu/Qwen2.5-3B-synth-step60
- https://huggingface.co/ZheqiWu/Qwen2.5-3B-PIPPA-r16
- https://huggingface.co/ZheqiWu/Qwen2.5-3B-CoSER-step500 (lessons-learned baseline)

---

## 10. Limitations

1. **Anime v1 has only 100 records.** That it produced the best Qwen3-4B SFT is partly luck on a small sample; a 200-record run with the same prompt could either confirm or regress.
2. **`best_of=64` is expensive at inference.** Chaiverse pays for it; production deployment would need cost-vs-quality balancing.
3. **No held-out evaluator beyond Chaiverse.** All decisions made on `win_ratio`, which has ~90 min latency. No CoSER GCA or local human eval.
4. **Gemma-4-E2B was unevaluated** due to Chai's vLLM stack failing to load the FP8 multimodal preprocessor. Excluded from rankings.
5. **DPO line not landed.** Scaffolding exists in `Onsite/dpo/`; ran out of time to validate against a sufficient pair count.
