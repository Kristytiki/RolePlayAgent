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

Complete Results: @/RolePlayAgent/Onsite/submissions.xlsx

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
