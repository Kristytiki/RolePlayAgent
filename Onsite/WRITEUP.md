# Onsite Write-Up — Chaiverse Submission

**Author:** Zheqi Wu
**Date:** 2026-06-20
**Repo:** https://github.com/Kristytiki/RolePlayAgent
**Goal:** Maximize Chaiverse win-rate within onsite time budget by exploring a 4-dimensional design space (base model × formatter × generation params × supervised fine-tune × LoRA ensemble).

---

## 1. Headline submissions

### Best individual submission — Qwen3-30B-A3B + long output

| Field | Value |
|---|---|
| **Submission ID** | `qwen-qwen3-30b-a3b-inst_16638_v7` |
| **Console URL** | https://console.chaiverse.com/models/qwen-qwen3-30b-a3b-inst_16638_v7 |
| **Win-rate** | **38.9%** (10,144 battles) |
| **Base model** | `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE, ~3B active) |
| **Method** | No SFT; off-the-shelf base + ChatML formatter + `max_output_tokens=80, stopping_words=[]` |

### Best fine-tuned submission — Qwen3-4B-Anime + best_of=64

| Field | Value |
|---|---|
| **Submission ID** | `zheqiwu-qwen3-4b-anime-step60_v5` |
| **Console URL** | https://console.chaiverse.com/models/zheqiwu-qwen3-4b-anime-step60_v5 |
| **HF model** | https://huggingface.co/ZheqiWu/Qwen3-4B-Anime-step60 |
| **Win-rate** | **36.4%** (1,730 battles) |
| **Base model** | `Qwen/Qwen3-4B-Instruct-2507` |
| **Fine-tune data** | 100-record Sonnet-4.6 anime ShareGPT corpus (`sft_chai_anime.json`) |
| **Method** | LoRA r=16 SFT, 60 steps, lr=1e-4, merged 16-bit |
| **Gen params** | `best_of=64`, others default |

Full per-submission ledger (slug, gen-params, formatter, win-rate, num_battles)
is in `submissions.json` (machine-readable) and `submissions.xlsx`
(human-readable, color-coded by Dim and leaderboard rank).

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

| ID | Dataset | Volume | Generator / Source | Style | File |
|---|---|---|---|---|---|
| (a) | **CoSER** | 30k of 305k | `Neph0s/CoSER`, distilled from 771 books | Literary, `[thought] (action)` | `sft_chai_aligned.json` |
| (b) | **PIPPA** | 1,944 of 16,832 (filtered) | `PygmalionAI/PIPPA` (character.ai user dump) | Real chat RP, `*action*` | `sft_chai_pippa.json` |
| (c) | **Hieunguyenminh** | 5,000 of 5,755 | `Hieunguyenminh/roleplay` HF dataset | QA-style "lecture" voice | `sft_chai_hieu.json` |
| (d) | **anime v1** ⭐ | 100 | Claude Sonnet 4.6, 8-line prompt | Single-line ≤50 tok, anime/RP | `sft_chai_anime.json` |
| (e) | **anime v2** | 1,000 | Claude Sonnet 4.6, onsite-style few-shot | 30-50 turn cinematic, weighted msgs | `sft_chai_anime_v2.json` |
| (f) | **anime v3_gpt** | 953 of 1,000 | GPT-OSS-120B (Bedrock), same prompt as v2 | Same shape as v2 | `sft_chai_anime_v3_gpt.json` |
| (g) | **synth** | 1,000 | Bedrock `claude-opus-4-8`, schema-controlled | Casual texting, 24 personas | `sft_chai_synth.json` |

**Preprocessing (universal across all sources):**
1. Strip leading `"<character>: "` from each assistant turn (Chai's bot_template prepends it at inference).
2. Collapse `\n\n` and internal newlines to single space (Chai uses `stopping_words=['\n']`).
3. Drop empty / multi-line / template-leftover turns.
4. Cap each turn at ~600 chars; cap conversations at 16-20 turns.

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

## 8. Final win-rate ranking (≥1k battles)

| Rank | Submission | Win | Battles | Approach |
|---|---|---|---|---|
| 🥇 | `qwen3_30b_a3b_long` | **38.9%** | 10,144 | A+C: MoE base + long output |
| 🥈 | `qwen3_4b_anime_step60_bo64` | **36.4%** | 1,730 | A+C+D: Qwen3-4B + anime-v1 SFT + best_of=64 |
| 🥉 | `qwen3_30b_a3b_bo16` | 36.2% | 10,149 | A+C: MoE + best_of=16 |
| 4 | `qwen3_4b_anime_step100_long` | 36.2% | 1,750 | A+C+D: anime SFT + long output |
| 5 | `qwen3_4b_long` | 35.3% | 10,509 | A+C: Qwen3-4B + long output |
| 6 | `qwen3_30b_a3b` | 34.9% | 10,684 | A: MoE base default |
| 7 | `qwen3_4b_anime_step60_long` | 34.8% | 1,732 | A+C+D |
| 8 | `qwen3_4b_anime_step60_bo16` | 33.8% | 7,582 | A+C+D |
| 9 | `qwen25_3b_long` | 33.3% | 10,356 | A+C: Qwen2.5-3B + long output |
| 10 | `qwen3_4b_long` | 33.3% | 10,270 | A+C: Qwen3-4B + long output |

**Takeaways:**
- The single biggest base-model lever is **Qwen3-30B-A3B** (MoE, 3B active params at inference) — it tops the leaderboard with no fine-tuning.
- Among ≤4B dense models, **Qwen3-4B + anime v1 SFT + best_of=64** is the winning fine-tuned combo (36.4%) and beats every Qwen2.5-3B variant we tried.
- `best_of` is the most cost-effective gen-param — going from 8 → 64 buys ~3 percentage points on SFT models.
- **Distribution match dominates volume** — 100 schema-matched records (anime v1) beat 30,000 literary records (CoSER) by ~10 percentage points.
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
│   ├── SCHEMA.md                     # data schema design doc
│   ├── download.sh                   # pull base models + datasets from HF
│   ├── preprocess.py                 # CoSER → ShareGPT
│   ├── preprocess_pippa.py           # PIPPA → ShareGPT (category + casual filter)
│   ├── preprocess_hieu.py            # Hieunguyenminh → ShareGPT
│   ├── gen_synthetic.py              # Bedrock Opus 4.8 schema-controlled synthesis
│   ├── setup_env.sh                  # uv venv + torch cu124 + Unsloth (pinned)
│   ├── train.py                      # Qwen base SFT (DATASET, RUN_TAG, LORA_R, LR env)
│   ├── train_llama.py                # Llama base SFT
│   └── run_after_first.sh            # serial watchdog (run B after A finishes)
├── dpo/                               # DPO scaffolding (gen_candidates / build_pairs / train_dpo)
└── train_model/                       # merged 16-bit checkpoints (safetensors gitignored)
```

### Reproduction
```bash
cd Onsite/sft
export HF_TOKEN=<read+write>
export CHAI_DEVELOPER_KEY=<your CR_...>

# 1. env
bash setup_env.sh && source .venv/bin/activate

# 2. data + base
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir assets/Qwen3-4B-Instruct-2507
# anime v1 generation lives in another workflow that fans out 100 Sonnet workers;
# the resulting json is committed under assets/sft_chai_anime.json.

# 3. train (winning recipe)
DATASET=anime LORA_R=16 LORA_ALPHA=32 LR=1e-4 MAX_STEPS=60 SNAPSHOT_AT=30,60 \
  python train.py

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
