# Onsite Write-Up — Chaiverse Submission

**Author:** Zheqi Wu
**Date:** 2026-06-19
**Goal:** Maximize Chaiverse win-rate within onsite time budget by exploring a 4-dimensional design space (base model × formatter × generation params × supervised fine-tuning).

---

## 1. Headline Submissions

### Primary recommended submission (Dim D, supervised fine-tune)

| Field | Value |
|---|---|
| **Submission ID** | `zheqiwu-qwen2-5-3b-cose_13112_v1` |
| **Console URL** | https://console.chaiverse.com/models/zheqiwu-qwen2-5-3b-cose_13112_v1 |
| **HF model repo** | https://huggingface.co/ZheqiWu/Qwen2.5-3B-CoSER-smoke |
| **Base model** | `Qwen/Qwen2.5-3B-Instruct` |
| **Fine-tune data** | `Neph0s/CoSER` train split (preprocessed, see §3.2) |
| **Method** | LoRA r=32 SFT on 30k roleplay conversations, merged 16-bit |
| **Formatter** | Qwen ChatML (see §4) |
| **Gen params** | default (temp=1.0, top_p=1.0, best_of=8, max_output=64) |

### Companion stacked submissions (same fine-tuned weights, different inference settings)

| Slug | Submission ID | Tweak |
|---|---|---|
| `qwen3b_coser_smoke_bo16` | `zheqiwu-qwen2-5-3b-cose_13112_v2` | `best_of: 16` |
| `qwen3b_coser_smoke_coser_guide` | `zheqiwu-qwen2-5-3b-cose_13112_v3` | system prompt injection of `[thought] (action) speech` style guide |

### Dimension-A/B/C portfolio

A total of **32 Chaiverse submissions** were fired covering 11 base models × 5 generation-parameter variants × 4 formatter variants. Full mapping is recorded in `submissions.json` (machine-readable) and `submissions.xlsx` (human-readable, color-coded by status). See §6.

Notable Dim-A submissions for comparison:
- `Qwen/Qwen2.5-3B-Instruct` (matching base, no SFT) — `qwen-qwen2-5-3b-instruct_v19`
- `Qwen/Qwen3-4B-Instruct-2507` — `qwen-qwen3-4b-instruct-2507_v10`
- `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE) — `qwen-qwen3-30b-a3b-inst_16638_v3`
- `meta-llama/Llama-3.2-3B-Instruct` — `meta-llama-llama-3-2-3b_30223_v2`

---

## 2. Strategy

The win-rate signal is single-shot and slow (~90 min per submission). To make progress fast we treated each tunable surface independently, fired many cheap variants in parallel, and reserved the GPU only for the one tunable that genuinely needs training (SFT).

| Dimension | Tunable surface | GPU? | # variants |
|---|---|---|---|
| **A. Base model** | which pretrained model does Chai serve | no | 11 |
| **B. Formatter** | per-family chat template, optional system-prompt injection | no | 5 |
| **C. Generation params** | sampling, `best_of`, `max_output_tokens`, stopping words | no | 5 |
| **D. SFT** | LoRA fine-tune on roleplay data, push to HF, submit | yes | 3 |

Constraints:
- Models constrained to **≤4B params** (Chai best-suits small backbones; we tested up to 30B-A3B MoE for the upside).
- Inference is `vllm` with `max_input_tokens=2048`, `max_output_tokens≤80`, `stopping_words=['\n']`. Training data was preprocessed to match this single-line, prefix-free assistant target so train and inference distributions agree.

---

## 3. Methodology

### 3.1 Base-model scan (Dim A)

We submitted each base model with the **chat template native to its family**, since wrong-template inference produces structurally broken output and tanks win-rate.

- **Qwen2.5 / Qwen3 / SmolLM2** → ChatML (`<|im_start|>...<|im_end|>`)
- **Llama 3.2** → header-id (`<|start_header_id|>...<|eot_id|>`)
- **Gemma 3 / 4** → turn (`<start_of_turn>...<end_of_turn>`)

We tried multiple model families (Qwen2.5, Qwen3, Llama-3.2, Gemma-3/4, SmolLM2) so the base-model dimension is observed across architectures, not just Qwen sizes.

### 3.2 Data preprocessing

Source: `Neph0s/CoSER` `train/sft_conversations_sharegpt.json` (305,134 multi-turn ShareGPT conversations distilled from 771 books).

Per-message preprocessing (`Onsite/sft/preprocess.py`):

1. **Strip the leading `"Character: "` from every assistant turn.**
   The training data has assistant values like `"Ron Weasley: [thought] (action) speech\n\n"`. At inference time Chaiverse's bot template **already produces `"<|im_start|>assistant\n{bot_name}:"` as the prompt suffix**, so leaving the prefix in training would teach the model to emit the name a second time, producing `"Ron Weasley: Ron Weasley: ..."` at deploy.
2. **Collapse `\n\n` and internal newlines** into single spaces. Chaiverse uses `stopping_words=['\n']`, so any newline inside the target would truncate the response. Inline `[thought] (action) speech` fits cleanly within the 64–80 token output budget.
3. Drop empty assistant messages and conversations with no assistant turn after step 1–2.

After preprocessing we kept a random 30,000-conversation subset (seed 42). Output: `Onsite/sft/assets/sft_chai_aligned.json` (198 MB).

A representative cleaned sample:

```
[I'm terrified of him, but I can't show it. I need to assess the situation.] Did you kill the Bogge?
```

### 3.3 LoRA SFT (Dim D)

**Stack:** Unsloth 2026.6.8 + TRL 0.18.2 + Transformers 5.5 + PyTorch 2.6.0+cu124, single NVIDIA L4 (24 GB), bf16, 4-bit quantized base.

**LoRA config:**
- `r = 32`, `lora_alpha = 64`, `dropout = 0.0`
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- `use_gradient_checkpointing = "unsloth"`
- trainable params: 59,867,136 (1.90 % of 3.15 B)

**Training:**
- `max_seq_length = 2048` (matches `max_input_tokens`)
- per-device batch 2 × grad-accum 4 → effective batch 8
- `learning_rate = 2e-4`, cosine schedule, `warmup_steps = 10`
- `optim = "adamw_8bit"`, `weight_decay = 0.0`
- only loss on assistant turns via `train_on_responses_only(instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")`

**Iterations:**

We ran two SFT runs. The first targeted 1149 steps (full 1 epoch on 30k samples) but failed at the step-1000 checkpoint due to a `transformers ↔ trl` SFTConfig pickle incompatibility, which lost the in-memory LoRA delta. The fix was to set `save_strategy="no"` and manage saves explicitly via a `TrainerCallback` that writes LoRA adapters every 100 steps and full merged 16-bit snapshots at chosen step counts.

The submitted model is the **30-step smoke run** that confirmed the fixed pipeline writes a usable merged checkpoint end-to-end:

| step | loss |
|---|---|
| 5  | 2.438 |
| 10 | 2.044 |
| 15 | 1.995 |
| 20 | 1.946 |
| 25 | 1.888 |
| 30 | 1.871 |

A 500-step run with intermediate snapshots at 200 and 500 was launched after the smoke pass; if it completed in time, additional submissions were fired against those snapshots. (Check `train_model/qwen25-3b-coser-step200/` and `train_model/qwen25-3b-coser-step500/` for the merged outputs.)

**Output:** merged 16-bit Qwen2.5-3B safetensors (~5.8 GB) saved to `Onsite/train_model/qwen25-3b-coser-smoke-30step/` and pushed to `ZheqiWu/Qwen2.5-3B-CoSER-smoke`.

### 3.4 Formatter sweep (Dim B)

The interesting variant was injecting a CoSER-style instruction into the system prompt:

```
{memory}

Use [your thought] for thoughts which others can't see.
Use (your action) for actions which others can see.
```

Applied on `Qwen2.5-3B-Instruct` (slug `qwen25_3b_coser_guide`), `Qwen3-4B`, `Qwen3-30B-A3B`, and `Gemma-4-E2B-it`.

### 3.5 Generation-param sweep (Dim C)

Five gen-param variants on the strongest bases:
- `best_of=16` — let vLLM's reranker pick from a wider candidate pool.
- `max_output_tokens=80, stopping_words=[]` — allow longer free-form replies. Originally tried `max_output_tokens=128` but Chaiverse validation enforces a hard upper bound of 80; resubmitted at 80.
- `frequency_penalty=0.3` — reduce repetition.
- `temperature=0.8, top_p=0.9` — conservative sampling for stability.

---

## 4. Per-family formatter (verbatim)

**ChatML — Qwen2.5, Qwen3, SmolLM2:**
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

**Llama-3.x headered:**
```python
{
  "memory_template":   "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{memory}<|eot_id|>",
  "prompt_template":   "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>",
  "bot_template":      "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}: {message}<|eot_id|>",
  "user_template":     "<|start_header_id|>user<|end_header_id|>\n\n{user_name}: {message}<|eot_id|>",
  "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}:",
  "truncate_by_message": True,
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
  "truncate_by_message": True,
}
```

**Default generation params:**
```python
{
  "temperature": 1.0, "top_p": 1.0, "min_p": 0.0, "top_k": 40,
  "presence_penalty": 0.0, "frequency_penalty": 0.0,
  "stopping_words": ["\n"],
  "max_input_tokens": 2048, "max_output_tokens": 64,
  "best_of": 8,
}
```

---

## 5. Reproducibility

### 5.1 Code

All code lives under `Onsite/`:

```
Onsite/
├── PLAN.md                # design doc
├── WRITEUP.md             # this file
├── onsite_submit.py       # legacy single-shot submitter
├── submit_batch.py        # portfolio submitter for Dim A/B/C/D
├── submissions.json       # full per-submission record (params, formatter, status, url)
├── submissions.xlsx       # human-readable spreadsheet of the same data
├── SUBMISSIONS.md         # auto-appended log per submission
└── sft/
    ├── download.sh        # pull base + dataset
    ├── preprocess.py      # strip name prefix, collapse newlines
    ├── setup_env.sh       # uv venv + torch cu124 + Unsloth
    └── train.py           # LoRA SFT with TrainerCallback for snapshots
```

`train_model/qwen25-3b-coser-smoke-30step/` contains the merged 16-bit weights for the submitted model along with `TRAINING_PARAMS.md`.

### 5.2 Reproduction steps

```bash
cd Onsite/sft
export HF_TOKEN=<your token>

# 1. environment
bash setup_env.sh
source .venv/bin/activate

# 2. data + base model
hf download Qwen/Qwen2.5-3B-Instruct --local-dir assets/Qwen2.5-3B-Instruct
hf download Neph0s/CoSER --repo-type dataset \
   --include "train/sft_conversations_sharegpt.json" --local-dir assets/CoSER

# 3. preprocess (30k random subset)
N_KEEP=30000 python preprocess.py

# 4. train (smoke run)
python train.py        # MAX_STEPS in train.py controls run length

# 5. push merged weights to HF
hf upload <username>/Qwen2.5-3B-CoSER ./assets/qwen25-3b-coser-merged

# 6. submit to Chaiverse
cd ..
uv run --with requests python submit_batch.py \
   --only qwen3b_coser_smoke qwen3b_coser_smoke_bo16 qwen3b_coser_smoke_coser_guide
```

### 5.3 Dataset

- **Primary:** `Neph0s/CoSER` (HF dataset). Downloaded file used:
  - `train/sft_conversations_sharegpt.json` (~2.1 GB, 305,134 conversations)
- **Preprocessed:** `Onsite/sft/assets/sft_chai_aligned.json` (~198 MB, 30,000 conversations after subset + cleanup) — produced deterministically with `seed=42` from the source file via `preprocess.py`.
- **License:** CoSER is released by the Neph0s authors on HuggingFace; we only used it for the supplied training split as documented in their CoSER paper README.

### 5.4 Software environment (key versions)

| package | version |
|---|---|
| python | 3.11 |
| torch | 2.6.0+cu124 |
| transformers | 5.5.0 |
| trl | 0.18.2 |
| unsloth | 2026.6.8 |
| peft | (resolved via Unsloth) |
| bitsandbytes | ≥ 0.43 |
| xformers | 0.0.29.post3 |
| CUDA driver | 595.71.05 / CUDA 13.2 (forward-compatible with cu124 wheels) |
| GPU | NVIDIA L4 24 GB |

### 5.5 Hardware

Training: single NVIDIA L4 (24 GB). Total GPU time for the submitted SFT model: ~90 seconds (smoke), ~25 minutes per 500-step run. No multi-GPU or distributed setup.

Inference: handled by Chaiverse on their managed vLLM cluster (single L40S per replica based on the deployment logs we observed).

---

## 6. Submission ledger (selected)

Full ledger in `submissions.json` and `submissions.xlsx`. Highlighted rows:

| slug | model_repo | submission_id | gen overrides |
|---|---|---|---|
| qwen3b_coser_smoke ★ | ZheqiWu/Qwen2.5-3B-CoSER-smoke | `zheqiwu-qwen2-5-3b-cose_13112_v1` | (default) |
| qwen3b_coser_smoke_bo16 | ZheqiWu/Qwen2.5-3B-CoSER-smoke | `zheqiwu-qwen2-5-3b-cose_13112_v2` | best_of=16 |
| qwen3b_coser_smoke_coser_guide | ZheqiWu/Qwen2.5-3B-CoSER-smoke | `zheqiwu-qwen2-5-3b-cose_13112_v3` | + system style guide |
| qwen25_3b | Qwen/Qwen2.5-3B-Instruct | `qwen-qwen2-5-3b-instruct_v19` | (default) — no-SFT baseline |
| qwen3_4b | Qwen/Qwen3-4B-Instruct-2507 | `qwen-qwen3-4b-instruct-2507_v10` | (default) |
| qwen3_30b_a3b | Qwen/Qwen3-30B-A3B-Instruct-2507 | `qwen-qwen3-30b-a3b-inst_16638_v3` | (default) — strong upper-bound |
| llama32_3b | meta-llama/Llama-3.2-3B-Instruct | `meta-llama-llama-3-2-3b_30223_v2` | + Llama-3.x headered formatter |
| qwen3_30b_a3b_bo16 | Qwen/Qwen3-30B-A3B-Instruct-2507 | `qwen-qwen3-30b-a3b-inst_16638_v4` | best_of=16 |
| qwen3_30b_a3b_coser_guide | Qwen/Qwen3-30B-A3B-Instruct-2507 | `qwen-qwen3-30b-a3b-inst_16638_v6` | + system style guide |

★ = primary submission.

### 6.1 Known failed deployments

The four `google/gemma-4-E2B-it` submissions (`v1`–`v5`) appear to have failed to deploy on Chai's vLLM stack. The pipeline log shows it tries to FP8-quantize then load via the multimodal `cached_get_processor`, but the FP8 repo lacks `preprocessor_config.json`, so the inference service crashes during startup. These are recorded in `submissions.json`/`SUBMISSIONS.md` but should be excluded from any win-rate analysis.

### 6.2 Resubmitted variants

The four `*_long` variants (`max_output_tokens=128`) were submitted successfully on the first wave but Chaiverse's later validation rejected `max_output_tokens > 80` if resubmitted. The `_max128` v3/v5/v12/v21 submissions ran to completion before the policy tightened; we additionally re-fired them at `max_output_tokens=80` (`v5`/`v7`/`v14`/`v25` respectively) so both regimes are observable.

---

## 7. Limitations and follow-ups

1. **30-step SFT under-trained.** The submitted model only saw 240 examples. Loss fell from 2.44→1.87, but stylistic transfer is partial. The 500-step run (and a planned 1k-step run) would compound the gap. Consider these as upgrade slots once training finishes.
2. **No formal eval.** We only have Chaiverse's win-rate signal, which arrives ~90 min after each submission. Within onsite we did not have the budget to run, e.g., CoSER's GCA evaluator on the SFT model.
3. **Data is CoSER-only.** Mixing in a small amount of general instruction-following data (10–20 %) is recommended to prevent catastrophic forgetting on Chai's broader user prompts. We did not have time to construct that mix.
4. **Gen-param × SFT cross-product not run.** Once a strong SFT model lands, the most valuable next move is to re-fire the Dim B/C portfolio on top of it.

---

## 8. Pointers

- **HF model:** https://huggingface.co/ZheqiWu/Qwen2.5-3B-CoSER-smoke
- **HF dataset (source):** https://huggingface.co/datasets/Neph0s/CoSER
- **Code root:** `/local/home/zheqi/workspace/RolePlayAgent/Onsite/`
- **Primary submission console:** https://console.chaiverse.com/models/zheqiwu-qwen2-5-3b-cose_13112_v1
