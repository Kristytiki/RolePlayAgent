# Onsite Chaiverse Plan

Goal: maximize Chai win-rate within onsite budget (~3-4h).
Single signal: **win-rate after 90min eval per submission**. Submissions run in parallel server-side, so fire-and-wait > sequential.

## Strategy: 4 dimensions × parallel fire

| Dim | What | Cost | Variants |
|---|---|---|---|
| **A. Base model (≤3B/4B)** | which pretrained LLM | 0 train | Qwen2.5-{0.5,1.5,3}B-Instruct, Llama-3.2-{1,3}B-Instruct, gemma-4-E2B-it, gemma-3-1b-it, SmolLM2-1.7B-Instruct |
| **B. Formatter** | chat template + system injection | 0 train | per-family default; +CoSER thought/action guide |
| **C. Generation params** | sampling + best_of + max_output | 0 train | conservative / current / best_of=16 / long-output / freq-penalty |
| **D. SFT** | LoRA on CoSER ShareGPT | GPU train | Qwen2.5-1.5B + 5k LoRA / Qwen2.5-3B + 5k / Qwen2.5-3B + 30k |

Stacking after Dim D wins: re-submit best SFT model × best Dim B/C combo.

## Critical pitfalls

1. **Per-family chat template**: Qwen=ChatML, Llama-3.x=`<|start_header_id|>`, Gemma=`<start_of_turn>...<end_of_turn>`. Wrong template → garbage output → win-rate floor.
2. **`stopping_words=['\n']`** truncates at first newline. Training assistant targets must therefore be **single-line** (strip leading `Character:` prefix; collapse `\n\n` to spaces).
3. **`max_output_tokens=64`** is short → roleplay style needs to fit `[thought] (action) speech` inline.
4. **Only Chaiverse `vllm` platform**: rules out GGUF community quants. AWQ/GPTQ untested with Chai.

## CoSER context

- Official models: 8B / 70B Llama-3.1 fine-tunes (no ≤3B official version).
- Repo provides **data only** for training; recommends LLaMA-Factory.
- Data format: ShareGPT, system = char profile + scenario, assistant = `"Name: [thought] (action) speech\n\n"`. Strip `Name: ` and `\n\n` before training.

## Timeline (rough, parallel)

```
T+0       fire submit_batch.py (12 training-free variants)
T+0..30   start Qwen2.5-1.5B + 5k LoRA (Unsloth, ~10-20min on L4)
T+30      push 1.5B SFT → submit
T+30..60  start Qwen2.5-3B + 5k LoRA (~20-30min)
T+60      push 3B-5k SFT → submit
T+60..150 start Qwen2.5-3B + 30k LoRA (~1.5h, big bet)
T+90      first batch results land → identify best base/B/C
T+150     push 3B-30k SFT → submit (default B/C + best B/C × 2-3 stacked)
T+150..240 final stacked submissions
```

## Variants to fire (preview)

Total **16** training-free variants. Submit them in one shot, results land ~90min later.

### Dim A — base model scan (all default formatter for the family, default gen params)

| # | slug | model_repo | size | formatter | notes |
|---|---|---|---|---|---|
| 1 | `qwen25_3b` | `Qwen/Qwen2.5-3B-Instruct` | 3B | ChatML | baseline (= already-submitted v18) |
| 2 | `qwen25_1p5b` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ChatML | smaller Qwen2.5 |
| 3 | `qwen25_0p5b` | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | ChatML | tiny |
| 4 | `qwen3_1p7b` | `Qwen/Qwen3-1.7B` | 1.7B | ChatML | **Qwen3 dense, 2025-07** |
| 5 | `qwen3_4b` | `Qwen/Qwen3-4B-Instruct-2507` | 4B | ChatML | **Qwen3 latest non-thinking, ~Sep 2025** ⭐ |
| 6 | `qwen3_30b_a3b` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 30B/3B-active MoE | ChatML | **risk play** — biggest model, likely strongest if Chai accepts ⭐⭐ |
| 7 | `llama32_3b` | `meta-llama/Llama-3.2-3B-Instruct` | 3B | Llama-3.x headered | |
| 8 | `llama32_1b` | `meta-llama/Llama-3.2-1B-Instruct` | 1B | Llama-3.x headered | |
| 9 | `gemma4_e2b` | `google/gemma-4-E2B-it` | 2B | Gemma turns | newest Gemma 4 (2026-06) |
| 10 | `gemma3_1b` | `google/gemma-3-1b-it` | 1B | Gemma turns | |
| 11 | `smollm2_1p7b` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | ChatML-ish | |

### Dim C — gen-param sweep (all on `Qwen/Qwen2.5-3B-Instruct` + ChatML)

| # | slug | overrides | hypothesis |
|---|---|---|---|
| 12 | `qwen25_3b_bo16` | `best_of=16` | reranker picks better of 16 candidates |
| 13 | `qwen25_3b_long` | `max_output_tokens=128`, `stopping_words=[]` | users prefer richer descriptions |
| 14 | `qwen25_3b_freqpen` | `frequency_penalty=0.3` | reduce repetition |
| 15 | `qwen25_3b_conservative` | `temperature=0.8`, `top_p=0.9` | safer, less off-rails |

### Dim B — formatter tweak (on `Qwen/Qwen2.5-3B-Instruct`)

| # | slug | tweak | hypothesis |
|---|---|---|---|
| 16 | `qwen25_3b_coser_guide` | system_template appends: *"Use [your thought] for thoughts which others can't see. Use (your action) for actions which others can see."* | CoSER-style format guidance steers model toward roleplay output |

### Default gen params (used unless overridden)

```json
{
  "temperature": 1.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 40,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stopping_words": ["\n"],
  "max_input_tokens": 2048,
  "best_of": 8,
  "max_output_tokens": 64
}
```

### Per-family formatters (verbatim)

**ChatML (Qwen, SmolLM2)**
```
memory_template:   <|im_start|>system\n{memory}<|im_end|>\n
prompt_template:   <|im_start|>user\n{prompt}<|im_end|>\n
bot_template:      <|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n
user_template:     <|im_start|>user\n{user_name}: {message}<|im_end|>\n
response_template: <|im_start|>assistant\n{bot_name}:
```

**Llama 3.x headered**
```
memory_template:   <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{memory}<|eot_id|>
prompt_template:   <|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>
bot_template:      <|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}: {message}<|eot_id|>
user_template:     <|start_header_id|>user<|end_header_id|>\n\n{user_name}: {message}<|eot_id|>
response_template: <|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}:
```

**Gemma turns** (no system role; memory delivered as leading user turn)
```
memory_template:   <start_of_turn>user\n{memory}<end_of_turn>\n
prompt_template:   <start_of_turn>user\n{prompt}<end_of_turn>\n
bot_template:      <start_of_turn>model\n{bot_name}: {message}<end_of_turn>\n
user_template:     <start_of_turn>user\n{user_name}: {message}<end_of_turn>\n
response_template: <start_of_turn>model\n{bot_name}:
```

## Files

- `Onsite/onsite_submit.py` — single-shot submit (legacy)
- `Onsite/submit_batch.py` — portfolio submitter (Dim A/B/C); appends to `submissions.json` + `SUBMISSIONS.md`
- `Onsite/SUBMISSIONS.md` — auto-generated log: slug, submission_id, URL, full gen_params + formatter for each fire
- `Onsite/sft/core/download.sh` — pull Qwen2.5-3B-Instruct + CoSER ShareGPT
- `Onsite/sft/data/preprocess_coser.py` — strip leading `Name: ` + collapse newlines
- `Onsite/sft/core/setup_env.sh` — uv venv + torch + Unsloth
- `Onsite/sft/core/train.py` — Unsloth LoRA r=32, 1 epoch, save merged 16-bit

## Run order

```bash
# 1. fire training-free portfolio NOW (don't wait for GPU)
cd Onsite
HF_TOKEN=hf_xxx python submit_batch.py --dry        # preview
HF_TOKEN=hf_xxx python submit_batch.py              # actually fire
# OR a subset:
HF_TOKEN=hf_xxx python submit_batch.py --only qwen25_3b llama32_3b gemma4_e2b

# 2. once nvidia-smi works, prep SFT
cd sft
HF_TOKEN=hf_xxx bash download.sh
N_KEEP=5000 python preprocess.py     # tiny first
bash setup_env.sh
source .venv/bin/activate
python train.py

# 3. push merged model to your HF account
huggingface-cli upload <your_user>/Qwen2.5-3B-CoSER-5k assets/qwen25-3b-coser-merged

# 4. submit fine-tuned model — copy submit_batch.py entry, swap model_repo
```

## Win-rate bookkeeping

Fill in `SUBMISSIONS.md` win_rate column after each Chaiverse eval (~90min).
Compare slugs to identify which dim's lever moved the needle, then stack.
