# Trained Models — local checkpoints

Each subdirectory is a merged 16-bit fine-tuned model produced by
`Onsite/sft/train.py` or its variants (`train_llama.py`,
`gen_synthetic.py + train.py`). The actual `model-*.safetensors`
weight shards (~5–8 GB each) are **gitignored**; only `config.json`,
`tokenizer_config.json`, `chat_template.jinja`, the safetensors index,
and `TRAINING_PARAMS.md` (where present) are committed so the recipe
is reproducible.

The actual weights are pushed to HuggingFace under
[`ZheqiWu`](https://huggingface.co/ZheqiWu) — see the per-model HF
links below.

## Top-3 best fine-tuned models (by Chaiverse win-rate)

| Rank | Local dir | HF repo | Best win-rate (battles) |
|---|---|---|---|
| 🥇 | `qwen3-4b-anime-step60` | [ZheqiWu/Qwen3-4B-Anime-step60](https://huggingface.co/ZheqiWu/Qwen3-4B-Anime-step60) | **36.4%** (1,730) — `step60_v5` with `best_of=64` |
| 🥈 | `qwen3-4b-anime-step100` | [ZheqiWu/Qwen3-4B-Anime-step100](https://huggingface.co/ZheqiWu/Qwen3-4B-Anime-step100) | 36.2% (1,750) — `step100_v5` long-output |
| 🥉 | `qwen3-4b-pippa-step60` | [ZheqiWu/Qwen3-4B-PIPPA-step60](https://huggingface.co/ZheqiWu/Qwen3-4B-PIPPA-step60) | TBD — newly fired |

## Other notable checkpoints

| Local dir | HF repo | Notes |
|---|---|---|
| `qwen3-4b-anime-step30` | [HF](https://huggingface.co/ZheqiWu/Qwen3-4B-Anime-step30) | early-stop comparison |
| `qwen3-4b-anime-merged` | — | final merged state of anime run (== step90+) |
| `qwen25-3b-synth-step60` | [HF](https://huggingface.co/ZheqiWu/Qwen2.5-3B-synth-step60) | best Qwen2.5-3B SFT (Opus 1k synth) |
| `qwen25-3b-pippa-step200` | [HF](https://huggingface.co/ZheqiWu/Qwen2.5-3B-PIPPA-r32) | PIPPA SFT on smaller base |
| `qwen25-3b-coser-step500` | [HF](https://huggingface.co/ZheqiWu/Qwen2.5-3B-CoSER-step500) | CoSER lessons-learned baseline (under 30%) |
| `llama32-3b-anime-step100` | [HF](https://huggingface.co/ZheqiWu/Llama-3.2-3B-Anime-step100) | Llama base for cross-family comparison |

## How to reproduce a checkpoint locally

```bash
cd Onsite/sft && source .venv/bin/activate

# 1. download base + dataset
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir assets/Qwen3-4B-Instruct-2507
# anime v1 dataset already committed at assets/sft_chai_anime.json

# 2. train (winning recipe — step60 LoRA with r=32)
DATASET=anime BASE_MODEL=qwen3-4b LORA_R=32 LORA_ALPHA=64 LR=1e-4 \
    MAX_STEPS=60 SNAPSHOT_AT=60 \
    python train.py
```

The merged 16-bit checkpoint will be written to
`Onsite/train_model/qwen3-4b-anime-step60/`. The corresponding LoRA adapter
(for ensembling) is written to
`Onsite/sft/assets/qwen3-4b-anime-lora/step60/`.

## Submission ledger

Per-checkpoint Chaiverse `submission_id`, `win_ratio`, full generation
parameters, and formatter strings are recorded in
`Onsite/submissions.json` (machine-readable) and
`Onsite/submissions.xlsx` (color-coded, sorted by Dim and rank).
