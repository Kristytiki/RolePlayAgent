# SFT pipeline

LoRA fine-tuning + LoRA ensembling on Qwen2.5-3B / Qwen3-4B / Llama-3.2-3B for the Chaiverse leaderboard.

## Layout

```
sft/
├── core/                       # generic tools, called by every run
│   ├── setup_env.sh            # uv venv + torch cu124 + Unsloth (pinned)
│   ├── download.sh             # pull base models + datasets from HF
│   ├── train.py                # Unsloth LoRA SFT — env-driven
│   └── ensemble_loras.py       # peft.add_weighted_adapter wrapper
├── data/                       # dataset construction
│   ├── SCHEMA.md               # data schema design doc
│   ├── preprocess_coser.py     # CoSER → Chai-aligned ShareGPT
│   ├── preprocess_pippa.py     # PIPPA → ShareGPT (category + casual filter)
│   ├── preprocess_hieu.py      # Hieunguyenminh → ShareGPT
│   └── gen_synthetic.py        # Bedrock Opus 4.8 schema-controlled synthesis
├── runs/
│   └── ensemble/run.sh         # build 3 ensembles + push HF
└── assets/                     # gitignored — base models, datasets, LoRA adapters
```

The 7 datasets we trained against are documented in `data/SCHEMA.md` and the project root `Onsite/WRITEUP.md` §6.

## Setup (one-time)

```bash
cd Onsite/sft
bash core/setup_env.sh
source .venv/bin/activate
HF_TOKEN=hf_... bash core/download.sh   # pulls Qwen + CoSER
```

## Run a fine-tune

`core/train.py` is fully env-driven:

| Env var | Default | Notes |
|---|---|---|
| `DATASET` | `coser` | one of `coser`, `pippa`, `hieu`, `synth`, `anime` |
| `BASE_MODEL` | `qwen25-3b` | or `qwen3-4b` |
| `LORA_R` | `32` | rank |
| `LORA_ALPHA` | `2*r` | scale |
| `LR` | `2e-4` | use `1e-4` for the lighter recipe that landed best |
| `MAX_STEPS` | `500` | |
| `SNAPSHOT_AT` | `200,500` | comma list of steps to save merged 16-bit |
| `SAVE_LORA_EVERY` | `100` | step interval for LoRA-adapter snapshots |
| `RUN_TAG` | empty | suffix appended to output dirs to avoid collisions |

Reproducing the **best individual SFT** (Qwen3-4B + anime-v1, 36.4% Chaiverse win-rate):

```bash
cd Onsite/sft && source .venv/bin/activate

DATASET=anime BASE_MODEL=qwen3-4b LORA_R=32 LORA_ALPHA=64 LR=1e-4 \
  MAX_STEPS=60 SNAPSHOT_AT=30,60 SAVE_LORA_EVERY=30 \
  python core/train.py
```

Outputs:
- LoRA adapter: `assets/qwen3-4b-anime-lora/step{30,60}/`
- Merged 16-bit: `../train_model/qwen3-4b-anime-step{30,60}/`

## Synthesize new training data

```bash
AWS_PROFILE=bedrock python data/gen_synthetic.py \
  --n 1000 --concurrency 16 \
  --out assets/sft_chai_synth.json
```

Schema-controlled (single line, *action* only, no [thought], ≤40 tokens, 30% emoji). See `data/SCHEMA.md` for the full design rationale.

## LoRA ensemble

Once two or more LoRA adapters with **the same base / r / alpha / target_modules** exist, weight-merge them into a single deployable checkpoint:

```bash
cd Onsite/sft && source .venv/bin/activate

python core/ensemble_loras.py \
  --base assets/Qwen3-4B-Instruct-2507 \
  --adapter anime=assets/qwen3-4b-anime-lora/step60 \
  --adapter pippa=assets/qwen3-4b-pippa-lora/step60 \
  --adapter synth=assets/qwen3-4b-synth-lora/step60 \
  --weights 0.5,0.25,0.25 \
  --combination linear \
  --out ../train_model/qwen3-4b-ens-linear-animeheavy
```

Combination strategies supported: `linear`, `ties`, `dare_linear`, `dare_ties`, `magnitude_prune`. The merged 16-bit checkpoint is ready to push to HuggingFace and submit via `Onsite/submit_batch.py`.

`runs/ensemble/run.sh` builds the three ensembles we shipped (linear-animeheavy, ties-animeheavy, linear-equal) and pushes them to HF.

## Submit to Chaiverse

```bash
cd Onsite
HF_TOKEN=... CHAI_DEVELOPER_KEY=CR_... \
  uv run --with requests python submit_batch.py --only qwen3_4b_anime_step60_bo64
```

`submit_batch.py` keeps the canonical variant list and writes every submission to `submissions.json` + `submissions.xlsx`.
