#!/usr/bin/env bash
# After PIPPA + synth LoRAs are trained, run the ensemble + push + submit pipeline.
# Pre-req: assets/qwen3-4b-anime-lora/step60, qwen3-4b-pippa-lora/step60,
#          qwen3-4b-synth-lora/step60 all exist.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

BASE=assets/Qwen3-4B-Instruct-2507
ANIME=assets/qwen3-4b-anime-lora/step60
PIPPA=assets/qwen3-4b-pippa-lora/step60
SYNTH=assets/qwen3-4b-synth-lora/step60

# 1) Linear merge with anime-heavy weighting (anime is the proven winner at 36.4%)
python ensemble_loras.py \
    --base "$BASE" \
    --adapter anime="$ANIME" \
    --adapter pippa="$PIPPA" \
    --adapter synth="$SYNTH" \
    --weights 0.5,0.25,0.25 \
    --combination linear \
    --out ../train_model/qwen3-4b-ensemble-linear-50_25_25

# 2) TIES merge — drops low-magnitude params per adapter, sign-vote then merge
python ensemble_loras.py \
    --base "$BASE" \
    --adapter anime="$ANIME" \
    --adapter pippa="$PIPPA" \
    --adapter synth="$SYNTH" \
    --weights 0.5,0.25,0.25 \
    --combination ties \
    --density 0.7 \
    --out ../train_model/qwen3-4b-ensemble-ties-50_25_25

# 3) Equal linear blend baseline for comparison
python ensemble_loras.py \
    --base "$BASE" \
    --adapter anime="$ANIME" \
    --adapter pippa="$PIPPA" \
    --adapter synth="$SYNTH" \
    --weights 0.34,0.33,0.33 \
    --combination linear \
    --out ../train_model/qwen3-4b-ensemble-linear-equal

echo "done; merged ensembles in ../train_model/"
ls -lah ../train_model/ | grep ensemble
