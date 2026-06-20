#!/usr/bin/env bash
# Build 3 LoRA ensembles on Qwen3-4B (anime + PIPPA + synth adapters).
# Pre-req: all three adapters trained at the same r/alpha and saved under
# Onsite/sft/assets/qwen3-4b-{anime,pippa,synth}-lora/step60/.
set -euo pipefail

# repo root (3 levels up from this file)
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SFT="$ROOT/sft"
cd "$SFT"
source .venv/bin/activate

BASE=assets/Qwen3-4B-Instruct-2507
ANIME=assets/qwen3-4b-anime-lora/step60
PIPPA=assets/qwen3-4b-pippa-lora/step60
SYNTH=assets/qwen3-4b-synth-lora/step60

for d in "$ANIME" "$PIPPA" "$SYNTH"; do
  [ -f "$d/adapter_config.json" ] || { echo "missing $d"; exit 1; }
done

build() {
  local tag="$1" weights="$2" combo="$3" extra="${4:-}"
  python core/ensemble_loras.py \
    --base "$BASE" \
    --adapter anime="$ANIME" --adapter pippa="$PIPPA" --adapter synth="$SYNTH" \
    --weights "$weights" --combination "$combo" $extra \
    --out "../train_model/qwen3-4b-ens-${tag}"
}

build "linear-animeheavy" 0.5,0.25,0.25 linear
build "ties-animeheavy"   0.5,0.25,0.25 ties "--density 0.7"
build "linear-equal"      0.34,0.33,0.33 linear

# Push to HF
for tag in linear-animeheavy ties-animeheavy linear-equal; do
  hf repos create "ZheqiWu/Qwen3-4B-ensemble-${tag}" --type model 2>/dev/null || true
  hf upload "ZheqiWu/Qwen3-4B-ensemble-${tag}" "../train_model/qwen3-4b-ens-${tag}" . | tail -2
done
