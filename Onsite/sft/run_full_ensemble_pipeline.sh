#!/usr/bin/env bash
# After both trainings done, build ensembles + push + submit.
# Self-contained background runner.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

# Wait for training PID (passed as $1)
TRAIN_PID="${1:-}"
if [ -n "$TRAIN_PID" ]; then
  echo "[watchdog] waiting for training PID $TRAIN_PID"
  while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 30; done
  echo "[watchdog] training done"
fi

# Verify both adapters exist
ANIME=assets/qwen3-4b-anime-lora/step60
PIPPA=assets/qwen3-4b-pippa-lora/step60
SYNTH=assets/qwen3-4b-synth-lora/step60
for d in "$ANIME" "$PIPPA" "$SYNTH"; do
  if [ ! -f "$d/adapter_config.json" ]; then
    echo "ERROR: $d/adapter_config.json missing — abort"
    exit 1
  fi
done

# Build 3 ensembles
echo "=== building ensemble: linear 0.5/0.25/0.25 anime-heavy ==="
python ensemble_loras.py \
    --base assets/Qwen3-4B-Instruct-2507 \
    --adapter anime="$ANIME" --adapter pippa="$PIPPA" --adapter synth="$SYNTH" \
    --weights 0.5,0.25,0.25 --combination linear \
    --out ../train_model/qwen3-4b-ens-linear-anime-heavy

echo "=== building ensemble: ties 0.5/0.25/0.25 ==="
python ensemble_loras.py \
    --base assets/Qwen3-4B-Instruct-2507 \
    --adapter anime="$ANIME" --adapter pippa="$PIPPA" --adapter synth="$SYNTH" \
    --weights 0.5,0.25,0.25 --combination ties --density 0.7 \
    --out ../train_model/qwen3-4b-ens-ties-anime-heavy

echo "=== building ensemble: linear equal ==="
python ensemble_loras.py \
    --base assets/Qwen3-4B-Instruct-2507 \
    --adapter anime="$ANIME" --adapter pippa="$PIPPA" --adapter synth="$SYNTH" \
    --weights 0.34,0.33,0.33 --combination linear \
    --out ../train_model/qwen3-4b-ens-linear-equal

# Push all three to HF
echo "=== pushing 3 ensembles to HF ==="
for tag in linear-anime-heavy ties-anime-heavy linear-equal; do
  repo="ZheqiWu/Qwen3-4B-ensemble-${tag}"
  hf repos create "$repo" --type model 2>&1 | tail -1 || true
  hf upload "$repo" "../train_model/qwen3-4b-ens-${tag}" . 2>&1 | tail -2
done

echo "=== ensemble pipeline complete ==="
ls ../train_model/ | grep ens-
