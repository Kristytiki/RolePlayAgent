#!/usr/bin/env bash
# Train PIPPA + synth LoRAs on Qwen3-4B base with the SAME r/alpha as the
# existing anime LoRA (r=32, alpha=64) so we can ensemble all three in weight
# space afterwards.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

COMMON_ARGS=(
  BASE_MODEL=qwen3-4b
  LORA_R=32
  LORA_ALPHA=64
  LR=1e-4
  MAX_STEPS=60
  SNAPSHOT_AT=60
)

echo "[1/2] training Qwen3-4B + PIPPA (r=32, lr=1e-4, 60 steps)..."
env DATASET=pippa "${COMMON_ARGS[@]}" python train.py 2>&1 | tee train_qwen3_4b_pippa.log

echo "[2/2] training Qwen3-4B + synth (r=32, lr=1e-4, 60 steps)..."
env DATASET=synth "${COMMON_ARGS[@]}" python train.py 2>&1 | tee train_qwen3_4b_synth.log

echo "done. LoRAs at:"
ls assets/qwen3-4b-pippa-lora/
ls assets/qwen3-4b-synth-lora/
