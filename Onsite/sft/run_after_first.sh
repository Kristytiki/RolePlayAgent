#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

echo "[watchdog] waiting for PID $1 to finish..."
while kill -0 "$1" 2>/dev/null; do
  sleep 30
done
echo "[watchdog] PID $1 done. starting lighter PIPPA run."

DATASET=pippa MAX_STEPS=200 SNAPSHOT_AT=200 RUN_TAG=light LORA_R=16 LORA_ALPHA=32 LR=1e-4 \
  python train.py
