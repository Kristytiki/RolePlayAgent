#!/usr/bin/env bash
# Download base model + CoSER SFT data into ./assets/.
# Run from inside the activated venv created by setup_env.sh, so the `hf` CLI is available.
# Needs HF_TOKEN env (read access fine; write needed later for push).
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p assets

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: hf CLI not on PATH. Activate the venv first: source .venv/bin/activate"
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN not set. Public CoSER data + Qwen base should still pull, but push later will fail."
fi

# Base model (~6GB safetensors)
hf download Qwen/Qwen2.5-3B-Instruct \
  --local-dir assets/Qwen2.5-3B-Instruct

# CoSER SFT data (single file, ShareGPT format)
hf download Neph0s/CoSER \
  --repo-type dataset \
  --include "train/sft_conversations_sharegpt.json" \
  --local-dir assets/CoSER

echo "Done. Model: assets/Qwen2.5-3B-Instruct/  Data: assets/CoSER/train/sft_conversations_sharegpt.json"
