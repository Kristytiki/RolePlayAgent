#!/usr/bin/env bash
# One-shot env setup for Unsloth + LoRA training on a CUDA box.
# Assumes nvidia-smi works (driver + CUDA already installed).
set -euo pipefail

# Always operate from sft/ (one level up from this script) so the venv lives
# at sft/.venv, matching what every train/download invocation expects.
cd "$(dirname "$0")/.."

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi missing — install NVIDIA driver first."
  exit 1
fi
nvidia-smi | head -20

# uv-managed venv to keep system python clean.
uv venv .venv --python 3.11
# shellcheck disable=SC1091
source .venv/bin/activate

# Torch + xformers built for cu124 — what we trained the released checkpoint with.
uv pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
uv pip install --index-url https://download.pytorch.org/whl/cu124 xformers==0.0.29.post3

# Unsloth pinned to the exact commit used for this run. Unsloth main is fast-moving;
# floating HEAD has produced API drift mid-week before. trl<0.19 needed for SFTConfig
# compatibility with the transformers stack below.
uv pip install \
  "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git@7ce8dc73ac52cf7a3e87501cce7546c30b66e010" \
  "trl>=0.18.2,<0.19.0" "transformers>=4.44" "datasets>=2.20" \
  "peft>=0.12" "accelerate>=0.33" "bitsandbytes>=0.43" \
  "torchao<0.10" setuptools \
  "huggingface_hub[cli]" sentencepiece protobuf \
  "anthropic>=0.40" "pydantic>=2"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "n", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

echo "env ready. activate with: source .venv/bin/activate"
