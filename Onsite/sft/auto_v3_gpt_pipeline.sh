#!/usr/bin/env bash
# Stage-3 pipeline: wait for the GPT-OSS dataset to land, then wait for
# the v2 SFT (Sonnet data) to release the GPU, then train a *second* Llama-3.2-3B
# SFT on the GPT corpus and push it to HF + Chaiverse.
set -euo pipefail

cd "$(dirname "$0")/.."
HERE="$(pwd)"
LOG="$HERE/sft/auto_v3_gpt_pipeline.log"

HF_TOKEN="${HF_TOKEN:-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}"
CHAI_DEVELOPER_KEY="${CHAI_DEVELOPER_KEY:-CR_5a47df3092764a8d9dfae7da900291ea}"
export HF_TOKEN CHAI_DEVELOPER_KEY

echo "$(date) === v3 GPT pipeline starting ===" | tee -a "$LOG"

# 1. Wait for GPT-OSS dataset
GPT_DATA="$HERE/dpo/assets/sft_chai_anime_v2_gpt.json"
echo "$(date) waiting for $GPT_DATA ..." | tee -a "$LOG"
while [ ! -s "$GPT_DATA" ]; do
  sleep 30
done
GPT_SIZE=$(wc -c < "$GPT_DATA")
echo "$(date) GPT dataset ready ($GPT_SIZE bytes)" | tee -a "$LOG"

# Defensive: ensure JSON is well-formed and has >=500 records
python3 - <<PY
import json, sys
data = json.load(open("$GPT_DATA"))
n = len(data)
print(f"records: {n}")
if n < 500:
    print("WARN: <500 records — continuing anyway", file=sys.stderr)
PY

# 2. Wait for the v2 SFT to release the GPU (final merge writes
#    trained_model/llama32-3b-anime-v2-merged/ — that's the signal it finished)
V2_FINAL="$HERE/trained_model/llama32-3b-anime-v2-merged"
echo "$(date) waiting for v2 SFT to release GPU (signal: $V2_FINAL) ..." | tee -a "$LOG"
while [ ! -d "$V2_FINAL" ]; do
  sleep 60
done
sleep 30  # let any final cleanup finish
echo "$(date) v2 SFT done — GPU should be free" | tee -a "$LOG"

# 3. Stage GPT data into the SFT assets dir under the canonical name our
#    train_llama_v2.py reads, then point at it.
GPT_DATA_LOCAL="$HERE/sft/assets/sft_chai_anime_v3_gpt.json"
cp "$GPT_DATA" "$GPT_DATA_LOCAL"

# 4. Patch train_llama_v2.py at runtime to swap data path + output names.
cd "$HERE/sft"
source .venv/bin/activate

python3 - <<'PY'
import re
p = 'train_llama_v2.py'
content = open(p).read()
content = re.sub(r'DATA = HERE / "[^"]*"', 'DATA = HERE / "assets/sft_chai_anime_v3_gpt.json"', content)
content = re.sub(r'OUT_LORA = HERE / "[^"]*"', 'OUT_LORA = HERE / "assets/llama32-3b-anime-v3-gpt-lora"', content)
content = re.sub(r'OUT_MERGED = TRAINED_DIR / "[^"]*"', 'OUT_MERGED = TRAINED_DIR / "llama32-3b-anime-v3-gpt-merged"', content)
content = content.replace('llama32-3b-anime-v2-step{step}', 'llama32-3b-anime-v3-gpt-step{step}')
open(p, 'w').write(content)
print('train_llama_v2.py patched for v3 (GPT data)')
PY

echo "$(date) === training Llama-3.2-3B + v3 (GPT-OSS data) ===" | tee -a "$LOG"
python train_llama_v2.py 2>&1 | tee train_llama_v3_gpt.log

# Restore train_llama_v2.py to pointing at v2 (in case anyone re-runs)
python3 - <<'PY'
import re
p = 'train_llama_v2.py'
content = open(p).read()
content = re.sub(r'DATA = HERE / "[^"]*"', 'DATA = HERE / "assets/sft_chai_anime_v2.json"', content)
content = re.sub(r'OUT_LORA = HERE / "[^"]*"', 'OUT_LORA = HERE / "assets/llama32-3b-anime-v2-lora"', content)
content = re.sub(r'OUT_MERGED = TRAINED_DIR / "[^"]*"', 'OUT_MERGED = TRAINED_DIR / "llama32-3b-anime-v2-merged"', content)
content = content.replace('llama32-3b-anime-v3-gpt-step{step}', 'llama32-3b-anime-v2-step{step}')
open(p, 'w').write(content)
print('train_llama_v2.py restored to v2')
PY

# 5. Upload 3 checkpoints to HF
echo "$(date) === uploading v3 GPT checkpoints to HF ===" | tee -a "$LOG"
for step in 100 200 300; do
  REPO="ZheqiWu/Llama-3.2-3B-Anime-v3-gpt-step${step}"
  DIR="../trained_model/llama32-3b-anime-v3-gpt-step${step}"
  if [ -d "$DIR" ]; then
    echo "$(date) uploading $REPO" | tee -a "$LOG"
    HF_TOKEN="$HF_TOKEN" hf upload "$REPO" "$DIR" 2>&1 | tail -3 | tee -a "$LOG"
  else
    echo "$(date) WARN: $DIR not found, skipping" | tee -a "$LOG"
  fi
done

# 6. Add v3 variants to submit_batch.py and fire to Chai
cd "$HERE"
python3 <<'PY'
import re
p = 'submit_batch.py'
content = open(p).read()
new_variants = '''
    # Llama-3.2-3B + Anime-v3-GPT SFT (1000 GPT-OSS-120B-synth onsite-schema records,
    # weight masking on w=0 turns, LR 5e-5). Mirrors the v2-Sonnet pipeline so we can
    # A/B Sonnet-flavoured vs GPT-flavoured synthetic data on Chai win-rate.
    ("llama32_3b_anime_v3_gpt_step100",
        "ZheqiWu/Llama-3.2-3B-Anime-v3-gpt-step100", LLAMA31_HEADERED, {}),
    ("llama32_3b_anime_v3_gpt_step200",
        "ZheqiWu/Llama-3.2-3B-Anime-v3-gpt-step200", LLAMA31_HEADERED, {}),
    ("llama32_3b_anime_v3_gpt_step300",
        "ZheqiWu/Llama-3.2-3B-Anime-v3-gpt-step300", LLAMA31_HEADERED, {}),
'''
if 'llama32_3b_anime_v3_gpt_step100' not in content:
    if not re.search(r'(\n\]\n)', content):
        raise SystemExit('cannot find VARIANTS terminator in submit_batch.py')
    content = re.sub(r'(\n\]\n)', new_variants + r'\1', content, count=1)
    open(p, 'w').write(content)
    print('added 3 v3-gpt variants')
else:
    print('v3-gpt variants already present')
PY

echo "$(date) === submitting v3 GPT to Chai ===" | tee -a "$LOG"
uv run --with requests python submit_batch.py --only \
    llama32_3b_anime_v3_gpt_step100 \
    llama32_3b_anime_v3_gpt_step200 \
    llama32_3b_anime_v3_gpt_step300 2>&1 | tee -a "$LOG"

echo "$(date) === v3 GPT pipeline complete ===" | tee -a "$LOG"
