#!/usr/bin/env bash
# Resume v2 pipeline from the SFT step (skip the workflow wait + aggregate
# phases — the dataset is already at assets/sft_chai_anime_v2.json).
set -euo pipefail

cd "$(dirname "$0")/.."
HERE="$(pwd)"
LOG="$HERE/sft/auto_v2_pipeline.log"

HF_TOKEN="${HF_TOKEN:-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}"
CHAI_DEVELOPER_KEY="${CHAI_DEVELOPER_KEY:-CR_5a47df3092764a8d9dfae7da900291ea}"
export HF_TOKEN CHAI_DEVELOPER_KEY

echo "$(date) === resume v2 pipeline at SFT step ===" | tee -a "$LOG"

# 3. Train Llama-3.2-3B on v2 data using train_llama_v2.py
echo "$(date) === training Llama-3.2-3B + v2 ===" | tee -a "$LOG"
cd "$HERE/sft"
source .venv/bin/activate
python train_llama_v2.py 2>&1 | tee train_llama_v2.log

# 4. Upload 3 checkpoints to HF
echo "$(date) === uploading to HF ===" | tee -a "$LOG"
cd "$HERE/sft"
for step in 100 200 300; do
  REPO="ZheqiWu/Llama-3.2-3B-Anime-v2-step${step}"
  DIR="../trained_model/llama32-3b-anime-v2-step${step}"
  if [ -d "$DIR" ]; then
    echo "$(date) uploading $REPO" | tee -a "$LOG"
    HF_TOKEN="$HF_TOKEN" hf upload "$REPO" "$DIR" 2>&1 | tail -3 | tee -a "$LOG"
  else
    echo "$(date) WARN: $DIR not found, skipping" | tee -a "$LOG"
  fi
done

# 5. Add new variants to submit_batch.py and fire Chai submissions
echo "$(date) === submitting to Chai ===" | tee -a "$LOG"
cd "$HERE"
python3 <<'PY'
import re
p = 'submit_batch.py'
content = open(p).read()
new_variants = '''
    # Llama-3.2-3B + Anime-v2 SFT (1000 Sonnet-synth onsite-schema records,
    # weight masking on w=0 turns, 850/150 hold-out, LR 5e-5).
    ("llama32_3b_anime_v2_step100",
        "ZheqiWu/Llama-3.2-3B-Anime-v2-step100", LLAMA31_HEADERED, {}),
    ("llama32_3b_anime_v2_step200",
        "ZheqiWu/Llama-3.2-3B-Anime-v2-step200", LLAMA31_HEADERED, {}),
    ("llama32_3b_anime_v2_step300",
        "ZheqiWu/Llama-3.2-3B-Anime-v2-step300", LLAMA31_HEADERED, {}),
'''
if 'llama32_3b_anime_v2_step100' not in content:
    if not re.search(r'(\n\]\n)', content):
        raise SystemExit('cannot find VARIANTS terminator in submit_batch.py')
    content = re.sub(r'(\n\]\n)', new_variants + r'\1', content, count=1)
    open(p, 'w').write(content)
    print('added 3 v2 variants')
else:
    print('v2 variants already present')
PY

uv run --with requests python submit_batch.py --only llama32_3b_anime_v2_step100 llama32_3b_anime_v2_step200 llama32_3b_anime_v2_step300 2>&1 | tee -a "$LOG"

echo "$(date) === pipeline complete ===" | tee -a "$LOG"
