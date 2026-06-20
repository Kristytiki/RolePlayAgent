#!/usr/bin/env bash
# v4 pipeline: wait for老prompt×1000 workflow, build dataset, train Qwen3-4B,
# upload, submit Chai. (Mirror of v3 GPT pipeline but base=Qwen3-4B + old prompt data.)
set -euo pipefail

cd "$(dirname "$0")/.."
HERE="$(pwd)"
LOG="$HERE/sft/auto_v4_qwen_pipeline.log"

WF_TASK=wq2muczoh
WF_OUT="/tmp/claude-22246397/-local-home-zheqi-workspace-RolePlayAgent/c7916a54-fab0-4960-8903-28c841855c4b/tasks/${WF_TASK}.output"

HF_TOKEN="${HF_TOKEN:-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}"
CHAI_DEVELOPER_KEY="${CHAI_DEVELOPER_KEY:-CR_5a47df3092764a8d9dfae7da900291ea}"
export HF_TOKEN CHAI_DEVELOPER_KEY

echo "$(date) === v4 pipeline starting ===" | tee -a "$LOG"

# 1. Wait for workflow output
echo "$(date) waiting for $WF_OUT ..." | tee -a "$LOG"
while [ ! -s "$WF_OUT" ]; do
  sleep 30
done
echo "$(date) workflow output ready ($(wc -c < $WF_OUT) bytes)" | tee -a "$LOG"

# 2. Aggregate to sft_chai_anime_v4.json (ShareGPT format — old prompt)
echo "$(date) === aggregating ===" | tee -a "$LOG"
python3 - <<PY 2>&1 | tee -a "$LOG"
import json
data = json.load(open("$WF_OUT"))
records = data['result']['convos']
print(f'raw records: {len(records)}')

out = []
for r in records:
    convo = [{'from':'system','value':r['system']}]
    for t in r['turns']:
        # strip newlines from assistant turns (safety net for Chai stop=['\n'])
        v = t['value']
        if t['from']=='assistant' and '\n' in v:
            v = v.split('\n',1)[0].strip()
        if v:
            convo.append({'from':t['from'],'value':v})
    out.append({'conversations':convo})

p = '$HERE/sft/assets/sft_chai_anime_v4.json'
json.dump(out, open(p,'w'), ensure_ascii=False)
print(f'wrote {len(out)} records to {p}')
PY

# 3. Train Qwen3-4B on v4 data — reuse train_qwen3.py (it already targets the
#    sft_chai_anime.json file; we patch DATA + output names at runtime).
echo "$(date) === training Qwen3-4B + v4 ===" | tee -a "$LOG"
cd "$HERE/sft"
source .venv/bin/activate

python3 - <<'PY'
import re
p = 'train_qwen3.py'
content = open(p).read()
content = re.sub(r'DATA = HERE / "[^"]*"', 'DATA = HERE / "assets/sft_chai_anime_v4.json"', content)
content = re.sub(r'OUT_LORA = HERE / "[^"]*"', 'OUT_LORA = HERE / "assets/qwen3-4b-anime-v4-lora"', content)
content = re.sub(r'OUT_MERGED = TRAINED_DIR / "[^"]*"', 'OUT_MERGED = TRAINED_DIR / "qwen3-4b-anime-v4-merged"', content)
content = content.replace('qwen3-4b-anime-step{step}', 'qwen3-4b-anime-v4-step{step}')
# Bump steps for 1000 records (was 100 for 100 records).
content = re.sub(r'MAX_STEPS = \d+', 'MAX_STEPS = 200', content)
content = re.sub(r'SAVE_LORA_EVERY = \d+', 'SAVE_LORA_EVERY = 50', content)
content = re.sub(r'SNAPSHOT_MERGED_AT = \{[^}]*\}', 'SNAPSHOT_MERGED_AT = {50, 100, 200}', content)
open(p,'w').write(content)
print('train_qwen3.py patched for v4')
PY

python train_qwen3.py 2>&1 | tee train_qwen3_v4.log

# Restore train_qwen3.py to v1 settings
python3 - <<'PY'
import re
p = 'train_qwen3.py'
content = open(p).read()
content = re.sub(r'DATA = HERE / "[^"]*"', 'DATA = HERE / "assets/sft_chai_anime.json"', content)
content = re.sub(r'OUT_LORA = HERE / "[^"]*"', 'OUT_LORA = HERE / "assets/qwen3-4b-anime-lora"', content)
content = re.sub(r'OUT_MERGED = TRAINED_DIR / "[^"]*"', 'OUT_MERGED = TRAINED_DIR / "qwen3-4b-anime-merged"', content)
content = content.replace('qwen3-4b-anime-v4-step{step}', 'qwen3-4b-anime-step{step}')
content = re.sub(r'MAX_STEPS = \d+', 'MAX_STEPS = 100', content)
content = re.sub(r'SAVE_LORA_EVERY = \d+', 'SAVE_LORA_EVERY = 30', content)
content = re.sub(r'SNAPSHOT_MERGED_AT = \{[^}]*\}', 'SNAPSHOT_MERGED_AT = {30, 60, 100}', content)
open(p,'w').write(content)
print('train_qwen3.py restored')
PY

# 4. Upload 3 checkpoints
echo "$(date) === uploading v4 checkpoints to HF ===" | tee -a "$LOG"
for step in 50 100 200; do
  REPO="ZheqiWu/Qwen3-4B-Anime-v4-step${step}"
  DIR="../trained_model/qwen3-4b-anime-v4-step${step}"
  if [ -d "$DIR" ]; then
    echo "$(date) uploading $REPO" | tee -a "$LOG"
    HF_TOKEN="$HF_TOKEN" hf upload "$REPO" "$DIR" 2>&1 | tail -3 | tee -a "$LOG"
  else
    echo "$(date) WARN: $DIR not found" | tee -a "$LOG"
  fi
done

# 5. Add v4 variants to submit_batch.py and fire 6 (3 default + 3 bo16)
cd "$HERE"
python3 <<'PY'
import re
p = 'submit_batch.py'
content = open(p).read()
new_variants = '''
    # Qwen3-4B + Anime-v4 SFT (1000 Sonnet records using the OLD simple ShareGPT prompt
    # — same prompt that produced v1\\'s 33.83% best — scaled 10x). Targets the proven
    # data recipe to see if 1000 > 100 holds.
    ("qwen3_4b_anime_v4_step50",
        "ZheqiWu/Qwen3-4B-Anime-v4-step50", QWEN_CHATML, {}),
    ("qwen3_4b_anime_v4_step100",
        "ZheqiWu/Qwen3-4B-Anime-v4-step100", QWEN_CHATML, {}),
    ("qwen3_4b_anime_v4_step200",
        "ZheqiWu/Qwen3-4B-Anime-v4-step200", QWEN_CHATML, {}),
    ("qwen3_4b_anime_v4_step50_bo16",
        "ZheqiWu/Qwen3-4B-Anime-v4-step50", QWEN_CHATML, {"best_of": 16}),
    ("qwen3_4b_anime_v4_step100_bo16",
        "ZheqiWu/Qwen3-4B-Anime-v4-step100", QWEN_CHATML, {"best_of": 16}),
    ("qwen3_4b_anime_v4_step200_bo16",
        "ZheqiWu/Qwen3-4B-Anime-v4-step200", QWEN_CHATML, {"best_of": 16}),
'''
if 'qwen3_4b_anime_v4_step50' not in content:
    if not re.search(r'(\n\]\n)', content):
        raise SystemExit('cannot find VARIANTS terminator')
    content = re.sub(r'(\n\]\n)', new_variants + r'\1', content, count=1)
    open(p,'w').write(content)
    print('added 6 v4 variants')
else:
    print('v4 variants already present')
PY

echo "$(date) === submitting v4 to Chai ===" | tee -a "$LOG"
uv run --with requests python submit_batch.py --only \
    qwen3_4b_anime_v4_step50 \
    qwen3_4b_anime_v4_step100 \
    qwen3_4b_anime_v4_step200 \
    qwen3_4b_anime_v4_step50_bo16 \
    qwen3_4b_anime_v4_step100_bo16 \
    qwen3_4b_anime_v4_step200_bo16 2>&1 | tee -a "$LOG"

echo "$(date) === v4 pipeline complete ===" | tee -a "$LOG"
