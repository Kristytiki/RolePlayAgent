#!/usr/bin/env bash
# Auto pipeline: wait for 4 v2 workflow outputs, build dataset, train Llama-3.2-3B, upload, submit Chai.
# Designed to run unattended after the 4 workflows are kicked off.
set -euo pipefail

cd "$(dirname "$0")/.."
HERE="$(pwd)"
LOG="$HERE/sft/auto_v2_pipeline.log"

WF_OUT_DIR="/tmp/claude-22246397/-local-home-zheqi-workspace-RolePlayAgent/c7916a54-fab0-4960-8903-28c841855c4b/tasks"
B0=woksna2yq
B1=w7iyp2mt1
B2=wbkh780cl
B3=wg56ajm3t

HF_TOKEN="${HF_TOKEN:-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}"
CHAI_DEVELOPER_KEY="${CHAI_DEVELOPER_KEY:-CR_5a47df3092764a8d9dfae7da900291ea}"
export HF_TOKEN CHAI_DEVELOPER_KEY

echo "$(date) === auto pipeline starting ===" | tee -a "$LOG"

# 1. Wait for 4 workflow outputs
echo "$(date) waiting for 4 workflow outputs..." | tee -a "$LOG"
for sid in $B0 $B1 $B2 $B3; do
  while [ ! -s "$WF_OUT_DIR/$sid.output" ]; do
    sleep 30
  done
  echo "$(date) $sid done ($(wc -c < $WF_OUT_DIR/$sid.output) bytes)" | tee -a "$LOG"
done

# 2. Aggregate to sft_chai_anime_v2.json with filter
echo "$(date) === aggregating + filtering ===" | tee -a "$LOG"
python3 <<PY 2>&1 | tee -a "$LOG"
import json
out = []
total_records = 0
total_turns_dropped = 0
total_turns_kept = 0
TURN_MAX = 320  # drop any w=1 turn that would generate > ~80 tokens (with cushion)
RECORD_TURNS_MIN = 10  # need >= 10 messages after filter

for sid in ['$B0', '$B1', '$B2', '$B3']:
    data = json.load(open(f'$WF_OUT_DIR/{sid}.output'))
    records = data['result']['records']
    print(f'{sid}: {len(records)} raw records')
    total_records += len(records)
    for r in records:
        clean_msgs = []
        for m in r['messages']:
            # Drop w=1 assistant turns that exceed TURN_MAX
            if m['role'] == 'assistant' and m.get('weight', 1) == 1 and len(m['content']) > TURN_MAX:
                total_turns_dropped += 1
                continue
            # Strip newlines from w=1 assistant turns (extra safety)
            if m['role'] == 'assistant' and m.get('weight', 1) == 1 and '\n' in m['content']:
                m = {**m, 'content': m['content'].split('\n', 1)[0].strip()}
            clean_msgs.append(m)
            total_turns_kept += 1
        # Keep only if structure intact
        if len(clean_msgs) >= RECORD_TURNS_MIN and clean_msgs[0]['role'] == 'system' and clean_msgs[1]['role'] == 'assistant':
            # Make sure we end on assistant turn (truncate trailing user)
            if clean_msgs[-1]['role'] == 'user':
                clean_msgs = clean_msgs[:-1]
            out.append({'messages': clean_msgs})

out_path = '$HERE/sft/assets/sft_chai_anime_v2.json'
with open(out_path, 'w') as f:
    json.dump(out, f, ensure_ascii=False)
print(f'\nFINAL: {len(out)} records (from {total_records} raw)')
print(f'turns: kept {total_turns_kept}, dropped (>320 chars) {total_turns_dropped}')
import os
print(f'file size: {os.path.getsize(out_path)} bytes')
PY

# 3. Train Llama-3.2-3B on v2 data
echo "$(date) === training Llama-3.2-3B + v2 ===" | tee -a "$LOG"
cd "$HERE/sft"
source .venv/bin/activate

# Patch train_llama.py to point at v2 data + new output names
python3 <<PY
import re
p = 'train_llama.py'
content = open(p).read()
content = re.sub(r'DATA = HERE / "assets/[^"]*"', 'DATA = HERE / "assets/sft_chai_anime_v2.json"', content)
content = re.sub(r'OUT_LORA = HERE / "assets/[^"]*"', 'OUT_LORA = HERE / "assets/llama32-3b-anime-v2-lora"', content)
content = re.sub(r'OUT_MERGED = TRAINED_DIR / "[^"]*"', 'OUT_MERGED = TRAINED_DIR / "llama32-3b-anime-v2-merged"', content)
content = content.replace('llama32-3b-anime-step{step}', 'llama32-3b-anime-v2-step{step}')
content = re.sub(r'MAX_STEPS = \d+', 'MAX_STEPS = 300', content)
content = re.sub(r'SAVE_LORA_EVERY = \d+', 'SAVE_LORA_EVERY = 50', content)
content = re.sub(r'SNAPSHOT_MERGED_AT = \{[^}]*\}', 'SNAPSHOT_MERGED_AT = {100, 200, 300}', content)
open(p, 'w').write(content)
print('train_llama.py patched for v2')
PY

python train_llama.py 2>&1 | tee train_llama_v2.log

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
python3 <<PY
import re
p = 'submit_batch.py'
content = open(p).read()
new_variants = '''
    # Llama-3.2-3B + Anime-v2 SFT (1000 Opus-synth onsite-schema records).
    ("llama32_3b_anime_v2_step100",
        "ZheqiWu/Llama-3.2-3B-Anime-v2-step100", LLAMA31_HEADERED, {}),
    ("llama32_3b_anime_v2_step200",
        "ZheqiWu/Llama-3.2-3B-Anime-v2-step200", LLAMA31_HEADERED, {}),
    ("llama32_3b_anime_v2_step300",
        "ZheqiWu/Llama-3.2-3B-Anime-v2-step300", LLAMA31_HEADERED, {}),
'''
# Insert before the final ']'
if 'llama32_3b_anime_v2_step100' not in content:
    content = re.sub(r'(\s*\]\s*\n\s*# ----)', new_variants + r'\1', content, count=1)
    if 'llama32_3b_anime_v2_step100' not in content:
        # Fallback: insert before the bare `]` that closes VARIANTS
        content = re.sub(r'(\n\]\n)', new_variants + r'\1', content, count=1)
    open(p, 'w').write(content)
    print('added 3 variants to submit_batch.py')
else:
    print('variants already present')
PY

uv run --with requests python submit_batch.py --only llama32_3b_anime_v2_step100 llama32_3b_anime_v2_step200 llama32_3b_anime_v2_step300 2>&1 | tee -a "$LOG"

echo "$(date) === pipeline complete ===" | tee -a "$LOG"
