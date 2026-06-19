# Onsite DPO — Llama-3.2-3B-CoSER-DPO

Two-stage training (SFT then DPO) on `meta-llama/Llama-3.2-3B-Instruct`,
mirroring the stock Chaiverse submission `meta-llama-llama-3-2-3b_30223_v2`
so the win-rate delta is directly readable.

Strategy: **Opus writes `chosen`, the SFT base model samples `rejected`** (N=4,
Opus picks worst). 500-pair smoke first, scale to 3000 only if win-rate moves.

## Outputs land in `Onsite/trained_model/`

```
trained_model/
├── llama32-3b-coser-step200/        SFT mid-snapshot
├── llama32-3b-coser-step500/        SFT mid-snapshot
├── llama32-3b-coser-merged/         final SFT (DPO start point)
└── llama32-3b-coser-dpo-smoke/      final DPO (uploaded to Chaiverse)
```

## Run order

```bash
# 0. env (one-time, also installs anthropic + pydantic for build_pairs.py)
cd Onsite/sft
bash setup_env.sh
source .venv/bin/activate

# 0a. pull Llama-3.2-3B-Instruct (gated; HF_TOKEN with model access)
HF_TOKEN=hf_xxx bash download.sh

# 1. SFT on Llama-3.2-3B over CoSER (~1-2h on L4)
python train_llama.py
#    -> trained_model/llama32-3b-coser-merged/

# 2. Build 500 DPO pairs with Opus (~$10-15, ~30-60 min including base sampling)
cd ../dpo
ANTHROPIC_API_KEY=sk-ant-... python build_pairs.py \
    --base ../trained_model/llama32-3b-coser-merged \
    --data ../sft/assets/sft_chai_aligned.json \
    --n 500 \
    --out assets/dpo_pairs_smoke.jsonl

# 3. DPO training (~30 min on L4 for 120 steps)
python train_dpo.py \
    --base ../trained_model/llama32-3b-coser-merged \
    --pairs assets/dpo_pairs_smoke.jsonl \
    --out  ../trained_model/llama32-3b-coser-dpo-smoke

# 4. Push to HF
huggingface-cli upload ZheqiWu/Llama-3.2-3B-CoSER-DPO-smoke \
    ../trained_model/llama32-3b-coser-dpo-smoke

# 5. Submit to Chaiverse
cd ..
HF_TOKEN=hf_xxx CHAI_DEVELOPER_KEY=ck_xxx \
    python submit_batch.py --only llama32_3b_dpo_smoke

# 6. Wait ~90 min, compare win-rate against meta-llama-llama-3-2-3b_30223_v2 in SUBMISSIONS.md
```

## Knobs

| File | Knob | Default |
|---|---|---|
| `build_pairs.py` | `--n` | 500 |
| `build_pairs.py` | `N_PER_PROMPT` | 4 |
| `build_pairs.py` | `OPUS_CONCURRENCY` | 8 |
| `build_pairs.py` | `MAX_CHOSEN_TOKENS` | 60 |
| `train_dpo.py` | `--max-steps` | 120 |
| `train_dpo.py` | `--beta` | 0.1 |
| `train_dpo.py` | `--lr` | 5e-6 |

## Scaling beyond smoke

If smoke shows Δwin-rate ≥ +2pp:
- `--n 3000`, `--max-steps 750`
- Cost: ~$80-120 in Opus calls, ~2h GPU
- Output dir: `trained_model/llama32-3b-coser-dpo/` (drop the `-smoke` suffix)

If smoke moves win-rate but is noisy: bump `N_PER_PROMPT` to 8 to give Opus a
wider rejected pool, and tighten the failure-mode list in `OPUS_SYSTEM` toward
the modes Chaiverse actually penalizes (newline truncation, OOC).
