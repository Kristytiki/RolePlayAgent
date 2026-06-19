# Training params — qwen25-3b-coser-smoke-30step

Smoke-test SFT: small step count, just to verify the pipeline trains + saves merged 16-bit successfully.

## Base
- Model: `Qwen/Qwen2.5-3B-Instruct` (loaded via Unsloth, 4-bit)
- max_seq_length: 2048

## LoRA
- r: 32
- lora_alpha: 64
- lora_dropout: 0.0
- target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- gradient_checkpointing: unsloth
- random_state: 42
- trainable params: 59,867,136 (1.90% of 3.15B)

## Data
- Source: `Neph0s/CoSER` train/sft_conversations_sharegpt.json (2.1GB, 305,134 conversations)
- Subset: 30,000 random samples (seed 42, shuffled)
- Preprocess: strip leading "Character: " from assistant; collapse `\n\n`/newlines into spaces; single-line targets
- After Unsloth's response-only filter: 18,372 effective training samples
- Output file: `Onsite/sft/assets/sft_chai_aligned.json`

## Training
- Steps: 30 (max_steps=30, smoke test)
- Effective batch size: 8 (per_device 2 × grad_accum 4)
- Optimizer: adamw_8bit
- Learning rate: 2e-4, cosine schedule, warmup_steps=2
- bf16: True
- save_strategy: "no" (only save merged at end)

## Loss curve
| step | loss | lr |
|---|---|---|
| 5  | 2.438 | 1.975e-4 |
| 10 | 2.044 | 1.707e-4 |
| 15 | 1.995 | 1.223e-4 |
| 20 | 1.946 | 6.697e-5 |
| 25 | 1.888 | 2.182e-5 |
| 30 | 1.871 | 6.288e-7 |

Final train_loss: 2.03 (avg)
Wallclock: 87 seconds on NVIDIA L4 (24GB)

## Output
- Format: merged 16-bit (LoRA delta merged into base, full safetensors)
- Files:
  - model-00001-of-00002.safetensors (3.7GB)
  - model-00002-of-00002.safetensors (2.1GB)
  - config.json, tokenizer files, chat_template.jinja

## Caveats
- Only 30 steps, very early stopping. Loss only fell from 2.44 → 1.87.
- Roleplay style/format only partially learned — production-quality SFT would need ~500-1000 steps.
- Use this as a "pipeline works" sentinel; expect baseline (or below) win-rate on Chaiverse.
