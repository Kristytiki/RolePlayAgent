"""
LoRA DPO training on Llama-3.2-3B-CoSER (the SFT checkpoint produced by
Onsite/sft/train_llama.py) using the preference dataset built by build_pairs.py.

Smoke run: 500 pairs, ~120 steps (≈2 epochs at effective batch 8), L4 24GB.

Usage:
    cd Onsite/sft && source .venv/bin/activate
    cd ../dpo
    python train_dpo.py \\
        --base ../trained_model/llama32-3b-coser-merged \\
        --pairs assets/dpo_pairs_smoke.jsonl \\
        --out  ../trained_model/llama32-3b-coser-dpo-smoke

Output:
    Onsite/trained_model/llama32-3b-coser-dpo-smoke/   merged 16-bit checkpoint
    Onsite/dpo/assets/llama32-3b-coser-dpo-lora/       LoRA adapter snapshots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from transformers import TrainerCallback
from trl import DPOConfig, DPOTrainer

PatchDPOTrainer()  # required by Unsloth before instantiating DPOTrainer

HERE = Path(__file__).parent
DEFAULT_BASE = HERE.parent / "trained_model" / "llama32-3b-coser-merged"
DEFAULT_PAIRS = HERE / "assets" / "dpo_pairs_smoke.jsonl"
DEFAULT_OUT = HERE.parent / "trained_model" / "llama32-3b-coser-dpo-smoke"

OUT_LORA = HERE / "assets" / "llama32-3b-coser-dpo-lora"

MAX_SEQ_LEN = 1024
MAX_PROMPT_LEN = 896
SAVE_LORA_EVERY = 40


def load_pairs(path: Path) -> Dataset:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({
                "prompt": obj["prompt"],
                "chosen": obj["chosen"],
                "rejected": obj["rejected"],
            })
    return Dataset.from_list(rows)


def validate_dataset(ds: Dataset) -> None:
    bad = []
    for i, r in enumerate(ds):
        if not r["prompt"] or not r["chosen"] or not r["rejected"]:
            bad.append((i, "empty field"))
        elif "\n" in r["chosen"]:
            bad.append((i, "newline in chosen"))
    if bad:
        sys.exit(f"dataset has {len(bad)} bad rows; first: {bad[:3]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE),
                    help="path to Llama-3.2-3B-CoSER SFT merged checkpoint")
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS), help="DPO pairs JSONL")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="merged 16-bit output directory")
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--chat-template", default="llama-3.1",
                    choices=["llama-3.1", "qwen-2.5"])
    args = ap.parse_args()

    base_dir = Path(args.base)
    pairs_path = Path(args.pairs)
    out_dir = Path(args.out)

    if not base_dir.exists():
        sys.exit(f"base dir not found: {base_dir}")
    if not pairs_path.exists():
        sys.exit(f"pairs file not found: {pairs_path}")

    print(f"loading base from {base_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(base_dir),
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"loading pairs from {pairs_path}")
    ds = load_pairs(pairs_path)
    validate_dataset(ds)
    print(f"dataset: {len(ds)} pairs")

    cfg = DPOConfig(
        output_dir=str(OUT_LORA),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.0,
        seed=42,
        report_to="none",
        beta=args.beta,
        loss_type="sigmoid",
        max_length=MAX_SEQ_LEN,
        max_prompt_length=MAX_PROMPT_LEN,
        remove_unused_columns=False,
    )

    # ref_model=None lets TRL+Unsloth use the frozen base of the PEFT adapter
    # as the reference policy — the standard LoRA-DPO setup, no extra GPU mem.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

    class StepwiseSaveCallback(TrainerCallback):
        def on_step_end(self, args_, state, control, **kwargs):
            step = state.global_step
            if step and step % SAVE_LORA_EVERY == 0:
                lora_dir = OUT_LORA / f"step{step}"
                print(f"\n[callback] step {step}: saving LoRA to {lora_dir}")
                lora_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(lora_dir))
                tokenizer.save_pretrained(str(lora_dir))

    trainer.add_callback(StepwiseSaveCallback())
    trainer.train()

    print(f"\nfinal save merged 16-bit to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(out_dir), tokenizer, save_method="merged_16bit")
    print("done")


if __name__ == "__main__":
    main()
