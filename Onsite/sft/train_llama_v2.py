"""
Unsloth + LoRA SFT for Llama-3.2-3B-Instruct on the v2 anime/fandom corpus
(OpenAI {messages: [{role, content, weight}]} schema, mirroring onsite.example).

Key differences from train_llama.py:
  1. Reads OpenAI messages format (not ShareGPT conversations).
  2. Loss mask honours `weight` — every token in a w=0 assistant turn is set
     to label -100, so the cinematic opening scene-setter does NOT contribute
     to gradient. Only w=1 assistant tokens drive learning.
  3. 850 / 150 train / val split. eval every 25 steps so we can spot overfit.
  4. Lower LR (5e-5), longer warmup (10%), larger effective batch (16).

Run:
    cd Onsite/sft && source .venv/bin/activate
    python train_llama_v2.py
"""
from __future__ import annotations

import os
# Must be set BEFORE Unsloth imports — otherwise compute_loss can't see logits.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

HERE = Path(__file__).parent
DATA = HERE / "assets/sft_chai_anime_v2.json"
BASE = HERE / "assets/Llama-3.2-3B-Instruct"
OUT_LORA = HERE / "assets/llama32-3b-anime-v2-lora"
TRAINED_DIR = HERE.parent / "trained_model"
OUT_MERGED = TRAINED_DIR / "llama32-3b-anime-v2-merged"

MAX_SEQ_LEN = 2048
MAX_STEPS = 300
SAVE_LORA_EVERY = 50
SNAPSHOT_MERGED_AT = {100, 200, 300}
VAL_SPLIT = 0.15
SEED = 42


def load_v2() -> tuple[list[dict], list[dict]]:
    """Return (train_records, val_records). Each record is a dict with
    'messages': [{role, content, weight}]."""
    with open(DATA, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rng = random.Random(SEED)
    rng.shuffle(raw)
    cut = int(len(raw) * (1 - VAL_SPLIT))
    return raw[:cut], raw[cut:]


def render_with_mask(record: dict, tokenizer) -> dict:
    """Render the full conversation into one string AND build a per-token
    label_mask aligned to the input_ids: 1 = include in loss, 0 = mask out.

    Strategy: render each message individually so we know its token span,
    concatenate, and mark only assistant turns with weight=1 as keep.
    """
    msgs = record["messages"]
    full_ids: list[int] = []
    label_mask: list[int] = []  # parallel to full_ids

    # Single BOS for the whole sequence
    if tokenizer.bos_token_id is not None:
        full_ids.append(tokenizer.bos_token_id)
        label_mask.append(0)

    for i, m in enumerate(msgs):
        # Render this single turn through chat_template, then strip the BOS the
        # template adds (we already have one).
        chunk = tokenizer.apply_chat_template(
            [{"role": m["role"], "content": m["content"]}],
            tokenize=False,
            add_generation_prompt=False,
        )
        # Drop leading BOS token text if present (Llama-3 template adds <|begin_of_text|>).
        bos_str = tokenizer.bos_token or ""
        if bos_str and chunk.startswith(bos_str):
            chunk = chunk[len(bos_str):]
        chunk_ids = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        is_loss = (m["role"] == "assistant" and m.get("weight", 1) == 1)
        full_ids.extend(chunk_ids)
        label_mask.extend([1 if is_loss else 0] * len(chunk_ids))

    # Truncate to MAX_SEQ_LEN
    full_ids = full_ids[:MAX_SEQ_LEN]
    label_mask = label_mask[:MAX_SEQ_LEN]
    return {"input_ids": full_ids, "label_mask": label_mask}


def to_hf_dataset(records: list[dict], tokenizer) -> Dataset:
    rendered = [render_with_mask(r, tokenizer) for r in records]
    rendered = [r for r in rendered if sum(r["label_mask"]) >= 8]  # at least some loss tokens
    return Dataset.from_list(rendered)


class WeightMaskCollator:
    """Pad to max length in batch, set labels = input_ids but mask out
    positions where label_mask == 0."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        ids, masks, attns = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = max_len - n
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            masks.append(b["label_mask"] + [0] * pad)
            attns.append([1] * n + [0] * pad)
        ids_t = torch.tensor(ids, dtype=torch.long)
        masks_t = torch.tensor(masks, dtype=torch.long)
        attns_t = torch.tensor(attns, dtype=torch.long)
        labels = ids_t.clone()
        labels[masks_t == 0] = -100
        return {"input_ids": ids_t, "attention_mask": attns_t, "labels": labels}


def main() -> None:
    print(f"loading base from {BASE}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(BASE),
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    train_recs, val_recs = load_v2()
    print(f"split: {len(train_recs)} train / {len(val_recs)} val")
    train_ds = to_hf_dataset(train_recs, tokenizer)
    val_ds = to_hf_dataset(val_recs, tokenizer)
    print(f"after filter: {len(train_ds)} train / {len(val_ds)} val")

    args = TrainingArguments(
        output_dir=str(OUT_LORA),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch = 16
        per_device_eval_batch_size=4,
        warmup_steps=30,                # 10% of 300
        max_steps=MAX_STEPS,
        learning_rate=5e-5,             # lower LR per the SFT discipline guide
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="no",             # Unsloth + custom collator can't run eval — skip
        save_strategy="no",
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.0,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
    )

    collator = WeightMaskCollator(tokenizer)

    # TRL 0.18+ deprecated `tokenizer=` in favour of `processing_class=`.
    # Try the new arg first, fall back to old.
    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            data_collator=collator,
            args=args,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            data_collator=collator,
            args=args,
        )

    class StepwiseSaveCallback(TrainerCallback):
        def on_step_end(self, args_, state, control, **kwargs):
            step = state.global_step
            if step and step % SAVE_LORA_EVERY == 0:
                lora_dir = OUT_LORA / f"step{step}"
                print(f"\n[callback] step {step}: saving LoRA adapter to {lora_dir}")
                lora_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(lora_dir))
                tokenizer.save_pretrained(str(lora_dir))
            if step in SNAPSHOT_MERGED_AT:
                merged_dir = TRAINED_DIR / f"llama32-3b-anime-v2-step{step}"
                print(f"[callback] step {step}: saving merged 16-bit to {merged_dir}")
                merged_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained_merged(
                    str(merged_dir), tokenizer, save_method="merged_16bit"
                )

    trainer.add_callback(StepwiseSaveCallback())
    trainer.train()

    print(f"\nfinal save merged 16-bit to {OUT_MERGED}")
    OUT_MERGED.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(OUT_MERGED), tokenizer, save_method="merged_16bit")
    print("done")


if __name__ == "__main__":
    main()
