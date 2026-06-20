"""
Unsloth + LoRA SFT for Qwen2.5-3B-Instruct on CoSER (preprocessed).

Run after download.sh + preprocess.py.

L4 24GB single GPU. 100k samples, 1 epoch ~= 2-3h.
Output: ./assets/qwen25-3b-coser-merged/  (16-bit merged checkpoint, ready for HF push)
"""
import json
import os
from pathlib import Path

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback

HERE = Path(__file__).parent.parent  # sft/  (this script lives in sft/core/)
# Switch DATASET via env. RUN_TAG appended to all output dirs so multiple
# runs don't overwrite each other. BASE_MODEL controls which base to fine-tune
# (default Qwen2.5-3B; set to "qwen3-4b" for ensemble runs).
_DATASET = os.environ.get("DATASET", "coser")
_TAG = os.environ.get("RUN_TAG", "")
_TAG_SUFFIX = f"-{_TAG}" if _TAG else ""
_BASE_MODEL = os.environ.get("BASE_MODEL", "qwen25-3b")
if _BASE_MODEL == "qwen3-4b":
    BASE = HERE / "assets/Qwen3-4B-Instruct-2507"
    BASE_PREFIX = "qwen3-4b"
else:
    BASE = HERE / "assets/Qwen2.5-3B-Instruct"
    BASE_PREFIX = "qwen25-3b"

if _DATASET == "pippa":
    DATA = HERE / "assets/sft_chai_pippa.json"
    OUT_LORA = HERE / f"assets/{BASE_PREFIX}-pippa{_TAG_SUFFIX}-lora"
    OUT_MERGED = HERE / f"assets/{BASE_PREFIX}-pippa{_TAG_SUFFIX}-merged"
    SNAPSHOT_PREFIX = f"{BASE_PREFIX}-pippa{_TAG_SUFFIX}"
elif _DATASET == "hieu":
    DATA = HERE / "assets/sft_chai_hieu.json"
    OUT_LORA = HERE / f"assets/{BASE_PREFIX}-hieu{_TAG_SUFFIX}-lora"
    OUT_MERGED = HERE / f"assets/{BASE_PREFIX}-hieu{_TAG_SUFFIX}-merged"
    SNAPSHOT_PREFIX = f"{BASE_PREFIX}-hieu{_TAG_SUFFIX}"
elif _DATASET == "synth":
    DATA = HERE / "assets/sft_chai_synth.json"
    OUT_LORA = HERE / f"assets/{BASE_PREFIX}-synth{_TAG_SUFFIX}-lora"
    OUT_MERGED = HERE / f"assets/{BASE_PREFIX}-synth{_TAG_SUFFIX}-merged"
    SNAPSHOT_PREFIX = f"{BASE_PREFIX}-synth{_TAG_SUFFIX}"
elif _DATASET == "anime":
    DATA = HERE / "assets/sft_chai_anime.json"
    OUT_LORA = HERE / f"assets/{BASE_PREFIX}-anime{_TAG_SUFFIX}-lora"
    OUT_MERGED = HERE / f"assets/{BASE_PREFIX}-anime{_TAG_SUFFIX}-merged"
    SNAPSHOT_PREFIX = f"{BASE_PREFIX}-anime{_TAG_SUFFIX}"
else:
    DATA = HERE / "assets/sft_chai_aligned.json"
    OUT_LORA = HERE / f"assets/{BASE_PREFIX}-coser{_TAG_SUFFIX}-lora"
    OUT_MERGED = HERE / f"assets/{BASE_PREFIX}-coser{_TAG_SUFFIX}-merged"
    SNAPSHOT_PREFIX = f"{BASE_PREFIX}-coser{_TAG_SUFFIX}"
TRAIN_MODEL_DIR = HERE.parent / "train_model"

MAX_SEQ_LEN = 2048
MAX_STEPS = int(os.environ.get("MAX_STEPS", "500"))
SAVE_LORA_EVERY = int(os.environ.get("SAVE_LORA_EVERY", "100"))
SNAPSHOT_MERGED_AT = {int(s) for s in os.environ.get("SNAPSHOT_AT", "200,500").split(",") if s.strip()}

def load_data():
    with open(DATA, "r", encoding="utf-8") as f:
        raw = json.load(f)

    role_map = {"system": "system", "human": "user", "user": "user", "assistant": "assistant"}

    def to_messages(sample):
        msgs = []
        for m in sample["conversations"]:
            role = role_map.get(m["from"], m["from"])
            msgs.append({"role": role, "content": m["value"]})
        return {"messages": msgs}

    return Dataset.from_list([to_messages(s) for s in raw])

def main():
    print(f"loading base from {BASE}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(BASE),
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,             # auto bf16/fp16
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    lora_r = int(os.environ.get("LORA_R", "32"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", str(2 * lora_r)))
    print(f"LoRA r={lora_r} alpha={lora_alpha}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    ds = load_data()
    print(f"dataset: {len(ds)} samples")

    def fmt(batch):
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch["messages"]
        ]
        return {"text": texts}

    ds = ds.map(fmt, batched=True, remove_columns=["messages"])

    args = TrainingArguments(
        output_dir=str(OUT_LORA),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=MAX_STEPS,
        learning_rate=float(os.environ.get("LR", "2e-4")),
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",                      # we save manually via callback
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.0,
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        args=args,
    )

    # Mask everything except assistant turns so loss is only on the bot's reply.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    class StepwiseSaveCallback(TrainerCallback):
        """At every SAVE_LORA_EVERY step: save LoRA adapter (cheap).
        At each step in SNAPSHOT_MERGED_AT: save merged 16-bit to train_model/{SNAPSHOT_PREFIX}-stepNNN.
        """
        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step
            if step % SAVE_LORA_EVERY == 0:
                lora_dir = OUT_LORA / f"step{step}"
                print(f"\n[callback] step {step}: saving LoRA adapter to {lora_dir}")
                lora_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(lora_dir))
                tokenizer.save_pretrained(str(lora_dir))
            if step in SNAPSHOT_MERGED_AT:
                merged_dir = TRAIN_MODEL_DIR / f"{SNAPSHOT_PREFIX}-step{step}"
                print(f"[callback] step {step}: saving merged 16-bit to {merged_dir}")
                merged_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    trainer.add_callback(StepwiseSaveCallback())
    trainer.train()

    print(f"\nfinal save merged 16-bit to {OUT_MERGED}")
    OUT_MERGED.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(OUT_MERGED), tokenizer, save_method="merged_16bit")
    print("done")

if __name__ == "__main__":
    main()
