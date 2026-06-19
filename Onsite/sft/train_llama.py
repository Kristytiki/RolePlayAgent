"""
Unsloth + LoRA SFT for Llama-3.2-3B-Instruct on CoSER (preprocessed).

Run after download.sh + preprocess.py.

L4 24GB single GPU. ~30k samples, 500 steps ~= 1-2h.
Output:
    Onsite/sft/assets/llama32-3b-coser-lora/        LoRA adapters per checkpoint
    Onsite/trained_model/llama32-3b-coser-step{200,500}/   merged 16-bit snapshots
    Onsite/trained_model/llama32-3b-coser-merged/   final merged 16-bit (DPO start point)
"""
import json
from pathlib import Path

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback

HERE = Path(__file__).parent
DATA = HERE / "assets/sft_chai_aligned.json"
BASE = HERE / "assets/Llama-3.2-3B-Instruct"
OUT_LORA = HERE / "assets/llama32-3b-coser-lora"
TRAINED_DIR = HERE.parent / "trained_model"
OUT_MERGED = TRAINED_DIR / "llama32-3b-coser-merged"

MAX_SEQ_LEN = 2048
MAX_STEPS = 500
SAVE_LORA_EVERY = 100
SNAPSHOT_MERGED_AT = {200, 500}


def load_data() -> Dataset:
    with open(DATA, "r", encoding="utf-8") as f:
        raw = json.load(f)

    role_map = {"system": "system", "human": "user", "user": "user", "assistant": "assistant"}

    def to_messages(sample: dict) -> dict:
        msgs = []
        for m in sample["conversations"]:
            role = role_map.get(m["from"], m["from"])
            msgs.append({"role": role, "content": m["value"]})
        return {"messages": msgs}

    return Dataset.from_list([to_messages(s) for s in raw])


def main() -> None:
    print(f"loading base from {BASE}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(BASE),
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    # Llama-3 / 3.1 / 3.2 share the same headered template (Unsloth alias "llama-3.1").
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

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

    ds = load_data()
    print(f"dataset: {len(ds)} samples")

    def fmt(batch: dict) -> dict:
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
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
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

    # Mask everything except assistant turns. Llama-3 headered markers.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    class StepwiseSaveCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step
            if step % SAVE_LORA_EVERY == 0:
                lora_dir = OUT_LORA / f"step{step}"
                print(f"\n[callback] step {step}: saving LoRA adapter to {lora_dir}")
                lora_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(lora_dir))
                tokenizer.save_pretrained(str(lora_dir))
            if step in SNAPSHOT_MERGED_AT:
                merged_dir = TRAINED_DIR / f"llama32-3b-coser-step{step}"
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
