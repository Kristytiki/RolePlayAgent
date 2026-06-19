"""
Phase 1 of DPO data construction: sample N candidate completions per CoSER prompt
from a CoSER-fine-tuned base model. Writes `candidates.jsonl` for the labeler.

The labeler (the assistant in this Claude Code session, OR a separate Opus call
from `label_pairs.py`) consumes candidates.jsonl, picks the worst as `rejected`
and writes a single-line `chosen`. See README.md.

Usage:
    cd Onsite/sft && source .venv/bin/activate
    cd ../dpo
    python gen_candidates.py \\
        --base ../sft/assets/qwen25-3b-coser-merged \\
        --data ../sft/assets/sft_chai_aligned.json \\
        --chat-template qwen-2.5 \\
        --n 100 \\
        --out assets/candidates_qwen_smoke.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from render import conv_to_messages, render_prompt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gen_candidates")

N_PER_PROMPT = 4
MAX_NEW = 64
STOP_TOKEN = "\n"


@dataclass(frozen=True)
class PromptSpec:
    prompt_text: str
    bot_name: str
    char_profile: str
    convo_messages: list


def _extract_bot_name(system: str) -> str:
    # CoSER style: "You will be portraying X from BOOK"
    m = re.search(r"portraying ([A-Z][\w' \-\.]+?)\s+from", system)
    if m:
        return m.group(1).strip()
    # PIPPA style: "You are X."  (X may include spaces, punctuation, "the")
    m = re.search(r"You are ([^.\n]+?)\.", system)
    if m:
        return m.group(1).strip()
    return "the character"


def _extract_profile(system: str) -> str:
    return system[:1200]


def sample_prompts(raw, n, tokenizer, rng):
    by_char: dict[str, list[dict]] = {}
    for sample in raw:
        sys_msg = next(
            (m["value"] for m in sample["conversations"] if m["from"] == "system"),
            "",
        )
        bot = _extract_bot_name(sys_msg)
        by_char.setdefault(bot, []).append(sample)

    chars = sorted(by_char.keys())
    rng.shuffle(chars)
    log.info("characters available: %d", len(chars))

    specs: list[PromptSpec] = []
    cursor = 0
    while len(specs) < n and cursor < len(chars) * 50:
        bot = chars[cursor % len(chars)]
        cursor += 1
        sample = rng.choice(by_char[bot])
        msgs = conv_to_messages(sample["conversations"])
        a_indices = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        if not a_indices:
            continue
        a_indices = [i for i in a_indices if i >= 2] or a_indices
        k = rng.choice(a_indices)
        prefix = msgs[:k]
        rendered = render_prompt(tokenizer, prefix)
        sys_msg = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
        specs.append(
            PromptSpec(
                prompt_text=rendered,
                bot_name=bot,
                char_profile=_extract_profile(sys_msg),
                convo_messages=prefix,
            )
        )
    return specs


def _stop_token_ids(tok):
    ids = tok.encode(STOP_TOKEN, add_special_tokens=False)
    extra = []
    for special in ("<|eot_id|>", "<|end_of_text|>", "<|im_end|>"):
        try:
            tid = tok.convert_tokens_to_ids(special)
            if tid is not None and tid >= 0:
                extra.append(tid)
        except KeyError:
            pass
    return [i for i in ids + extra if i is not None and i >= 0]


def _trim(text: str) -> str:
    return text.split("\n", 1)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--chat-template", required=True,
                    choices=["llama-3.1", "qwen-2.5"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-per-prompt", type=int, default=N_PER_PROMPT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base_dir = Path(args.base)
    if not base_dir.exists():
        sys.exit(f"base dir not found: {base_dir}")

    log.info("loading SFT data from %s", args.data)
    raw = json.loads(Path(args.data).read_text(encoding="utf-8"))
    log.info("loaded %d conversations", len(raw))

    log.info("loading base model from %s", base_dir)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    # Reuse Unsloth-flavor chat template via tokenizer's own apply_chat_template.
    # The local merged checkpoint already has chat_template baked in by the SFT run,
    # so we just trust tokenizer.apply_chat_template (called inside render_prompt).

    log.info("sampling %d prompts", args.n)
    specs = sample_prompts(raw, args.n, tok, rng)
    log.info("got %d prompts; %d distinct characters",
             len(specs), len({s.bot_name for s in specs}))

    log.info("generating %d candidates per prompt (greedy seeds)", args.n_per_prompt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stop_ids = _stop_token_ids(tok)

    with out_path.open("w", encoding="utf-8") as f:
        for i, spec in enumerate(specs):
            inputs = tok(spec.prompt_text, return_tensors="pt",
                         add_special_tokens=False).to(model.device)
            cands = []
            for j in range(args.n_per_prompt):
                torch.manual_seed(args.seed + i * 7 + j)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=40,
                        max_new_tokens=MAX_NEW,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=stop_ids if stop_ids else tok.eos_token_id,
                    )
                new = out[0, inputs["input_ids"].shape[1]:]
                cands.append(_trim(tok.decode(new, skip_special_tokens=True)))
            record = {
                "idx": i,
                "prompt": spec.prompt_text,
                "char": spec.bot_name,
                "char_profile": spec.char_profile,
                "convo_tail": [
                    {"role": m["role"], "content": m["content"]}
                    for m in spec.convo_messages[-6:]
                ],
                "candidates": cands,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                log.info("  %d/%d prompts done", i + 1, len(specs))

    log.info("wrote %d records to %s", len(specs), out_path)


if __name__ == "__main__":
    main()
