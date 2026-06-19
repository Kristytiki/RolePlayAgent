"""
Build a DPO preference dataset for Llama-3.2-3B-CoSER using:
  * a Llama-3.2-3B SFT checkpoint to sample N=4 candidate completions per prompt
  * Opus to (a) pick the worst candidate as `rejected`, (b) write a single-line
    in-character `chosen`.

Cost target: ~$10-15 for 500 pairs (Strategy B from the plan).

Usage:
    cd Onsite/sft && source .venv/bin/activate
    cd ../dpo
    ANTHROPIC_API_KEY=sk-ant-... \\
        python build_pairs.py \\
            --base ../trained_model/llama32-3b-coser-merged \\
            --data ../sft/assets/sft_chai_aligned.json \\
            --n 500 --out assets/dpo_pairs_smoke.jsonl

Determinism: --seed (default 42) controls prompt sampling and base-model
generation seeds. Opus is non-deterministic per call (we only ask for a small
JSON object so the variance does not matter much for DPO).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from render import conv_to_messages, first_assistant_idx, render_prompt  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("build_pairs")

OPUS_MODEL = "claude-opus-4-8"
N_PER_PROMPT = 4
MAX_NEW = 64
STOP_TOKEN = "\n"
MAX_CHOSEN_TOKENS = 60  # Llama tokens, must fit Chaiverse max_output=64
OPUS_CONCURRENCY = 8

# ---- Schemas ---------------------------------------------------------------


class OpusPick(BaseModel):
    worst_idx: int = Field(..., ge=0, le=N_PER_PROMPT - 1)
    reason: str
    chosen: str


VALID_REASONS = {
    "out_of_character",
    "multi_line_truncated",
    "refusal",
    "meta_narration",
    "template_opener",
    "repetition",
    "incoherent",
}


# ---- Prompt sampling ------------------------------------------------------


@dataclass(frozen=True)
class PromptSpec:
    prompt_text: str       # rendered with chat template + assistant header trailing
    bot_name: str
    char_profile: str      # excerpt of system message
    convo_messages: list   # the full message list up to (but not including) the assistant turn


def _extract_bot_name(system: str) -> str:
    m = re.search(r"portraying ([A-Z][\w' \-\.]+?)\s+from", system)
    return m.group(1).strip() if m else "the character"


def _extract_profile(system: str) -> str:
    return system[:1200]


def sample_prompts(
    raw: list[dict], n: int, tokenizer, rng: random.Random
) -> list[PromptSpec]:
    """Stratify by character so we cover ≥100 distinct chars in 500."""
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
    char_cursor = 0
    while len(specs) < n and char_cursor < len(chars) * 50:
        bot = chars[char_cursor % len(chars)]
        char_cursor += 1
        candidates = by_char[bot]
        sample = rng.choice(candidates)

        msgs = conv_to_messages(sample["conversations"])
        # Pick a random assistant turn to use as the supervision target boundary.
        a_indices = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        if not a_indices:
            continue
        # Skip the very first assistant turn — too little context.
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


# ---- Base-model sampling ---------------------------------------------------


def load_base(base_dir: Path):
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
    return model, tok


def _stop_token_ids(tok) -> list[int]:
    # Encode "\n" without adding special tokens so we can stop at first newline.
    ids = tok.encode(STOP_TOKEN, add_special_tokens=False)
    extra = []
    for special in ("<|eot_id|>", "<|end_of_text|>"):
        try:
            extra.append(tok.convert_tokens_to_ids(special))
        except KeyError:
            pass
    return [i for i in ids + extra if i is not None and i >= 0]


def _trim_at_first_newline(text: str) -> str:
    return text.split("\n", 1)[0].strip()


def sample_n_completions(
    model, tok, prompt_text: str, n: int, base_seed: int
) -> list[str]:
    """Sample `n` independent completions matching Chaiverse defaults."""
    inputs = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    stop_ids = _stop_token_ids(tok)
    completions: list[str] = []
    for i in range(n):
        torch.manual_seed(base_seed + i)
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
        text = tok.decode(new, skip_special_tokens=True)
        completions.append(_trim_at_first_newline(text))
    return completions


# ---- Opus call -------------------------------------------------------------


OPUS_SYSTEM = (
    "You are an expert role-play data labeler. You are given a partial role-play "
    "conversation, the character profile, and 4 candidate continuations sampled "
    "from a model. You will:\n"
    "  1. Pick the WORST candidate by index (0-3). Cite ONE failure mode from: "
    + ", ".join(sorted(VALID_REASONS))
    + ".\n"
    "  2. Write a single-line in-character `chosen` reply for the same turn. "
    "Hard constraints on `chosen`:\n"
    "     - Single line. Absolutely no newline characters.\n"
    "     - At most ~50 tokens.\n"
    "     - In character; no meta narration, no AI-assistant disclaimers.\n"
    "     - If the system message tells the model to use [thought] and (action), "
    "use that format. Otherwise just speak naturally.\n"
    "     - Do NOT prefix with the character's name (e.g. no 'Feyre:' prefix).\n"
    "Return ONLY a JSON object with keys worst_idx (int), reason (str), chosen (str). "
    "No prose, no markdown fences."
)


def _format_user(spec: PromptSpec, candidates: list[str]) -> str:
    convo_str = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in spec.convo_messages[-6:]
    )
    cand_str = "\n".join(f"[{i}] {c!r}" for i, c in enumerate(candidates))
    return (
        f"CHARACTER: {spec.bot_name}\n"
        f"CHARACTER PROFILE (excerpt):\n{spec.char_profile}\n\n"
        f"RECENT CONVERSATION (last 6 turns):\n{convo_str}\n\n"
        f"NEXT TURN: {spec.bot_name}\n\n"
        f"CANDIDATES:\n{cand_str}\n\n"
        "Return JSON only."
    )


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(s: str) -> str:
    return _FENCE_RE.sub("", s).strip()


async def opus_pick(
    client: AsyncAnthropic, spec: PromptSpec, candidates: list[str]
) -> OpusPick | None:
    user = _format_user(spec, candidates)
    try:
        resp = await client.messages.create(
            model=OPUS_MODEL,
            max_tokens=400,
            system=OPUS_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as e:
        log.warning("Opus call failed: %s", e)
        return None
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    text = _strip_fences(text)
    try:
        obj = json.loads(text)
        return OpusPick.model_validate(obj)
    except (json.JSONDecodeError, ValidationError) as e:
        log.warning("Opus output unparseable: %s\nraw=%s", e, text[:300])
        return None


# ---- Filter ---------------------------------------------------------------


def passes_filter(chosen: str, rejected: str, tok) -> tuple[bool, str]:
    if not chosen.strip():
        return False, "chosen empty"
    if "\n" in chosen:
        return False, "chosen has newline"
    if not rejected.strip():
        return False, "rejected empty"
    if chosen.strip().lower() == rejected.strip().lower():
        return False, "chosen == rejected"
    n_tok = len(tok.encode(chosen, add_special_tokens=False))
    if n_tok > MAX_CHOSEN_TOKENS:
        return False, f"chosen {n_tok}>{MAX_CHOSEN_TOKENS} tokens"
    return True, ""


# ---- Main pipeline --------------------------------------------------------


async def run(args) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("set ANTHROPIC_API_KEY before running")

    rng = random.Random(args.seed)
    base_dir = Path(args.base)
    if not base_dir.exists():
        sys.exit(f"base model dir not found: {base_dir}")

    data_path = Path(args.data)
    log.info("loading SFT data from %s", data_path)
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    log.info("loaded %d conversations", len(raw))

    model, tok = load_base(base_dir)

    log.info("sampling %d prompts (stratified by character)", args.n)
    specs = sample_prompts(raw, args.n, tok, rng)
    log.info("got %d prompts; %d distinct characters",
             len(specs), len({s.bot_name for s in specs}))

    log.info("phase 1: sampling %d candidates per prompt from base model", N_PER_PROMPT)
    all_candidates: list[list[str]] = []
    for i, spec in enumerate(specs):
        cands = sample_n_completions(model, tok, spec.prompt_text, N_PER_PROMPT,
                                     base_seed=args.seed + i * 7)
        all_candidates.append(cands)
        if (i + 1) % 25 == 0:
            log.info("  %d/%d prompts sampled", i + 1, len(specs))

    log.info("phase 2: Opus pick + chosen (concurrency=%d)", OPUS_CONCURRENCY)
    client = AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(OPUS_CONCURRENCY)

    progress = {"done": 0, "total": len(specs)}

    async def call_one(spec: PromptSpec, candidates: list[str]) -> OpusPick | None:
        async with sem:
            result = await opus_pick(client, spec, candidates)
        progress["done"] += 1
        if progress["done"] % 25 == 0:
            log.info("  %d/%d Opus calls done", progress["done"], progress["total"])
        return result

    picks_aligned: list[OpusPick | None] = await asyncio.gather(
        *(call_one(spec, cands) for spec, cands in zip(specs, all_candidates))
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_kept = 0
    n_dropped = 0
    drop_reasons: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as f:
        for spec, cands, pick in zip(specs, all_candidates, picks_aligned):
            if pick is None:
                n_dropped += 1
                drop_reasons["opus_failed"] = drop_reasons.get("opus_failed", 0) + 1
                continue
            if pick.reason not in VALID_REASONS:
                # don't reject — just normalize
                pass
            rejected = cands[pick.worst_idx]
            chosen = pick.chosen.strip()
            ok, why = passes_filter(chosen, rejected, tok)
            if not ok:
                n_dropped += 1
                drop_reasons[why] = drop_reasons.get(why, 0) + 1
                continue
            record = {
                "prompt": spec.prompt_text,
                "chosen": chosen,
                "rejected": rejected,
                "meta": {
                    "char": spec.bot_name,
                    "reason": pick.reason,
                    "worst_idx": pick.worst_idx,
                    "n_candidates": len(cands),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_kept += 1

    log.info("wrote %d records (kept) / %d dropped to %s", n_kept, n_dropped, out_path)
    if drop_reasons:
        log.info("drop reasons: %s", drop_reasons)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="Llama-3.2-3B SFT checkpoint dir (produced by train_llama.py)")
    ap.add_argument("--data", required=True, help="path to sft_chai_aligned.json")
    ap.add_argument("--n", type=int, default=500, help="number of pairs to attempt")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
