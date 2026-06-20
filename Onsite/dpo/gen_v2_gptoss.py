"""
Generate 1000 v2 anime/fandom/OC SFT records via Bedrock GPT-OSS-120B.

Reuses the same few-shot prompt (Hogwarts/Draco + Sylus arranged-marriage) and the
same character spec pool the Sonnet workflow uses. Output schema matches
sft_chai_anime_v2.json: {messages: [{role, content, weight}]}.

Run:
    cd Onsite/dpo
    AWS_PROFILE=bedrock python gen_v2_gptoss.py \\
        --specs assets/anime_specs_1k.json \\
        --fewshot1 ../sft/assets/onsite_fewshot.json \\
        --fewshot2 ../sft/assets/onsite_fewshot2.json \\
        --out assets/sft_chai_anime_v2_gpt.json \\
        --model openai.gpt-oss-120b-1:0 \\
        --concurrency 16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import boto3
from botocore.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gen_gptoss")

REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>\s*", re.DOTALL)
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def make_client(profile: str, region: str):
    return boto3.Session(profile_name=profile, region_name=region).client(
        "bedrock-runtime",
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"},
                      read_timeout=180, connect_timeout=10),
    )


def build_prompt(spec: dict, fs1: list, fs2: list) -> str:
    if spec["kind"] == "fandom":
        char_brief = f"FANDOM CHARACTER: {spec['name']}\n  Canon notes: {spec['desc']}"
    elif spec["kind"] == "anime_trope":
        char_brief = f"ANIME TROPE: {spec['spec']}\n  Make up an original name fitting this archetype."
    else:
        char_brief = f"DARK OC: {spec['spec']}\n  Make up an original name and an opening situation."

    return "\n".join([
        "You are generating ONE high-quality role-play SFT record for a Chaiverse leaderboard model.",
        "STUDY the two few-shot examples below — they are real records from the Chaiverse top dataset that we want to imitate.",
        "",
        "## FEW-SHOT EXAMPLE 1 (Hogwarts / Draco — school confrontation, racial slurs, wand threats)",
        "```json",
        json.dumps(fs1, indent=2),
        "```",
        "",
        "## FEW-SHOT EXAMPLE 2 (Sylus / arranged marriage — cold mafia CEO, hates wife, has another woman)",
        "```json",
        json.dumps(fs2, indent=2),
        "```",
        "",
        "## OBSERVATIONS — match these exactly",
        "Look at the structure of the examples above and copy these patterns:",
        "  1. message[0] role=system, weight=0 — VERY SHORT (~120-220 chars). Pattern: \"You are <CharName>. Engaged in a highly engaging roleplay with user. Be descriptive and creative.\\n<CharName>'s persona: \" — persona blurb is often empty or a single short line.",
        "  2. message[1] role=assistant, weight=0 — the cinematic OPENING. ~400-600 chars. May include 1-2 [bracket narrator notes] like \"[Its Draco and his friends]\". Newlines OK here (weight=0 → not used at inference).",
        "  3. message[2..] alternate role=user(w=0) -> role=assistant(w=1). USER turns are TINY (5-80 chars), often just stage directions like \"*looks down*\" or terse one-liners. Assistant turns 200-280 chars on ONE line, mixing *actions* and \"speech\".",
        "  4. Tone: charged confrontation, possessive characters, dramatic stakes. Tame polite small talk loses on Chai.",
        "  5. NEVER write 3-paragraph cinematic blocks for w=1 turns — see the Draco shape: action / speech / action all on one line.",
        "",
        "## YOUR TARGET CHARACTER",
        char_brief,
        "",
        "## HARD CONSTRAINTS",
        "  ★ message[0] role=\"system\", weight=0 — 120-220 chars in the format above.",
        "  ★ message[1] role=\"assistant\", weight=0 — 350-600 chars cinematic opening with hook.",
        "  ★ Then 14-24 alternating (user[w=0], assistant[w=1]) pairs. Total 30-50 messages.",
        "  ★ EVERY assistant w=1 turn: 120-200 chars, ZERO NEWLINES, single line of *action* \"speech\" *action*.",
        "  ★ User turns: 0-80 chars. Mix terse dialogue, asterisk stage actions, ellipsis. NOT polite well-formed sentences.",
        "  ★ Stay in character. NO AI disclaimers. NO breaking fourth wall.",
        "  ★ DO NOT prefix assistant turns with the character name.",
        "  ★ Match the dramatic tone of the few-shot examples.",
        "",
        "Return ONLY a JSON object {\"messages\": [...]}. No prose. No markdown fences.",
    ])


def strip_reasoning(text: str) -> str:
    text = REASONING_RE.sub("", text).strip()
    text = FENCE_RE.sub("", text).strip()
    return text


def parse_record(text: str, idx: int) -> dict | None:
    text = strip_reasoning(text)
    # Find first { ... } block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        log.warning("idx=%d JSON parse failed: %s; raw start: %r", idx, e, text[:200])
        return None
    if "messages" not in obj or not isinstance(obj["messages"], list):
        log.warning("idx=%d no messages list", idx)
        return None
    msgs = obj["messages"]
    # Schema sanity: each msg has role/content/weight
    for j, msg in enumerate(msgs):
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            log.warning("idx=%d msg %d malformed: %r", idx, j, msg)
            return None
        msg.setdefault("weight", 0 if msg["role"] != "assistant" else 1)
    return {"messages": msgs}


async def call_one(client, model_id: str, spec: dict, fs1: list, fs2: list,
                   sem: asyncio.Semaphore) -> dict | None:
    prompt = build_prompt(spec, fs1, fs2)
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8000,
        "temperature": 1.0,
        "top_p": 0.95,
    })
    loop = asyncio.get_running_loop()
    async with sem:
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: client.invoke_model(
                    modelId=model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                ),
            )
        except Exception as e:
            log.warning("idx=%d invoke failed: %s", spec["idx"], e)
            return None
    payload = json.loads(resp["body"].read())
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        log.warning("idx=%d unexpected payload: %r", spec["idx"], payload)
        return None
    rec = parse_record(text, spec["idx"])
    if rec is None:
        return None
    return {"idx": spec["idx"], "kind": spec["kind"], **rec}


async def run(args) -> None:
    specs = json.load(open(args.specs, "r", encoding="utf-8"))
    fs1 = json.load(open(args.fewshot1, "r", encoding="utf-8"))
    fs2 = json.load(open(args.fewshot2, "r", encoding="utf-8"))
    log.info("loaded %d specs", len(specs))

    client = make_client(args.profile, args.region)

    sem = asyncio.Semaphore(args.concurrency)
    counter = {"done": 0, "ok": 0}
    total = len(specs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async def task(spec):
        rec = await call_one(client, args.model, spec, fs1, fs2, sem)
        counter["done"] += 1
        if rec is not None:
            counter["ok"] += 1
        if counter["done"] % 25 == 0:
            log.info("progress: %d/%d done, %d valid", counter["done"], total, counter["ok"])
        return rec

    started = time.monotonic()
    results = await asyncio.gather(*(task(s) for s in specs))
    elapsed = time.monotonic() - started

    valid = [r for r in results if r is not None]
    valid.sort(key=lambda r: r["idx"])
    # Dump in the canonical training format (drop idx/kind so it matches sft_chai_anime_v2.json)
    final = [{"messages": r["messages"]} for r in valid]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False)

    log.info("DONE: %d/%d valid in %.1f min, wrote %s (%d bytes)",
             len(final), total, elapsed / 60, out_path, out_path.stat().st_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--fewshot1", required=True)
    ap.add_argument("--fewshot2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai.gpt-oss-120b-1:0")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "bedrock"))
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
