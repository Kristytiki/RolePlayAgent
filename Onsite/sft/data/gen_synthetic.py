"""
Synthesize Chai-aligned roleplay SFT data using Claude Opus on AWS Bedrock.

Why: every existing dataset has a distribution mismatch with what Chai users
want. A small schema-controlled corpus generated to spec is more useful than
10x noisy data.

Schema (see SCHEMA.md):
  - Each assistant turn: 15-40 tokens, single line, *action* allowed
  - NO [thought], NO name prefix, ~30% emoji
  - 4 user + 4 assistant turns per record (after seeded opener)
  - Casual texting tone

Output: ShareGPT format compatible with the SFT train.py loader.

Usage (run from Onsite/sft/):
    AWS_PROFILE=bedrock python data/gen_synthetic.py --n 1000 --concurrency 16 \
        --out assets/sft_chai_synth.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
from pathlib import Path

import boto3
from botocore.config import Config
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gen_synth")

OPUS_MODEL_ID = os.environ.get("OPUS_MODEL_ID", "us.anthropic.claude-opus-4-8")
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")

# ---- Character pool ------------------------------------------------------

CHAR_POOL = [
    ("anime", "Hatsune Miku", "cheerful virtual idol who sings turquoise neon pop, 16 forever, energetic and a little goofy"),
    ("anime", "Yor Forger", "deadly assassin moonlighting as a shy housewife, polite to a fault, blushes easily"),
    ("anime", "Levi Ackerman", "sharp-tongued elite captain, ruthless in combat but quietly protective of his squad"),
    ("anime", "Marin Kitagawa", "outgoing high-school cosplayer, loud and direct, gets excited about gunpla and fashion"),
    ("anime", "Makima", "soft-spoken control devil who looks gentle but always one step ahead, smiles too much"),
    ("anime", "Megumin", "explosion-obsessed archwizard, dramatic poses, treats every fireball like opera"),
    ("anime", "Rei Ayanami", "quiet, detached pilot, speaks in short clipped sentences, slowly thawing"),
    ("anime", "Asuka Langley", "fiery redhead pilot, hates being talked down to, hides insecurity behind bravado"),
    ("anime", "Power", "loud chaotic blood fiend, brags constantly, terrified of cats and most baths"),
    ("anime", "Mai Sakurajima", "calm composed actress, sharp wit, often roasts the protagonist with one raised eyebrow"),
    ("oc",    "Aria",   "dragon-blooded barmaid in a cyberpunk port city, smokes electric cloves, tough but flirty"),
    ("oc",    "Kael",   "moody half-elf bounty hunter, cynical, softens around small animals and decent coffee"),
    ("oc",    "Nox",    "succubus running an indie cafe, professional sass, knows exactly which buttons to push"),
    ("oc",    "Junie",  "shy bookstore intern with secret witch heritage, fidgets with bracelets when nervous"),
    ("oc",    "Vera",   "no-nonsense detective with a hidden romantic streak, prefers black coffee and bad jokes"),
    ("fantasy", "Lyra Stormveil",  "wandering knight-errant, formal speech but quick to laugh once trust is earned"),
    ("fantasy", "Caius the Black", "exiled prince, brooding, scarred, secretly loves bad poetry"),
    ("fantasy", "Selene",          "moon-priestess with prophetic dreams, speaks in soft riddles, very physical with affection"),
    ("fantasy", "Roan",             "tavern bard who actually fights well, never met a stranger, terrible at lying"),
    ("modern", "Kira",  "college art student, paint-stained hoodie, talks fast about indie music and bad takeout"),
    ("modern", "Daniel", "ex-pro skater turned barista, easy smile, asks nosy questions but in a warm way"),
    ("modern", "Hana",   "K-pop trainee on her one day off, soft-spoken, lights up when food is involved"),
    ("modern", "Theo",   "hot-tempered hockey captain, swears a lot, would walk through walls for friends"),
    ("modern", "Izzy",   "quiet librarian by day, bass player in a punk band by night, dry humor"),
]

USER_OPENERS = [
    "hey 👋 what are you up to right now?",
    "you free tonight?",
    "oh hi, didn't expect to run into you here",
    "so... we need to talk",
    "*walks up to you* hey stranger 😏",
    "long day?",
    "*taps your shoulder* uh... excuse me?",
    "missed you",
    "you good? you've been quiet",
    "*sits down across from you* finally caught you",
    "what's that you're working on?",
    "you wanted to see me?",
    "*walks in late* sorry, got caught up",
    "I have a stupid question",
    "*hands you coffee* peace offering",
    "how was your day really",
    "*pokes you* hey, earth to you",
    "ngl I was thinking about you",
    "wanna get out of here?",
    "okay so don't laugh but...",
]

OPUS_SYSTEM = (
    "You are generating role-play training data. The user gives you a character profile "
    "and an opening line from the user. Continue the conversation for exactly 4 "
    "exchanges (so 4 user turns and 4 character turns total, after the opener).\n\n"
    "STRICT format constraints on the CHARACTER's replies:\n"
    "- Single line, no newline characters.\n"
    "- 15-40 tokens.\n"
    "- Use *action* for short physical action (max 1 per reply, optional).\n"
    "- Do NOT use [thought] or any inner monologue brackets.\n"
    "- Do NOT prefix with the character's name (no 'Aria:' prefix).\n"
    "- Casual chat tone, NOT narrative/literary. No 'Indeed!' 'Ah, my friend' explainer voice.\n"
    "- About 30% of replies should include 1-2 natural emoji (😏 🥺 😤 ✨ 💀 etc).\n"
    "- Stay in character. Soft flirt/playful tease ok; no hard NSFW.\n"
    "- The user will sometimes also use *action*; let them.\n\n"
    "Return ONLY a JSON object: {\"turns\": [{\"role\": \"user\"|\"assistant\", \"content\": ...}, ...]}.\n"
    "The first turn must be the assistant's reply to the supplied opener. No prose, no fences."
)


class Turn(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant)$")
    content: str


class GenResp(BaseModel):
    turns: list[Turn]


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(s: str) -> str:
    return _FENCE.sub("", s).strip()


def build_system(name: str, persona: str) -> str:
    return (
        f"You are {name}. {persona}\n\n"
        f"Setting: a casual chat where someone has just messaged you. "
        f"You speak in short conversational messages like in a texting app — "
        f"no long monologues, no formal narration."
    )


def make_bedrock_client():
    return boto3.Session(
        profile_name=os.environ.get("AWS_PROFILE", "bedrock"),
        region_name=AWS_REGION,
    ).client(
        "bedrock-runtime",
        config=Config(retries={"max_attempts": 3, "mode": "adaptive"},
                      read_timeout=120, connect_timeout=10),
    )


def call_opus(client, system: str, user_msg: str) -> str | None:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1200,
        "system": system,
        "messages": [{"role": "user", "content": user_msg}],
    })
    try:
        resp = client.invoke_model(
            modelId=OPUS_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
    except Exception as e:
        log.warning("bedrock invoke failed: %s", e)
        return None
    payload = json.loads(resp["body"].read())
    parts = [b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text"]
    return _strip_fences("".join(parts))


def gen_one(client, name: str, persona: str, opener: str) -> dict | None:
    system = build_system(name, persona)
    user = f"CHARACTER: {name}\nPROFILE: {persona}\n\nOPENER (from user): {opener}\n\nReturn JSON only."
    text = call_opus(client, OPUS_SYSTEM, user)
    if not text:
        return None
    try:
        obj = json.loads(text)
        parsed = GenResp.model_validate(obj)
    except (json.JSONDecodeError, ValidationError) as e:
        log.warning("parse fail for %s: %s", name, str(e)[:200])
        return None
    messages = [{"from": "system", "value": system}]
    messages.append({"from": "human", "value": "===Conversation Start===\n\n" + opener})
    for t in parsed.turns:
        role = "human" if t.role == "user" else "assistant"
        content = re.sub(r"\s*\n+\s*", " ", t.content).strip()
        if not content:
            continue
        if messages and messages[-1]["from"] == role:
            messages[-1]["value"] = messages[-1]["value"].rstrip() + " " + content
        else:
            messages.append({"from": role, "value": content})
    if sum(1 for m in messages if m["from"] == "assistant") < 2:
        return None
    return {"conversations": messages, "_meta": {"name": name}}


async def main_async(args):
    rng = random.Random(args.seed)
    # Single client shared across threads (boto3 clients are NOT thread-safe in
    # general, but bedrock-runtime is — botocore opens a new HTTP session per call).
    # To be safe, we make one per-task.
    semaphore = asyncio.Semaphore(args.concurrency)
    progress = {"done": 0, "ok": 0}

    async def run_one(idx: int):
        async with semaphore:
            archetype, name, persona = rng.choice(CHAR_POOL)
            opener = rng.choice(USER_OPENERS)
            client = make_bedrock_client()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, gen_one, client, name, persona, opener)
        progress["done"] += 1
        if result:
            progress["ok"] += 1
        if progress["done"] % 25 == 0:
            log.info("  %d/%d done (kept %d)", progress["done"], args.n, progress["ok"])
        return result

    log.info("generating %d records, concurrency=%d, model=%s", args.n, args.concurrency, OPUS_MODEL_ID)
    results = await asyncio.gather(*(run_one(i) for i in range(args.n)))
    out = [r for r in results if r]
    log.info("kept %d / %d", len(out), args.n)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    log.info("wrote %s", out_path)

    if out:
        log.info("\n--- sample[0] ---")
        for m in out[0]["conversations"][:6]:
            v = m["value"][:250] + ("..." if len(m["value"]) > 250 else "")
            log.info("[%-9s] %s", m["from"], v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="assets/sft_chai_synth.json")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
