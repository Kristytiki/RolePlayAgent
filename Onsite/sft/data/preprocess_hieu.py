"""
Convert Hieunguyenminh/roleplay (anime/fandom RP, 5755 records) to Chai-aligned
ShareGPT format.

Source schema: name, description, text. The `text` field is one block with
<|system|>, <|user|>, <|assistant|> tags separated by </s>. Most records have
~5-10 turns of QA-style RP between user and the character.

Strip leading "Character:" prefixes from assistant turns and collapse newlines
so output stays single-line at Chai inference (stopping_words=['\n']).
"""
import json
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd

random.seed(42)

HERE = Path(__file__).parent.parent  # sft/
SRC = HERE / "assets/Hieunguyenminh-roleplay/data/train-00000-of-00001.parquet"
OUT = HERE / "assets/sft_chai_hieu.json"
N_KEEP = int(os.environ.get("N_KEEP", "5000"))
MAX_TURNS = 16
MAX_CHARS_PER_TURN = 800

LEADING_NAME_RE = re.compile(r"^[A-Z][A-Za-z0-9 .'\-]{0,60}:\s*")


def clean(value: str, strip_name: bool = False) -> str:
    s = value
    if strip_name:
        s = LEADING_NAME_RE.sub("", s, count=1)
    s = re.sub(r"\s*\n+\s*", " ", s)
    s = re.sub(r" {2,}", " ", s).strip()
    if len(s) > MAX_CHARS_PER_TURN:
        s = s[:MAX_CHARS_PER_TURN].rsplit(" ", 1)[0] + "..."
    return s


def parse_text(text: str) -> list[tuple[str, str]]:
    """Return list of (role, content) where role in {system, user, assistant}."""
    turns = []
    # Pattern: <|role|>content</s>  (also handle missing trailing </s>)
    pat = re.compile(r"<\|(system|user|assistant)\|>(.*?)(?=<\|(?:system|user|assistant)\|>|$)", re.DOTALL)
    for m in pat.finditer(text):
        role = m.group(1)
        body = m.group(2).rstrip().rstrip("</s>").strip()
        if body:
            turns.append((role, body))
    return turns


def build_system(rec: dict, sys_text: str) -> str:
    name = (rec.get("name") or "Character").strip()
    desc = (rec.get("description") or "").strip()
    parts = [f"You are {name}."]
    if desc:
        parts.append(desc)
    if sys_text:
        parts.append(sys_text)
    return "\n\n".join(parts)


def main():
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    df = pd.read_parquet(SRC)
    print(f"loaded {len(df)} records")

    out = []
    n_skip_short = 0
    n_skip_no_assistant = 0

    for _, row in df.iterrows():
        rec = row.to_dict()
        text = rec.get("text") or ""
        turns = parse_text(text)
        if len(turns) < 3:
            n_skip_short += 1
            continue

        system_chunks = [t[1] for t in turns if t[0] == "system"]
        sys_combined = "\n\n".join(system_chunks)
        system = build_system(rec, sys_combined)

        msgs = [{"from": "system", "value": system}]
        msgs.append({"from": "human", "value": "===Conversation Start===\n\n"})

        turn_count = 0
        for role, body in turns:
            if role == "system":
                continue
            if turn_count >= MAX_TURNS:
                break
            if role == "user":
                cleaned = clean(body, strip_name=False)
                target = "human"
            else:
                cleaned = clean(body, strip_name=True)
                target = "assistant"
            if not cleaned:
                continue
            if msgs and msgs[-1]["from"] == target:
                msgs[-1]["value"] = (msgs[-1]["value"].rstrip() + " " + cleaned).strip()
            else:
                msgs.append({"from": target, "value": cleaned})
            turn_count += 1

        if not any(m["from"] == "assistant" for m in msgs):
            n_skip_no_assistant += 1
            continue

        out.append({"conversations": msgs})

    print(f"kept {len(out)}; skipped short={n_skip_short} no_assistant={n_skip_no_assistant}")

    random.shuffle(out)
    if N_KEEP and len(out) > N_KEEP:
        out = out[:N_KEEP]
        print(f"truncated to {len(out)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"wrote {OUT}")

    if not out:
        sys.exit("0 samples — abort")

    print("\n--- sample[0] ---")
    for m in out[0]["conversations"][:6]:
        v = m["value"]
        if len(v) > 250:
            v = v[:250] + "..."
        print(f"[{m['from']:9}] {v}")


if __name__ == "__main__":
    main()
