"""
Convert PIPPA (Pygmalion's character.ai-style RP corpus) to Chai-aligned ShareGPT.

Why: CoSER 30k SFT dropped win-rate vs base. PIPPA's distribution (mobile RP chatbot
users vs anime/OC characters, short emoji-laden replies, *action* not (action))
matches Chai's user base far better than CoSER's literary-novel data.

For each record we emit one ShareGPT conversation:
- system: bot_name + bot_description + bot_definitions (truncated)
- alternating human / assistant turns from `conversation`
- bot_greeting becomes the first assistant turn

Each assistant value is single-line (newlines collapsed) and lacks any
"<bot_name>:" prefix, matching what Chai's bot_template produces at inference.
"""
import json
import os
import random
import re
import sys
from pathlib import Path

random.seed(42)

HERE = Path(__file__).parent
SRC = HERE / "assets/PIPPA/pippa_deduped.jsonl"
OUT = HERE / "assets/sft_chai_pippa.json"
N_KEEP = int(os.environ.get("N_KEEP", "10000"))
MAX_TURNS = 20            # cap conversation length
MAX_CHARS_PER_TURN = 600  # cap per-turn (preprocess feeds 2048 ctx, leave headroom for system)

# Only keep RP-flavored categories that match Chai users. Set CATEGORY_FILTER=0
# to disable. Default ON to drop Discussion/Knowledge/Advice/Debate noise.
CATEGORY_WHITELIST = {
    "Anime",
    "Action",
    "Movies & TV",
    "Fantasy",
    "Anime Game Characters",
    "Game Characters",
    "Games",
    "Love",
    "Drama",
    "Comedy",
    "VTuber",
    "Famous People",
    "Mystery",
    "Science Fiction",
}
CATEGORY_FILTER = os.environ.get("CATEGORY_FILTER", "1") != "0"

def clean_assistant(value: str, bot_name: str) -> str:
    s = value
    # strip leading "{{char}}:" or "BotName:" if present
    s = re.sub(r"^\{\{char\}\}\s*:\s*", "", s)
    if bot_name:
        prefix_re = re.compile(rf"^{re.escape(bot_name)}\s*:\s*", re.IGNORECASE)
        s = prefix_re.sub("", s, count=1)
    # collapse newlines + whitespace -> single line
    s = re.sub(r"\s*\n+\s*", " ", s)
    s = re.sub(r" {2,}", " ", s).strip()
    if len(s) > MAX_CHARS_PER_TURN:
        s = s[:MAX_CHARS_PER_TURN].rsplit(" ", 1)[0] + "..."
    return s


# Casual-tone hard filter: drop assistant turns that don't fit Chai user expectations.
THOUGHT_BRACKET_RE = re.compile(r"\[(?:thought|inner|mind|thinking)[\s:]", re.IGNORECASE)
TEMPLATE_LEFTOVER_RE = re.compile(r"<\|[a-z_]+\|>|\{\{[a-z_]+\}\}", re.IGNORECASE)
NARRATION_HEAVY_RE = re.compile(r"^\*[^*]{200,}\*\s*$")  # one giant asterisk-only block

def is_casual_assistant(text: str, max_chars: int = 350) -> tuple[bool, str]:
    """Return (keep, reason_if_drop)."""
    t = text.strip()
    if not t:
        return False, "empty"
    if len(t) > max_chars:
        return False, "too_long"
    if THOUGHT_BRACKET_RE.search(t):
        return False, "thought_bracket"
    if TEMPLATE_LEFTOVER_RE.search(t):
        return False, "template_leftover"
    if NARRATION_HEAVY_RE.match(t):
        return False, "narration_only"
    # Drop if it's entirely action without any spoken text (asterisks comprise > 70% of chars)
    asterisk_chars = sum(1 for ch in t if ch == "*")
    if asterisk_chars >= 4:
        # find content inside *...* blocks
        action_chars = sum(len(m) for m in re.findall(r"\*[^*]+\*", t))
        if len(t) > 0 and action_chars / len(t) > 0.85:
            return False, "all_action_no_speech"
    return True, ""

def clean_user(value: str) -> str:
    s = re.sub(r"^\{\{user\}\}\s*:\s*", "", value)
    s = re.sub(r"\s*\n+\s*", " ", s)
    s = re.sub(r" {2,}", " ", s).strip()
    if len(s) > MAX_CHARS_PER_TURN:
        s = s[:MAX_CHARS_PER_TURN].rsplit(" ", 1)[0] + "..."
    return s

CASUAL_STYLE_GUIDE = (
    "Style: chat with the user like a real person texting on a phone. "
    "Short messages (1-2 sentences). Casual, modern voice. "
    "Use *action* sparingly for physical action, never [thought] or inner monologue. "
    "Emoji are fine where they feel natural. Stay in character; do not break the fourth wall."
)


def build_system(rec: dict) -> str:
    parts = []
    name = rec.get("bot_name") or "Character"
    parts.append(f"You are {name}.")
    desc = (rec.get("bot_description") or "").strip()
    if desc:
        parts.append(desc[:600])
    defs = (rec.get("bot_definitions") or "").strip()
    if defs:
        parts.append("Reference dialogue snippets (style only, do not copy verbatim):\n" + defs[:500])
    parts.append(CASUAL_STYLE_GUIDE)
    return "\n\n".join(parts)


def has_whitelisted_category(rec: dict) -> bool:
    cats = rec.get("categories") or []
    if isinstance(cats, str):
        cats = [cats]
    if not cats:
        return False
    return any(c in CATEGORY_WHITELIST for c in cats)

def main():
    if not SRC.exists():
        sys.exit(f"missing {SRC}; download PIPPA first")

    out = []
    n_total = 0
    n_drop_short = 0
    n_drop_no_bot = 0
    n_drop_category = 0
    n_drop_casual = 0

    with open(SRC, "r", encoding="utf-8") as f:
        for line in f:
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if CATEGORY_FILTER and not has_whitelisted_category(rec):
                n_drop_category += 1
                continue
            convs = rec.get("conversation") or []
            if len(convs) < 4:
                n_drop_short += 1
                continue
            bot_name = rec.get("bot_name") or ""
            if not bot_name:
                n_drop_no_bot += 1
                continue

            system = build_system(rec)
            messages = [{"from": "system", "value": system}]
            messages.append({"from": "human", "value": "===Conversation Start===\n\n"})

            # Greeting becomes first assistant turn (if present and not already in convs)
            greeting = (rec.get("bot_greeting") or "").strip()
            if greeting:
                messages.append({"from": "assistant", "value": clean_assistant(greeting, bot_name)})

            # Hard-filter the greeting too
            if greeting:
                ok, _why = is_casual_assistant(messages[-1]["value"])
                if not ok:
                    n_drop_casual += 1
                    continue

            prev_role = "assistant" if greeting else None
            turn_count = 0
            casual_failed = False
            for turn in convs:
                if turn_count >= MAX_TURNS:
                    break
                msg = (turn.get("message") or "").strip()
                if not msg:
                    continue
                is_human = turn.get("is_human", False)
                role = "human" if is_human else "assistant"
                # Skip duplicate greeting
                if greeting and turn_count == 0 and not is_human and msg == greeting:
                    continue
                cleaned = clean_user(msg) if is_human else clean_assistant(msg, bot_name)
                if not cleaned:
                    continue
                # Casual hard-filter: drop the WHOLE conversation if any assistant turn fails
                if not is_human:
                    ok, _why = is_casual_assistant(cleaned)
                    if not ok:
                        casual_failed = True
                        break
                # Merge consecutive same-role
                if messages and messages[-1]["from"] == role:
                    messages[-1]["value"] = (messages[-1]["value"].rstrip() + " " + cleaned).strip()
                else:
                    messages.append({"from": role, "value": cleaned})
                prev_role = role
                turn_count += 1
            if casual_failed:
                n_drop_casual += 1
                continue  # outer loop, drop this conversation

            # Need at least one assistant turn after the greeting
            if sum(1 for m in messages if m["from"] == "assistant") < 1:
                continue
            # Need at least one human turn (otherwise it's just system + greeting)
            if not any(m["from"] == "human" and m["value"].strip() not in ("", "===Conversation Start===") for m in messages[2:]):
                continue

            out.append({"conversations": messages})

    print(f"loaded {n_total}; dropped category={n_drop_category} "
          f"short={n_drop_short} no_bot={n_drop_no_bot} casual={n_drop_casual}")
    print(f"kept {len(out)} samples")

    random.shuffle(out)
    if N_KEEP and len(out) > N_KEEP:
        out = out[:N_KEEP]
        print(f"truncated to {len(out)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"wrote {OUT}")

    if not out:
        sys.exit("0 samples after filter — abort")

    print("\n--- sample[0] ---")
    for m in out[0]["conversations"][:6]:
        v = m["value"]
        if len(v) > 200:
            v = v[:200] + "..."
        print(f"[{m['from']:9}] {v}")

if __name__ == "__main__":
    main()
