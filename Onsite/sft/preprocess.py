"""
Preprocess CoSER ShareGPT data so it matches the Chaiverse inference format.

Chaiverse will assemble prompts as:
  ...<|im_start|>assistant
  {bot_name}:[model generates here, stops at \n]

CoSER training samples have assistant.value like:
  "Ron Weasley: [thought] (action) speech\n\n"

If we train on the raw value the model learns to emit "Ron Weasley: ..." again,
which gets concatenated after the prompt's "{bot_name}:" -> double prefix.
The \n\n inside the value also gets cut by Chai's stopping_words=['\n'],
so multi-paragraph outputs are wasted.

Fix: per assistant message,
  1) strip leading "<character>: " (any character name)
  2) collapse all internal whitespace newlines to spaces (single line)
  3) trim
This way assistant target is just "[thought] (action) speech" inline.
"""
import json
import os
import random
import re
import sys
from pathlib import Path

random.seed(42)

HERE = Path(__file__).parent
SRC = HERE / "assets/CoSER/train/sft_conversations_sharegpt.json"
OUT = HERE / "assets/sft_chai_aligned.json"
N_KEEP = int(os.environ.get("N_KEEP", "100000"))

# Strip leading "Name: " or "Name Foo: " (allow letters, spaces, apostrophes, hyphens, dots).
LEADING_NAME_RE = re.compile(r"^[A-Z][A-Za-z0-9 .'\-]{0,60}:\s*")

def clean_assistant(value: str) -> str:
    s = value
    # Drop leading character prefix once.
    s = LEADING_NAME_RE.sub("", s, count=1)
    # Collapse newlines + redundant whitespace.
    s = re.sub(r"\s*\n+\s*", " ", s)
    s = re.sub(r" {2,}", " ", s).strip()
    return s

def main():
    if not SRC.exists():
        sys.exit(f"missing {SRC}; run download.sh first")

    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"loaded {len(data)} raw conversations")

    cleaned = []
    dropped_empty = 0
    for sample in data:
        convs = sample.get("conversations", [])
        new_convs = []
        ok = True
        for m in convs:
            role = m.get("from")
            val = m.get("value", "")
            if role == "assistant":
                val = clean_assistant(val)
                if not val:
                    ok = False
                    break
            new_convs.append({"from": role, "value": val})
        if not ok:
            dropped_empty += 1
            continue
        # need at least one assistant turn
        if not any(m["from"] == "assistant" for m in new_convs):
            dropped_empty += 1
            continue
        cleaned.append({"conversations": new_convs})

    print(f"kept {len(cleaned)} / dropped {dropped_empty} empty")

    random.shuffle(cleaned)
    if N_KEEP and len(cleaned) > N_KEEP:
        cleaned = cleaned[:N_KEEP]
        print(f"truncated to {len(cleaned)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False)
    print(f"wrote {OUT}")

    if not cleaned:
        sys.exit("preprocess produced 0 samples — abort")

    # quick sanity print
    print("\n--- sample[0] assistant msgs (first 200 chars each) ---")
    for m in cleaned[0]["conversations"]:
        if m["from"] == "assistant":
            print(repr(m["value"][:200]))

if __name__ == "__main__":
    main()
