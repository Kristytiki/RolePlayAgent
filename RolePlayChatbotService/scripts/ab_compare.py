"""A/B comparison: same prompts, RAG+GCA off vs on, write a markdown report.

Workflow:
1. Run all prompts against http://localhost:8000 (whatever config is running).
2. Save replies as JSON.
3. Re-run after restarting server with feature flag toggled.
4. Build EXAMPLES.md from the two captures.

Usage:
    uv run python scripts/ab_compare.py capture --tag on
    # ... restart server with EVERYTHING off ...
    uv run python scripts/ab_compare.py capture --tag off
    uv run python scripts/ab_compare.py report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

OUT_ROOT = Path(__file__).resolve().parent.parent / ".cache" / "ab"

PROMPTS = [
    ("mr_darcy",        "Mr. Darcy, your manners at the ball were terribly cold."),
    ("atticus_finch",   "Atticus, I'm afraid I might lose my temper in court tomorrow."),
    ("sherlock_holmes", "Sherlock, why do you bother helping people who hate you?"),
    ("scarlett_ohara",  "Scarlett, the war is here. What will you do?"),
]


def capture(tag: str, base: str = "http://127.0.0.1:8000") -> None:
    out = OUT_ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60) as c:
        for persona_id, prompt in PROMPTS:
            sid = c.post(f"{base}/chat/sessions",
                         json={"persona_id": persona_id, "user_name": "Reader"}).json()["session_id"]
            r = c.post(f"{base}/chat/sessions/{sid}/messages",
                       json={"message": prompt}).json()
            r["_prompt"] = prompt
            r["_persona_id"] = persona_id
            (out / f"{persona_id}.json").write_text(
                json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[{tag}] {persona_id}: scene_score={r.get('retrieved_scene_score')} chunks={r.get('retrieved_chunk_count')}")


def report(out_md: Path) -> None:
    on_dir = OUT_ROOT / "on"
    off_dir = OUT_ROOT / "off"
    if not on_dir.exists() or not off_dir.exists():
        raise SystemExit(f"missing capture(s); run `capture --tag on` and `capture --tag off` first")

    lines: list[str] = ["# A/B comparison — RAG+GCA off vs on", ""]
    lines.append("Same model (CHAI Guanaco), same persona, same prompt. The only difference between")
    lines.append("the two columns is whether persona-RAG (UtteranceIndex) and GCA scene retrieval")
    lines.append("are enabled.")
    lines.append("")

    for persona_id, prompt in PROMPTS:
        on = json.loads((on_dir / f"{persona_id}.json").read_text(encoding="utf-8"))
        off = json.loads((off_dir / f"{persona_id}.json").read_text(encoding="utf-8"))

        lines.append(f"## {persona_id}")
        lines.append("")
        lines.append(f"**Prompt:** _{prompt}_")
        lines.append("")
        lines.append("### Without RAG/GCA — only `profile_markdown` injected")
        lines.append("")
        lines.append("> " + (off.get("reply") or "").replace("\n", "\n> "))
        lines.append("")
        lines.append(f"_(scene_score={off.get('retrieved_scene_score')!r}, chunks={off.get('retrieved_chunk_count')})_")
        lines.append("")
        lines.append("### With RAG + GCA — utterance retrieval + scene match + S·A·T format")
        lines.append("")
        lines.append("> " + (on.get("reply") or "").replace("\n", "\n> "))
        lines.append("")
        meta = []
        if on.get("retrieved_scene_score") is not None:
            meta.append(f"scene_score={on['retrieved_scene_score']:.3f}")
        meta.append(f"chunks={on.get('retrieved_chunk_count', 0)}")
        if on.get("action"):
            meta.append(f"action={on['action']!r}")
        if on.get("thought"):
            meta.append(f"thought={on['thought']!r}")
        lines.append("_(" + ", ".join(meta) + ")_")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--tag", required=True, choices=["on", "off"])
    cap.add_argument("--base", default="http://127.0.0.1:8000")
    rep = sub.add_parser("report")
    rep.add_argument("--out", default="EXAMPLES.md")
    args = ap.parse_args()

    if args.cmd == "capture":
        capture(args.tag, args.base)
    else:
        report(Path(args.out))


if __name__ == "__main__":
    main()
