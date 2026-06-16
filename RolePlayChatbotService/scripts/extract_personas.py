"""Extract raw data for our 17 target personas from the cached CoSER and
ChatHaruhi datasets, and write one JSON file per persona under data/raw/.

Run after `inspect_datasets.py` has populated .cache/. This script only reads
local files; no network calls.

    uv run python scripts/extract_personas.py

Output layout:
    src/roleplaychatbotservice/personas/data/raw/
        coser/
            mr_darcy.json
            heathcliff.json
            ...
        haruhi/
            qiaofeng.json
            haruhi.json
            ...

Each CoSER per-persona file shape (mirrors CoSER's `character_datasets[name]`,
which is exactly the GCA-ready bundle the paper recommends):
    {
        "name": "Fitzwilliam Darcy",
        "books": ["Pride and Prejudice", "The Complete Novels"],
        "profile_markdown": "**Name:** Fitzwilliam Darcy\\n**Background:** ...",
        "plots":        [{"name", "description", "experience"}],   # per-plot character role
        "conversations":[{"name", "thought"}],                     # per-scene inner thought
        "utterances":   [{"character", "message"}],                # actual lines spoken (with embedded [thought])
        "scenes":       [                                           # scenes the character participates in
            {"book": "...", "scenario": "...", "topic": "...",
             "key_characters": [...], "dialogues": [...]}
        ]
    }

Each Haruhi per-persona file shape (aligned with CoSER's structure where
possible — Haruhi has no profile/plot data, only utterances + Q&A pairs):
    {
        "agent_role_name_en": "qiaofeng",
        "agent_role": "乔峰",
        "display_name": "乔峰 (Qiao Feng)",
        "profile_markdown": "",         # filled in manually via manual_profiles.json
        "utterances":  [
            {"character": "乔峰", "message": "...", "source": "story"|"synthesized"}
        ],
        "scenes": [
            # Each ChatHaruhi row → one mini scene with two turns
            {"scenario": "synth: 王语嫣 asked about ...", "topic": "...",
             "dialogues": [{"character": "王语嫣", "message": "..."},
                           {"character": "乔峰",   "message": "..."}]}
        ]
    }
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache"
RAW_OUT = ROOT / "src" / "roleplaychatbotservice" / "personas" / "data" / "raw"


# ─── target rosters ─────────────────────────────────────────────────────────

# CoSER targets: dataset key → (filename slug, display name)
COSER_TARGETS: dict[str, tuple[str, str]] = {
    "Fitzwilliam Darcy":         ("mr_darcy",          "Mr. Darcy"),
    "Heathcliff":                ("heathcliff",        "Heathcliff"),
    "Sherlock Holmes":           ("sherlock_holmes",   "Sherlock Holmes"),
    "Jay Gatsby":                ("jay_gatsby",        "Jay Gatsby"),
    "Hermione Granger":          ("hermione_granger",  "Hermione Granger"),
    "Elizabeth Bennet":          ("elizabeth_bennet",  "Elizabeth Bennet"),
    "Anna Arkadyevna Karenina":  ("anna_karenina",     "Anna Karenina"),
    "Atticus Finch":             ("atticus_finch",     "Atticus Finch"),
    "Scarlett O'Hara":           ("scarlett_ohara",    "Scarlett O'Hara"),
}

# ChatHaruhi targets: dataset key → (filename slug, display name)
HARUHI_TARGETS: dict[str, tuple[str, str]] = {
    "qiaofeng":     ("qiaofeng",     "乔峰 (Qiao Feng)"),
    "weixiaobao":   ("weixiaobao",   "韦小宝 (Wei Xiaobao)"),
    "wangyuyan":    ("wangyuyan",    "王语嫣 (Wang Yuyan)"),
    "tongxiangyu":  ("tongxiangyu",  "佟湘玉 (Tong Xiangyu)"),
    "hutao":        ("hutao",        "胡桃 (Hu Tao)"),
    "zhongli":      ("zhongli",      "钟离 (Zhongli)"),
    "Sheldon":      ("sheldon",      "Sheldon Cooper"),
    "haruhi":       ("haruhi",       "凉宫春日 (Haruhi Suzumiya)"),
}


def _resolve_coser_full_dir() -> Path:
    """Find the CoSER full/ directory inside the HF snapshot cache."""
    hf_root = CACHE_DIR / "hf"
    if not hf_root.exists():
        raise RuntimeError(f"missing {hf_root}; run inspect_datasets.py first")
    matches = list(hf_root.rglob("Neph0s/CoSER/snapshots/*/full"))
    if not matches:
        # Try alternative layout (snapshot_download writes nested dirs)
        matches = [p for p in hf_root.rglob("full") if "CoSER" in str(p)]
    if not matches:
        raise RuntimeError(f"could not locate CoSER full/ under {hf_root}")
    return matches[0]


def extract_coser() -> None:
    """CoSER stores per-character GCA-ready bundles under
    `character_datasets[name]`. We pull those verbatim and additionally
    capture the scenes (top-level `plots[*].conversation[*]`) the character
    participates in so the loader has scenario/topic context for retrieval.
    """
    full_dir = _resolve_coser_full_dir()
    out_dir = RAW_OUT / "coser"
    out_dir.mkdir(parents=True, exist_ok=True)
    book_files = sorted(full_dir.glob("*.json"))
    logger.info("scanning %d CoSER book files for %d targets ...",
                len(book_files), len(COSER_TARGETS))

    aggregated: dict[str, dict[str, Any]] = {
        key: {
            "name": name,
            "slug": slug,
            "books": [],
            "profile_markdown": "",
            "plots": [],
            "conversations": [],
            "utterances": [],
            "scenes": [],
        }
        for key, (slug, name) in COSER_TARGETS.items()
    }

    for book_file in book_files:
        book_name = book_file.stem
        try:
            data = json.loads(book_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("skip %s: %s", book_name, exc)
            continue
        if not isinstance(data, dict):
            continue

        char_datasets = data.get("character_datasets") or {}
        if not isinstance(char_datasets, dict):
            char_datasets = {}

        present_targets = [key for key in aggregated if key in char_datasets]
        if not present_targets:
            continue

        for key in present_targets:
            agg = aggregated[key]
            if book_name not in agg["books"]:
                agg["books"].append(book_name)
            cd = char_datasets[key]

            # Profile is markdown — keep the longest version we see across books.
            prof = cd.get("profile") or ""
            if isinstance(prof, str) and len(prof) > len(agg["profile_markdown"]):
                agg["profile_markdown"] = prof

            for field in ("plots", "conversations", "utterances"):
                items = cd.get(field) or []
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            agg[field].append({"book": book_name, **it})

            # Scenes: walk plots[*].conversation[*] for ones the character
            # participates in (named in key_characters or speaks in dialogues).
            for plot in (data.get("plots") or []):
                if not isinstance(plot, dict):
                    continue
                for conv in (plot.get("conversation") or []):
                    if not isinstance(conv, dict):
                        continue
                    kc_names = {
                        kc["name"] for kc in conv.get("key_characters", []) or []
                        if isinstance(kc, dict) and isinstance(kc.get("name"), str)
                    }
                    speaker_names = {
                        d["character"] for d in conv.get("dialogues") or []
                        if isinstance(d, dict) and isinstance(d.get("character"), str)
                    }
                    if key not in kc_names and key not in speaker_names:
                        continue
                    agg["scenes"].append({
                        "book": book_name,
                        "chapter": plot.get("chapter"),
                        "plot_summary": plot.get("summary"),
                        "scenario": conv.get("scenario"),
                        "topic": conv.get("topic"),
                        "key_characters": conv.get("key_characters") or [],
                        "dialogues": conv.get("dialogues") or [],
                    })

    written = 0
    for key, agg in aggregated.items():
        has_data = bool(agg["profile_markdown"] or agg["utterances"] or agg["scenes"])
        if not has_data:
            logger.warning("no data found for CoSER target %r — skipped", key)
            continue
        slug = agg.pop("slug")
        out_path = out_dir / f"{slug}.json"
        out_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "wrote %s → profile=%dch, plots=%d, conv-thoughts=%d, utt=%d, scenes=%d, books=%d",
            slug, len(agg["profile_markdown"]),
            len(agg["plots"]), len(agg["conversations"]),
            len(agg["utterances"]), len(agg["scenes"]), len(agg["books"]),
        )
        written += 1
    logger.info("CoSER extraction: %d/%d personas written", written, len(COSER_TARGETS))


def extract_haruhi() -> None:
    """Map ChatHaruhi's flat (Q,A) rows into the same shape as CoSER (minus
    the fields ChatHaruhi doesn't carry: profile/plots/conversation-thoughts).

    Each row becomes:
      - one entry in `utterances` (the agent_response, tagged with question_source)
      - one entry in `scenes` (the (Q,A) pair as a 2-turn mini-dialogue)
    """
    from datasets import load_dataset  # lazy import

    out_dir = RAW_OUT / "haruhi"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading ChatHaruhi from local cache ...")
    ds = load_dataset(
        "silk-road/ChatHaruhi-54K-Role-Playing-Dialogue",
        cache_dir=str(CACHE_DIR / "hf"),
    )["train"]

    by_key: dict[str, list[dict[str, Any]]] = {k: [] for k in HARUHI_TARGETS}
    native_names: dict[str, str] = {}
    for row in ds:
        key = row["agent_role_name_en"]
        if key not in by_key:
            continue
        by_key[key].append(row)
        native_names.setdefault(key, row["agent_role"])

    for key, (slug, display) in HARUHI_TARGETS.items():
        rows = by_key[key]
        if not rows:
            logger.warning("no rows found for Haruhi target %r — skipped", key)
            continue

        agent_native = native_names.get(key, key)
        utterances: list[dict[str, Any]] = []
        scenes: list[dict[str, Any]] = []
        for row in rows:
            user_role = row.get("user_role") or "User"
            q = row.get("user_question") or ""
            a = row.get("agent_response") or ""
            src = row.get("question_source") or "synthesized"
            if not a:
                continue
            utterances.append({
                "character": agent_native,
                "message": a,
                "source": src,
            })
            scenes.append({
                "scenario": f"{user_role} addresses {agent_native}",
                "topic": q[:80],
                "source": src,
                "dialogues": [
                    {"character": user_role, "message": q},
                    {"character": agent_native, "message": a},
                ],
            })
            # Some rows carry multi-turn continuations.
            for extra in row.get("more_dialogues") or []:
                if not isinstance(extra, dict):
                    continue
                ea = extra.get("agent_response")
                if ea:
                    utterances.append({
                        "character": agent_native, "message": ea, "source": src,
                    })

        out_path = out_dir / f"{slug}.json"
        out_path.write_text(
            json.dumps({
                "agent_role_name_en": key,
                "agent_role": agent_native,
                "display_name": display,
                "profile_markdown": "",  # filled in by manual_profiles.json
                "utterances": utterances,
                "scenes": scenes,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        story = sum(1 for u in utterances if u["source"] == "story")
        synth = len(utterances) - story
        logger.info(
            "wrote %s (%s) → utt=%d (story=%d, synth=%d), scenes=%d",
            slug, agent_native, len(utterances), story, synth, len(scenes),
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    try:
        extract_coser()
    except Exception as exc:
        logger.exception("CoSER extraction failed: %s", exc)
    try:
        extract_haruhi()
    except Exception as exc:
        logger.exception("Haruhi extraction failed: %s", exc)


if __name__ == "__main__":
    main()
