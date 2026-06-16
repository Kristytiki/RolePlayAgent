"""Inspect CoSER and ChatHaruhi datasets to enumerate available characters.

Run:
    uv run python scripts/inspect_datasets.py

Outputs two files:
    .cache/coser_characters.json     — every character → list of books they appear in
    .cache/haruhi_characters.json    — agent_role_name_en → row count

Why: the HF previews are unreliable. Before we commit to a 17-character
roster we need to know which characters are actually present in the data.
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def inspect_coser() -> None:
    """Download CoSER's per-book JSON files and enumerate characters.

    Strategy:
    - snapshot_download only the `full/` directory (still ~2GB but skips train/SFT)
    - walk every per-book JSON, collect distinct character names from
      character_profiles dict + key_characters list
    - emit a sorted index: character → [books]
    """
    logger.info("downloading Neph0s/CoSER full/ snapshot (this can take a few minutes)...")
    repo_path = snapshot_download(
        repo_id="Neph0s/CoSER",
        repo_type="dataset",
        allow_patterns=["full/*.json"],
        cache_dir=str(CACHE_DIR / "hf"),
    )
    full_dir = Path(repo_path) / "full"
    if not full_dir.exists():
        raise RuntimeError(f"expected full/ at {full_dir}")

    character_to_books: dict[str, set[str]] = defaultdict(set)
    book_files = sorted(full_dir.glob("*.json"))
    logger.info("found %d book files", len(book_files))

    for book_path in book_files:
        book_name = book_path.stem
        try:
            data = json.loads(book_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("skipping %s: %s", book_name, exc)
            continue

        # CoSER 'full' files: top-level dict with 'plots' / 'character_profiles' etc.
        # Different versions of the file may use slightly different shapes; be defensive.
        names: set[str] = set()
        if isinstance(data, dict):
            cp = data.get("character_profiles")
            if isinstance(cp, dict):
                names.update(cp.keys())
            for plot in data.get("plots", []) or []:
                if not isinstance(plot, dict):
                    continue
                for kc in plot.get("key_characters", []) or []:
                    if isinstance(kc, dict) and isinstance(kc.get("name"), str):
                        names.add(kc["name"])
                    elif isinstance(kc, str):
                        names.add(kc)
                for d in plot.get("dialogues", []) or []:
                    if isinstance(d, dict) and isinstance(d.get("character"), str):
                        names.add(d["character"])
        for n in names:
            character_to_books[n].add(book_name)

    out = {
        name: sorted(books) for name, books in sorted(character_to_books.items(), key=lambda kv: kv[0].lower())
    }
    out_path = CACHE_DIR / "coser_characters.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %d characters to %s", len(out), out_path)


def inspect_haruhi() -> None:
    """Enumerate agent_role_name_en values + row counts in ChatHaruhi 54K."""
    from datasets import load_dataset  # imported lazily to avoid heavy import on coser-only runs

    logger.info("loading silk-road/ChatHaruhi-54K-Role-Playing-Dialogue ...")
    ds = load_dataset(
        "silk-road/ChatHaruhi-54K-Role-Playing-Dialogue",
        cache_dir=str(CACHE_DIR / "hf"),
    )
    train = ds["train"]
    counts: Counter[str] = Counter(train["agent_role_name_en"])
    out = dict(counts.most_common())
    out_path = CACHE_DIR / "haruhi_characters.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %d characters to %s", len(out), out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Don't fail the whole run if one dataset is offline.
    try:
        inspect_coser()
    except Exception as exc:
        logger.error("CoSER inspection failed: %s", exc)
    try:
        inspect_haruhi()
    except Exception as exc:
        logger.error("Haruhi inspection failed: %s", exc)


if __name__ == "__main__":
    # Some HF servers serve large files faster with hf_transfer.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    main()
