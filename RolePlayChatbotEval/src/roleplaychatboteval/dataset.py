"""Load + filter the CoSER test set."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CharacterEntry:
    name: str
    motivation: str = ""


@dataclass(frozen=True)
class DialogueTurn:
    character: str
    message: str


@dataclass(frozen=True)
class TestCase:
    """One held-out CoSER conversation = one evaluation unit."""
    book: str
    plot: str
    scenario: str
    topic: str
    key_characters: tuple[CharacterEntry, ...]
    major_characters: tuple[str, ...]
    speaking_characters: tuple[str, ...]
    character_profiles: dict[str, str]
    ground_truth: tuple[DialogueTurn, ...]


def _resolve_test_path(cache_root: str | Path) -> Path:
    cache_root = Path(cache_root)
    matches = list(cache_root.rglob("test/test_set.json"))
    if not matches:
        raise FileNotFoundError(
            f"CoSER test set not found under {cache_root}; "
            f"download via `huggingface_hub.snapshot_download(... allow_patterns=['test/test_set.json'])`"
        )
    return matches[0]


def load_test_cases(cache_root: str | Path = ".cache/hf") -> list[TestCase]:
    path = _resolve_test_path(cache_root)
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[TestCase] = []
    for row in raw:
        kcs = tuple(
            CharacterEntry(
                name=(kc.get("name") or "").strip(),
                motivation=(kc.get("motivation") or "").strip(),
            )
            for kc in row.get("key_characters") or []
            if isinstance(kc, dict) and kc.get("name")
        )
        dialogues = tuple(
            DialogueTurn(
                character=(d.get("character") or "").strip(),
                message=(d.get("message") or "").strip(),
            )
            for d in row.get("dialogues") or []
            if isinstance(d, dict)
        )
        cases.append(TestCase(
            book=row.get("book", ""),
            plot=row.get("plot", ""),
            scenario=row.get("scenario", ""),
            topic=row.get("topic", ""),
            key_characters=kcs,
            major_characters=tuple(row.get("major_characters") or []),
            speaking_characters=tuple(row.get("speaking_characters_w_env") or []),
            character_profiles=dict(row.get("character_profiles") or {}),
            ground_truth=dialogues,
        ))
    logger.info("loaded %d CoSER test cases from %s", len(cases), path)
    return cases


def filter_to_personas(cases: list[TestCase], persona_names: set[str]) -> list[TestCase]:
    """Keep only cases where at least one of our personas is a major_character."""
    out = [c for c in cases if any(n in persona_names for n in c.major_characters)]
    logger.info("filtered %d → %d cases for %d personas", len(cases), len(out), len(persona_names))
    return out
