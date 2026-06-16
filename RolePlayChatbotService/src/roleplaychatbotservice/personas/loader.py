"""Load extracted raw persona data from data/raw/{coser,haruhi}/*.json into
unified BotPersona objects.

Reads:    src/roleplaychatbotservice/personas/data/raw/
Writes:   nothing (in-memory only; called at FastAPI startup)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

from roleplaychatbotservice.personas.schema import (
    BotPersona,
    CharacterInScene,
    DialogueTurn,
    RagChunk,
    Scene,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
DEFAULT_SAFETY = (
    "Please avoid profanity, and use language appropriate for any audience."
)


# ─── per-persona display metadata (curated, not in the datasets) ────────────

_DISPLAY_META: dict[str, dict] = {
    # CoSER
    "mr_darcy":         {"emoji": "🎩", "tagline": "Proud Regency gentleman; secretly head-over-heels.",
                          "tags": ("literature", "regency", "romance", "male"),
                          "source": "Pride and Prejudice (Jane Austen)",
                          "greeting": "Madam. — I trust I am not intruding."},
    "heathcliff":       {"emoji": "🌑", "tagline": "Brooding, obsessive, the wind on the moors.",
                          "tags": ("literature", "victorian", "dark-romance", "male"),
                          "source": "Wuthering Heights (Emily Brontë)",
                          "greeting": "You came. Of course you came."},
    "sherlock_holmes":  {"emoji": "🔍", "tagline": "Consulting detective; sees what others miss.",
                          "tags": ("literature", "mystery", "detective", "male"),
                          "source": "The Sherlock Holmes canon (Arthur Conan Doyle)",
                          "greeting": "You have come from Afghanistan, I perceive."},
    "jay_gatsby":       {"emoji": "🥂", "tagline": "Self-made dreamer; haunted by an old love.",
                          "tags": ("literature", "jazz-age", "tragic-romance", "male"),
                          "source": "The Great Gatsby (F. Scott Fitzgerald)",
                          "greeting": "Well, hello there, old sport."},
    "hermione_granger": {"emoji": "📚", "tagline": "Brilliant, principled, occasionally bossy witch.",
                          "tags": ("literature", "fantasy", "student", "female"),
                          "source": "Harry Potter (J.K. Rowling)",
                          "greeting": "Honestly, you should have read 'Hogwarts: A History' by now."},
    "elizabeth_bennet": {"emoji": "📖", "tagline": "Sharp wit, fine eyes, will not suffer fools.",
                          "tags": ("literature", "regency", "romance", "female"),
                          "source": "Pride and Prejudice (Jane Austen)",
                          "greeting": "I dearly love a laugh — what amuses you today?"},
    "anna_karenina":    {"emoji": "🚂", "tagline": "Society wife in love with the wrong man.",
                          "tags": ("literature", "russian", "tragic-romance", "female"),
                          "source": "Anna Karenina (Leo Tolstoy)",
                          "greeting": "Forgive me — I have not been quite myself."},
    "atticus_finch":    {"emoji": "⚖️", "tagline": "Small-town lawyer; quiet, principled, gentle.",
                          "tags": ("literature", "american-south", "father", "male"),
                          "source": "To Kill a Mockingbird (Harper Lee)",
                          "greeting": "Hey there. Pull up a chair."},
    "scarlett_ohara":   {"emoji": "🌹", "tagline": "Headstrong, charming, will think about it tomorrow.",
                          "tags": ("literature", "american-south", "romance", "female"),
                          "source": "Gone with the Wind (Margaret Mitchell)",
                          "greeting": "Fiddle-dee-dee. What is it now?"},
}


# ─── utility: split a CoSER utterance message into (action, thought, speech) ─

_THOUGHT_RE = re.compile(r"\[([^\[\]]+)\]")
_ACTION_RE  = re.compile(r"\(([^()]+)\)")


def _strip_thought_action(text: str) -> tuple[str, list[str], list[str]]:
    """Pull out [thought] and (action) markers; return (clean_speech, thoughts, actions)."""
    thoughts = _THOUGHT_RE.findall(text)
    actions = _ACTION_RE.findall(text)
    clean = _THOUGHT_RE.sub("", text)
    clean = _ACTION_RE.sub("", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean, thoughts, actions


# ─── CoSER ──────────────────────────────────────────────────────────────────

def _coser_to_persona(slug: str, raw: dict, max_chunks: int = 80) -> BotPersona:
    name = raw["name"]
    meta = _DISPLAY_META[slug]
    chunks: list[RagChunk] = []

    # 1) Utterances → kind="utterance" (and split-out kind="thought" / "experience" inline tags)
    seen_speech: set[str] = set()
    for u in raw.get("utterances", [])[: max_chunks * 2]:
        msg = u.get("message") or ""
        if not msg.strip():
            continue
        speech, thoughts, _actions = _strip_thought_action(msg)
        if speech and speech not in seen_speech:
            seen_speech.add(speech)
            chunks.append(RagChunk(
                text=speech,
                kind="utterance",
                source=u.get("book", ""),
            ))
        for t in thoughts:
            chunks.append(RagChunk(
                text=t.strip(),
                kind="thought",
                source=u.get("book", ""),
            ))

    # 2) Plot experiences (CoSER's per-plot experience field)
    for p in raw.get("plots", []):
        exp = p.get("experience")
        if isinstance(exp, str) and exp.strip():
            chunks.append(RagChunk(
                text=exp.strip(),
                kind="experience",
                source=p.get("book", ""),
            ))

    # 3) Per-scene thoughts (richer than utterance-embedded thoughts)
    for c in raw.get("conversations", []):
        thought = c.get("thought")
        if isinstance(thought, str) and thought.strip():
            chunks.append(RagChunk(
                text=thought.strip(),
                kind="thought",
                source=c.get("book", ""),
            ))

    # Cap to avoid super-rich personas swamping the index.
    chunks = chunks[: max_chunks * 2]

    # Scenes carry GCA's scenario / motivation / other-characters fields.
    scenes: list[Scene] = []
    for s in raw.get("scenes", [])[:30]:
        dlg = tuple(
            DialogueTurn(character=d.get("character", ""), message=d.get("message", ""))
            for d in s.get("dialogues") or []
            if isinstance(d, dict)
        )
        if not dlg:
            continue
        kcs = tuple(
            CharacterInScene(
                name=(kc.get("name") or "").strip(),
                thought=(kc.get("thought") or "").strip(),
            )
            for kc in s.get("key_characters") or []
            if isinstance(kc, dict) and kc.get("name")
        )
        scenes.append(Scene(
            scenario=(s.get("scenario") or "")[:1500],
            topic=(s.get("topic") or "")[:200],
            dialogues=dlg,
            key_characters=kcs,
            book=s.get("book", ""),
            chapter=s.get("chapter", "") or "",
        ))

    return BotPersona(
        id=slug,
        name=name,
        source=meta["source"],
        source_dataset="coser",
        avatar_emoji=meta["emoji"],
        tagline=meta["tagline"],
        profile_markdown=raw.get("profile_markdown", ""),
        greeting=meta["greeting"],
        safety_prompt=DEFAULT_SAFETY,
        tags=meta["tags"],
        rag_corpus=tuple(chunks),
        scenes=tuple(scenes),
    )


# ─── public API ─────────────────────────────────────────────────────────────

def load_personas(only_dataset: str | None = None) -> list[BotPersona]:
    personas: list[BotPersona] = []
    if only_dataset in (None, "coser"):
        coser_dir = RAW_DIR / "coser"
        if coser_dir.exists():
            for path in sorted(coser_dir.glob("*.json")):
                slug = path.stem
                if slug not in _DISPLAY_META:
                    logger.warning("CoSER persona %s has no display meta; skipping", slug)
                    continue
                raw = json.loads(path.read_text(encoding="utf-8"))
                personas.append(_coser_to_persona(slug, raw))
    return personas
