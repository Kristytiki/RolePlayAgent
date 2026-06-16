"""Unified persona schema. CoSER and ChatHaruhi data are normalised into this.

Design notes:
- The five GCA prompt elements (CoSER paper §3) all have a home here:
    profile_markdown        ← CoSER's character profile
    utterances              ← exemplar lines, used for ChatHaruhi-style few-shot
    scenes                  ← scenario + topic + dialogues, used for situation matching
    relationships (in profile_markdown for now; we can promote later)
    output format           ← rendered at prompt-build time, not stored
- We keep things flat and frozen. Loader builds these once at startup.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RagChunk:
    """One unit of retrieval-time context."""
    text: str
    kind: str          # "utterance" | "thought" | "experience" | "scene"
    source: str = ""   # e.g. "story" / "synthesized" for ChatHaruhi; book name for CoSER
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class DialogueTurn:
    character: str
    message: str


@dataclass(frozen=True)
class CharacterInScene:
    """One participant in a scene (CoSER's `key_characters[*]`).

    `thought` is what makes this GCA-grade: it captures the character's
    in-scene motivation. For OTHER characters in the scene (not the persona
    we're playing), the same field doubles as a brief profile line.
    """
    name: str
    thought: str = ""


@dataclass(frozen=True)
class Scene:
    scenario: str
    topic: str
    dialogues: tuple[DialogueTurn, ...]
    key_characters: tuple[CharacterInScene, ...] = ()
    book: str = ""
    chapter: str = ""


@dataclass(frozen=True)
class BotPersona:
    id: str                              # slug, e.g. "mr_darcy"
    name: str                            # display name
    source: str                          # "Pride and Prejudice (Jane Austen)"
    source_dataset: str                  # "coser" | "haruhi"
    avatar_emoji: str
    tagline: str
    profile_markdown: str                # CoSER's full markdown profile (Background/Personality/Relationships/...)
    greeting: str
    safety_prompt: str
    tags: tuple[str, ...]
    rag_corpus: tuple[RagChunk, ...]     # flat list of typed chunks for utterance RAG
    scenes: tuple[Scene, ...] = ()       # for future scenario-matching retrieval
