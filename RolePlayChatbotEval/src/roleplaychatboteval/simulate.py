"""Multi-agent simulation of one CoSER test case.

For each speaking character in the case, we open one chat session with the
RolePlayChatbotService and feed it the running transcript turn by turn.
This relies on the service's two-layer RAG + GCA + Strands memory — i.e.
exactly the same pipeline a user hits.

If a CoSER `major_character` matches one of our service-side personas
(by display name), we use that persona id directly. Otherwise we fall
back to a generic "stranger" persona — flagged in the result so the judge
can discount fidelity scoring on those turns.

Turn ordering: fixed alternation through `speaking_characters_w_env`,
skipping the environment narrator. Capped at `max_turns` (paper uses 20).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from roleplaychatboteval.dataset import DialogueTurn, TestCase

logger = logging.getLogger(__name__)

# Display-name → persona_id mapping for our 9 service personas.
# Matches the names CoSER uses in `character_profiles` / `major_characters`.
PERSONA_BY_DISPLAY: dict[str, str] = {
    "Fitzwilliam Darcy": "mr_darcy",
    "Mr. Darcy":         "mr_darcy",
    "Heathcliff":        "heathcliff",
    "Sherlock Holmes":   "sherlock_holmes",
    "Jay Gatsby":        "jay_gatsby",
    "Hermione Granger":  "hermione_granger",
    "Elizabeth Bennet":  "elizabeth_bennet",
    "Anna Arkadyevna Karenina": "anna_karenina",
    "Anna Karenina":     "anna_karenina",
    "Atticus Finch":     "atticus_finch",
    "Scarlett O'Hara":   "scarlett_ohara",
}


@dataclass
class SimulatedTurn:
    speaker: str            # the canonical CoSER name
    persona_id: str | None  # service persona id, if matched
    speech: str
    action: str | None = None
    thought: str | None = None
    raw: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    case_book: str
    case_topic: str
    case_scenario: str
    ground_truth: tuple[DialogueTurn, ...]
    generated: list[SimulatedTurn]
    speaker_persona_map: dict[str, str | None]


def _resolve_persona(name: str) -> str | None:
    return PERSONA_BY_DISPLAY.get(name.strip())


def _service_speakers(case: TestCase) -> list[str]:
    """Pick the human speakers (drop 'Environment' narrator etc.).

    CoSER puts non-character turns (scene descriptions) under names like
    'Environment'. Filter them out.
    """
    skip = {"Environment", "Narrator", "Setting", ""}
    out: list[str] = []
    for s in case.speaking_characters:
        if s in skip:
            continue
        if s not in out:
            out.append(s)
    if not out:
        # Fallback: use major_characters in order.
        out = [c for c in case.major_characters if c not in skip]
    return out


def simulate_case(
    case: TestCase,
    base_url: str = "http://127.0.0.1:8000",
    max_turns: int = 20,
    seed_first_message: str | None = None,
) -> SimulationResult:
    """Run multi-agent simulation against a running RolePlayChatbotService.

    The first speaker receives `seed_first_message` (defaults to the case's
    topic) as the opening user input. Each subsequent speaker sees the
    previous speaker's reply as their user input. We open one chat session
    per speaker and reuse it across the simulation.
    """
    speakers = _service_speakers(case)
    if len(speakers) < 2:
        raise ValueError(f"need ≥2 speakers; got {speakers!r}")

    persona_map: dict[str, str | None] = {s: _resolve_persona(s) for s in speakers}
    sessions: dict[str, str] = {}

    seed = seed_first_message or case.topic or case.scenario[:200]
    generated: list[SimulatedTurn] = []

    with httpx.Client(timeout=120) as client:
        # Open a session per speaker (fall back to a literary stand-in if no match).
        for speaker, persona_id in persona_map.items():
            if persona_id is None:
                logger.info("speaker %r not in our roster; skipping (no fallback)", speaker)
                continue
            r = client.post(
                f"{base_url}/chat/sessions",
                json={"persona_id": persona_id, "user_name": "Reader"},
            )
            r.raise_for_status()
            sessions[speaker] = r.json()["session_id"]

        if not sessions:
            logger.warning("no service-resolvable speakers in case; skipping")
            return SimulationResult(
                case_book=case.book, case_topic=case.topic, case_scenario=case.scenario,
                ground_truth=case.ground_truth, generated=[],
                speaker_persona_map=persona_map,
            )

        next_input = seed
        # Fixed alternation: cycle through speakers we have sessions for.
        ring = list(sessions)
        for turn_idx in range(max_turns):
            speaker = ring[turn_idx % len(ring)]
            sid = sessions[speaker]
            try:
                resp = client.post(
                    f"{base_url}/chat/sessions/{sid}/messages",
                    json={"message": next_input},
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("turn %d (%s) failed: %s", turn_idx, speaker, exc)
                break
            body = resp.json()
            turn = SimulatedTurn(
                speaker=speaker,
                persona_id=persona_map[speaker],
                speech=body.get("speech") or body.get("reply") or "",
                action=body.get("action"),
                thought=body.get("thought"),
                raw=body.get("raw") or body.get("reply") or "",
                meta={
                    "scene_score": body.get("retrieved_scene_score"),
                    "chunks": body.get("retrieved_chunk_count"),
                },
            )
            generated.append(turn)
            next_input = turn.speech or turn.raw

        # Best-effort cleanup.
        for sid in sessions.values():
            try:
                client.delete(f"{base_url}/chat/sessions/{sid}")
            except httpx.HTTPError:
                pass

    return SimulationResult(
        case_book=case.book,
        case_topic=case.topic,
        case_scenario=case.scenario,
        ground_truth=case.ground_truth,
        generated=generated,
        speaker_persona_map=persona_map,
    )
