"""GCA (Given-Circumstance Acting) prompt elements.

Implements 4 of the 5 GCA elements from the CoSER paper §3 (the 5th —
character profile — is in `header.py`, since it's static per persona).

For each turn we feed in:
  • element 1 — Scenario description (from retrieved scene)
  • element 3 — Character motivation in this scene (this persona's `thought`
                in the retrieved scene's key_characters)
  • element 4 — Other characters' profiles (the OTHER key_characters' `thought`
                lines in the retrieved scene)
  • element 5 — Output format instruction (Speech / Action / Thought)

These render as Bot-tagged messages in CHAI chat_history (CHAI ignores sender
labels anyway, so framing has to live in the message text).
"""
from __future__ import annotations

from roleplaychatbotservice.chai import ChaiMessage
from roleplaychatbotservice.personas.retrieval import RetrievedScene
from roleplaychatbotservice.personas.schema import BotPersona


SAT_INSTRUCTION = (
    "Respond in role-play form. Use this 3-line structure when natural:\n"
    "Speech: <what you say aloud>\n"
    "Action: (what you physically do or your facial expression)\n"
    "Thought: [your private inner voice — others cannot hear this]\n"
    "If only Speech fits the moment, just write Speech."
)


def render_scenario_block(rs: RetrievedScene) -> ChaiMessage | None:
    """Element #1 — what kind of moment we're in.

    We prefer `topic` because it's terser; fall back to a clipped scenario
    if topic is empty.
    """
    if rs.score < 0.30:
        return None
    sc = rs.scene.topic.strip() or rs.scene.scenario.strip()[:300]
    if not sc:
        return None
    return ChaiMessage(
        sender="Bot",
        message=f"[Setting] The current moment resembles a past scene: {sc}",
    )


def render_motivation_block(rs: RetrievedScene, persona: BotPersona) -> ChaiMessage | None:
    """Element #3 — this character's in-scene drive."""
    if rs.score < 0.30:
        return None
    own = next(
        (kc for kc in rs.scene.key_characters
         if _is_same_character(kc.name, persona.name)),
        None,
    )
    if own is None or not own.thought:
        return None
    return ChaiMessage(
        sender="Bot",
        message=f"[My motivation in such moments] {own.thought}",
    )


def render_others_block(rs: RetrievedScene, persona: BotPersona) -> ChaiMessage | None:
    """Element #4 — short profile lines of the other people involved."""
    if rs.score < 0.30:
        return None
    others = [
        kc for kc in rs.scene.key_characters
        if not _is_same_character(kc.name, persona.name) and kc.thought
    ]
    if not others:
        return None
    lines = "; ".join(f"{kc.name}: {kc.thought}" for kc in others[:3])
    return ChaiMessage(
        sender="Bot",
        message=f"[Others present in such moments] {lines}",
    )


def render_sat_instruction() -> ChaiMessage:
    """Element #5 — output format. Always emitted when GCA is on."""
    return ChaiMessage(sender="Bot", message=SAT_INSTRUCTION)


def render_gca_block(retrieved_scene: RetrievedScene | None, persona: BotPersona,
                      enable_sat: bool) -> list[ChaiMessage]:
    """Compose all GCA elements into chat_history lines."""
    out: list[ChaiMessage] = []
    if retrieved_scene is not None:
        for fn in (render_scenario_block, render_motivation_block, render_others_block):
            if fn is render_scenario_block:
                msg = fn(retrieved_scene)
            else:
                msg = fn(retrieved_scene, persona)  # type: ignore[arg-type]
            if msg is not None:
                out.append(msg)
    if enable_sat:
        out.append(render_sat_instruction())
    return out


def _is_same_character(a: str, b: str) -> bool:
    """Loose match — CoSER uses 'Fitzwilliam Darcy' but persona display name is 'Mr. Darcy'."""
    a, b = a.strip().lower(), b.strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    # "Mr. Darcy" matches "Fitzwilliam Darcy" etc — last token compare.
    a_last = a.rsplit(maxsplit=1)[-1]
    b_last = b.rsplit(maxsplit=1)[-1]
    return a_last == b_last
