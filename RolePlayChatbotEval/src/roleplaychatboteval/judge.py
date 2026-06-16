"""Penalty-based 4-dimension judge (per CoSER paper §4).

Each dimension starts at 100. The judge LLM lists issues, each with a
severity label that maps to a deduction:
  - minor    →  -5
  - major    → -10
  - critical → -20

Final score = max(0, 100 - sum(deductions)).

Judge backend: Anthropic Claude. Looks for `ANTHROPIC_API_KEY` in env;
without it `score_case` raises `JudgeUnavailable` so the pipeline can
emit transcripts but skip scoring.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

import httpx

from roleplaychatboteval.dataset import DialogueTurn, TestCase
from roleplaychatboteval.simulate import SimulationResult

logger = logging.getLogger(__name__)


DIMENSIONS: tuple[str, ...] = (
    "anthropomorphism",
    "character_fidelity",
    "storyline_quality",
    "storyline_consistency",
)

DIMENSION_RUBRICS: dict[str, str] = {
    "anthropomorphism": (
        "Anthropomorphism — does it sound like a human, not a chatbot?\n"
        "  - Minor (-5): occasionally too polished or verbose for natural speech\n"
        "  - Major (-10): obvious AI tells (\"As an AI...\", over-helpfulness, hedging)\n"
        "  - Critical (-20): completely breaks the conversational illusion"
    ),
    "character_fidelity": (
        "Character Fidelity — how true is it to the character's canon?\n"
        "  - Minor (-5): slight word/register mismatches with the character's voice\n"
        "  - Major (-10): character takes an action or stance clearly out-of-character\n"
        "  - Critical (-20): character knowledge or identity is plainly wrong"
    ),
    "storyline_quality": (
        "Storyline Quality — is the conversation interesting and coherent on its own?\n"
        "  - Minor (-5): a turn is flat or filler\n"
        "  - Major (-10): logical contradictions inside the dialogue\n"
        "  - Critical (-20): the dialogue makes no narrative sense"
    ),
    "storyline_consistency": (
        "Storyline Consistency — does it respect the original book's plot direction?\n"
        "  - Minor (-5): small detail differs from the source but is plausible\n"
        "  - Major (-10): plot direction diverges from canon in a meaningful way\n"
        "  - Critical (-20): contradicts a core fact / outcome of the source"
    ),
}


_DEDUCTION = {"minor": 5, "major": 10, "critical": 20}


class JudgeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Issue:
    severity: str  # "minor" | "major" | "critical"
    note: str
    deduction: int


@dataclass(frozen=True)
class DimensionScore:
    dimension: str
    score: int
    issues: tuple[Issue, ...] = ()
    justification: str = ""


@dataclass
class JudgeScore:
    book: str
    topic: str
    by_dimension: dict[str, DimensionScore] = field(default_factory=dict)

    @property
    def average(self) -> float:
        if not self.by_dimension:
            return 0.0
        return sum(d.score for d in self.by_dimension.values()) / len(self.by_dimension)


# ─── prompt template ────────────────────────────────────────────────────────


_JUDGE_SYSTEM = (
    "You are a meticulous drama critic. You evaluate role-play chat simulations "
    "against the original book scene. You output strictly valid JSON, never prose."
)


def _build_judge_prompt(
    case: TestCase,
    sim: SimulationResult,
    dimension: str,
) -> str:
    rubric = DIMENSION_RUBRICS[dimension]
    gt_lines = "\n".join(
        f"  {t.character}: {t.message}" for t in case.ground_truth[:25]
    ) or "  (no ground truth turns)"
    sim_lines = "\n".join(
        f"  {t.speaker}: {t.speech}" + (f"  (action: {t.action})" if t.action else "")
        + (f"  (thought: {t.thought})" if t.thought else "")
        for t in sim.generated[:25]
    ) or "  (no generated turns)"

    char_summary = "\n".join(
        f"  - {name}: {profile[:300]}"
        for name, profile in (case.character_profiles or {}).items()
    ) or "  (no profiles)"

    return f"""## Original scene
Book: {case.book}
Setting: {case.scenario[:1500]}
Topic: {case.topic}
Characters:
{char_summary}

## Original Dialogue (ground truth)
{gt_lines}

## Generated Simulation
{sim_lines}

## Evaluation: {dimension}
{rubric}

## Output format
Return ONLY a JSON object with this shape:
{{
  "issues": [
    {{"severity": "minor"|"major"|"critical", "note": "<short description>"}}
  ],
  "deductions": <integer total>,
  "score": <integer 0-100, equal to max(0, 100 - deductions)>,
  "justification": "<1-2 sentences>"
}}
"""


# ─── Anthropic call ─────────────────────────────────────────────────────────


def _anthropic_call(
    prompt: str,
    *,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 800,
    api_key: str | None = None,
) -> str:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise JudgeUnavailable("ANTHROPIC_API_KEY not set; judging skipped")
    with httpx.Client(timeout=90) as c:
        r = c.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "system": _JUDGE_SYSTEM,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
    if r.status_code != 200:
        raise JudgeUnavailable(f"anthropic {r.status_code}: {r.text[:200]}")
    blocks = r.json().get("content") or []
    return "".join(b.get("text", "") for b in blocks if isinstance(b, dict))


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(text: str) -> dict:
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"no JSON in judge output: {text[:200]!r}")
    return json.loads(m.group(0))


def _to_issues(raw_issues: list[dict] | None) -> tuple[Issue, ...]:
    out: list[Issue] = []
    for it in raw_issues or []:
        if not isinstance(it, dict):
            continue
        sev = (it.get("severity") or "").strip().lower()
        if sev not in _DEDUCTION:
            continue
        out.append(Issue(
            severity=sev,
            note=(it.get("note") or "")[:300],
            deduction=_DEDUCTION[sev],
        ))
    return tuple(out)


# ─── public API ─────────────────────────────────────────────────────────────


def score_dimension(
    case: TestCase,
    sim: SimulationResult,
    dimension: str,
    *,
    model: str = "claude-haiku-4-5-20251001",
    api_key: str | None = None,
) -> DimensionScore:
    if dimension not in DIMENSION_RUBRICS:
        raise ValueError(f"unknown dimension: {dimension}")
    prompt = _build_judge_prompt(case, sim, dimension)
    raw = _anthropic_call(prompt, model=model, api_key=api_key)
    data = _parse_judge_json(raw)
    issues = _to_issues(data.get("issues"))
    deductions = int(data.get("deductions") or sum(i.deduction for i in issues))
    score = int(data.get("score") if "score" in data else max(0, 100 - deductions))
    score = max(0, min(100, score))
    return DimensionScore(
        dimension=dimension,
        score=score,
        issues=issues,
        justification=(data.get("justification") or "")[:500],
    )


def score_case(
    case: TestCase,
    sim: SimulationResult,
    *,
    dimensions: tuple[str, ...] = DIMENSIONS,
    model: str = "claude-haiku-4-5-20251001",
    api_key: str | None = None,
) -> JudgeScore:
    out = JudgeScore(book=case.book, topic=case.topic)
    for d in dimensions:
        try:
            out.by_dimension[d] = score_dimension(case, sim, d, model=model, api_key=api_key)
        except JudgeUnavailable:
            raise
        except Exception as exc:
            logger.warning("scoring %s failed: %s", d, exc)
            out.by_dimension[d] = DimensionScore(dimension=d, score=0,
                                                 justification=f"scoring error: {exc!s}")
    return out
