"""Aggregate per-case JudgeScores into a summary table."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from roleplaychatboteval.judge import DIMENSIONS, JudgeScore


@dataclass(frozen=True)
class DimensionSummary:
    dimension: str
    n: int
    mean: float
    stdev: float


def aggregate(scores: list[JudgeScore]) -> dict[str, DimensionSummary]:
    out: dict[str, DimensionSummary] = {}
    for d in DIMENSIONS:
        vals = [s.by_dimension[d].score for s in scores if d in s.by_dimension]
        if not vals:
            continue
        out[d] = DimensionSummary(
            dimension=d,
            n=len(vals),
            mean=mean(vals),
            stdev=pstdev(vals) if len(vals) > 1 else 0.0,
        )
    return out


def format_table(summary: dict[str, DimensionSummary]) -> str:
    if not summary:
        return "(no scored cases)"
    lines = ["dimension              | n  | mean   | stdev",
             "-----------------------+----+--------+-------"]
    for d, s in summary.items():
        lines.append(f"{d:22s} | {s.n:2d} | {s.mean:6.2f} | {s.stdev:5.2f}")
    return "\n".join(lines)
