"""Command-line entry point for the CoSER evaluation harness.

Run:
    # Service must already be running on :8000
    uv run roleplaychatboteval --base http://127.0.0.1:8000 --max-cases 4

    # Skip judging (no ANTHROPIC_API_KEY needed)
    uv run roleplaychatboteval --no-judge

    # Run only on cases involving particular personas
    uv run roleplaychatboteval --persona Hermione --persona Sherlock
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

from roleplaychatboteval.dataset import filter_to_personas, load_test_cases
from roleplaychatboteval.judge import (
    DIMENSIONS,
    JudgeScore,
    JudgeUnavailable,
    score_case,
)
from roleplaychatboteval.report import aggregate, format_table
from roleplaychatboteval.simulate import PERSONA_BY_DISPLAY, simulate_case

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="roleplaychatboteval")
    p.add_argument("--base", default="http://127.0.0.1:8000",
                   help="RolePlayChatbotService base URL")
    p.add_argument("--cache", default=".cache/hf",
                   help="HF cache root holding Neph0s/CoSER snapshot")
    p.add_argument("--max-cases", type=int, default=4,
                   help="Cap on test cases to run (full set is small after filtering)")
    p.add_argument("--max-turns", type=int, default=12,
                   help="Max turns per simulated conversation (paper: 20)")
    p.add_argument("--no-judge", action="store_true",
                   help="Skip the LLM judging step; emit transcripts only")
    p.add_argument("--persona", action="append", default=None,
                   help="Filter to test cases featuring this CoSER name "
                        "(repeatable; matches are OR-ed)")
    p.add_argument("--out-dir", default=".eval_runs",
                   help="Where to write per-case json + summary.txt")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_test_cases(args.cache)
    if args.persona:
        wanted = set(args.persona)
    else:
        wanted = set(PERSONA_BY_DISPLAY)
    cases = filter_to_personas(cases, wanted)
    if not cases:
        logger.error("no test cases match %r", wanted)
        return 2

    cases = cases[: args.max_cases]
    logger.info("running %d case(s) against %s", len(cases), args.base)

    scores: list[JudgeScore] = []
    judge_available = bool(os.environ.get("ANTHROPIC_API_KEY")) and not args.no_judge

    for i, case in enumerate(cases):
        logger.info("[%d/%d] %r — topic=%r", i + 1, len(cases), case.book, case.topic[:60])
        sim = simulate_case(case, base_url=args.base, max_turns=args.max_turns)
        case_dir = out_dir / f"case_{i:03d}"
        case_dir.mkdir(exist_ok=True)
        (case_dir / "case.json").write_text(json.dumps({
            "book": case.book, "topic": case.topic, "scenario": case.scenario,
            "major_characters": list(case.major_characters),
            "ground_truth": [asdict(t) for t in case.ground_truth],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (case_dir / "simulation.json").write_text(json.dumps({
            "speaker_persona_map": sim.speaker_persona_map,
            "generated": [
                {
                    "speaker": t.speaker, "persona_id": t.persona_id,
                    "speech": t.speech, "action": t.action, "thought": t.thought,
                    "meta": t.meta,
                }
                for t in sim.generated
            ],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        if judge_available:
            try:
                score = score_case(case, sim)
            except JudgeUnavailable as exc:
                logger.warning("judge skipped: %s", exc)
                judge_available = False
            else:
                scores.append(score)
                (case_dir / "score.json").write_text(json.dumps({
                    "book": score.book, "topic": score.topic,
                    "average": round(score.average, 2),
                    "by_dimension": {
                        d: {
                            "score": s.score,
                            "justification": s.justification,
                            "issues": [
                                {"severity": i.severity, "note": i.note,
                                 "deduction": i.deduction}
                                for i in s.issues
                            ],
                        }
                        for d, s in score.by_dimension.items()
                    },
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("    avg=%.2f  %s",
                            score.average,
                            ", ".join(f"{d}={score.by_dimension[d].score}"
                                      for d in DIMENSIONS if d in score.by_dimension))

    summary_text = format_table(aggregate(scores)) if scores else (
        "Judging skipped (set ANTHROPIC_API_KEY to enable). "
        f"{len(cases)} simulation transcripts saved under {out_dir}/."
    )
    (out_dir / "summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
