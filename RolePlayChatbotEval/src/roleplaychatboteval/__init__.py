"""CoSER-style evaluation harness for the role-play chatbot service.

Pipeline (per the CoSER paper §4):
  1. dataset.load_test_cases()  — 200 held-out conversations from Neph0s/CoSER
  2. dataset.filter_to_personas()  — keep cases where one of our personas
     is a major_character
  3. simulate.run_simulation(case)  — multi-agent role-play producing M̄
  4. judge.score_case(M, M_bar)  — penalty-based 4-dim scoring
  5. report.aggregate(results)  — mean/std per dimension, persona, book

The judge LLM is plug-pointed: ANTHROPIC_API_KEY enables it; absent,
the runner emits the simulated transcripts only.
"""
from roleplaychatboteval.dataset import (
    CharacterEntry,
    DialogueTurn,
    TestCase,
    filter_to_personas,
    load_test_cases,
)

__all__ = [
    "CharacterEntry",
    "DialogueTurn",
    "TestCase",
    "filter_to_personas",
    "load_test_cases",
]
