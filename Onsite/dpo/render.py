"""
Chat-template rendering helpers shared by build_pairs.py and train_dpo.py.

Llama-3.2 uses the same headered template as Llama-3.1 (Unsloth alias
"llama-3.1"). We always render the prompt with `add_generation_prompt=True`
so the assistant header is the last thing in the string and the model is
positioned to start the next turn — required for both base sampling and
DPOTrainer.
"""
from __future__ import annotations

from typing import Iterable

ROLE_MAP = {"system": "system", "human": "user", "user": "user", "assistant": "assistant"}


def conv_to_messages(conv: list[dict]) -> list[dict]:
    """ShareGPT-ish {from,value} -> OpenAI-ish {role,content}."""
    return [{"role": ROLE_MAP.get(m["from"], m["from"]), "content": m["value"]} for m in conv]


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """Render `messages` with the assistant header trailing, ready for generation."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def first_assistant_idx(messages: Iterable[dict]) -> int | None:
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            return i
    return None
