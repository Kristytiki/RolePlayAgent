"""
Batch-submit a portfolio of model variants to Chaiverse for parallel eval.

Each variant = (label, model_repo, formatter, gen_params override).
Per-base formatter matters: Qwen uses ChatML, Llama-3.x uses header_id, Gemma uses
start_of_turn/model. Wrong template tanks win rate.

Usage:
    HF_TOKEN=hf_xxx python submit_batch.py
    HF_TOKEN=hf_xxx python submit_batch.py --only qwen25_3b llama32_3b
    HF_TOKEN=hf_xxx python submit_batch.py --dry      # print, don't POST
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

CHAI_DEVELOPER_KEY = os.environ.get("CHAI_DEVELOPER_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
if not CHAI_DEVELOPER_KEY or not HF_TOKEN:
    sys.exit("set CHAI_DEVELOPER_KEY and HF_TOKEN env vars before running")
SUBMIT_URL = (
    "http://guanaco-submitter-v2.guanaco-backend.kchai-google-us-east4.chaiverse.com"
    "/models/submit"
)

# ---- Formatters (per-family chat template) ------------------------------

QWEN_CHATML = {
    "memory_template":   "<|im_start|>system\n{memory}<|im_end|>\n",
    "prompt_template":   "<|im_start|>user\n{prompt}<|im_end|>\n",
    "bot_template":      "<|im_start|>assistant\n{bot_name}: {message}<|im_end|>\n",
    "user_template":     "<|im_start|>user\n{user_name}: {message}<|im_end|>\n",
    "response_template": "<|im_start|>assistant\n{bot_name}:",
    "truncate_by_message": True,
}

LLAMA31_HEADERED = {
    "memory_template":   "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{memory}<|eot_id|>",
    "prompt_template":   "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>",
    "bot_template":      "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}: {message}<|eot_id|>",
    "user_template":     "<|start_header_id|>user<|end_header_id|>\n\n{user_name}: {message}<|eot_id|>",
    "response_template": "<|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}:",
    "truncate_by_message": True,
}

GEMMA_TURNS = {
    # Gemma has no system role; we treat memory + prompt as leading user turns.
    "memory_template":   "<start_of_turn>user\n{memory}<end_of_turn>\n",
    "prompt_template":   "<start_of_turn>user\n{prompt}<end_of_turn>\n",
    "bot_template":      "<start_of_turn>model\n{bot_name}: {message}<end_of_turn>\n",
    "user_template":     "<start_of_turn>user\n{user_name}: {message}<end_of_turn>\n",
    "response_template": "<start_of_turn>model\n{bot_name}:",
    "truncate_by_message": True,
}

SMOLLM_CHATML = QWEN_CHATML  # SmolLM2 uses ChatML-like tokens (<|im_start|>...).

# ---- Default gen params (Chai onsite default) ---------------------------

DEFAULT_GEN = {
    "temperature": 1.0,
    "top_p": 1.0,
    "min_p": 0.0,
    "top_k": 40,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "stopping_words": ["\n"],
    "max_input_tokens": 2048,
    "best_of": 8,
    "max_output_tokens": 64,
}

# ---- Variant portfolio --------------------------------------------------

# Each entry: (slug, model_repo, formatter, gen_param_overrides_dict)
VARIANTS = [
    # Dim A: scan small base models with correct templates
    # NOTE: qwen25_3b default-config already submitted as v18 + v19; do not re-fire.
    ("qwen25_1p5b",     "Qwen/Qwen2.5-1.5B-Instruct",        QWEN_CHATML,      {}),
    ("qwen25_0p5b",     "Qwen/Qwen2.5-0.5B-Instruct",        QWEN_CHATML,      {}),
    ("qwen3_1p7b",      "Qwen/Qwen3-1.7B",                   QWEN_CHATML,      {}),
    ("qwen3_4b",        "Qwen/Qwen3-4B-Instruct-2507",       QWEN_CHATML,      {}),
    ("qwen3_30b_a3b",   "Qwen/Qwen3-30B-A3B-Instruct-2507",  QWEN_CHATML,      {}),
    ("llama32_3b",      "meta-llama/Llama-3.2-3B-Instruct",  LLAMA31_HEADERED, {}),
    ("llama32_1b",      "meta-llama/Llama-3.2-1B-Instruct",  LLAMA31_HEADERED, {}),
    ("gemma4_e2b",      "google/gemma-4-E2B-it",             GEMMA_TURNS,      {}),
    ("gemma3_1b",       "google/gemma-3-1b-it",              GEMMA_TURNS,      {}),
    ("smollm2_1p7b",    "HuggingFaceTB/SmolLM2-1.7B-Instruct", SMOLLM_CHATML,   {}),

    # Dim C: gen-param sweep on best-guess base (Qwen2.5-3B)
    ("qwen25_3b_bo16",       "Qwen/Qwen2.5-3B-Instruct", QWEN_CHATML,
        {"best_of": 16}),
    ("qwen25_3b_long",       "Qwen/Qwen2.5-3B-Instruct", QWEN_CHATML,
        {"max_output_tokens": 80, "stopping_words": []}),
    ("qwen25_3b_freqpen",    "Qwen/Qwen2.5-3B-Instruct", QWEN_CHATML,
        {"frequency_penalty": 0.3}),
    ("qwen25_3b_conservative", "Qwen/Qwen2.5-3B-Instruct", QWEN_CHATML,
        {"temperature": 0.8, "top_p": 0.9}),

    # Dim B: formatter tweak on Qwen — inject CoSER thought/action style guide.
    ("qwen25_3b_coser_guide", "Qwen/Qwen2.5-3B-Instruct",
        {**QWEN_CHATML,
         "memory_template":
            "<|im_start|>system\n{memory}\n\n"
            "Use [your thought] for thoughts which others can't see. "
            "Use (your action) for actions which others can see.<|im_end|>\n"},
        {}),

    # Dim C × strong bases: best_of=16 / long output sweep on Qwen3-4B, Qwen3-30B-A3B, Gemma-4-E2B
    ("qwen3_30b_a3b_bo16",  "Qwen/Qwen3-30B-A3B-Instruct-2507", QWEN_CHATML,
        {"best_of": 16}),
    ("qwen3_30b_a3b_long",  "Qwen/Qwen3-30B-A3B-Instruct-2507", QWEN_CHATML,
        {"max_output_tokens": 80, "stopping_words": []}),
    ("qwen3_4b_bo16",       "Qwen/Qwen3-4B-Instruct-2507",      QWEN_CHATML,
        {"best_of": 16}),
    ("qwen3_4b_long",       "Qwen/Qwen3-4B-Instruct-2507",      QWEN_CHATML,
        {"max_output_tokens": 80, "stopping_words": []}),
    ("gemma4_e2b_bo16",     "google/gemma-4-E2B-it",            GEMMA_TURNS,
        {"best_of": 16}),
    ("gemma4_e2b_long",     "google/gemma-4-E2B-it",            GEMMA_TURNS,
        {"max_output_tokens": 80, "stopping_words": []}),

    # Dim B × strong bases: CoSER thought/action injection on Qwen3-4B, Qwen3-30B-A3B, Gemma-4-E2B
    ("qwen3_30b_a3b_coser_guide", "Qwen/Qwen3-30B-A3B-Instruct-2507",
        {**QWEN_CHATML,
         "memory_template":
            "<|im_start|>system\n{memory}\n\n"
            "Use [your thought] for thoughts which others can't see. "
            "Use (your action) for actions which others can see.<|im_end|>\n"},
        {}),
    ("qwen3_4b_coser_guide", "Qwen/Qwen3-4B-Instruct-2507",
        {**QWEN_CHATML,
         "memory_template":
            "<|im_start|>system\n{memory}\n\n"
            "Use [your thought] for thoughts which others can't see. "
            "Use (your action) for actions which others can see.<|im_end|>\n"},
        {}),
    ("gemma4_e2b_coser_guide", "google/gemma-4-E2B-it",
        {**GEMMA_TURNS,
         # Gemma has no system role — prepend the guide to the first user-turn memory.
         "memory_template":
            "<start_of_turn>user\n{memory}\n\n"
            "Use [your thought] for thoughts which others can't see. "
            "Use (your action) for actions which others can see.<end_of_turn>\n"},
        {}),

    # Dim D: CoSER fine-tuned models pushed to HF
    # smoke = 30-step LoRA SFT (sanity check the pipeline; not expected to beat baseline)
    ("qwen3b_coser_smoke",         "ZheqiWu/Qwen2.5-3B-CoSER-smoke", QWEN_CHATML, {}),
    ("qwen3b_coser_smoke_bo16",    "ZheqiWu/Qwen2.5-3B-CoSER-smoke", QWEN_CHATML,
        {"best_of": 16}),
    ("qwen3b_coser_smoke_coser_guide", "ZheqiWu/Qwen2.5-3B-CoSER-smoke",
        {**QWEN_CHATML,
         "memory_template":
            "<|im_start|>system\n{memory}\n\n"
            "Use [your thought] for thoughts which others can't see. "
            "Use (your action) for actions which others can see.<|im_end|>\n"},
        {}),
]

# ---- Submitter ----------------------------------------------------------

def build_submission(variant):
    slug, repo, fmt, gen_override = variant
    gen = {**DEFAULT_GEN, **gen_override}
    return slug, {
        "model_repo": repo,
        "generation_params": gen,
        "hf_token": HF_TOKEN,
        "formatter": fmt,
        "platform": "vllm",
    }

def submit(model_submission, key):
    headers = {"Authorization": f"Bearer {key}"}
    r = requests.post(SUBMIT_URL, headers=headers, json=model_submission, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()

MD_PATH = Path(__file__).with_name("SUBMISSIONS.md")

def _append_md(slug, sub, sid, url):
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MD_PATH.exists():
        MD_PATH.write_text(
            "# Chaiverse Submissions Log\n\n"
            "Each entry: slug, submission_id, URL, full generation_params + formatter.\n"
            "Win-rate / preferences are filled in manually after ~90min eval.\n\n"
        )
    ts = time.strftime("%Y-%m-%d %H:%M")
    gen = sub["generation_params"]
    fmt = sub["formatter"]
    md = []
    md.append(f"## `{slug}`  —  {ts}\n")
    md.append(f"- **submission_id**: `{sid}`")
    md.append(f"- **url**: {url}")
    md.append(f"- **model_repo**: `{sub['model_repo']}`")
    md.append(f"- **platform**: `{sub['platform']}`")
    md.append(f"- **win_rate**: _TBD_\n")
    md.append("**generation_params**:\n```json")
    md.append(json.dumps(gen, indent=2, ensure_ascii=False))
    md.append("```\n")
    md.append("**formatter**:\n```json")
    md.append(json.dumps(fmt, indent=2, ensure_ascii=False))
    md.append("```\n")
    md.append("---\n")
    with MD_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="slugs to submit (default: all)")
    ap.add_argument("--dry", action="store_true", help="print but do not POST")
    ap.add_argument("--out", default=str(Path(__file__).with_name("submissions.json")))
    args = ap.parse_args()

    targets = VARIANTS
    if args.only:
        wanted = set(args.only)
        targets = [v for v in VARIANTS if v[0] in wanted]
        unknown = wanted - {v[0] for v in VARIANTS}
        if unknown:
            sys.exit(f"unknown slugs: {unknown}")

    print(f"firing {len(targets)} variant(s)" + (" [DRY]" if args.dry else ""))

    history = []
    out_path = Path(args.out)
    if out_path.exists():
        history = json.loads(out_path.read_text())

    for v in targets:
        slug, sub = build_submission(v)
        print(f"\n--- {slug} :: {sub['model_repo']} ---")
        print(f"    gen: best_of={sub['generation_params']['best_of']} "
              f"temp={sub['generation_params']['temperature']} "
              f"max_out={sub['generation_params']['max_output_tokens']} "
              f"stop={sub['generation_params']['stopping_words']}")
        if args.dry:
            continue
        try:
            res = submit(sub, CHAI_DEVELOPER_KEY)
            sid = res.get("submission_id") if isinstance(res, dict) else None
            if not sid:
                raise RuntimeError(f"no submission_id in response: {res}")
            url = f"https://console.chaiverse.com/models/{sid}"
            print(f"    submission_id: {sid}")
            print(f"    URL: {url}")
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "slug": slug,
                "model_repo": sub["model_repo"],
                "submission_id": sid,
                "url": url,
                "generation_params": sub["generation_params"],
                "formatter": sub["formatter"],
                "gen_overrides": v[3],
            }
            history.append(entry)
            out_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))
            _append_md(slug, sub, sid, url)
        except Exception as e:
            print(f"    FAILED: {e}")
            history.append({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "slug": slug,
                "model_repo": sub["model_repo"],
                "error": str(e),
            })
            out_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))
        time.sleep(1)  # be polite to submit endpoint

    print(f"\nlog: {out_path}")

if __name__ == "__main__":
    main()
