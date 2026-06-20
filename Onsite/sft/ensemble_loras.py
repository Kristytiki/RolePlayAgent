"""
Weight-space LoRA ensemble for Qwen3-4B.

Combines multiple LoRA adapters trained on the SAME base model into one
merged checkpoint via peft.add_weighted_adapter, then saves merged 16-bit
ready to push to HF and submit to Chaiverse.

Hard requirements (enforced):
- All adapters share the same base model (Qwen3-4B-Instruct-2507).
- All adapters use identical r / alpha / target_modules.

Usage:
    cd Onsite/sft && source .venv/bin/activate

    # 3-way equal blend, linear merge, of anime+pippa+synth at step 60:
    python ensemble_loras.py \
      --base assets/Qwen3-4B-Instruct-2507 \
      --adapter anime=assets/qwen3-4b-anime-lora/step60 \
      --adapter pippa=assets/qwen3-4b-pippa-lora/step60 \
      --adapter synth=assets/qwen3-4b-synth-lora/step60 \
      --weights 0.5,0.3,0.2 \
      --combination linear \
      --out ../train_model/qwen3-4b-ensemble-linear-v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_adapter_arg(values: list[str]) -> dict[str, Path]:
    """Each entry: 'name=path/to/adapter'."""
    out: dict[str, Path] = {}
    for v in values:
        if "=" not in v:
            sys.exit(f"--adapter must be name=path, got: {v}")
        name, path = v.split("=", 1)
        out[name] = Path(path)
    return out


def parse_weights(s: str | None, n: int) -> list[float]:
    if s is None:
        return [1.0 / n] * n
    parts = [float(x) for x in s.split(",")]
    if len(parts) != n:
        sys.exit(f"--weights count {len(parts)} != #adapters {n}")
    total = sum(parts)
    if abs(total - 1.0) > 1e-3:
        print(f"[warn] weights sum to {total:.3f}, not 1.0; will normalize.")
        parts = [p / total for p in parts]
    return parts


def check_compat(adapters: dict[str, Path]) -> dict:
    """Verify all adapters use identical base / r / alpha / target_modules."""
    refs: dict | None = None
    for name, path in adapters.items():
        cfg_path = path / "adapter_config.json"
        if not cfg_path.exists():
            sys.exit(f"missing adapter_config.json at {cfg_path}")
        cfg = json.loads(cfg_path.read_text())
        keys = {
            "base_model": cfg.get("base_model_name_or_path", ""),
            "r": cfg.get("r", 0),
            "alpha": cfg.get("lora_alpha", 0),
            "targets": tuple(sorted(cfg.get("target_modules", []))),
        }
        if refs is None:
            refs = keys
            print(f"[ref] {name}: r={refs['r']} alpha={refs['alpha']} "
                  f"targets={len(refs['targets'])} base={refs['base_model']}")
        else:
            for k in ("r", "alpha", "targets"):
                if keys[k] != refs[k]:
                    sys.exit(
                        f"adapter {name} has {k}={keys[k]}, expected {refs[k]} — "
                        "ensemble requires identical config"
                    )
            print(f"[ok ] {name}: r={keys['r']} alpha={keys['alpha']}")
    return refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="path to base model dir")
    ap.add_argument("--adapter", action="append", required=True,
                    help="repeated; each as name=path")
    ap.add_argument("--weights", default=None,
                    help="comma-separated weights (default: equal)")
    ap.add_argument("--combination", default="linear",
                    choices=["linear", "ties", "dare_linear", "dare_ties",
                             "magnitude_prune"],
                    help="peft add_weighted_adapter combination_type")
    ap.add_argument("--density", type=float, default=0.7,
                    help="density for ties / dare (fraction kept)")
    ap.add_argument("--out", required=True,
                    help="output dir for merged 16-bit checkpoint")
    args = ap.parse_args()

    adapters = parse_adapter_arg(args.adapter)
    if len(adapters) < 2:
        sys.exit("need >=2 adapters to ensemble")
    weights = parse_weights(args.weights, len(adapters))
    print(f"\nadapters: {list(adapters)}")
    print(f"weights:  {weights}")
    print(f"merge:    {args.combination}")

    print("\n=== checking adapter compatibility ===")
    check_compat(adapters)

    print(f"\nloading base from {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    tok = AutoTokenizer.from_pretrained(args.base)

    # Load first adapter and attach the rest as named adapters.
    first_name, first_path = next(iter(adapters.items()))
    print(f"\nattaching {first_name} ({first_path})")
    model = PeftModel.from_pretrained(base, str(first_path), adapter_name=first_name)
    for name, path in list(adapters.items())[1:]:
        print(f"attaching {name} ({path})")
        model.load_adapter(str(path), adapter_name=name)

    # Build ensemble adapter via weight-space combination.
    ensemble_name = "ensemble"
    print(f"\nmerging {len(adapters)} adapters into '{ensemble_name}' ({args.combination})")
    extra: dict = {}
    if args.combination in {"ties", "dare_linear", "dare_ties", "magnitude_prune"}:
        # Density needed for ties/dare; pass per-adapter or scalar
        extra["density"] = args.density
    model.add_weighted_adapter(
        adapters=list(adapters.keys()),
        weights=weights,
        adapter_name=ensemble_name,
        combination_type=args.combination,
        **extra,
    )
    model.set_adapter(ensemble_name)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nmerging ensemble adapter into base weights and saving to {out_dir}")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))
    print("done")


if __name__ == "__main__":
    main()
