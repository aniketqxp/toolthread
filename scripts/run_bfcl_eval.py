#!/usr/bin/env python
"""Entry point: run the FULL official BFCL multi-turn eval against ANY given model
and log the official score to W&B.

The model is fully parameterized — pass a model config (YAML) and/or override on
the CLI. No model id is hardcoded anywhere.

Examples
--------
  # Baseline candidate A (off-the-shelf), vLLM (default, any GPU):
  python scripts/run_bfcl_eval.py \
      --model-config configs/model/qwen3-1.7b.yaml \
      --eval-config  configs/eval/bfcl_multi_turn.yaml

  # Same, on Lightning Ampere+ via the sglang fast path:
  python scripts/run_bfcl_eval.py \
      --model-config configs/model/qwen3-1.7b.yaml \
      --eval-config  configs/eval/bfcl_multi_turn.yaml \
      --backend sglang

  # A different base, parameterized purely from the CLI:
  python scripts/run_bfcl_eval.py \
      --model-config configs/model/qwen3-1.7b.yaml \
      --eval-config  configs/eval/bfcl_multi_turn.yaml \
      --model-id some-org/some-model --bfcl-model some-org/some-model-FC

  # A fine-tuned checkpoint, enforcing parity against the recorded baseline:
  python scripts/run_bfcl_eval.py \
      --model-config configs/model/qwen3-1.7b.yaml \
      --eval-config  configs/eval/bfcl_multi_turn.yaml \
      --local-model-path /path/to/merged \
      --baseline-manifest runs/Qwen_Qwen3-1.7B-FC.baseline.json
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# --- import bootstrap: make src/ importable whether or not `pip install -e .` ran
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import config as cfg                       # src/config.py
from eval import bfcl_runner               # src/eval/bfcl_runner.py
from eval import scores as scores_mod
from eval import wandb_logger


def _bfcl_version() -> str:
    try:
        import bfcl_eval
        return getattr(bfcl_eval, "__version__", "unknown")
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full official BFCL multi-turn eval.")
    p.add_argument("--model-config", required=True, help="path to a configs/model/*.yaml")
    p.add_argument("--eval-config", required=True, help="path to a configs/eval/*.yaml")
    p.add_argument("--model-id", help="override hf_repo_id from the model config")
    p.add_argument("--bfcl-model", help="override the BFCL handler (--model)")
    p.add_argument("--local-model-path", help="merged fine-tuned weights; omit for baseline")
    p.add_argument("--backend", choices=list(bfcl_runner.VALID_BACKENDS),
                   help="override the eval-config backend")
    p.add_argument("--num-gpus", type=int)
    p.add_argument("--gpu-memory-utilization", type=float)
    p.add_argument("--bfcl-project-root", help="overrides $BFCL_PROJECT_ROOT")
    p.add_argument("--baseline-manifest",
                   help="enforce prompt/tokenizer parity against this baseline run's manifest")
    p.add_argument("--no-wandb", action="store_true", help="skip W&B logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg.load_env()

    mcfg = cfg.load_yaml(args.model_config)
    ecfg = cfg.load_yaml(args.eval_config)

    backend = args.backend or ecfg.get("backend", "vllm")
    spec = bfcl_runner.RunSpec(
        bfcl_model=args.bfcl_model or mcfg.get("bfcl_model"),
        hf_repo_id=args.model_id or mcfg.get("hf_repo_id"),
        test_category=ecfg.get("test_category", "multi_turn"),
        backend=backend,
        local_model_path=args.local_model_path or mcfg.get("local_model_path"),
        num_gpus=args.num_gpus or mcfg.get("num_gpus", 1),
        gpu_memory_utilization=args.gpu_memory_utilization or mcfg.get("gpu_memory_utilization", 0.9),
        num_threads=ecfg.get("num_threads"),
    )

    # BFCL_PROJECT_ROOT is required by pip-installed bfcl-eval.
    project_root = args.bfcl_project_root or cfg.get_env("BFCL_PROJECT_ROOT")
    if not project_root:
        raise RuntimeError(
            "BFCL_PROJECT_ROOT is not set (env or --bfcl-project-root). See .env.example."
        )
    os.environ["BFCL_PROJECT_ROOT"] = project_root
    Path(project_root).mkdir(parents=True, exist_ok=True)

    expected = ecfg.get("expected_categories")
    if not expected:
        raise RuntimeError("eval-config is missing 'expected_categories' (full-set guard).")

    # --- parity checks for fine-tuned runs (loud failure on divergence) ---
    if spec.local_model_path:
        if args.baseline_manifest:
            bfcl_runner.assert_handler_parity(args.baseline_manifest, spec)
        else:
            print("[parity] WARNING: fine-tuned run without --baseline-manifest; "
                  "handler parity is not cross-checked against a baseline.", file=sys.stderr)
        if spec.hf_repo_id:
            bfcl_runner.assert_tokenizer_parity(spec.hf_repo_id, spec.local_model_path)

    weights = spec.local_model_path or spec.hf_repo_id
    print(f"[run] handler={spec.bfcl_model} weights={weights} backend={spec.backend} "
          f"category={spec.test_category}")

    # --- official BFCL: generate then evaluate (full set, never --partial-eval) ---
    bfcl_runner.generate(spec)
    bfcl_runner.evaluate(spec)

    result = scores_mod.parse_multi_turn(project_root, spec.bfcl_model, expected)

    # --- manifest, so a later fine-tuned run can prove parity against this one ---
    runs_dir = REPO_ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)
    tag = "baseline" if spec.local_model_path is None else "finetuned"
    manifest_path = runs_dir / f"{spec.bfcl_model.replace('/', '_')}.{tag}.json"
    bfcl_runner.write_manifest(manifest_path, spec, _bfcl_version())

    # --- report ---
    print("\n=== BFCL multi-turn (official scores) ===")
    print(f"handler: {spec.bfcl_model} (mode={bfcl_runner.prompt_mode(spec.bfcl_model)})   "
          f"backend: {spec.backend}   bfcl: {_bfcl_version()}")
    for cat in expected:
        print(f"  {cat:24s} {result['per_category'][cat]['accuracy']}")
    print(f"  {'multi_turn_overall':24s} {result['overall']}")
    print(f"manifest: {manifest_path}")

    wandb_cfg = ecfg.get("wandb", {})
    wandb_logger.log_scores(
        project=wandb_cfg.get("project", "toolthread-bfcl"),
        run_name=f"{spec.bfcl_model}-{tag}-{spec.backend}",
        config={
            "bfcl_model": spec.bfcl_model,
            "prompt_mode": bfcl_runner.prompt_mode(spec.bfcl_model),
            "hf_repo_id": spec.hf_repo_id,
            "local_model_path": spec.local_model_path,
            "backend": spec.backend,
            "test_category": spec.test_category,
            "bfcl_version": _bfcl_version(),
            "is_baseline": spec.local_model_path is None,
        },
        scores=result,
        enabled=(wandb_cfg.get("enabled", True) and not args.no_wandb),
    )


if __name__ == "__main__":
    main()
