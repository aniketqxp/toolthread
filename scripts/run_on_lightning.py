#!/usr/bin/env python
"""Drive BFCL eval or LoRA SFT training on a Lightning AI Studio from a local box.

No browser. Reads Lightning credentials from the environment (or a local,
gitignored ``.env.lightning``), provisions a Studio, clones the repo there,
installs the pinned env, runs the requested pipeline, and stops the Studio
in a ``finally`` so billing always ends.

Modes:
  eval   — run the official BFCL multi-turn benchmark (default)
  train  — run LoRA SFT on preprocessed multi-turn data

Authentication (pick one):
  A) Run ``lightning login`` once in your own terminal. It writes
     ~/.lightning/credentials.json, which the SDK reads automatically. Your
     API key never passes through this chat. (recommended)
  B) Put LIGHTNING_USER_ID + LIGHTNING_API_KEY in .env.lightning (gitignored).

Always required:
  LIGHTNING_USERNAME   your lightning.ai/<username> handle (for teamspace lookup),
                       or pass --user. Not a secret.
  LIGHTNING_TEAMSPACE  optional; auto-selected when the account has exactly one.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/aniketqxp/toolthread.git"


def log(msg: str) -> None:
    print(f"[lightning] {msg}", flush=True)


def load_local_env(path: Path) -> None:
    """Minimal .env loader (no dependency); does not override real env vars."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def pick_teamspace(user, requested: str | None):
    spaces = list(user.teamspaces)
    if not spaces:
        sys.exit("[lightning] this account has no teamspaces.")
    names = [ts.name for ts in spaces]
    if requested:
        for ts in spaces:
            if ts.name == requested:
                return ts
        sys.exit(f"[lightning] teamspace {requested!r} not found. Have: {names}")
    if len(spaces) == 1:
        return spaces[0]
    sys.exit(
        f"[lightning] multiple teamspaces {names}; set LIGHTNING_TEAMSPACE to one."
    )


def _clone_and_setup() -> list[str]:
    """Common steps: clone repo, pull latest, install base deps."""
    return [
        "set -e",
        f"if [ ! -d toolthread ]; then git clone {REPO_URL}; fi",
        "cd toolthread",
        "git pull --ff-only",
        "pip install -q -e . --no-deps",
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true",
    ]


def build_eval_pipeline(no_wandb: bool) -> str:
    eval_cmd = (
        "python scripts/run_bfcl_eval.py "
        "--model-config configs/model/qwen3-1.7b.yaml "
        "--eval-config configs/eval/bfcl_multi_turn.yaml "
        "--backend vllm"
    )
    if no_wandb:
        eval_cmd += " --no-wandb"
    steps = _clone_and_setup() + [
        "export BFCL_PROJECT_ROOT=$PWD/bfcl_runs",
        "pip install -q -r requirements.txt",
        "pip install -q soundfile",
        'pip install -q "bfcl-eval[oss_eval_vllm,wandb]==2026.3.23"',
        "python -c \""
        "import vllm.transformers_utils.tokenizer as t; "
        "p = t.__file__; "
        "s = open(p).read(); "
        "open(p,'w').write(s.replace('all_special_tokens_extended','all_special_tokens')); "
        "print('[patch] fixed', p)"
        "\"",
        eval_cmd,
    ]
    return "\n".join(steps)


def build_train_pipeline(train_args: argparse.Namespace) -> str:
    """Build the shell pipeline for LoRA SFT training."""
    train_cmd = (
        "python scripts/run_lora_sft.py "
        f"--config {train_args.train_config}"
    )
    if train_args.lora_rank:
        train_cmd += f" --lora-rank {train_args.lora_rank}"
    if train_args.batch_size:
        train_cmd += f" --per-device-batch-size {train_args.batch_size}"
    if train_args.learning_rate:
        train_cmd += f" --learning-rate {train_args.learning_rate}"
    if train_args.push_to_hub:
        train_cmd += " --push-to-hub"
    if train_args.hub_model_id:
        train_cmd += f" --hub-model-id {train_args.hub_model_id}"

    steps = _clone_and_setup() + [
        "pip install -q datasets peft trl accelerate bitsandbytes",
        "pip install -q torch --upgrade || true",
        "python scripts/prep_sft_data.py",
        train_cmd,
    ]

    if train_args.push_to_hub and train_args.hub_model_id:
        steps.append(f"echo '[lightning] merged model pushed to {train_args.hub_model_id}'")

    return "\n".join(steps)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run eval or training on Lightning AI.")
    ap.add_argument("--mode", choices=["eval", "train"], default="eval",
                    help="eval = BFCL benchmark, train = LoRA SFT (default: eval)")
    ap.add_argument("--studio", default="toolthread-baseline", help="Studio name")
    ap.add_argument("--machine", default="L4", help="Machine enum name (default L4)")
    ap.add_argument("--max-runtime", type=int, default=5400,
                    help="hard auto-stop ceiling in seconds (billing safety)")
    ap.add_argument("--keep-running", action="store_true",
                    help="do not stop the Studio when the run finishes")
    ap.add_argument("--wandb", action="store_true",
                    help="log to W&B (requires WANDB_API_KEY set as a Studio secret)")
    ap.add_argument("--user", help="lightning.ai username (overrides LIGHTNING_USERNAME)")
    # Training-specific args
    ap.add_argument("--train-config", default="configs/train/lora_sft.yaml")
    ap.add_argument("--lora-rank", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--hub-model-id")
    args = ap.parse_args()

    load_local_env(REPO_ROOT / ".env.lightning")

    has_env_creds = bool(os.environ.get("LIGHTNING_USER_ID")
                         and os.environ.get("LIGHTNING_API_KEY"))
    has_creds_file = (Path.home() / ".lightning" / "credentials.json").exists()
    if not (has_env_creds or has_creds_file):
        sys.exit(
            "[lightning] not authenticated. Either run 'lightning login' once, "
            "or set LIGHTNING_USER_ID + LIGHTNING_API_KEY in .env.lightning."
        )

    username = args.user or os.environ.get("LIGHTNING_USERNAME")
    if not username:
        sys.exit("[lightning] need your username: pass --user or set LIGHTNING_USERNAME.")

    from lightning_sdk import Machine, Studio, User

    machine = getattr(Machine, args.machine, None)
    if machine is None:
        sys.exit(f"[lightning] unknown machine {args.machine!r}.")

    log(f"authenticating as {username} ...")
    user = User(name=username)
    teamspace = pick_teamspace(user, os.environ.get("LIGHTNING_TEAMSPACE"))
    log(f"teamspace: {teamspace.name}")

    if args.mode == "train" and args.max_runtime == 5400:
        args.max_runtime = 25000

    studio = Studio(name=args.studio, teamspace=teamspace, user=user, create_ok=True)
    log(f"studio '{studio.name}' status={studio.status}")

    try:
        status = str(studio.status)
        if status == "Running" and studio.machine != machine:
            log(f"running on {studio.machine}; switching to {args.machine} ...")
            studio.switch_machine(machine)
        elif status != "Running":
            log(f"starting on {args.machine} (max_runtime={args.max_runtime}s) ...")
            studio.start(machine=machine, interruptible=False,
                         max_runtime=args.max_runtime)
        log(f"studio is up on {studio.machine}")

        if args.mode == "train":
            pipeline = build_train_pipeline(args)
            log("running pipeline (clone -> install -> prep data -> train). This blocks until done.")
        else:
            pipeline = build_eval_pipeline(no_wandb=not args.wandb)
            log("running pipeline (clone -> install -> eval). This blocks until done.")
        out, code = studio.run_with_exit_code(pipeline)
        log_file = REPO_ROOT / "runs" / "lightning_output.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(out, encoding="utf-8")
        log(f"full output saved to {log_file}")
        for line in out.splitlines():
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                print(line.encode("ascii", errors="replace").decode(), flush=True)
        log(f"pipeline exit code: {code}")
        if code != 0:
            sys.exit(code)
    finally:
        if args.keep_running:
            log("leaving Studio running (--keep-running).")
        elif str(studio.status) == "Running":
            log("stopping Studio to end billing ...")
            try:
                studio.stop()
                log("Studio stopped.")
            except Exception as exc:  # noqa: BLE001
                log(f"WARNING: failed to stop Studio: {exc!r}. Stop it in the UI.")
        else:
            log(f"Studio already {studio.status}; no stop needed.")


if __name__ == "__main__":
    main()
