"""Log the OFFICIAL BFCL scores to Weights & Biases.

The W&B run config records which handler / backend / BFCL version produced the
number, so baseline and fine-tuned runs are directly comparable and auditable.
"""
from __future__ import annotations

from typing import Any


def log_scores(
    *,
    project: str,
    run_name: str,
    config: dict[str, Any],
    scores: dict[str, Any],
    enabled: bool = True,
):
    """Push parsed official scores to W&B. Returns the run, or None if disabled."""
    if not enabled:
        print("[wandb] disabled; skipping logging.")
        return None

    import wandb  # imported lazily so the eval can run without wandb installed

    run = wandb.init(project=project, name=run_name, config=config)

    metrics = {"bfcl/multi_turn_overall": scores["overall"]}
    for cat, d in scores["per_category"].items():
        metrics[f"bfcl/{cat}"] = d["accuracy"]

    wandb.log(metrics)
    wandb.summary.update(metrics)
    run.finish()
    return run
