"""Thin wrapper around the OFFICIAL bfcl CLI (`bfcl generate` / `bfcl evaluate`).

We never compute scores ourselves — BFCL's own evaluator does the scoring. This
module only:
  (a) builds and runs the official commands,
  (b) enforces full-set, no-subset multi-turn runs (never passes --partial-eval),
  (c) guarantees prompt/template parity between the off-the-shelf baseline and
      any future fine-tuned checkpoint.

WHY PARITY IS SAFE HERE
-----------------------
BFCL's OSS handlers do NOT format prompts from the checkpoint's tokenizer
chat_template. Each handler subclass implements `_format_prompt(messages, function)`,
and the handler is selected by the `--model` string (verified against
bfcl_eval/model_handler/local_inference/base_oss_handler.py, which raises
NotImplementedError and requires subclasses to implement prompt formatting).

Consequently: if the baseline run and the fine-tuned run pass the IDENTICAL
`--model` handler, the prompt construction is byte-identical by construction; only
`--local-model-path` swaps the weights. We additionally fingerprint the tokenizer
to catch a checkpoint that silently changed the vocab / special tokens (which
would alter tokenization even with an identical prompt string).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

VLLM = "vllm"
SGLANG = "sglang"
VALID_BACKENDS = (VLLM, SGLANG)


def prompt_mode(bfcl_model: str | None) -> str:
    """The BFCL prompting regime implied by the handler name. BFCL's "-FC" suffix is
    the native function-calling handler; anything else is prompt mode. Recorded so
    baseline and fine-tuned runs are provably in the same regime."""
    return "fc" if bfcl_model and bfcl_model.endswith("-FC") else "prompt"


@dataclass
class RunSpec:
    bfcl_model: str | None       # the --model handler; DEFINES the prompt format
    hf_repo_id: str | None       # off-the-shelf HF weights (baseline) / base of a checkpoint
    test_category: str
    backend: str
    local_model_path: str | None = None
    num_gpus: int = 1
    gpu_memory_utilization: float = 0.9
    num_threads: int | None = None


def _run(cmd: list[str]) -> None:
    print("[bfcl-runner] $ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")


def _check_backend(backend: str) -> None:
    if backend not in VALID_BACKENDS:
        raise ValueError(f"backend must be one of {VALID_BACKENDS}, got {backend!r}")


def _preflight_handler(spec: RunSpec) -> None:
    """Refuse to run a model that has no registered BFCL handler — scoring it would
    use a mismatched/undefined prompt format."""
    if not spec.bfcl_model:
        raise RuntimeError(
            f"No BFCL handler set for {spec.hf_repo_id!r} (bfcl_model is null / "
            f"bfcl_supported: false). This model is not in BFCL's SUPPORTED_MODELS; "
            f"register a custom handler (implement _format_prompt) before evaluating. "
            f"Refusing to score under an undefined prompt format."
        )


def generate(spec: RunSpec) -> list[str]:
    """Run `bfcl generate` for the full requested category set."""
    _check_backend(spec.backend)
    _preflight_handler(spec)
    cmd = [
        "bfcl", "generate",
        "--model", spec.bfcl_model,
        "--test-category", spec.test_category,
        "--backend", spec.backend,
        "--num-gpus", str(spec.num_gpus),
        "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
    ]
    if spec.local_model_path:
        cmd += ["--local-model-path", spec.local_model_path]
    if spec.num_threads:
        cmd += ["--num-threads", str(spec.num_threads)]
    _run(cmd)
    return cmd


def evaluate(spec: RunSpec) -> list[str]:
    """Run `bfcl evaluate`. NOTE: --partial-eval is intentionally NEVER passed, so
    scoring is over the full set; missing categories surface as an error in
    scores.parse_multi_turn rather than a silent subset."""
    _preflight_handler(spec)
    cmd = [
        "bfcl", "evaluate",
        "--model", spec.bfcl_model,
        "--test-category", spec.test_category,
    ]
    _run(cmd)
    return cmd


# ----------------------------- parity guarantees -----------------------------

def write_manifest(path: str | Path, spec: RunSpec, bfcl_version: str) -> dict:
    """Persist what produced a score, so a later fine-tuned run can prove parity."""
    manifest = {
        "bfcl_model": spec.bfcl_model,
        "prompt_mode": prompt_mode(spec.bfcl_model),
        "test_category": spec.test_category,
        "backend": spec.backend,
        "bfcl_version": bfcl_version,
        "is_baseline": spec.local_model_path is None,
        "hf_repo_id": spec.hf_repo_id,
        "local_model_path": spec.local_model_path,
    }
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def assert_handler_parity(baseline_manifest_path: str | Path, spec: RunSpec) -> None:
    """Fail loudly if a fine-tuned run would use a different handler (=> different
    prompt format) or category than the baseline it claims to improve on."""
    m = json.loads(Path(baseline_manifest_path).read_text(encoding="utf-8"))
    if m.get("bfcl_model") != spec.bfcl_model:
        raise RuntimeError(
            f"PROMPT-PARITY VIOLATION: baseline was scored with handler "
            f"{m.get('bfcl_model')!r} but this run uses {spec.bfcl_model!r}. The "
            f"before/after comparison would measure the prompt format, not training."
        )
    if m.get("test_category") != spec.test_category:
        raise RuntimeError(
            f"CATEGORY-PARITY VIOLATION: baseline {m.get('test_category')!r} vs "
            f"{spec.test_category!r}."
        )


def assert_tokenizer_parity(hf_repo_id: str, local_model_path: str) -> None:
    """Fail loudly if the fine-tuned checkpoint's tokenizer diverged from the base.
    A changed vocab / special tokens alters tokenization even when the prompt
    string is identical, which would invalidate the comparison."""
    from transformers import AutoTokenizer  # heavy import; only when actually checking

    base = AutoTokenizer.from_pretrained(hf_repo_id)
    ft = AutoTokenizer.from_pretrained(local_model_path)

    def fingerprint(tok) -> dict:
        return {
            "vocab_size": tok.vocab_size,
            "len": len(tok),
            "special_tokens_map": tok.special_tokens_map,
            "added_tokens": sorted(tok.get_added_vocab().keys()),
        }

    b, f = fingerprint(base), fingerprint(ft)
    if b != f:
        diffs = {k: (b[k], f[k]) for k in b if b[k] != f[k]}
        raise RuntimeError(
            f"TOKENIZER-PARITY VIOLATION between base {hf_repo_id!r} and checkpoint "
            f"{local_model_path!r}: {diffs}. A changed tokenizer breaks the "
            f"before/after comparison — keep the base tokenizer untouched during "
            f"fine-tuning, or re-baseline with the new tokenizer."
        )
