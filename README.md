# Toolthread

Fine-tuning a small (~1.5–3B) language model to improve **multi-turn tool / function calling**, benchmarked on the **Berkeley Function Calling Leaderboard (BFCL)**.

## Thesis & target

**Thesis.** A small open model, fine-tuned specifically for multi-turn tool use, can beat its own off-the-shelf multi-turn tool-calling performance — the capability that actually matters for agents: calling the right function, with the right arguments, across a stateful, multi-step conversation.

**Target.** Beat **the chosen base model's off-the-shelf BFCL multi-turn score.** The base model is **not locked yet** — we baseline two candidates and let the eval decide:

| Candidate | HF repo (verified live, 2026-06) | BFCL handler (`--model`) | In BFCL registry? |
|---|---|---|---|
| A | `Qwen/Qwen3-1.7B` | `Qwen/Qwen3-1.7B-FC` | ✅ yes |
| B | `HuggingFaceTB/SmolLM3-3B` | — (needs a custom handler) | ❌ not yet |

> Note: at the 1.7B size Qwen3 has **no `-Instruct` suffix** — `Qwen/Qwen3-1.7B` *is* the post-trained model (base is `Qwen/Qwen3-1.7B-Base`). SmolLM3-3B exists and is the instruct model, but is **not** in BFCL's `SUPPORTED_MODELS` yet, so it needs a registered handler before it can be scored. It is **shelved as a deliberate follow-up** until the Qwen3 pipeline is proven end-to-end; the stub and its `bfcl_supported: false` guard are kept (see `configs/model/smollm3-3b.yaml`).

**Prompt regime: FC (native function calling).** BFCL defines `FC` as the native path for models that support tool calling and `Prompt` as the workaround for those that don't. Qwen3 supports native tool calling, so its leaderboard-comparable row is the `-FC` handler (`Qwen/Qwen3-1.7B-FC`). The mode is recorded in every run manifest as `prompt_mode` and must stay identical across baseline and fine-tuned runs. (FunReason-MT, the closest prior art on Qwen3 multi-turn, does **not** disclose its BFCL mode, so exact cross-paper equivalence isn't guaranteed under either choice; locking the mode and comparing against the same-mode leaderboard row is the mitigation.)

The first task is to **measure the baseline**, so the harness runs on an off-the-shelf model with no fine-tuning.

> **Status: setup phase.** No training, data-generation, or reward code yet — only the project skeleton, a pinned environment, and the BFCL eval harness.

## What the harness does

It wraps the **official `bfcl` CLI** (`bfcl generate` → `bfcl evaluate`) — BFCL's own scorer, never a homemade one — and runs the **full official multi-turn set**.

`--test-category multi_turn` expands (verified against `bfcl_eval`'s `category_mapping` → `MULTI_TURN_CATEGORY`) to exactly:

`multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`

— i.e. the public leaderboard's **Multi-Turn** column. The harness **never passes `--partial-eval`** and **refuses to report a subset**: if any of the four score files is missing, it errors instead of averaging a partial set.

## Layout

```
configs/        YAML only — model + eval params (no code)
  model/        base-model candidates (hf id, BFCL handler, serving knobs)
  eval/         BFCL category set, backend, W&B
src/            importable code (module `config`, package `eval`)
  config.py     load YAML + read secrets from env (no hardcoding)
  eval/         the BFCL harness:
    bfcl_runner.py   thin wrapper over `bfcl generate`/`evaluate` + parity guards
    scores.py        parse BFCL's official score JSON (full-set enforced)
    wandb_logger.py  push official scores to W&B
scripts/
  run_bfcl_eval.py   entry point: model id fully parameterized via config + CLI
notebooks/
  lightning_eval.ipynb   thin runner (Lightning AI; vLLM default, sglang opt-in)
  kaggle_eval.ipynb      thin runner (Kaggle T4; vLLM only)
```

## Run a baseline

**Locally / on a GPU box:**

```bash
cp .env.example .env          # fill in HF_TOKEN, WANDB_API_KEY, BFCL_PROJECT_ROOT
pip install -r requirements.txt
pip install "bfcl-eval[oss_eval_vllm,wandb]==2026.3.23"   # GPU serving layer
pip install -e . --no-deps

python scripts/run_bfcl_eval.py \
  --model-config configs/model/qwen3-1.7b.yaml \
  --eval-config  configs/eval/bfcl_multi_turn.yaml
```

The model is fully parameterized — swap `--model-config`, or override with `--model-id` / `--bfcl-model`, without editing code.

**On the platforms:** open `notebooks/lightning_eval.ipynb` (Lightning AI, primary) or `notebooks/kaggle_eval.ipynb` (Kaggle, secondary). Each clones this repo, installs the pinned env, and runs the baseline.

## Prompt / template parity (baseline vs fine-tuned)

A before/after comparison is only valid if the baseline and the fine-tuned model are scored under an **identical** prompt format. This is enforced, not hoped for:

- BFCL's OSS handlers do **not** format prompts from the checkpoint's tokenizer `chat_template`; each handler implements `_format_prompt`, selected by the `--model` string. So reusing the **same `bfcl_model` handler** for both runs makes prompt construction byte-identical by construction; only `--local-model-path` swaps the weights.
- `bfcl_runner.assert_handler_parity(...)` compares a fine-tuned run's handler/category against the recorded baseline manifest and **raises** on any mismatch.
- `bfcl_runner.assert_tokenizer_parity(...)` fingerprints the base vs checkpoint tokenizer (vocab size, length, special tokens, added tokens) and **raises** if they diverge — a changed tokenizer would alter tokenization even with identical prompt text.

Run a fine-tuned checkpoint against its baseline:

```bash
python scripts/run_bfcl_eval.py \
  --model-config configs/model/qwen3-1.7b.yaml \
  --eval-config  configs/eval/bfcl_multi_turn.yaml \
  --local-model-path /path/to/merged \
  --baseline-manifest runs/Qwen_Qwen3-1.7B-FC.baseline.json
```

## Backends

- **vLLM** — default everywhere; works on Kaggle's T4 (SM75) and all newer GPUs.
- **sglang** — faster for multi-turn but requires **SM80+ (Ampere+)**; available on Lightning's L4 / A10G / A100. Select with `--backend sglang` (after installing the sglang extra).

The backend is recorded in both the W&B run config and the run manifest, so every score is attributable to the backend that produced it.

## Secrets / env

Loaded from `.env` (gitignored) via `python-dotenv`, or from the platform secret manager. Never hardcoded. See `.env.example`.

| Var | Purpose | Read by |
|---|---|---|
| `HF_TOKEN` | pull models; later push adapters | `huggingface_hub` / `transformers` |
| `WANDB_API_KEY` | eval/run logging | `wandb` |
| `BFCL_PROJECT_ROOT` | where BFCL writes responses + scores (**required** by pip-installed bfcl-eval) | BFCL CLI |
| `HF_HOME` *(optional)* | pin model cache to fast/ephemeral disk | `huggingface_hub` |

## Environment

`requirements.txt` pins a stable, platform-agnostic layer exactly. The GPU serving layer (`torch` / `vllm` | `sglang` / `transformers` / `huggingface_hub`) is installed via the `bfcl-eval` extra so vLLM dictates a compatible torch, then frozen into a **per-platform** `requirements.lock.txt` after the first clean GPU install. On Kaggle, do **not** reinstall torch (keep the image's CUDA-matched build). `numpy` / `pyarrow` are left unpinned (platform base images own them).
