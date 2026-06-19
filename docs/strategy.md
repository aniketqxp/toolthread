# Strategy: beating the multi-turn cliff at 1.7B

How a ~1.7B model can reach a respectable BFCL v3 **multi-turn** score — the
capability that collapses fastest as parameters shrink — and what the staged
plan to get there is. This is the project's research thesis made concrete; the
eval harness in this repo measures progress against it.

## Goal

Take `Qwen/Qwen3-1.7B` from its off-the-shelf multi-turn score into a band that
**beats the off-the-shelf 4B** and every sub-2B model on the board. Matching
FunReason-MT-4B's 56.5 at 1.7B is not the promise; proving the
parameter-efficiency frontier is.

## The cliff (BFCL v3 multi-turn)

| Model | Params | Multi-turn | Note |
|---|---:|---:|---|
| xLAM-2-1b-fc-r | 1B | 8.4 | purpose-built for tools, still craters |
| Qwen3-1.7B | 1.7B | 16.9 | our base (off-the-shelf, prompt mode) |
| Qwen3-4B-Instruct-2507 | 4B | 15.75 | FunReason-MT's reported baseline |
| Qwen3-4B (thinking) | 4B | 35.25 | reasoning mode roughly doubles the instruct base |
| xLAM-2-3b-fc-r | 3B | 55.6 | trained |
| EGPO-4B | 4B | 56.25 | trained (GRPO) |
| FunReason-MT-4B | 4B | 56.5 | trained (SFT + RL) |

The decisive observation is not the low end — it is that the **4B base is also at
the bottom (15.75)**. Size is not what separates the mid-50s from the teens.
Training is. Qwen3-1.7B already doubles xLAM's 1B (16.9 vs 8.4); the difference
is reasoning capability, which is the lever this plan leans on.

## How FunReason-MT actually won

FunReason-MT raised Qwen3-4B from 15.75 -> 46.90 (SFT) -> 56.50 (RL). Its
method is a three-phase data-synthesis pipeline:

1. **Environment-API Graph Interactions** — model tool interdependencies as a
   graph; a directed sampler walks legal tool sequences toward a target tool,
   collecting execution traces with real environment feedback.
2. **Advanced Tool-Query Synthesis** — reverse a trace into a hard query: one
   agent abstracts the multi-step trace into a single "advanced tool," another
   writes a query that *requires* it.
3. **Guided Iterative Chain** — a reasoning agent produces CoT; it is validated
   against ground-truth calls; on failure a critiquing agent gives feedback and
   it retries up to K_max. Only samples that pass are kept.

**The key fact for us: their contribution is, in practice, a dataset, and it is
released.** The expensive pipeline already ran. That reframes the project from
"build a data factory" to "distill a 4B-grade training signal into 1.7B, then
reclaim the capacity gap with RL."

## The recipe is over-determined

Three independent teams converge on the same answer:

| Team | Base | Method | Multi-turn |
|---|---|---|---|
| FunReason-MT | Qwen3-4B | SFT on CoT data -> RL | 15.75 -> 46.9 -> 56.5 |
| EGPO (2508.05118) | Qwen3-4B | GRPO + binary reward + CoT-entropy bonus | -> 56.25 |
| xLAM-2 (APIGen-MT) | 1B-70B | SFT on verified multi-turn data | 3B -> 55.6 |

Pattern: **SFT installs the format and reasoning behavior (most of the gain);
GRPO with a binary, executable correctness reward adds the last ~10 points.**
BFCL multi-turn is executable, so ground-truth state checks give a verifiable
reward for free — no reward model needed. That is the ideal RL setup for a small
model.

## Plan

### Stage 0 — Baseline (current next step)
Run the harness on off-the-shelf `Qwen/Qwen3-1.7B-FC`. Establishes the honest
number to beat and the parity manifest every later run is checked against.
Expected ~12-17.

### Stage 1 — Distillation SFT
Fine-tune Qwen3-1.7B on the released FunReason-MT trajectories, mixed with
APIGen-MT for tool-format diversity. Largest single lever (FunReason's own SFT
did 15.75 -> 46.9; a 1.7B captures less, but most of our gain lives here).

### Stage 2 — GRPO RL
GRPO with a binary executable reward on multi-turn rollouts. Most
parameter-efficient gain; where a small model reclaims capability. This is the
hardest piece to stand up (multi-turn rollouts against a live environment).

### Stage 3 — Reasoning edge (differentiator)
FunReason and EGPO both show CoT is the engine of multi-turn performance. Qwen3
has a native thinking mode xLAM's 1B lacks — the likely reason it doubles xLAM
at the same size. Experiment: preserve/exploit reasoning rather than let the FC
handler suppress it.

## Target

A 1.7B in the **35-45** multi-turn band beats the off-the-shelf 4B (35.25
thinking / 15.75 instruct) and every sub-2B model. That is the result: the
thesis that 4B is not required, demonstrated.

## Data assets

| Asset | Size | License | Use |
|---|---|---|---|
| `Bingguang/FunReason-MT` (dataset) | ~16K multi-turn, CoT + tool traces + injected error responses | Apache-2.0 | Stage 1 primary |
| `Salesforce/APIGen-MT-5k` | 5K verified trajectories | check (likely CC-BY-NC) | Stage 1 diversity mix |
| `Qwen/Qwen3-1.7B` | base weights | Apache-2.0 | model under test |

> The injected erroneous environment responses in the FunReason-MT data target
> exactly the hardest BFCL categories (miss_func / miss_param), which stay
> lowest even at 4B.

## Open questions / risks

- **FC handler vs thinking mode.** Whether BFCL's `-FC` handler suppresses
  Qwen3's reasoning could cap the ceiling. Resolve early (Stage 3).
- **1.7B capacity floor.** xLAM-1B's 8.4 warns some floor may be unbreakable;
  Qwen3-1.7B's 16.9 says there is headroom.
- **GRPO infrastructure.** Multi-turn rollouts with live environment execution
  is the most complex component.
- **APIGen-MT license.** Likely non-commercial; fine for research/leaderboard,
  flag before any commercial use.

## References

- FunReason-MT: arXiv 2510.24645 — https://arxiv.org/html/2510.24645v1
- FunReason-MT data/weights: https://huggingface.co/datasets/Bingguang/FunReason-MT
- EGPO: arXiv 2508.05118 — https://arxiv.org/html/2508.05118v4
- APIGen-MT / xLAM-2: https://apigen-mt.github.io/ , https://huggingface.co/Salesforce/APIGen-MT-5k
- Small-model BFCL eval: arXiv 2511.22138 — https://arxiv.org/pdf/2511.22138
