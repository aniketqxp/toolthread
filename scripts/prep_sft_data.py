#!/usr/bin/env python
"""Download FunReason-MT + APIGen-MT-5k and convert to TRL SFT format.

No GPU required. Writes a single shuffled JSONL to data/sft_train.jsonl with
each row containing a ``messages`` list in OpenAI/TRL chat format.

Usage
-----
  python scripts/prep_sft_data.py                    # defaults
  python scripts/prep_sft_data.py --out data/custom.jsonl --seed 42
  python scripts/prep_sft_data.py --preview 3        # inspect 3 rows, don't write
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# FunReason-MT  (Bingguang/FunReason-MT)
# ---------------------------------------------------------------------------

def _convert_funreason(data) -> list[dict]:
    """Convert FunReason-MT to TRL messages format.

    The dataset is a JSON array of conversations, where each conversation is
    itself a list of message dicts with ``role`` and ``content`` fields.
    When loaded via HF datasets, each row may be a dict wrapping that list,
    or (when loaded from raw JSON) directly a list of messages.
    """
    role_map = {
        "user": "user",
        "human": "user",
        "assistant": "assistant",
        "system": "system",
        "tool": "tool",
        "function": "tool",
        "observation": "tool",
    }
    converted = []
    for row in data:
        if isinstance(row, list):
            raw_msgs = row
        elif isinstance(row, dict):
            raw_msgs = (row.get("messages") or row.get("conversation")
                        or row.get("trajectory"))
            if raw_msgs is None:
                cols = list(row.keys())
                if len(cols) == 1:
                    raw_msgs = row[cols[0]]
                else:
                    continue
        else:
            continue

        messages = []
        for msg in raw_msgs:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "").strip().lower()
            content = msg.get("content", "")
            if not role or content is None:
                continue
            mapped = role_map.get(role, role)
            messages.append({"role": mapped, "content": str(content)})

        if len(messages) >= 2:
            converted.append({"messages": messages, "source": "funreason-mt"})

    return converted


# ---------------------------------------------------------------------------
# APIGen-MT-5k  (Salesforce/APIGen-MT-5k)
# ---------------------------------------------------------------------------

def _convert_apigen(ds) -> list[dict]:
    """Convert APIGen-MT-5k rows (ShareGPT-like) to TRL messages format.

    Schema: {conversations: [{from, value}], system, tools}
    """
    converted = []
    for row in ds:
        convos = row.get("conversations", [])
        if not convos:
            continue

        messages = []
        sys_prompt = row.get("system")
        if sys_prompt:
            messages.append({"role": "system", "content": str(sys_prompt)})

        for turn in convos:
            from_role = turn.get("from", "").strip().lower()
            value = turn.get("value", "")
            role_map = {
                "human": "user",
                "gpt": "assistant",
                "function_call": "assistant",
                "observation": "tool",
                "tool": "tool",
                "system": "system",
            }
            mapped = role_map.get(from_role, from_role)
            messages.append({"role": mapped, "content": str(value)})

        if len(messages) >= 2:
            converted.append({"messages": messages, "source": "apigen-mt-5k"})

    return converted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare SFT training data from HF datasets.")
    ap.add_argument("--out", default=str(OUT_DIR / "sft_train.jsonl"),
                    help="output JSONL path (default: data/sft_train.jsonl)")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed")
    ap.add_argument("--preview", type=int, default=0,
                    help="print N sample rows and exit (don't write)")
    ap.add_argument("--skip-funreason", action="store_true")
    ap.add_argument("--skip-apigen", action="store_true")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets  (no GPU needed)")

    hf_token = os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HF_TOKEN")

    all_samples: list[dict] = []

    if not args.skip_funreason:
        print("[prep] loading Bingguang/FunReason-MT ...", flush=True)
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("Bingguang/FunReason-MT", "bfcl_multi_turn.json",
                               repo_type="dataset", token=hf_token)
        with open(path, "r", encoding="utf-8") as f:
            fr_data = json.load(f)
        print(f"[prep] FunReason-MT: {len(fr_data)} raw conversations from JSON")
        fr_samples = _convert_funreason(fr_data)
        print(f"[prep] FunReason-MT: {len(fr_samples)} converted samples")
        all_samples.extend(fr_samples)

    if not args.skip_apigen:
        print("[prep] loading Salesforce/APIGen-MT-5k ...", flush=True)
        ag_ds = load_dataset("Salesforce/APIGen-MT-5k", split="train", token=hf_token)
        print(f"[prep] APIGen-MT-5k: {len(ag_ds)} raw rows, columns={ag_ds.column_names}")
        ag_samples = _convert_apigen(ag_ds)
        print(f"[prep] APIGen-MT-5k: {len(ag_samples)} converted samples")
        all_samples.extend(ag_samples)

    if not all_samples:
        sys.exit("[prep] no samples converted — check dataset schemas.")

    random.seed(args.seed)
    random.shuffle(all_samples)

    if args.preview:
        for i, s in enumerate(all_samples[: args.preview]):
            print(f"\n--- sample {i} (source={s['source']}) ---")
            for msg in s["messages"][:4]:
                print(f"  [{msg['role']}] {msg['content'][:120]}...")
            if len(s["messages"]) > 4:
                print(f"  ... ({len(s['messages'])} messages total)")
        print(f"\n[prep] total: {len(all_samples)} samples (preview only, nothing written)")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[prep] wrote {len(all_samples)} samples to {out_path}")

    sources = {}
    for s in all_samples:
        sources[s["source"]] = sources.get(s["source"], 0) + 1
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")

    msg_lens = [len(s["messages"]) for s in all_samples]
    print(f"  messages/sample: min={min(msg_lens)} median={sorted(msg_lens)[len(msg_lens)//2]} max={max(msg_lens)}")


if __name__ == "__main__":
    main()
