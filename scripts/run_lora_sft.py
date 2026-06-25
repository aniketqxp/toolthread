#!/usr/bin/env python
"""LoRA SFT on multi-turn function-calling data using TRL + PEFT.

Loads a YAML config (configs/train/lora_sft.yaml) and a preprocessed JSONL
(data/sft_train.jsonl from prep_sft_data.py).  Trains a LoRA adapter on
Qwen3-1.7B, saves the adapter, and optionally merges + pushes to HF.

Usage
-----
  # Default config, default data:
  python scripts/run_lora_sft.py

  # Override hyperparams from CLI:
  python scripts/run_lora_sft.py --lora-rank 32 --num-epochs 2 --per-device-batch-size 2

  # Custom paths:
  python scripts/run_lora_sft.py --config configs/train/lora_sft.yaml \
      --data data/sft_train.jsonl --output-dir checkpoints/run2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import config as cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA SFT training for multi-turn tool calling.")
    p.add_argument("--config", default="configs/train/lora_sft.yaml",
                   help="training config YAML")
    p.add_argument("--data", default="data/sft_train.jsonl",
                   help="preprocessed JSONL from prep_sft_data.py")
    # CLI overrides for key hyperparams
    p.add_argument("--model-id", help="override model_id from config")
    p.add_argument("--lora-rank", type=int)
    p.add_argument("--lora-alpha", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--num-epochs", type=int)
    p.add_argument("--per-device-batch-size", type=int)
    p.add_argument("--max-seq-length", type=int)
    p.add_argument("--output-dir")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-model-id")
    p.add_argument("--no-merge", action="store_true",
                   help="skip merging LoRA into base (save adapter only)")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    cfg.load_env()
    tcfg = cfg.load_yaml(args.config)

    model_id = args.model_id or tcfg["model_id"]
    lora_rank = args.lora_rank or tcfg.get("lora_rank", 16)
    lora_alpha = args.lora_alpha or tcfg.get("lora_alpha", 32)
    lora_target = tcfg.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
    lora_dropout = tcfg.get("lora_dropout", 0.05)
    lr = args.learning_rate or tcfg.get("learning_rate", 2e-4)
    num_epochs = args.num_epochs or tcfg.get("num_epochs", 3)
    batch_size = args.per_device_batch_size or tcfg.get("per_device_batch_size", 4)
    grad_accum = tcfg.get("gradient_accumulation_steps", 4)
    max_seq_len = args.max_seq_length or tcfg.get("max_seq_length", 2048)
    output_dir = args.output_dir or tcfg.get("output_dir", "checkpoints/qwen3-1.7b-lora")
    push_to_hub = args.push_to_hub or tcfg.get("push_to_hub", False)
    hub_model_id = args.hub_model_id or tcfg.get("hub_model_id")

    use_bf16 = tcfg.get("bf16", True)
    grad_ckpt = tcfg.get("gradient_checkpointing", True)
    packing = tcfg.get("packing", False)
    assistant_only = tcfg.get("assistant_only_loss", True)
    warmup_ratio = tcfg.get("warmup_ratio", 0.03)
    weight_decay = tcfg.get("weight_decay", 0.01)
    lr_scheduler = tcfg.get("lr_scheduler_type", "cosine")
    logging_steps = tcfg.get("logging_steps", 10)
    save_steps = tcfg.get("save_steps", 500)
    save_total_limit = tcfg.get("save_total_limit", 2)

    # --- load data ---
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    if not data_path.exists():
        sys.exit(f"[train] data not found: {data_path}\n"
                 f"  run: python scripts/prep_sft_data.py")

    print(f"[train] loading data from {data_path} ...", flush=True)
    raw_data = load_jsonl(data_path)
    print(f"[train] {len(raw_data)} training samples loaded")

    # --- imports (heavy, so deferred) ---
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    device_info = "CPU"
    if torch.cuda.is_available():
        device_info = torch.cuda.get_device_name(0)
        print(f"[train] GPU: {device_info}  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[train] WARNING: no GPU detected — training will be very slow")

    # --- tokenizer + model ---
    print(f"[train] loading {model_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        trust_remote_code=True,
    )

    # --- dataset ---
    train_ds = Dataset.from_list(raw_data)

    # --- LoRA config ---
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
    )

    # --- SFT config ---
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler,
        gradient_checkpointing=grad_ckpt,
        bf16=use_bf16,
        fp16=(not use_bf16),
        max_seq_length=max_seq_len,
        packing=packing,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        report_to="none",
        remove_unused_columns=False,
    )

    print(f"[train] LoRA r={lora_rank} alpha={lora_alpha} targets={lora_target}")
    print(f"[train] epochs={num_epochs} batch={batch_size} grad_accum={grad_accum} "
          f"lr={lr} max_seq={max_seq_len}")
    print(f"[train] output_dir={output_dir}")

    # --- train ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print("[train] starting training ...", flush=True)

    trainer.train()
    print("[train] training complete", flush=True)

    # --- save adapter ---
    adapter_dir = Path(output_dir) / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[train] adapter saved to {adapter_dir}")

    # --- merge LoRA into base ---
    if not args.no_merge:
        print("[train] merging LoRA into base model ...", flush=True)
        merged_dir = Path(output_dir) / "merged"
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"[train] merged model saved to {merged_dir}")

        if push_to_hub and hub_model_id:
            hf_token = cfg.get_env("HUGGINGFACE_API_TOKEN") or cfg.get_env("HF_TOKEN")
            if hf_token:
                print(f"[train] pushing merged model to {hub_model_id} ...", flush=True)
                merged_model.push_to_hub(hub_model_id, token=hf_token)
                tokenizer.push_to_hub(hub_model_id, token=hf_token)
                print(f"[train] pushed to https://huggingface.co/{hub_model_id}")
            else:
                print("[train] WARNING: push_to_hub requested but no HF token found")

    print("[train] done.")


if __name__ == "__main__":
    main()
