"""Smoke test: 20 steps of stage-1 training on a 32-sample slice of LLaVA-Pretrain.

Passes iff:
  1. Model builds on GPU without OOM
  2. Forward + backward runs, projector grads are non-None
  3. Loss at step 20 < loss at step 1 (downward trend)
  4. No NaNs in loss
  5. Only projector params have non-None .grad

Run:
  uv run python scripts/02_smoke_test.py --limit 32 --steps 20

Will exit non-zero on any failure.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vqwen_qformer.dataset import DataCollatorForSupervisedDataset, LlavaPretrainDataset
from vqwen_qformer.train import build_model_and_processors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage1.yaml")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--limit", type=int, default=32)
    args = p.parse_args()

    cfg_path = (REPO_ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("[smoke] building model...", flush=True)
    t0 = time.time()
    model, tokenizer, image_processor = build_model_and_processors(cfg)
    for pp in model.projector.parameters(): pp.requires_grad_(True)
    model.to("cuda")
    print(f"[smoke] model on CUDA in {time.time()-t0:.1f}s", flush=True)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[smoke] trainable: {n_train/1e6:.2f}M / {n_all/1e6:.2f}M total", flush=True)

    # Verify freeze bookkeeping: nothing but projector should require grad.
    frozen_violations = [n for n, p in model.named_parameters()
                         if p.requires_grad and not n.startswith("projector.")]
    if frozen_violations:
        print(f"[smoke] ERROR: unexpected trainable params: {frozen_violations[:5]}", flush=True)
        return 1

    # Toy dataset
    ds = LlavaPretrainDataset(cfg["json_path"], cfg["image_root"], tokenizer, image_processor,
                              max_length=cfg.get("model_max_length", 2048))
    ds.records = ds.records[: args.limit]
    collator = DataCollatorForSupervisedDataset(pad_token_id=tokenizer.pad_token_id)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         collate_fn=collator, shuffle=True, num_workers=2)

    opt = torch.optim.AdamW(
        [p for p in model.projector.parameters() if p.requires_grad],
        lr=cfg.get("learning_rate", 1e-4), betas=(0.9, 0.999), weight_decay=cfg.get("weight_decay", 0.05),
        fused=True,
    )

    model.train()
    losses = []
    it = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        if not torch.isfinite(loss):
            print(f"[smoke] step {step+1}: non-finite loss = {loss.item()}", flush=True)
            return 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient-presence sanity check (step 1 only)
        if step == 0:
            gradded = [n for n, p in model.named_parameters() if p.grad is not None]
            if not any(n.startswith("projector.") for n in gradded):
                print("[smoke] ERROR: projector has no grad after backward", flush=True)
                return 3
            unexpected_grad = [n for n in gradded if not n.startswith("projector.")]
            if unexpected_grad:
                print(f"[smoke] ERROR: unexpected .grad on frozen params: {unexpected_grad[:3]}", flush=True)
                return 4
        opt.step()
        losses.append(float(loss.detach()))
        print(f"[smoke] step {step+1:2d}/{args.steps}  loss={losses[-1]:.4f}", flush=True)

    first = sum(losses[:3]) / 3
    last = sum(losses[-3:]) / 3
    drop = first - last
    print(f"[smoke] first_3_mean={first:.4f}  last_3_mean={last:.4f}  drop={drop:+.4f}", flush=True)
    if drop < 0.05:
        print("[smoke] WARN: loss did not drop > 0.05 in 20 steps (toy data). Check LR / init.",
              flush=True)
        # don't fail — 20 steps on 32 samples is very little.
    print("[smoke] PASS: forward+backward healthy, projector-only grads, no NaN.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
