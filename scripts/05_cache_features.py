"""Precompute frozen ViT-G + Q-Former features for every unique image in a
conversation dataset and persist them to disk.

Why: vision_model + qformer + query_tokens are all frozen in our setup, so
their per-image output (shape [32, 768]) is deterministic. Computing it once
offline and loading it at training time skips the entire vision stack per step
and saves ~20% wall time on stage-2 (and any future runs that reuse the cache).

Output layout (default `--cache_dir data/stage2/qformer_cache/`):
  features.bin         # one big (N, 32, 768) bf16 tensor, torch.save
  image_to_idx.json    # {image_relpath: int idx}
  meta.json            # {"num_query_tokens": 32, "hidden_size": 768,
                        #  "dtype": "bfloat16", "source_model": "...",
                        #  "count": N}

Run on the same GPU used for training. Uses large batches since no backward.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vqwen_qformer.model import VQwenQFormerForCausalLM, build_image_processor  # noqa: E402


class ImageOnlyDataset(Dataset):
    def __init__(self, image_paths, image_root, image_processor):
        self.image_paths = image_paths
        self.image_root = Path(image_root)
        self.image_processor = image_processor

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        rel = self.image_paths[idx]
        try:
            img = Image.open(self.image_root / rel).convert("RGB")
            pv = self.image_processor(images=img, return_tensors="pt")["pixel_values"][0]
            return {"idx": idx, "pixel_values": pv, "ok": True}
        except Exception as e:
            # Placeholder on failure; index marked for post-hoc removal
            return {"idx": idx, "pixel_values": torch.zeros(3, 224, 224), "ok": False,
                    "err": f"{type(e).__name__}: {e}"}


def collate(batch):
    return {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "ok": torch.tensor([b["ok"] for b in batch], dtype=torch.bool),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json_path", required=True,
                   help="Conversation dataset (mix665k or LLaVA-Pretrain 558k)")
    p.add_argument("--image_root", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--blip2_bundle_path",
                   default=str(REPO_ROOT / "models" / "blip2-frozen"))
    p.add_argument("--batch_size", type=int, default=256,
                   help="Forward-only is memory-light; push it high")
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--limit", type=int, default=0, help="Cap for smoke; 0 = all")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(args.json_path) as f:
        records = json.load(f)
    # Collect unique image paths; skip text-only records.
    unique = {}
    for r in records:
        img = r.get("image")
        if img and img not in unique:
            unique[img] = len(unique)
    if args.limit > 0:
        unique = {k: i for i, k in enumerate(list(unique)[: args.limit])}
    image_paths = list(unique.keys())
    print(f"[cache] unique images: {len(image_paths)}", file=sys.stderr)

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    # Build a vision-only helper with the frozen submodules. We *don't* need
    # Qwen3 here, but VQwenQFormerForCausalLM loads it too. To skip that, just
    # build the two pieces directly.
    from transformers import Blip2Config, Blip2QFormerModel, Blip2VisionModel
    from torch import nn

    bundle = Path(args.blip2_bundle_path)
    blip2_cfg = Blip2Config.from_pretrained(str(bundle))
    vision_model = Blip2VisionModel(blip2_cfg.vision_config).to(dtype=dtype).cuda().eval()
    qformer = Blip2QFormerModel(blip2_cfg.qformer_config).to(dtype=dtype).cuda().eval()
    num_q = blip2_cfg.num_query_tokens
    q_hidden = blip2_cfg.qformer_config.hidden_size

    vision_model.load_state_dict(
        torch.load(bundle / "vision_model.bin", map_location="cpu", weights_only=True), strict=True)
    qformer.load_state_dict(
        torch.load(bundle / "qformer.bin", map_location="cpu", weights_only=True), strict=True)
    query_tokens = torch.load(bundle / "query_tokens.bin", map_location="cuda",
                              weights_only=True)["query_tokens"].to(dtype=dtype)

    image_processor = build_image_processor(str(bundle))

    ds = ImageOnlyDataset(image_paths, args.image_root, image_processor)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=collate, pin_memory=True, persistent_workers=True,
                        prefetch_factor=4, shuffle=False)

    out = torch.zeros(len(image_paths), num_q, q_hidden, dtype=dtype)
    failures = []
    t0 = time.time()
    with torch.no_grad():
        for b_i, batch in enumerate(loader):
            pv = batch["pixel_values"].to("cuda", dtype=dtype, non_blocking=True)
            img_out = vision_model(pv).last_hidden_state                # (B, 257, 1408)
            B = img_out.size(0)
            queries = query_tokens.expand(B, -1, -1)
            attn = torch.ones(img_out.shape[:2], dtype=torch.long, device="cuda")
            qf_out = qformer(query_embeds=queries,
                             encoder_hidden_states=img_out,
                             encoder_attention_mask=attn).last_hidden_state[:, :num_q, :]
            qf_cpu = qf_out.detach().to(dtype=dtype).cpu()
            for i, idx in enumerate(batch["idx"].tolist()):
                out[idx] = qf_cpu[i]
            ok = batch["ok"].tolist()
            for i, is_ok in enumerate(ok):
                if not is_ok:
                    failures.append(int(batch["idx"][i].item()))
            if (b_i + 1) % 10 == 0:
                done = (b_i + 1) * args.batch_size
                dt = time.time() - t0
                rate = done / dt
                eta = (len(image_paths) - done) / max(rate, 1e-6)
                print(f"[cache] {done}/{len(image_paths)}  {rate:.1f} img/s  "
                      f"eta={eta/60:.1f} min", file=sys.stderr)

    # Drop failed indices so users know to skip them (simplest: write alongside).
    torch.save(out, cache_dir / "features.bin")
    with open(cache_dir / "image_to_idx.json", "w") as f:
        json.dump(unique, f)
    with open(cache_dir / "meta.json", "w") as f:
        json.dump({"num_query_tokens": num_q, "hidden_size": q_hidden,
                   "dtype": args.dtype, "count": len(image_paths),
                   "failed_indices": failures,
                   "bundle": str(bundle)}, f, indent=2)
    print(f"[cache] wrote features ({len(image_paths)} x {num_q} x {q_hidden} {args.dtype}) "
          f"+ index to {cache_dir}  ({len(failures)} failed)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
